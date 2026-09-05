# Architecture — US Tax & Legal RAG

This document explains the architecture of the US Tax & Legal RAG application
implemented in a single file: `app.py`.

## 1. Problem

US tax and legal documents are dense, citation-heavy, and unforgiving of
hallucinations. A junior tax analyst or associate product engineer needs a
trustworthy retrieval system that:

* Reads arbitrary PDFs (one or many)
* Returns evidence with **exact page citations**
* Abstains when evidence is insufficient
* Survives infrastructure failures (missing tokens, rate limits, network errors)
* Summarizes very long PDFs without truncation
* Operates entirely on **free, open-source** models

## 2. Architecture (high-level)

```
                         ┌───────────────────┐
                         │     PDF Upload    │
                         └─────────┬─────────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │   PyMuPDF    │
                            └──────┬───────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Page-Aware Legal       │
                       │ Chunking + Metadata    │
                       └───────────┬────────────┘
                                   │
                     ┌─────────────┴──────────────┐
                     │                            │
                     ▼                            ▼
              ┌──────────────┐             ┌──────────────┐
              │ BGE-small    │             │    BM25      │
              │ Embeddings   │             │ Sparse Search│
              │ (HF API /    │             │ (legal tok.) │
              │  local CPU)  │             │              │
              └──────┬───────┘             └──────┬───────┘
                     │                            │
                     ▼                            │
              ┌──────────────┐                    │
              │    FAISS     │                    │
              │ IndexFlatIP  │                    │
              └──────┬───────┘                    │
                     │                            │
                     └────────────┬───────────────┘
                                  ▼
                         ┌────────────────┐
                         │      RRF       │
                         └───────┬────────┘
                                 ▼
                       ┌────────────────────┐
                       │ Evidence Selection │
                       └──────────┬─────────┘
                                  │
                         ┌────────┴─────────┐
                         │                  │
                         ▼                  ▼
                   Qwen3-8B          Extractive
                  Hugging Face        Fallback
                (HF_LLM_TOKEN)        (always available)
                         │                  │
                         └────────┬─────────┘
                                  ▼
                       ┌────────────────────┐
                       │ Answer + Program-  │
                       │ matic Citations    │
                       └────────────────────┘
```

Long-document branch:

```
PDF
 ↓
Pages
 ↓
Legal Chunks
 ↓
Token-Aware Batches (≤ SUMMARY_BATCH_CHARS)
 ↓
MAP Summaries (preserve pp.X–Y)
 ↓
REDUCE
 ↓
Hierarchical Reduction if needed
 ↓
Final Structured Legal Summary
 ↓
Original PDF Page Citations
```

## 3. PDF ingestion

* `pymupdf` opens the PDF from in-memory bytes (`fitz.open(stream=...)`).
* For every page we keep: `document_id`, `document_name`, `sha256`,
  `page_number` (1-indexed, original PDF page), `page_text`.
* Each page's text is lightly normalized (strip NULs, collapse 3+ blank
  lines).
* Empty pages and image-only PDFs are handled without crashing — they
  simply produce no chunks and a warning in the document registry.

## 4. Parsing — error handling

`parse_pdf_bytes` **never raises**. On any exception (corrupt PDF, I/O error,
encryption) it returns `([], error_message)`. The calling transactional
ingestion code refuses to commit a parse failure if a previous valid version
existed, so a bad upload never destroys an existing valid document.

## 5. Chunking (legal-aware, page-aware)

Implemented in `chunk_pages()`. Key properties:

* Each chunk is associated with **exactly one page** (no cross-page chunks).
* Headings (Article / Section / Chapter / Title / numbered headings) are
  detected and stored on the chunk.
* Legal citation patterns (e.g. `384 U.S. 436`, `26 U.S.C. § 1`,
  `5 U.S. 137`, `No. 23-1234`) are recognized; chunk boundaries avoid
  splitting them mid-citation when possible.
* Sliding window (default 700 chars, 120 overlap) attempts to break on
  sentence boundaries; very small paragraphs are merged.
* Both tiny meaningless chunks (< 80 chars) and massive chunks (> 2000
  chars) are avoided.

## 6. Embeddings — TWO SEPARATE TOKENS

This is the central design choice that satisfies the user's requirement of
**using different HF tokens for embedding and LLM**.

* `HF_EMBEDDING_TOKEN` — used **only** for the embedding endpoint.
* `HF_LLM_TOKEN` — used **only** for the LLM endpoint.

### Why two tokens?

* **Independent rate limits** — Hugging Face's free Inference API rate-limits
  per-token per-model. Using separate tokens means heavy embedding traffic
  cannot starve the LLM, and vice versa.
* **Independent rotation / revocation** — If one token is compromised, you
  can revoke just that one without disrupting the other pipeline.
* **Independent scoping** — A read-only token issued specifically for the
  embedding model has a minimal blast radius.
* **Auditability** — Logs (which are secret-filtered) clearly show which
  model/token was used in each step.

### How it is implemented

In `app.py`:

```python
HF_EMBEDDING_TOKEN_ENV = "HF_EMBEDDING_TOKEN"
HF_LLM_TOKEN_ENV = "HF_LLM_TOKEN"

def get_embedding_token() -> str: ...
def get_llm_token() -> str: ...
```

These helpers read from environment variables first, then fall back to
Streamlit secrets. Tokens are never logged (a `_SecretFilter` in the logger
strips any token-like substring) and never sent to the browser.

### Embedding model

* **`BAAI/bge-small-en-v1.5`** (384-dim, normalized).
  * Open weights — Apache-2.0 family license.
  * Free to use via the HF Inference API.
* Vectors are L2-normalized so that `faiss.IndexFlatIP` behaves like cosine
  similarity.
* Query embeddings use the recommended BGE instruction prefix
  `"Represent this sentence for searching relevant passages: "`.

### Embedding backend

`EmbeddingBackend.initialize()` first tries the **HF Inference API** with
`HF_EMBEDDING_TOKEN`. If that fails (no token, 401, 404, 503 cold start after
retries, or any network error), and `ALLOW_LOCAL_EMBEDDING_FALLBACK=true`,
it falls back to **local `sentence-transformers`** (CPU-only torch).

This gives us three operating modes:

| Mode     | Behavior                                             |
|----------|-----------------------------------------------------|
| `remote` | HF Inference API for BGE-small (token present)      |
| `local`  | CPU sentence-transformers (fallback)                |
| `uninitialized` | No embedding path available → BM25-only mode |

### Free / open-source guarantee

* **BGE-small-en-v1.5**: open weights, freely downloadable, MIT-style license.
* **sentence-transformers / transformers / torch (CPU)**: all open source.
* **Hugging Face Inference API**: free tier, no paid GPU needed for either
  BGE-small or Qwen3-8B.

## 7. FAISS

* `faiss.IndexFlatIP` with L2-normalized embeddings → cosine similarity.
* Dimension is read from the embedding backend (not hardcoded) so the system
  stays correct if a different BGE variant is substituted.
* **Always rebuild from authoritative chunks** — never partial mutation.
  The `FaissIndex` is reset and re-added whenever the corpus changes.
* Vector → chunk_id mapping is stored as an in-memory list and validated
  (`n_vectors == n_chunks`) on every rebuild.

## 8. BM25

* `rank_bm25.BM25Okapi` over the current authoritative chunk collection.
* Legal-aware tokenizer (`_legal_tokenize`) preserves:
  * numbers and legal abbreviations
  * U.S.C. / U.S. / S.Ct. / L.Ed. / F.3d / F.Supp. / No. § ¶
  * case citations as multi-word phrases
* BM25 rebuilds whenever the corpus changes; it operates **independently**
  of embeddings, so embedding failures never break BM25.

## 9. RRF (Reciprocal Rank Fusion)

```python
rrf_score += 1 / (rrf_k + rank)   # ranks start at 1
```

* `rrf_k = 60` (configurable via env).
* Fuse by `chunk_id` so a chunk retrieved by both dense and sparse paths gets
  two contributions; a chunk retrieved by only one path gets one.
* Final results sorted descending by RRF score.

## 10. Evidence selection & abstention

After RRF fusion, we apply a conservative **abstention check**:

* If the best BGE cosine score is below `ABSTENTION_THRESHOLD` (default 0.30)
  **and** the best BM25 score is below `BM25_MIN_SCORE` (default 1.0), the
  system abstains and returns:
  > "The available documents do not contain sufficient evidence to answer
  > this question reliably."
* Thresholds are **configurable** and **documented** — the README and
  evaluation report explicitly warn that thresholds need recalibration when
  the embedding model, chunking, corpus, or retrieval config changes.

## 11. Qwen3-8B via Hugging Face Inference API

* **`Qwen/Qwen3-8B`** — Qwen3 family is open source (Apache-2.0 license).
  We call it via the HF free Inference API. **No local 8B model is downloaded.**
* Endpoint: `https://api-inference.huggingface.co/models/Qwen/Qwen3-8B`.
* Authorization: `Bearer <HF_LLM_TOKEN>`.
* The system prompt enforces evidence-only grounding:
  > "Answer ONLY from the supplied retrieved evidence. Do NOT use unsupported
  > external knowledge. If the evidence is insufficient, say so explicitly.
  > Never invent citations, page numbers, statutes, or case holdings…"
* HTTP retries handle 503 (cold start) and 429 (rate limit) with exponential
  backoff. 401 / 403 / 404 / 5xx / timeout all fall through to extractive
  fallback — the user is never shown a stack trace.

## 12. Long-document summarization (Map → Reduce → Hierarchical)

1. **Map stage** — chunks of the target document are batched by character
   budget (`SUMMARY_BATCH_CHARS = 6000`). Each batch is summarized in
   isolation, preserving a `[pp.X–Y]` reference.
2. **Reduce stage** — if the joined map summaries fit inside
   `SUMMARY_MAX_CONTEXT_CHARS = 12000`, a single LLM call produces the final
   structured summary using the canonical legal-summary headings (Parties,
   Procedural History, Issues, Relevant Law, Holding, etc.).
3. **Hierarchical reduction** — if the joined map summaries exceed the
   context budget, they are grouped (default 5 per group), each group is
   reduced, then the group summaries are reduced into the final summary.
   This allows arbitrary-length documents to be summarized without
   truncation.
4. **Page preservation** — every intermediate summary carries its source
   page range; the final summary references original PDF page numbers
   (e.g. `Judgment.pdf — pp. 14–19`), never "Summary chunk 4".
5. **Faithfulness** — the LLM is instructed to use only supplied material,
   never invent. If the LLM is unavailable, the map / reduce stages fall
   back to extractive summaries (top sentences) so the user always gets
   *something* useful.

## 13. Citation strategy

Citations are **programmatically generated** from retrieval metadata,
never from LLM output:

```text
Sources:
- Judgment.pdf — p. 12
- Tax_Code.pdf — p. 7
```

The retrieval layer already knows `document_name`, `page_number`, and
`chunk_id`. We render them after the answer; the LLM is told the sources
inside the prompt for grounding, but the citation strings shown to the user
are constructed from retrieved chunks — there is no risk of the LLM
hallucinating "page 25" if page 25 was never retrieved.

## 14. Abstention

Three abstention paths:

1. **Empty retrieval** — no chunks found at all.
2. **Insufficient evidence** — top hits below abstention thresholds.
3. **Explicit LLM-stated abstention** — the system prompt instructs Qwen3
   to reply with the abstention sentence when evidence is insufficient.

All three paths return the same standardized abstention message, so the
caller cannot tell (and shouldn't care) which path triggered it.

## 15. Error handling & graceful degradation

The application **must not crash** under any of the following conditions:

* no documents, empty PDF, malformed PDF
* embedding failure, missing `HF_EMBEDDING_TOKEN`, invalid token, 503 cold
  start, 429 rate limit, 401 / 403 / 404, network failure
* missing `HF_LLM_TOKEN`, LLM 5xx, timeout, model unavailable
* empty retrieval, insufficient evidence
* corrupted index files

Recovery is always the same:

1. Keep the **authoritative** state (`registry.json` + `chunks.json`) intact.
2. Rebuild **derived** state (FAISS, BM25) from the authoritative state.
3. If dense rebuild fails, fall back to BM25-only mode and show the
   🟡 status banner.
4. If LLM fails, fall back to extractive answer (top sentences from top
   evidence chunks) and show the 🟡 LLM-unavailable banner.

## 16. Persistence

* `data/registry.json` — authoritative document registry.
* `data/chunks.json` — authoritative chunk collection.
* FAISS is held in memory; the authoritative chunks are enough to rebuild
  it on demand. (FAISS persistence can be added later, but is not strictly
  required because rebuilding is cheap for typical corpora.)

The application **never** persists an index it cannot later map to chunks.
If a persisted FAISS file is missing or its metadata doesn't match the
authoritative chunks, the application logs:

> "Index inconsistency detected. Rebuilding indexes from authoritative
> document metadata."

…and rebuilds from scratch.

## 17. Evaluation

Implemented in `evaluate_golden_set()`:

* **Retrieval**: Hit Rate, Recall@K, MRR.
* **Abstention**: count of queries where the system's decision to abstain
  matched `expected_abstain`.
* **Citation correctness**: count of queries where all cited pages fall
  within the truth page set.

The function reuses the same `hybrid_retrieve()` call for retrieval and
abstention — never re-embedding the same query twice.

## 18. Deployment

* **Python**: 3.12 (target; the code uses no 3.13+ features).
* **Local dev**: `streamlit run app.py`.
* **Deployment platforms**: any platform that runs Streamlit and exposes
  outbound HTTPS to `api-inference.huggingface.co` (Hugging Face, Streamlit
  Community Cloud, a small VM, Docker, etc.).
* **No CUDA required**: BGE-small runs on CPU when the HF embedding API is
  unavailable. Qwen3-8B is never downloaded locally.
* **Persistence caveat**: If you deploy on a platform with **ephemeral
  storage** (e.g. Streamlit Community Cloud free tier), the `data/`
  directory is reset on app restart. The application handles this
  gracefully — the user just re-uploads PDFs. If you need durable state,
  deploy on a VM / container with persistent volume.

## 19. Security

* `HF_EMBEDDING_TOKEN` and `HF_LLM_TOKEN` are read **only** from env or
  Streamlit secrets — never hardcoded.
* Tokens are never sent to the browser.
* The logger installs a `_SecretFilter` that strips any `hf_…` token or
  `Bearer …` header before the log line is written.
* `.env` is git-ignored; `.env.example` contains no real tokens.
* The README, evaluation report, and this architecture document contain no
  tokens.
* All errors shown to the user are **type + status code** only — never the
  response body of an auth-failed request.

## 20. Limitations

* Free-tier HF Inference API can be slow (cold starts on Qwen3-8B can take
  10–30s for the first request).
* BM25 is exact-match; semantic paraphrases (e.g. "What is the standard
  deduction?" vs. "Define standard deduction") rely on dense retrieval.
* The abstention threshold needs recalibration when the corpus / chunking /
  embedding model changes.
* Multi-document summarization is not implemented; per-document summarization
  is.
* No vector store is used (FAISS in-memory); this scales comfortably to
  tens of thousands of chunks on a laptop, but not millions. For million-
  scale corpora, swap FAISS `IndexFlatIP` for `IndexIVFPQ`.

## 21. Future improvements

* Add a vector store backend (FAISS file persistence or Qdrant local).
* Add a faithfulness evaluator (LLM judge) for the answer step.
* Add a "re-index all" button to rebuild FAISS+BM25 from authoritative
  chunks after a model swap.
* Add citation-level page verification: ensure the LLM never cites a page
  that wasn't in the retrieved evidence.
* Support OCR for image-only PDFs (PyMuPDF has `page.get_text("ocr")`).
* Add document-level deduplication across uploads (same SHA from a
  different filename).
