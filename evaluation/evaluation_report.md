# Evaluation Report — US Tax & Legal RAG

## Status

> **Evaluation not yet executed.**

The application is fully wired to evaluate the Golden Set (`evaluation/golden_set.csv`)
through the Streamlit UI (section 5 — "Golden Set Evaluation") and through the
`evaluate_golden_set()` function in `app.py`. The metrics below describe what the
application *will* compute when you click **Run evaluation** in the UI. The numbers
are not filled in here because no run was performed — per the project's strict
"do not fabricate" rule, we never invent scores.

## Dataset description

* **Golden Set file**: `evaluation/golden_set.csv`
* **Number of questions**: 30
* **Question types** (column `question_type`):
  * factual (direct fact lookup)
  * statutory (statute-specific)
  * case_law (case-specific)
  * multi_document (questions spanning more than one document)
  * cross_page (answer spans multiple pages of a single document)
  * reasoning (requires extracting legal reasoning, not just facts)
  * unsupported (intentionally outside the corpus — should trigger abstention)
* **Difficulty levels**: easy, medium, hard, abstention
* **Distribution**:
  * ~10 factual / statutory easy questions
  * ~10 medium cross-page or case-law questions
  * ~6 hard reasoning or multi-document questions
  * ~4 unsupported questions testing abstention
* **Columns**:
  * `query_id`, `question_type`, `difficulty`
  * `query`, `ground_truth_answer`
  * `source_document`, `relevant_page_numbers`
  * `expected_key_points`, `expected_abstain`
  * `retrieval_notes`

## Retrieval configuration (defaults)

| Parameter              | Value |
|-----------------------|-------|
| Embedding model        | `BAAI/bge-small-en-v1.5` (384-dim, normalized) |
| Vector index           | FAISS `IndexFlatIP` |
| Sparse retrieval       | BM25 (legal-aware tokenizer) |
| Fusion                 | Reciprocal Rank Fusion (RRF), `k = 60` |
| `DENSE_TOP_K`          | 20 |
| `BM25_TOP_K`           | 20 |
| `FINAL_TOP_K`          | 8 |
| `ABSTENTION_THRESHOLD` | 0.30 (BGE cosine) |
| `BM25_MIN_SCORE`       | 1.0 |

## Metrics implemented

The `evaluate_golden_set()` function in `app.py` computes:

### Retrieval metrics
* **Hit Rate** — fraction of queries for which at least one retrieved hit
  matches the truth document or a true page.
* **Recall@K** — fraction of true pages that appear among the retrieved pages
  of the truth document (averaged over all queries).
* **MRR** — Mean Reciprocal Rank of the first relevant hit (1/rank).

### Generation / abstention metrics
* **Abstention correctness** — count of queries where the system correctly
  abstained (or correctly chose not to abstain) against `expected_abstain`.
* **Citation correctness** — count of queries where all cited pages fall
  within the truth page set.

### Per-query table
Each query records: `hit`, `recall`, `rr`, `mode` (hybrid / bm25_only / none),
`abstained`, `expected_abstain`, `top_chunk`, `top_page`, `top_doc`.

## Results

> **Not yet executed.** Run the evaluation in the Streamlit UI (section 5) and
> fill this section in with the actual numbers reported by the application.

## Known failure cases (expected a priori)

These are expected limitations of the current configuration. They are not
fabricated results — they are honest predictions based on the design:

* **Low-corpus queries** — If a question in the golden set references a document
  that has not been uploaded, retrieval will fail (expected: hit rate drops).
* **Year-specific facts** — Questions like "What is the standard deduction for
  2025?" will trigger abstention unless the uploaded document contains that
  specific number. The golden set includes a few of these intentionally.
* **Cross-domain unsupported** — Questions asking about topics outside the
  corpus (e.g. minimum wage, NY sales tax) are marked `expected_abstain=yes`
  and should trigger the abstention path.
* **Long-document MRR** — For 100+ page PDFs, recall and MRR may be lower
  because page-level ground truth may not match the highest-scoring chunk
  when many chunks are similarly relevant.

## Interpretation guide

* **Hit Rate ≥ 80%** is the target for the easy/medium factual and statutory
  questions.
* **MRR ≥ 0.5** is a reasonable bar when the first relevant hit is in the top 2.
* **Abstention correctness** should be **4/4** for the four `expected_abstain=yes`
  questions in the current golden set.

## Limitations

* The `ABSTENTION_THRESHOLD = 0.30` default is calibrated for BGE-small on
  legal English. If you change the embedding model, chunking, or corpus,
  you MUST re-calibrate this threshold.
* The golden set references source documents by filename
  (e.g. `Miranda_v_Arizona.pdf`, `26_U.S._Tax_Code.pdf`). Make sure the PDFs
  you upload have these exact base names so the source-document matching
  works correctly.
* Answer faithfulness is measured indirectly via abstention behavior, not via
  a semantic faithfulness model. A real faithfulness evaluation would require
  an additional LLM judge, which is out of scope for this interview project.
