# `data/` directory

This directory holds **runtime, user-generated** state for the US Tax & Legal RAG
application. It is intentionally **git-ignored** (see `../.gitignore`) — only this
`README.md` is tracked.

## What lives here

When you run the application, the following files are created automatically:

| File                | Purpose                                                  |
|---------------------|----------------------------------------------------------|
| `registry.json`     | Authoritative document registry (one record per PDF).   |
| `chunks.json`       | Authoritative chunk collection (one record per chunk).   |
| `faiss.index`       | (Optional, if persisted) FAISS `IndexFlatIP` snapshot.   |
| `meta.json`         | (Optional) metadata about the FAISS snapshot.             |
| `app.log`           | Application log file (secret-filtered).                  |

## Why it is git-ignored

* **Privacy**: Uploaded PDFs are user content. Their parsed text and derived
  chunks should not be checked into source control.
* **Reproducibility**: The authoritative state can always be rebuilt from the
  uploaded PDFs by re-running ingestion. Persisting it just saves time.
* **Secrets safety**: Although secrets are never persisted in `data/`, keeping
  this directory out of git is defense-in-depth.

## How to reset

To completely reset the knowledge base:

```bash
# from project root
rm -f data/registry.json data/chunks.json data/faiss.index data/meta.json data/app.log
```

Then restart Streamlit. The application will rebuild the indexes from scratch
the next time you upload PDFs.

## Recovery from corruption

If FAISS or BM25 ever becomes corrupted or fails to load, the application will
**automatically rebuild them from `registry.json` + `chunks.json`** — the
authoritative source of truth. You do not need to delete anything; just
restart the app.

If `registry.json` itself is corrupted, the application will start with an
empty knowledge base. Re-upload your PDFs to rebuild.
