# REPORT - RAG V1

## Team details

- Jakub

## Decision Log

**1. Corpus**

Picked: Vellora Duo manual (`docs/vellora-duo-espresso-machine.txt`).

Why: Fictional product forces retrieval dependence. Has ~99 structured error codes (E01-E99) for fine-grained lookups. Small size (31 KB, ~45 chunks at 800 chars).

**2. Stack**

Picked: HTML/JS + FastAPI, local `all-MiniLM-L6-v2`, local ChromaDB, `gemini-3.5-flash-lite`.

Why: Aligns with course baseline (`ask_rag_google.py`). No frontend build step or embedding quota limits.

**3. Question set**

Picked: 17 questions (12 answerable, 5 unanswerable) in `eval/questions.jsonl`.

Why: Mixes exact lookups, paraphrases, sibling codes, specs, and out-of-scope queries (price, Alexa) to evaluate failure modes.

**4. Retrieval hit definition**

Picked: Character-range overlap between top-k chunks and verbatim ground-truth quotes (`eval/run_eval.py 800 10 5`).

Why: Stable across changing chunk sizes, unlike chunk IDs.

**5. Fallback behavior**

Picked: No similarity threshold; relies on prompt instructions for the LLM to flag unanswerable queries.

Why: Keeps V1 naive as a baseline.

**6. Schema & Citations**

Picked: Pydantic schema `{answerable, answer, citations: [int]}` using passage index numbers.

Why: Simple format prevents invalid citation IDs and rejects ungrounded answers.

**7. Reference display**

Picked: Document name, character range (start-end), and chunk text.

Why: Matches exact LLM context for fast UI verification.

**8. Parameter changes**

Picked: Persistent Chroma collections keyed by `(chunk_size, overlap)`.

Why: Prevents redundant re-embedding during parameter tweaks.

**9. Error handling**

Mapped to clear statuses: HTTP 400 (bad input/file), 404 (missing doc), 415 (unsupported format), 422 (invalid params), 502 (LLM down), `invalid_llm_output`, or `not_in_documents`.

**Baseline V1:** 800 chars / 10% overlap / top-5: **hit rate 0.75 (9/12)**.

## Known Failures

**1.** Question: "What does error E47 mean and how do I fix it?" (q01)

Observed: `not_in_documents`. Target chunk (#26) missed due to high structural similarity across error entries.

**2.** Question: "The machine shows E88, how many coffees has it made?" (q02)

Observed: `not_in_documents`. Target chunk (#36-#37) missed in favor of earlier error entries.

**3.** Question: "What is the capacity of the water tank?" (q08)

Observed: `not_in_documents`. Spec chunk (#2) outranked by error entries mentioning water tank issues (e.g., E01).

**4.** Question: "What is the price of the Vellora Duo?"

Observed: `not_in_documents`. Correctly identified as unanswerable.

**Takeaway:** Vector similarity struggles with exact codes and short specs vs. dense keyword overlaps. V2 focus: hybrid/keyword search and structural chunking (one chunk per error code).