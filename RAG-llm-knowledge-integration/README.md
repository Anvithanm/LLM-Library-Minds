# RAG over 20 Books: Pipeline and Evaluation

Retrieval-Augmented Generation (RAG) question answering over the first 20 books of the
[`IsmaelMousa/books`](https://huggingface.co/datasets/IsmaelMousa/books) dataset, using
FAISS + BM25, a cross-encoder re-ranker, and `mistralai/Mistral-7B-Instruct-v0.3`.

| File | What it is |
|---|---|
| [RAG_Book_Based_LLM.ipynb](RAG_Book_Based_LLM.ipynb) | **v1**: the original LangChain + FAISS + LangGraph walkthrough (kept as the baseline) |
| [RAG_v2_Improved_Evaluation.ipynb](RAG_v2_Improved_Evaluation.ipynb) | **v2**: improved pipeline, plus a full evaluation of v1 vs v2 |
| [rag_pipeline.py](rag_pipeline.py) | Chunking, hybrid index, re-ranking, abstention gate, generation, LangGraph wrapper |
| [rag_evaluation.py](rag_evaluation.py) | Test-set construction, retrieval/generation/abstention metrics, LLM judge, reporting |
| [eval_data/](eval_data/) | Hand-checked gold questions, negative questions, cached synthetic questions |

Run the v2 notebook on a Colab GPU (L4 or better). It clones this repo to import the two modules.

---

## What changed from v1 to v2

| Problem in v1 | Fix in v2 |
|---|---|
| `HuggingFacePipeline` defaults to `return_full_text=True`, so every "answer" started with the system prompt and the retrieved context. This also inflated v1's faithfulness and "context used" scores, because the context was literally inside the answer. | Mistral chat template, only newly generated tokens decoded, greedy decoding (reproducible) |
| Dense MiniLM search only. Rare names (e.g. *Kreutznaer*) and exact phrases are often missed. | Hybrid search: BGE-small dense + BM25, fused with Reciprocal Rank Fusion |
| First-stage ranking only | Cross-encoder re-ranker (`BAAI/bge-reranker-base`) over 30 candidates |
| Chunks carry no hint of which book they come from, so "Which book has X?" or "In Treasure Island, ..." can't match | Each chunk is indexed as `Book: <title> by <author>` + text |
| Searches all 20 books for every question | Title routing: if the question names a book, search only that book |
| Relevance gate = hand-set L2 threshold + "question must contain what/who/how/...". It rejected valid questions such as *"which book has the character Robinson Kreutznaer?"* ("which" wasn't in the list). | A single threshold on the re-ranker score, **calibrated on a dev split** |
| No source attribution | Numbered excerpts with `[n]` citations, which are then scored |

Each change is a flag in `RAGConfig`, and `RAGConfig.baseline()` reproduces v1's retrieval, so the evaluation can measure each change separately.

---

## Evaluation methodology

v1 evaluated one question with metrics that couldn't really fail. Examples: a substring check on an answer that contained the prompt, word-overlap "faithfulness", and a hand-weighted "overall score". v2 evaluates the system the way it would be judged in practice.

### Test set (fixed, versioned, and split into dev/test)
- **Curated (23):** hand-written questions, each with an exact evidence quote checked against the book text. They cover factoid, explanatory, cross-book ("Which book has a narrator called Robinson Kreutznaer?") and metadata ("Which books are by Dickens?") questions.
- **Synthetic (~60):** chunks are sampled evenly across all 20 books. The LLM writes a question, an answer and an *exact* quote. Items are dropped if the quote can't be found in the chunk, if the question refers to "the passage", or if it's a duplicate. The quote's location becomes the gold span.
- **Negatives (21):** off-topic questions ("capital of France"), books that are in the dataset but not indexed (*Frankenstein*, *Monte Cristo*, *The Time Machine*...), false premises ("Why does Alice kill the Queen of Hearts?"), and gibberish.
- Stratified **dev/test** split: the abstention threshold is tuned on dev and every number is reported on test.

### Retrieval metrics (no LLM needed, run as a 7-configuration ablation)
- **Passage-level Hit@k, MRR, nDCG@10.** A retrieved chunk counts only if it covers the gold *evidence span*, so the scores stay comparable across chunk sizes. v1 only checked whether the right *book* appeared, which hides most retrieval failures (see results).
- **Book-level Hit@k**, for comparison with v1's metric.
- **Abstention AUROC:** how well the retrieval score separates answerable from unanswerable questions.

### End-to-end metrics
- **Answer accuracy:** LLM judge, 1-5 against the reference (4 or above counts as correct). Also token F1 and embedding similarity.
- **Faithfulness:** the answer is split into claims and each claim is checked against the retrieved context. **Hallucination rate** = share of answers with at least one unsupported claim.
- **Context recall:** was the gold evidence among the chunks given to the LLM?
- **Citation validity and precision:** do cited excerpts exist, and do they contain the evidence?
- **False-refusal rate** on answerable questions, and the share of **unanswerable questions handled correctly** (refused, or the false premise rejected).
- **End-to-end success:** share of all questions handled correctly.
- **Failure attribution** for every wrong answer: *retrieval miss*, *hallucination*, *generation error*, or *false refusal*.
- **Latency** p50/p95 per stage, and output tokens/s.
- **95% bootstrap confidence intervals** on headline metrics.
- **Judge validation:** export about 30 graded answers, label them by hand, and compute Cohen's kappa between judge and human.

---

## Results

### Retrieval ablation

Measured on the 23 curated answerable questions and 17 unanswerable ones, on the real 19,899-chunk index. Each row adds one change to the row above, except "BM25 only", which is shown for reference. The generator sees the top 4 chunks.

| Config | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Book Hit@5 | MRR | nDCG@10 | Abstain AUROC |
|---|---|---|---|---|---|---|---|---|
| v1 baseline (MiniLM, dense) | 0.30 | 0.43 | 0.48 | 0.57 | 0.91 | 0.38 | 0.40 | 0.94 |
| + BGE-small embeddings | 0.30 | 0.57 | 0.65 | 0.70 | 1.00 | 0.45 | 0.49 | 0.81 |
| + contextual headers | 0.48 | 0.70 | 0.74 | 0.74 | 1.00 | 0.58 | 0.62 | 0.93 |
| BM25 only (+headers) | 0.43 | 0.65 | 0.74 | 0.83 | 1.00 | 0.57 | 0.61 | 0.92 |
| + hybrid (dense + BM25, RRF) | 0.57 | 0.65 | 0.65 | 0.74 | 1.00 | 0.62 | 0.62 | 0.93 |
| + cross-encoder re-rank | **0.61** | **0.78** | **0.87** | **0.91** | 1.00 | **0.72** | **0.76** | **0.97** |
| + title routing (full v2) | 0.61 | 0.78 | 0.83 | 0.91 | 1.00 | 0.72 | 0.76 | 0.97 |

What the table shows:
- **v1's book-level metric hid most retrieval failures.** v1 found the right *book* 91% of the time, but the passage that answers the question only 48% of the time.
- **The biggest gains are contextual headers and the cross-encoder re-ranker.** Together they nearly double MRR (0.38 → 0.72).
- **Title routing adds nothing measurable here** once headers and re-ranking are in place: one question moved from rank 5 to 6. It stays on because it guarantees that "In *Book X*, ..." questions never pull passages from other books.
- **The re-ranker score is the best abstention signal** (AUROC 0.97). Raw BGE cosine is worse than MiniLM's (0.81), which is why v2 gates on the re-ranker instead.
- Still missed by every config: *"What name does the narrator of Moby Dick ask to be called?"* (the "Call me Ishmael" opening line).
- **Cost:** re-ranking 30 candidates took about 0.5 s per query on a laptop CPU/MPS. On a Colab GPU it's usually tens of milliseconds. Dense + BM25 alone take under 20 ms.
- With 23 questions, one question is worth about 4 points, so treat small differences as noise. The synthetic set in the notebook adds about 60 questions.

### End-to-end results
Run sections 5-6 of the v2 notebook on a GPU. They print a side-by-side table (v1 retrieval / v2 without gate / v2) with confidence intervals, a per-question-type breakdown, and a failure-attribution table. Everything is saved to `eval_outputs/`.

---

## Known limitations
- **Dataset labels:** some titles don't match the text. `"The Mysterious Island"` (Anthony Hope) is not Verne's novel, `"The Life of Julius Caesar"` (Melville) is *Bartleby, the Scrivener*, and `"The Lord of the Rings"` (Tolkien) is Machiavelli's *The Prince*. Apostrophes are stripped throughout. The system answers using the dataset's labels.
- **Self-judging:** with the default setup, Mistral judges Mistral, which is lenient. Use a different judge model and run the human spot check.
- **Synthetic questions** come from single chunks, so they favor simple lookups. Use the curated per-type breakdown for harder question types.
- **Sample size:** with about 70 test questions, differences of a few points are within the confidence intervals.
- **Whole-book questions** ("What is the main theme of Moby Dick?") need summarization or map-reduce over many chunks. Top-k retrieval isn't designed for them.
