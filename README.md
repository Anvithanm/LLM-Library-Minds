# LLM Library Minds: Question Answering Over 20 Classic Novels

Two ways to make an LLM answer questions about books, built and measured side by side:

1. **RAG (Retrieval-Augmented Generation):** search the books for the right passages, then answer from them with citations.
2. **Fine-tuning:** train Mistral-7B on the book text with LoRA.

The RAG system comes with a full evaluation on held-out questions. It measures retrieval, answers and refusals separately, reports confidence intervals, and validates the LLM judge against human labels.

### RAG results at a glance

| | |
|---|---|
| Answer accuracy (held-out questions) | **67%** |
| Accuracy when the right passage is retrieved | **83%** |
| Unanswerable questions handled correctly | **93%** |
| Right passage in the top 4 / right book in the top 5 | **71% / 98%** |
| MRR / nDCG@10 | **0.61 / 0.65** |
| LLM judge vs human labels (Cohen's kappa, 30 answers) | **1.0** |
| Median latency | **4.5 s** |

---

## Contents

- [Dataset](#dataset)
- [Approach 1: RAG](#approach-1-rag)
- [How the RAG system is evaluated](#how-the-rag-system-is-evaluated)
- [RAG results](#rag-results)
- [Approach 2: Fine-tuning](#approach-2-fine-tuning)
- [RAG vs fine-tuning](#rag-vs-fine-tuning)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Known limitations](#known-limitations)

---

## Dataset

**Source:** [IsmaelMousa/books](https://huggingface.co/datasets/IsmaelMousa/books) on Hugging Face (first 20 books of the train split).

- **Size:** about 2.6 million words in total; books range from 8,600 to 569,000 words (median about 105,000).
- **Fields:** `title`, `author`, `category`, and the full text in `EN`.
- **Books:** *Robinson Crusoe*, *The Adventures of Tom Sawyer*, *Treasure Island*, *Little Women*, *Peter Pan*, *Alice in Wonderland*, *Anna Karenina*, *Great Expectations*, *Moby Dick*, *The Adventures of Sherlock Holmes*, *The Picture of Dorian Gray*, *Wuthering Heights*, *The Wonderful Wizard of Oz*, *A Tale of Two Cities*, *Les Misérables*, *Pride and Prejudice*, and four more.
- **Label quirks:** some titles don't match the text. "The Mysterious Island" (Anthony Hope) is not Verne's novel, "The Life of Julius Caesar" (Melville) is *Bartleby, the Scrivener*, and "The Lord of the Rings" (Tolkien) is Machiavelli's *The Prince*. Apostrophes are stripped ("won t"). The system answers using the dataset's labels.

---

## Approach 1: RAG

Code: [RAG-llm-knowledge-integration/](RAG-llm-knowledge-integration/)

### Architecture

```
OFFLINE
  20 books → 19,899 chunks (1,000 chars, 200 overlap, character positions kept)
           → "Book: <title> by <author>" header on each chunk
           → BGE-small embeddings in FAISS  +  BM25 keyword index

PER QUESTION
  question
    → title routing: if the question names a book, search only that book
    → dense search (top 30)  +  BM25 search (top 30)
    → Reciprocal Rank Fusion merges the two lists
    → cross-encoder re-ranker re-orders the 30 → keep the top 4
    → gate: best re-ranker score below threshold?
          yes → "I don't know based on the available books."  (no LLM call)
          no  → Mistral-7B answers from the 4 numbered excerpts, citing them as [n]
```

### Components

| Component | What it does | Why |
|---|---|---|
| **Chunking** | 1,000-character chunks with 200 overlap, split at paragraphs and sentences; each chunk stores its start/end position | Positions let the evaluation check whether a chunk contains the answer |
| **Contextual headers** | Each chunk is indexed as `Book: <title> by <author>` + text | Mid-book chunks rarely mention the title; questions like "In *Treasure Island*, …" need something to match |
| **Dense search** | `BAAI/bge-small-en-v1.5` (384-dim) in a FAISS inner-product index, with BGE's query prefix | Finds paraphrases and meaning |
| **Keyword search** | BM25 (`rank_bm25`) with a stopword-filtered tokenizer | Finds rare names like *Kreutznaer* that small embedding models miss |
| **Reciprocal Rank Fusion** | Score = Σ 1 / (60 + rank) over both lists | Merges by rank, so BM25 and cosine scores never need a common scale |
| **Cross-encoder re-ranker** | `BAAI/bge-reranker-base` scores each (question, chunk) pair; the top 4 go to the LLM | More accurate than the embeddings, but one model call per pair, so it only sees 30 candidates |
| **Title routing** | Restricts both searches to a book named in the question | "In *Book X*, …" never pulls passages from another book |
| **Refusal gate** | Refuses when the best re-ranker score is below a threshold tuned on a dev split | Off-topic questions get "I don't know" in under a second, with no LLM call |
| **Cited generation** | Mistral-7B Instruct, chat template, greedy decoding; the prompt requires `[n]` citations, flags false premises and allows refusal | Answers are checkable and reproducible |

Every component is a flag in `RAGConfig`, so each one can be switched off and measured.

### Usage

```python
from rag_pipeline import RAGConfig, RAGPipeline, HFChatLLM, BookIndex, load_books, chunk_books

cfg = RAGConfig(abstain_threshold=0.068)   # threshold from calibrate_abstention() on the dev split
books = load_books(n_books=20)
index = BookIndex(chunk_books(books, cfg.chunk_size, cfg.chunk_overlap), cfg)

llm = HFChatLLM("mistralai/Mistral-7B-Instruct-v0.3")   # or any callable llm(messages) -> str, e.g. Ollama
rag = RAGPipeline(index, llm, cfg)

out = rag.answer("Which book has the character Robinson Kreutznaer?")
print(out["answer"])        # The character Robinson Kreutznaer is from the book "Robinson Crusoe" by Daniel Defoe [1].
print(out["citations"])     # [1]
```

The same pipeline is also available as a LangGraph graph (`retrieve → gate → generate | abstain`) via `build_langgraph(rag)`.

---

## How the RAG system is evaluated

### Test set: 104 questions, split 31 dev / 73 test

| Set | Count | What it contains |
|---|---|---|
| Hand-written | 23 | Facts, explanations, cross-book and metadata questions, each with an exact evidence quote located in the book |
| Generated | 60 | Questions written by the LLM from chunks sampled evenly across all 20 books; dropped if the quote can't be found or the question refers to "the passage" |
| Should be refused | 21 | Off-topic, books not in the index (*Frankenstein*), false premises, gibberish |

The split is stratified by question type. The refusal threshold is tuned on **dev** only, and all numbers below are on **test**.

### Metrics

A retrieved chunk counts as correct only if it covers at least half of the gold evidence span. Finding the right book is not enough.

| Layer | Metrics |
|---|---|
| **Retrieval** | Hit@1/3/5/10, MRR, nDCG@10, book-level Hit@k, gate AUROC (answerable vs unanswerable) |
| **Answers** | Accuracy (LLM judge, 4 or 5 out of 5 against the reference), token F1, semantic similarity, faithfulness (claim-by-claim check against the passages), citation validity and precision |
| **Refusals** | False-refusal rate on answerable questions, share of unanswerable questions handled correctly |
| **System** | End-to-end success, latency p50/p95 per stage, failure label per question (retrieval miss, hallucination, generation error, wrong refusal) |
| **Trust** | 95% bootstrap confidence intervals; human labels on 30 answers to validate the judge (Cohen's kappa) |

---

## RAG results

Run on the 73 test questions (58 answerable, 15 unanswerable), with Mistral-7B served locally by Ollama.

### End to end

| Metric | Result | 95% CI |
|---|---|---|
| Handled correctly, all questions | 72.6% (53/73) | 62–82% |
| Answer accuracy | 67.2% (39/58) | 57–79% |
| Accuracy when the right passage was retrieved | 83% (34/41) | |
| Right passage among the 4 sent to the LLM | 70.7% | 59–81% |
| Cited passage contains the answer | 70.6% | |
| Answers with valid citations | 100% | |
| Answerable questions wrongly refused | 12.1% | 5–21% |
| Unanswerable questions handled correctly | 93.3% (14/15) | 80–100% |
| Latency, median / p95 | 4.5 s / 7.5 s | |

Of the median latency, generation takes about 4.0 s, re-ranking 0.65 s and search 0.13 s on a MacBook. Gate refusals return in 0.5–1.1 s.

### What each retrieval component contributes

Each row adds one component (73 test questions):

| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| Dense search only (BGE-small) | 0.28 | 0.50 | 0.59 | 0.40 | 0.45 |
| + contextual headers | 0.33 | 0.50 | 0.57 | 0.43 | 0.48 |
| BM25 only (with headers) | 0.43 | 0.57 | 0.62 | 0.51 | 0.55 |
| + hybrid (dense + BM25, RRF) | 0.40 | 0.64 | 0.66 | 0.53 | 0.57 |
| + cross-encoder re-ranker | **0.52** | **0.71** | **0.72** | **0.62** | **0.65** |
| + title routing (full system) | 0.52 | 0.71 | 0.71 | 0.61 | 0.65 |

- **The re-ranker is the most valuable component:** Hit@1 rises from 0.40 to 0.52. It is also the slowest retrieval step.
- **Hybrid search helps the top 3–5** more than the top 1.
- **Title routing has no measurable effect** here, but guarantees book-specific questions stay in their book.
- **Gate AUROC is 0.89:** the re-ranker score separates answerable from unanswerable questions well, but not perfectly.

### Where it fails

Of 20 failed questions, 7 were broken test questions (too vague, or with a wrong reference answer). Of the 13 real failures:

| Cause | Count |
|---|---|
| Retrieval missed the passage | 9 |
| Refused although the right passage was retrieved | 2 |
| Generation error | 1 |
| Answered an unanswerable question | 1 |

**Retrieval is the bottleneck.** Two examples:
- *"What name does the narrator of Moby Dick ask to be called?"* The question shares almost no words with "Call me Ishmael", so the right chunk ranked 250th by meaning and 1,449th by keywords.
- *"How does A Tale of Two Cities describe the period in its opening line?"* Both searches found the right chunk, but the re-ranker placed it 6th, just outside the top 4.

### Judge validation

The LLM judge agreed with hand labels on all 30 checked answers (Cohen's kappa = 1.0), so the accuracy figures are reliable. The faithfulness score is **not**: the judge rated every answer as fully supported, but manual review found unsupported claims in at least 2 of 51 answers. A different judge model is needed before reporting faithfulness.

---

## Approach 2: Fine-tuning

Model card: [fine-tuning-exp-LLM/README.md](fine-tuning-exp-LLM/README.md) · [W&B training run](https://wandb.ai/anvithanm22-northeastern-university/llm-finetuning-books/runs/gxtqvo6g)

### Pipeline

```
Books dataset
    ↓
Instruction-format training examples from book passages
    ↓
LoRA fine-tuning of Mistral-7B-Instruct-v0.3 (4-bit, TRL SFTTrainer)
    ↓
Fine-tuned model → answer
```

### Training configuration

```python
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,      # effective batch size 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    weight_decay=0.01,
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    dataset_text_field="text",
    max_length=512,
    report_to="wandb",
)
```

### Training data format

```
### Instruction:
Summarize the following passage from 'Book Title' by Author:

### Input:
[Book passage text...]

### Response:
[Expected output...]
```

Training was tracked in Weights & Biases (loss curves, GPU use, hyperparameters, checkpoints): [project dashboard](https://wandb.ai/anvithanm22-northeastern-university/llm-finetuning-books).

---

## RAG vs fine-tuning

| | RAG | Fine-tuning | Better |
|---|---|---|---|
| Accuracy | 67% (LLM judge, held-out questions) | 87% | See note |
| Speed (typical answer) | 4.5 s median (MacBook, Ollama) | 2.1 s | Fine-tuning |
| Cites its sources | Yes, `[n]` citations on every answer | No | RAG |
| Says "I don't know" | Yes, 93% of unanswerable questions handled | No built-in mechanism | RAG |
| Updating the knowledge | Re-index the books (minutes) | Retrain (hours on a GPU) | RAG |
| GPU needed | No (runs on a laptop) | Yes, for training | RAG |

**Note:** the two accuracy numbers come from different test sets and scoring methods, and the latencies from different hardware. They show each approach's own performance, not a head-to-head result.

**Use RAG when** you need citations, refusals on out-of-scope questions, frequently changing content, or no training GPU.
**Use fine-tuning when** you need a specific style or tone, fast answers, and the knowledge is fixed.

---

## Project structure

```
LLM-Library-Minds/
├── RAG-llm-knowledge-integration/
│   ├── rag_pipeline.py                    # Chunking, indexes, fusion, re-ranker, routing, gate, LLM wrapper, LangGraph
│   ├── rag_evaluation.py                  # Test-set building, metrics, threshold tuning, LLM judge, reports
│   ├── RAG_v2_Improved_Evaluation.ipynb   # Runs the pipeline and the full evaluation step by step
│   ├── eval_data/
│   │   ├── curated_questions.json         # 23 hand-written questions with evidence quotes
│   │   ├── negative_questions.json        # 21 questions the system should refuse
│   │   └── synthetic_questions.json       # 60 generated questions (cached for reproducible runs)
│   └── eval_outputs/                      # Per-question results, summaries, human review labels
├── fine-tuning-exp-LLM/
│   └── README.md                          # Model card for the LoRA fine-tuned model
└── README.md
```

---

## Getting started

### Prerequisites
- Python 3.10+
- For RAG: a laptop is enough. The LLM runs through [Ollama](https://ollama.com) locally, or Hugging Face Transformers on a GPU (Colab L4 or better). Mistral-7B-Instruct on Hugging Face is gated: accept its license and set `HF_TOKEN`.
- For fine-tuning: a GPU with 16 GB+ VRAM.

### Setup

```bash
git clone https://github.com/Anvithanm/LLM-Library-Minds.git
cd LLM-Library-Minds
python -m venv .venv
source .venv/bin/activate

# RAG
pip install langchain-text-splitters faiss-cpu sentence-transformers rank_bm25 langgraph datasets accelerate pandas jupyter

# Optional: run the LLM locally with Ollama
pip install ollama
ollama pull mistral

# Fine-tuning
pip install transformers peft trl bitsandbytes wandb
```

### Run the evaluation

Open [RAG_v2_Improved_Evaluation.ipynb](RAG-llm-knowledge-integration/RAG_v2_Improved_Evaluation.ipynb) and run it top to bottom. A full run takes about 45–60 minutes on a Colab L4; most of that is the LLM judge. Set `MAX_TEST_ITEMS = 30` for a quick run. Results are saved to `eval_outputs/`.

---

## Known limitations

- **Whole-book questions** ("What is the main theme of Moby Dick?") need summarization across many chunks; top-4 retrieval isn't built for them.
- **Self-judging:** the same model generates and judges answers, which made the faithfulness judge too lenient.
- **Small test set:** with 73 test questions, one question moves a percentage by 1–4 points. Treat small differences as noise.
- **Generated questions** come from single chunks and favor simple lookups. A few are vague or have flawed reference answers.
- **Dataset labels** don't always match the text (see [Dataset](#dataset)).

### Next steps
1. Rewrite questions before searching (query expansion or HyDE) to fix vocabulary mismatches.
2. Send 6 passages to the LLM instead of 4.
3. Try newer embedding and re-ranker models.
4. Use a different model as the judge.
5. Clean up the generated questions and add more hand-written ones.

---

## Resources

**Papers**
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [C-Pack: BGE embeddings and re-rankers](https://arxiv.org/abs/2309.07597)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [FAISS: Efficient Similarity Search](https://arxiv.org/abs/1702.08734)

**Documentation**
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) · [Sentence Transformers](https://www.sbert.net/) · [FAISS](https://faiss.ai/) · [LangGraph](https://langchain-ai.github.io/langgraph/) · [PEFT](https://huggingface.co/docs/peft) · [TRL](https://huggingface.co/docs/trl) · [Weights & Biases](https://docs.wandb.ai/)

## Acknowledgments

- **Dataset:** [IsmaelMousa/books](https://huggingface.co/datasets/IsmaelMousa/books)
- **Models:** Mistral AI (Mistral-7B-Instruct), BAAI (BGE embeddings and re-ranker)
- **Frameworks:** Hugging Face, FAISS, LangChain, LangGraph, Ollama, Weights & Biases

## Citation

```bibtex
@misc{llm-library-minds,
  author    = {Anvithanm},
  title     = {LLM Library Minds: RAG vs Fine-Tuning for Question Answering Over Books},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/Anvithanm/LLM-Library-Minds}
}
```
