"""
Realistic evaluation for the book RAG system.

The v1 notebook evaluated a single question with metrics that could not fail in
a meaningful way: substring checks against an answer that contained the whole
prompt, word-overlap "faithfulness" and a hand-weighted composite score. This
module evaluates the system the way it would be judged in production:

Test set (fixed, versioned, reused across runs)
  * curated   - hand-written questions with exact evidence quotes from the books
  * synthetic - questions generated from randomly sampled chunks of every book,
                with the supporting quote located in the text (gold span)
  * negatives - out-of-domain, books not in the library, false premises, gibberish
  * dev/test split: thresholds are tuned on dev and reported on test only

Retrieval (no LLM needed, cheap enough to ablate many configurations)
  * Hit@k, MRR@10, nDCG@10 at the *passage* level: a retrieved chunk counts only if
    it overlaps the gold evidence span, which works for any chunk size
  * Book-level Hit@k, and AUROC of the abstention score (answerable vs negatives)

Generation / end-to-end (LLM-as-judge plus deterministic metrics)
  * Answer correctness vs reference (judge 1-5), token F1, embedding similarity
  * Faithfulness: answer decomposed into claims, each checked against the context
  * Hallucination rate, citation validity and citation precision
  * Abstention: correct-refusal rate on negatives, false-refusal rate on answerable
  * End-to-end success rate (every question handled correctly), per question type
  * Failure attribution: retrieval miss vs. generation error vs. hallucination
  * Latency p50/p95 per stage, output tokens/s
  * 95% bootstrap confidence intervals, so small differences are not over-read
  * Export for human spot checks + judge/human agreement (Cohen's kappa)
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from rag_pipeline import (
    BookIndex,
    Chunk,
    RAGConfig,
    RAGPipeline,
    chunk_books,
    format_context,
    is_abstention,
    strip_citations,
)

NEGATIVE_TYPES = {"out_of_domain", "missing_book", "false_premise", "gibberish"}
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CURATED = os.path.join(_HERE, "eval_data", "curated_questions.json")
DEFAULT_NEGATIVES = os.path.join(_HERE, "eval_data", "negative_questions.json")


# ======================================================================================
# Test-set construction
# ======================================================================================


def _ws_regex(quote: str) -> re.Pattern:
    """Pattern matching the quote's words in order, ignoring case, whitespace and
    punctuation. The dataset strips apostrophes/quotes ("won t"), while LLMs quote
    them back normally ("won't"), so exact matching would reject faithful quotes."""
    words = re.findall(r"[A-Za-z0-9]+", quote)
    if not words:
        return re.compile(r"(?!x)x")  # matches nothing
    return re.compile(r"\b" + r"[\W_]+".join(re.escape(w) for w in words) + r"\b", re.IGNORECASE)


def _best_partial_match(quote: str, text: str, min_words: int = 6):
    """Longest run of >= `min_words` consecutive quote words that appears in `text`.
    LLMs often quote most of a sentence faithfully but alter its ending."""
    words = re.findall(r"[A-Za-z0-9]+", quote)
    for length in range(len(words), min_words - 1, -1):
        for i in range(len(words) - length + 1):
            m = _ws_regex(" ".join(words[i : i + length])).search(text)
            if m:
                return m
    return None


def locate_spans(book_text: str, quote: str, max_hits: int = 5) -> List[List[int]]:
    return [[m.start(), m.end()] for _, m in zip(range(max_hits), _ws_regex(quote).finditer(book_text))]


def resolve_gold(items: List[dict], books: Sequence[dict]) -> List[dict]:
    """Attach `gold_spans` [[book_id, start, end], ...] and `gold_book_ids` to each item."""
    by_title = {b["title"]: b for b in books}
    resolved = []
    for it in items:
        it = dict(it)
        it.setdefault("answerable", it.get("qtype") not in NEGATIVE_TYPES)
        if it.get("gold_spans") is None:
            spans = []
            titles = it.get("gold_titles") or ([it["title"]] if it.get("title") else [])
            for t in titles:
                if t not in by_title:
                    raise ValueError(f"{it['id']}: title {t!r} is not in the indexed books")
            if it.get("title") and it.get("evidence"):
                book = by_title[it["title"]]
                for q in it["evidence"]:
                    found = locate_spans(book["text"], q)
                    if not found:
                        raise ValueError(f"{it['id']}: evidence not found in {it['title']!r}: {q!r}")
                    spans += [[book["book_id"], s, e] for s, e in found]
            it["gold_spans"] = spans
            it["gold_book_ids"] = sorted({by_title[t]["book_id"] for t in titles})
        resolved.append(it)
    return resolved


def load_json(path: str) -> list:
    with open(path) as f:
        return json.load(f)


SYNTH_PROMPT = """Here is an excerpt from the book "{title}" by {author}:

<excerpt>
{text}
</excerpt>

Write ONE question that a reader of this book might realistically ask, which can be answered from this excerpt alone.
Requirements:
- The question must make sense without seeing the excerpt: mention the book title or a specific named character or place. Never refer to "the excerpt", "the passage" or "the text".
- Ask about a concrete fact, event, or reason (who / what / where / why / how). Avoid yes/no questions.
- Give a short answer (one sentence) and copy an exact supporting quote of 5-30 words from the excerpt.

Respond with JSON only, no other text:
{{"question": "...", "answer": "...", "evidence": "..."}}"""

_LEAKY = re.compile(r"\b(excerpt|passage|the text|this text|the author mention|according to the)\b", re.I)


def parse_json_object(text: str) -> Optional[dict]:
    """Extract the first JSON object from an LLM response."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _call(llm, messages, max_new_tokens):
    try:
        return llm(messages, max_new_tokens=max_new_tokens)
    except TypeError:
        return llm(messages)


def generate_synthetic_testset(
    books: Sequence[dict],
    chunks: Sequence[Chunk],
    llm: Callable,
    n_questions: int = 60,
    seed: int = 42,
    out_path: Optional[str] = None,
    min_chunk_chars: int = 600,
    skip_front_matter: float = 0.03,
    verbose: bool = True,
) -> List[dict]:
    """Generate QA pairs from chunks sampled evenly across books.

    The gold span is the location of the model's quoted evidence in the book, so
    retrieval relevance is independent of how the corpus is chunked. Results are
    cached to `out_path`; delete the file to regenerate. Prefer a stronger model
    than the generator under test if you have one (avoids self-preference)."""
    if out_path and os.path.exists(out_path):
        items = load_json(out_path)
        if verbose:
            print(f"Loaded {len(items)} cached synthetic questions from {out_path}")
        return items

    rng = random.Random(seed)
    book_len = {b["book_id"]: len(b["text"]) for b in books}
    pools: Dict[int, List[Chunk]] = defaultdict(list)
    for c in chunks:
        if len(c.text) >= min_chunk_chars and c.start > skip_front_matter * book_len[c.book_id]:
            pools[c.book_id].append(c)
    for p in pools.values():
        rng.shuffle(p)
    order = sorted(pools)

    items, seen, attempts, rejected = [], set(), 0, Counter()
    max_attempts = n_questions * 3
    while len(items) < n_questions and attempts < max_attempts and any(pools.values()):
        bid = order[attempts % len(order)]
        attempts += 1
        if not pools[bid]:
            continue
        c = pools[bid].pop()
        prompt = SYNTH_PROMPT.format(title=c.title, author=c.author, text=c.text.strip())
        obj = parse_json_object(_call(llm, [{"role": "user", "content": prompt}], 256))
        if not obj or not all(isinstance(obj.get(k), str) for k in ("question", "answer", "evidence")):
            rejected["bad_json"] += 1
            continue
        q, a, ev = obj["question"].strip(), obj["answer"].strip(), obj["evidence"].strip()
        reason = (
            "question_length" if not 5 <= len(q.split()) <= 40
            else "mentions_passage" if _LEAKY.search(q)
            else "duplicate" if q.lower() in seen
            else "evidence_too_short" if len(ev.split()) < 4
            else None
        )
        m = None if reason else (_ws_regex(ev).search(c.text) or _best_partial_match(ev, c.text))
        if not reason and not m:  # model paraphrased instead of quoting -> unverifiable
            reason = "evidence_not_in_chunk"
        if reason:
            rejected[reason] += 1
            continue
        seen.add(q.lower())
        items.append(
            dict(
                id=f"syn-{len(items) + 1:03d}",
                qtype="synthetic",
                source="synthetic",
                title=c.title,
                question=q,
                reference_answer=a,
                evidence=[c.text[m.start() : m.end()]],
                gold_spans=[[c.book_id, c.start + m.start(), c.start + m.end()]],
                gold_book_ids=[c.book_id],
                answerable=True,
                source_chunk=[c.book_id, c.start, c.end],
            )
        )
        if verbose and len(items) % 10 == 0:
            print(f"  generated {len(items)}/{n_questions} (attempts: {attempts})")
    if verbose:
        print(f"Synthetic test set: {len(items)} questions kept from {attempts} attempts; rejected: {dict(rejected)}")
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(items, f, indent=2)
    return items


def build_testset(
    books: Sequence[dict],
    synthetic: Optional[List[dict]] = None,
    curated_path: str = DEFAULT_CURATED,
    negatives_path: str = DEFAULT_NEGATIVES,
) -> List[dict]:
    curated = [dict(it, source="curated") for it in load_json(curated_path)]
    negatives = [dict(it, source="negative", answerable=False) for it in load_json(negatives_path)]
    return resolve_gold(curated + negatives + list(synthetic or []), books)


def split_dev_test(items: List[dict], dev_frac: float = 0.3, seed: int = 0):
    """Stratified (by qtype) split. Tune thresholds on dev, report on test."""
    rng = random.Random(seed)
    groups: Dict[str, List[dict]] = defaultdict(list)
    for it in items:
        groups[it["qtype"]].append(it)
    dev, test = [], []
    for _, g in sorted(groups.items()):
        g = g[:]
        rng.shuffle(g)
        n_dev = int(round(len(g) * dev_frac))
        if len(g) >= 2:
            n_dev = min(max(n_dev, 1), len(g) - 1)
        else:
            n_dev = 0
        dev += g[:n_dev]
        test += g[n_dev:]
    return dev, test


# ======================================================================================
# Small metric helpers
# ======================================================================================


def is_relevant(chunk: Chunk, item: dict, min_overlap: float = 0.5) -> bool:
    """A chunk is relevant if it covers >= `min_overlap` of a gold evidence span,
    or (for book-level questions without spans) comes from a gold book."""
    if item.get("gold_spans"):
        for bid, s, e in item["gold_spans"]:
            if chunk.book_id == bid:
                overlap = min(chunk.end, e) - max(chunk.start, s)
                if overlap >= min_overlap * (e - s):
                    return True
        return False
    return chunk.book_id in set(item.get("gold_book_ids") or [])


def token_f1(pred: str, ref: str) -> float:
    p = re.findall(r"[a-z0-9]+", (pred or "").lower())
    r = re.findall(r"[a-z0-9]+", (ref or "").lower())
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def auroc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    """P(score of a random positive > score of a random negative)."""
    if not pos_scores or not neg_scores:
        return float("nan")
    pos, neg = np.asarray(pos_scores), np.asarray(neg_scores)
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def bootstrap_ci(values: Iterable, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    vals = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))], float)
    if len(vals) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(axis=1)
    return float(vals.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _pct(values, q):
    vals = [v for v in values if v is not None]
    return float(np.percentile(vals, q)) if vals else float("nan")


def _mean(values):
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


# ======================================================================================
# Retrieval evaluation
# ======================================================================================


def evaluate_retrieval(index: BookIndex, items: List[dict], ks: Sequence[int] = (1, 3, 5, 10), verbose=False):
    max_k = max(ks)
    per_item = []
    for it in items:
        res = index.search(it["question"], top_k=max_k)
        rec = dict(
            id=it["id"],
            qtype=it["qtype"],
            answerable=it["answerable"],
            gate_score=res.gate_score,
            routed=bool(res.routed_book_ids),
            latency_s=sum(res.timings.values()),
        )
        if it["answerable"]:
            rel = [is_relevant(h.chunk, it) for h in res.hits]
            book_rel = [h.chunk.book_id in set(it["gold_book_ids"]) for h in res.hits]
            first = next((i + 1 for i, r in enumerate(rel) if r), None)
            n_rel_total = sum(is_relevant(c, it) for c in index.chunks if c.book_id in set(it["gold_book_ids"]))
            dcg = sum(1.0 / np.log2(i + 2) for i, r in enumerate(rel[:10]) if r)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(n_rel_total, 10))) or 1.0
            rec.update(
                gold_rank=first,
                mrr=1.0 / first if first else 0.0,
                ndcg10=dcg / idcg,
                **{f"hit@{k}": float(any(rel[:k])) for k in ks},
                **{f"book_hit@{k}": float(any(book_rel[:k])) for k in ks},
            )
        per_item.append(rec)
        if verbose:
            print(rec)
    return summarize_retrieval(per_item, ks), per_item


def summarize_retrieval(per_item: List[dict], ks=(1, 3, 5, 10)) -> dict:
    ans = [r for r in per_item if r["answerable"]]
    neg = [r for r in per_item if not r["answerable"] and r["qtype"] != "false_premise"]
    s = {f"hit@{k}": _mean(r[f"hit@{k}"] for r in ans) for k in ks}
    s.update({f"book_hit@{k}": _mean(r[f"book_hit@{k}"] for r in ans) for k in ks})
    s.update(
        mrr=_mean(r["mrr"] for r in ans),
        ndcg10=_mean(r["ndcg10"] for r in ans),
        abstain_auroc=auroc([r["gate_score"] for r in ans], [r["gate_score"] for r in neg]),
        latency_p50_ms=1000 * _pct([r["latency_s"] for r in per_item], 50),
        latency_p95_ms=1000 * _pct([r["latency_s"] for r in per_item], 95),
        n_answerable=len(ans),
        n_negative=len(neg),
    )
    return s


class IndexFactory:
    """Builds BookIndex objects for many configs while sharing models, chunks and
    embeddings, so a retrieval ablation does not re-embed the corpus needlessly."""

    def __init__(self, books: Sequence[dict], device: Optional[str] = None, cache_dir: Optional[str] = None):
        self.books, self.device, self.cache_dir = books, device, cache_dir
        self._chunks, self._emb, self._models = {}, {}, {}

    def _model(self, kind, name):
        key = (kind, name)
        if key not in self._models:
            from sentence_transformers import CrossEncoder, SentenceTransformer

            from rag_pipeline import _auto_device

            cls = SentenceTransformer if kind == "embed" else CrossEncoder
            self._models[key] = cls(name, device=_auto_device(self.device))
        return self._models[key]

    def build(self, cfg: RAGConfig) -> BookIndex:
        ckey = (cfg.chunk_size, cfg.chunk_overlap)
        if ckey not in self._chunks:
            self._chunks[ckey] = chunk_books(self.books, cfg.chunk_size, cfg.chunk_overlap)
        chunks = self._chunks[ckey]
        dense = cfg.retrieval_mode in ("dense", "hybrid")
        embedder = self._model("embed", cfg.embedding_model) if dense else None
        reranker = self._model("rerank", cfg.reranker_model) if cfg.reranker_model else None
        emb = None
        if dense:
            ekey = (cfg.embedding_model, *ckey, cfg.contextual_headers)
            path = None
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                path = os.path.join(self.cache_dir, re.sub(r"[^A-Za-z0-9_.-]", "_", "_".join(map(str, ekey))) + ".npy")
            if ekey in self._emb:
                emb = self._emb[ekey]
            elif path and os.path.exists(path):
                emb = self._emb[ekey] = np.load(path)
        index = BookIndex(chunks, cfg, embeddings=emb, embedder=embedder, reranker=reranker)
        if dense and emb is None:
            self._emb[ekey] = index.embeddings
            if path:
                np.save(path, index.embeddings)
        return index


def default_ablation_configs() -> Dict[str, RAGConfig]:
    """Each row adds one change, so the table shows what each change contributes."""
    return {
        "v1 baseline (MiniLM dense)": RAGConfig.baseline(),
        "+ BGE-small embeddings": RAGConfig.baseline(
            embedding_model="BAAI/bge-small-en-v1.5", query_prefix=RAGConfig.query_prefix
        ),
        "+ contextual headers": RAGConfig.baseline(
            embedding_model="BAAI/bge-small-en-v1.5", query_prefix=RAGConfig.query_prefix, contextual_headers=True
        ),
        "BM25 only (+headers)": RAGConfig(retrieval_mode="bm25", reranker_model=None, title_routing=False),
        "+ hybrid (dense+BM25, RRF)": RAGConfig(reranker_model=None, title_routing=False),
        "+ cross-encoder rerank": RAGConfig(title_routing=False),
        "+ title routing (full v2)": RAGConfig(),
    }


def compare_retrieval_configs(factory: IndexFactory, configs: Dict[str, RAGConfig], items: List[dict], ks=(1, 3, 5, 10)):
    import pandas as pd

    rows, details = [], {}
    for name, cfg in configs.items():
        t0 = time.perf_counter()
        index = factory.build(cfg)
        summary, per_item = evaluate_retrieval(index, items, ks)
        details[name] = per_item
        rows.append(dict(config=name, **summary))
        print(f"{name:32s} hit@1={summary['hit@1']:.2f} hit@5={summary['hit@5']:.2f} MRR={summary['mrr']:.3f} "
              f"nDCG@10={summary['ndcg10']:.3f} gate-AUROC={summary['abstain_auroc']:.2f} "
              f"({time.perf_counter() - t0:.0f}s)")
    return pd.DataFrame(rows).set_index("config"), details


# ======================================================================================
# Abstention-threshold calibration
# ======================================================================================


def calibrate_abstention(index: BookIndex, dev_items: List[dict], min_answer_coverage: float = 0.9) -> dict:
    """Choose the retrieval-score threshold below which the system abstains.

    Uses answerable vs. unanswerable dev questions (false-premise questions are
    on-topic and excluded; the LLM must handle those). Picks the threshold with
    the best balanced accuracy, subject to answering at least
    `min_answer_coverage` of the answerable questions."""
    pos = [index.search(it["question"]).gate_score for it in dev_items if it["answerable"]]
    neg = [
        index.search(it["question"]).gate_score
        for it in dev_items
        if not it["answerable"] and it["qtype"] != "false_premise"
    ]
    if not pos or not neg:
        raise ValueError("Need both answerable and unanswerable dev questions to calibrate")
    cands = sorted(set(pos + neg))
    cands = [cands[0] - 1e-6] + [(a + b) / 2 for a, b in zip(cands, cands[1:])] + [cands[-1] + 1e-6]
    table = []
    for thr in cands:
        coverage = float(np.mean([s >= thr for s in pos]))  # answerable questions we still answer
        refusal = float(np.mean([s < thr for s in neg]))  # negatives correctly refused
        table.append(dict(threshold=thr, answer_coverage=coverage, negative_refusal=refusal, balanced_acc=(coverage + refusal) / 2))
    feasible = [r for r in table if r["answer_coverage"] >= min_answer_coverage] or table
    best = max(feasible, key=lambda r: (r["balanced_acc"], -r["threshold"]))
    return dict(threshold=best["threshold"], best=best, auroc=auroc(pos, neg), table=table, pos_scores=pos, neg_scores=neg)


# ======================================================================================
# LLM-as-judge
# ======================================================================================

CORRECTNESS_PROMPT = """You are grading a question-answering system about books.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Grade the system answer's correctness against the reference answer on a 1-5 scale:
5 = fully correct and complete
4 = correct with minor omissions
3 = partially correct
2 = mostly incorrect
1 = incorrect, irrelevant, or a refusal to answer
Extra details that are not in the reference are fine as long as they do not contradict it.

Respond with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}"""

FAITHFULNESS_PROMPT = """You check whether an answer is supported by the retrieved context.

Context:
{context}

Answer: {answer}

Split the answer into its atomic factual claims (ignore citation markers like [1] and statements of uncertainty). For each claim decide whether the context directly supports it.

Respond with JSON only: {{"claims": [{{"claim": "...", "supported": true}}]}}"""

FALSE_PREMISE_PROMPT = """A user asked a question that contains a false premise.

Question: {question}
Why the premise is false: {note}
System answer: {answer}

Does the system answer go along with the false premise as if it were true? Answering "I don't know" or pointing out that the premise is wrong both count as NOT accepting it.

Respond with JSON only: {{"accepts_premise": true or false}}"""


class LLMJudge:
    """Wraps any chat callable `llm(messages) -> str` for grading.
    Using the same model as the generator is convenient but biased; a different,
    stronger judge model gives more trustworthy numbers."""

    def __init__(self, llm: Callable, max_context_chars: int = 6000):
        self.llm, self.max_context_chars = llm, max_context_chars
        self.parse_failures = 0

    def _ask(self, prompt: str, max_new_tokens: int = 256) -> Optional[dict]:
        obj = parse_json_object(_call(self.llm, [{"role": "user", "content": prompt}], max_new_tokens))
        if obj is None:
            self.parse_failures += 1
        return obj

    def correctness(self, question, reference, answer) -> Optional[float]:
        obj = self._ask(CORRECTNESS_PROMPT.format(question=question, reference=reference, answer=answer), 128)
        try:
            return float(min(5, max(1, int(obj["score"]))))
        except (TypeError, KeyError, ValueError):
            return None

    def faithfulness(self, context: str, answer: str) -> Optional[dict]:
        obj = self._ask(FAITHFULNESS_PROMPT.format(context=context[: self.max_context_chars], answer=answer), 512)
        if not obj or not isinstance(obj.get("claims"), list):
            return None
        claims = [c for c in obj["claims"] if isinstance(c, dict) and "supported" in c]
        if not claims:
            return dict(score=1.0, n_claims=0, unsupported=[])
        supported = [c for c in claims if c["supported"] is True or str(c["supported"]).lower() == "true"]
        unsupported = [c.get("claim", "") for c in claims if c not in supported]
        return dict(score=len(supported) / len(claims), n_claims=len(claims), unsupported=unsupported)

    def accepts_false_premise(self, question, note, answer) -> Optional[bool]:
        obj = self._ask(FALSE_PREMISE_PROMPT.format(question=question, note=note, answer=answer), 64)
        if not obj or "accepts_premise" not in obj:
            return None
        v = obj["accepts_premise"]
        return v is True or str(v).lower() == "true"


# ======================================================================================
# End-to-end evaluation
# ======================================================================================


def evaluate_end_to_end(
    pipeline: RAGPipeline,
    items: List[dict],
    judge: Optional[LLMJudge] = None,
    embedder=None,
    correct_threshold: float = 4.0,
    verbose: bool = True,
):
    """Run every test question through the full pipeline and score it.
    Returns (summary, per_item)."""
    per_item = []
    t_start = time.perf_counter()
    for n, it in enumerate(items, 1):
        out = pipeline.answer(it["question"])
        hits = out["hits"]
        answer = out["answer"]
        clean = strip_citations(answer)
        rec = dict(
            id=it["id"],
            source=it.get("source"),
            qtype=it["qtype"],
            answerable=it["answerable"],
            question=it["question"],
            reference=it.get("reference_answer", ""),
            answer=answer,
            abstained=out["abstained"],
            abstained_by_gate=out["abstained_by_gate"],
            gate_score=out["gate_score"],
            routed=bool(out["routed_book_ids"]),
            retrieved=[f"{h.chunk.title} @{h.chunk.start}" for h in hits],
            output_tokens=out["output_tokens"],
            **{k: v for k, v in out["timings"].items()},
        )
        n_cited = out["citations"]
        rec["citation_valid"] = all(1 <= c <= len(hits) for c in n_cited) if n_cited else None

        if it["answerable"]:
            rel = [is_relevant(h.chunk, it) for h in hits]
            rec["context_recall"] = float(any(rel))
            rec["gold_rank"] = next((i + 1 for i, r in enumerate(rel) if r), None)
            rec["cited_relevant"] = (
                float(any(rel[c - 1] for c in n_cited if 1 <= c <= len(hits))) if n_cited and not out["abstained"] else None
            )
            rec["token_f1"] = 0.0 if out["abstained"] else token_f1(clean, rec["reference"])
            if embedder is not None:
                if out["abstained"]:
                    rec["semantic_sim"] = 0.0
                else:
                    a, r = embedder.encode([clean, rec["reference"]], normalize_embeddings=True)
                    rec["semantic_sim"] = float(np.dot(a, r))
            if judge is not None:
                rec["judge_score"] = 1.0 if out["abstained"] else judge.correctness(it["question"], rec["reference"], clean)
                rec["correct"] = None if rec["judge_score"] is None else float(rec["judge_score"] >= correct_threshold)
                if not out["abstained"]:
                    f = judge.faithfulness(format_context(hits), clean)
                    if f:
                        rec["faithfulness"] = f["score"]
                        rec["hallucinated"] = float(f["score"] < 1.0)
                        rec["unsupported_claims"] = f["unsupported"]
            rec["outcome"] = _classify_answerable(rec)
        else:
            if it["qtype"] == "false_premise" and not out["abstained"] and judge is not None:
                accepts = judge.accepts_false_premise(it["question"], it.get("premise_note", ""), clean)
                rec["handled_correctly"] = None if accepts is None else float(not accepts)
            elif it["qtype"] == "false_premise" and not out["abstained"]:
                rec["handled_correctly"] = None  # needs a judge
            else:
                rec["handled_correctly"] = float(out["abstained"])
            rec["outcome"] = (
                "correct_refusal" if rec["handled_correctly"] == 1.0
                else "ungraded" if rec["handled_correctly"] is None
                else "answered_unanswerable"
            )
        per_item.append(rec)
        if verbose:
            elapsed = time.perf_counter() - t_start
            print(f"[{n}/{len(items)}] {it['id']:8s} {rec['outcome']:26s} ({elapsed:.0f}s) {it['question'][:60]}")
    return summarize_end_to_end(per_item), per_item


def _classify_answerable(rec: dict) -> str:
    if rec.get("correct") == 1.0:
        return "correct"
    if rec["abstained"]:
        return "false_refusal (gold retrieved)" if rec["context_recall"] else "false_refusal (retrieval miss)"
    if not rec["context_recall"]:
        return "wrong: retrieval miss"
    if rec.get("hallucinated") == 1.0:
        return "wrong: hallucination"
    if rec.get("correct") is None:
        return "ungraded"
    return "wrong: generation error"


def summarize_end_to_end(per_item: List[dict], n_boot: int = 1000) -> dict:
    ans = [r for r in per_item if r["answerable"]]
    answered = [r for r in ans if not r["abstained"]]
    neg = [r for r in per_item if not r["answerable"]]
    ci = lambda vals: bootstrap_ci(vals, n_boot)  # noqa: E731

    success = [r.get("correct") if r["answerable"] else r.get("handled_correctly") for r in per_item]
    s = dict(
        n_items=len(per_item),
        n_answerable=len(ans),
        n_negative=len(neg),
        end_to_end_success=ci(success),
        answer_accuracy=ci(r.get("correct") for r in ans),
        answer_correctness_1to5=_mean(r.get("judge_score") for r in ans),
        token_f1=ci(r.get("token_f1") for r in ans),
        semantic_sim=_mean(r.get("semantic_sim") for r in ans),
        faithfulness=ci(r.get("faithfulness") for r in answered),
        hallucination_rate=ci(r.get("hallucinated") for r in answered),
        context_recall=ci(r.get("context_recall") for r in ans),
        citation_rate=_mean(float(r["citation_valid"] is not None) for r in answered),
        citation_validity=_mean(r.get("citation_valid") for r in answered if r.get("citation_valid") is not None),
        citation_precision=_mean(r.get("cited_relevant") for r in answered),
        false_refusal_rate=ci(float(r["abstained"]) for r in ans),
        negative_handled_rate=ci(r.get("handled_correctly") for r in neg),
        latency_total_p50_s=_pct([r.get("total_s") for r in per_item], 50),
        latency_total_p95_s=_pct([r.get("total_s") for r in per_item], 95),
        latency_retrieve_p50_ms=1000 * _pct([r.get("retrieve_s") for r in per_item], 50),
        latency_rerank_p50_ms=1000 * _pct([r.get("rerank_s") for r in per_item], 50),
        latency_generate_p50_s=_pct([r.get("generate_s") for r in per_item], 50),
        output_tokens_per_s=_mean(
            r["output_tokens"] / r["generate_s"] for r in per_item if r.get("generate_s") and r.get("output_tokens")
        ),
    )
    by_type = {}
    for qtype in sorted({r["qtype"] for r in per_item}):
        group = [r for r in per_item if r["qtype"] == qtype]
        key = "correct" if group[0]["answerable"] else "handled_correctly"
        by_type[qtype] = dict(n=len(group), success=_mean(r.get(key) for r in group), abstained=_mean(float(r["abstained"]) for r in group))
    s["by_qtype"] = by_type
    s["outcomes"] = dict(Counter(r["outcome"] for r in per_item).most_common())
    return s


# ======================================================================================
# Reporting
# ======================================================================================

HEADLINE = [
    ("end_to_end_success", "End-to-end success (all questions)", True),
    ("answer_accuracy", "Answer accuracy (judge >= 4/5)", True),
    ("answer_correctness_1to5", "Answer correctness (1-5)", True),
    ("token_f1", "Token F1 vs reference", True),
    ("faithfulness", "Faithfulness (supported claims)", True),
    ("hallucination_rate", "Hallucination rate", False),
    ("context_recall", "Context recall (gold in top-k)", True),
    ("citation_precision", "Citation precision", True),
    ("false_refusal_rate", "False-refusal rate", False),
    ("negative_handled_rate", "Unanswerable handled correctly", True),
    ("latency_total_p50_s", "Latency p50 (s)", False),
    ("latency_total_p95_s", "Latency p95 (s)", False),
]


def _fmt(v, digits=3):
    if isinstance(v, (tuple, list)):
        m, lo, hi = v
        return "n/a" if np.isnan(m) else f"{m:.{digits}f} [{lo:.2f}, {hi:.2f}]"
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def comparison_table(summaries: Dict[str, dict]):
    """Side-by-side headline metrics (with 95% CIs) for several systems."""
    import pandas as pd

    rows = []
    for key, label, higher_better in HEADLINE:
        row = {"metric": label + (" ↑" if higher_better else " ↓")}
        for name, s in summaries.items():
            row[name] = _fmt(s.get(key))
        rows.append(row)
    return pd.DataFrame(rows).set_index("metric")


def qtype_table(summaries: Dict[str, dict]):
    import pandas as pd

    rows = []
    for name, s in summaries.items():
        for qtype, d in s["by_qtype"].items():
            rows.append(dict(system=name, qtype=qtype, n=d["n"], success=round(d["success"], 3), abstained=round(d["abstained"], 3)))
    return pd.DataFrame(rows).pivot(index="qtype", columns="system", values="success")


def save_results(prefix: str, summary: dict, per_item: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    with open(prefix + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    keys = sorted({k for r in per_item for k in r})
    with open(prefix + "_items.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in per_item:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    print(f"Saved {prefix}_summary.json and {prefix}_items.csv")


def show_failures(per_item: List[dict], outcome_prefix: str = "wrong", n: int = 5) -> None:
    for r in [r for r in per_item if r["outcome"].startswith(outcome_prefix)][:n]:
        print(f"--- {r['id']} [{r['outcome']}] {r['question']}")
        print(f"    reference: {r['reference']}")
        print(f"    answer:    {r['answer'][:300]}")
        print(f"    retrieved: {r['retrieved']}")
        if r.get("unsupported_claims"):
            print(f"    unsupported: {r['unsupported_claims']}")


# ======================================================================================
# Human spot-check (validates the LLM judge itself)
# ======================================================================================


def export_for_human_review(per_item: List[dict], path: str, n: int = 30, seed: int = 0) -> None:
    rows = [r for r in per_item if r["answerable"] and r.get("correct") is not None]
    random.Random(seed).shuffle(rows)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "reference", "answer", "judge_correct", "human_correct (0/1)"])
        for r in rows[:n]:
            w.writerow([r["id"], r["question"], r["reference"], r["answer"], int(r["correct"]), ""])
    print(f"Wrote {min(n, len(rows))} rows to {path}; fill in the last column, then call judge_human_agreement().")


def judge_human_agreement(path: str) -> dict:
    j, h = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row["human_correct (0/1)"].strip()
            if v in ("0", "1"):
                j.append(int(row["judge_correct"]))
                h.append(int(v))
    if not j:
        return dict(n=0)
    j, h = np.array(j), np.array(h)
    po = float((j == h).mean())
    pe = float(j.mean() * h.mean() + (1 - j.mean()) * (1 - h.mean()))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return dict(n=len(j), agreement=po, cohen_kappa=kappa, judge_accuracy=float(j.mean()), human_accuracy=float(h.mean()))
