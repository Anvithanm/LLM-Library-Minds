"""
Improved RAG pipeline for question answering over the IsmaelMousa/books dataset.

What changed compared to the v1 notebook (RAG_Book_Based_LLM.ipynb)
-------------------------------------------------------------------
1. Generation bug fix: v1 used `HuggingFacePipeline` with the default
   `return_full_text=True`, so every "answer" began with the full system prompt
   and the retrieved context. That also inflated v1's faithfulness/"context used"
   scores. We now apply Mistral's chat template and decode only new tokens.
2. Contextual chunk headers: each chunk is indexed as "Book: <title> by <author>"
   + text, so questions such as "Which book has the character X?" or
   "In Treasure Island, ..." can match on the title/author.
3. Hybrid retrieval: dense (BGE embeddings, cosine/inner product on FAISS) +
   sparse BM25, fused with Reciprocal Rank Fusion. BM25 rescues rare proper
   nouns (e.g. "Kreutznaer") that small embedding models handle poorly.
4. Cross-encoder re-ranking of the fused candidates (BAAI/bge-reranker-base).
5. Title routing: if the question names a book in the library, retrieval is
   restricted to that book.
6. Calibrated abstention: the brittle v1 gate (an L2-distance threshold plus a
   "does the question contain what/who/how" word list, which rejected valid
   questions like "which book has the character Robinson Kreutznaer?") is
   replaced by a single threshold on the re-ranker score. The threshold is
   chosen on a dev split by `rag_evaluation.calibrate_abstention`.
7. Deterministic (greedy) decoding, numbered sources with [n] citations, and
   per-stage latency instrumentation.

Every improvement is a config flag, so `RAGConfig.baseline()` reproduces the v1
retrieval setup and the evaluation can run an honest ablation.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, TypedDict

import numpy as np

# Import faiss before torch initializes OpenMP: on macOS the two bundled libomp
# runtimes otherwise segfault on the first search.
import faiss  # noqa: E402

ABSTAIN_ANSWER = "I don't know based on the available books."

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class RAGConfig:
    # data / chunking
    n_books: int = 20
    chunk_size: int = 1000
    chunk_overlap: int = 200
    contextual_headers: bool = True
    # dense retrieval
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # BGE v1.5 models expect this prefix on queries (not on passages).
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    # "dense" | "bm25" | "hybrid"
    retrieval_mode: str = "hybrid"
    n_candidates: int = 30  # candidates per retriever before fusion / re-ranking
    rrf_k: int = 60
    # re-ranking (None disables)
    reranker_model: Optional[str] = "BAAI/bge-reranker-base"
    # restrict search to a book whose title is mentioned in the question
    title_routing: bool = True
    # number of chunks passed to the LLM
    top_k: int = 4
    # abstain when the top retrieval score is below this value. The score is the
    # re-ranker score when a re-ranker is configured, else dense cosine similarity.
    # None = no retrieval gate (the LLM prompt still allows "I don't know").
    abstain_threshold: Optional[float] = None
    # generation
    max_new_tokens: int = 256
    device: Optional[str] = None  # None = auto (cuda > mps > cpu)

    @classmethod
    def baseline(cls, **overrides) -> "RAGConfig":
        """Retrieval setup of the v1 notebook: MiniLM, dense-only, no headers,
        no re-ranker, no routing. (Generation still uses the fixed decoding,
        otherwise answers contain the prompt and cannot be scored at all.)"""
        cfg = dict(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            query_prefix="",
            retrieval_mode="dense",
            contextual_headers=False,
            reranker_model=None,
            title_routing=False,
        )
        cfg.update(overrides)
        return cls(**cfg)

    def to_dict(self) -> dict:
        return asdict(self)


def _auto_device(device: Optional[str]) -> str:
    if device:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# --------------------------------------------------------------------------------------
# Data loading and chunking
# --------------------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: int
    book_id: int
    title: str
    author: str
    text: str
    start: int  # character offset in the book text (inclusive)
    end: int  # character offset (exclusive)

    def header(self) -> str:
        return f"Book: {self.title} by {self.author}"


def load_books(n_books: int = 20, split: str = "train", json_path: Optional[str] = None) -> List[dict]:
    """Return [{book_id, title, author, category, text}] for the first `n_books` rows.
    `json_path` lets you read a local copy of the dataset's train.json instead."""
    if json_path:
        with open(json_path) as f:
            rows = json.load(f)
    else:
        from datasets import load_dataset

        rows = load_dataset("IsmaelMousa/books", split=split)
    books = []
    for i in range(min(n_books, len(rows))):
        r = rows[i]
        books.append(
            dict(
                book_id=i,
                title=r.get("title") or f"Book {i}",
                author=r.get("author") or "Unknown",
                category=r.get("category") or "",
                text=r.get("EN") or "",
            )
        )
    return books


def chunk_books(books: Sequence[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Chunk]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks: List[Chunk] = []
    for b in books:
        for d in splitter.create_documents([b["text"]]):
            start = d.metadata["start_index"]
            chunks.append(
                Chunk(
                    chunk_id=len(chunks),
                    book_id=b["book_id"],
                    title=b["title"],
                    author=b["author"],
                    text=d.page_content,
                    start=start,
                    end=start + len(d.page_content),
                )
            )
    return chunks


# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------

_STOPWORDS = set(
    """a an and are as at be but by for from had has have he her his i if in into is it its
    me my no not of on or our she so than that the their them then there these they this to
    was we were what when where which who whom why will with would you your does did do how""".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def _title_aliases(title: str) -> List[str]:
    t = title.lower().strip()
    aliases = {t}
    for prefix in ("the adventures of ", "the life of ", "the picture of ", "the wonderful ", "the "):
        if t.startswith(prefix) and len(t) - len(prefix) >= 4:
            aliases.add(t[len(prefix):])
    return sorted(aliases, key=len, reverse=True)


@dataclass
class Hit:
    chunk: Chunk
    rank: int
    score: float  # final score used for ordering (rerank > rrf > dense/bm25)
    dense: Optional[float] = None
    bm25: Optional[float] = None
    rerank: Optional[float] = None


@dataclass
class RetrievalResult:
    hits: List[Hit]
    routed_book_ids: List[int]
    gate_score: float  # score compared with abstain_threshold
    timings: Dict[str, float] = field(default_factory=dict)


class BookIndex:
    """Dense (FAISS) + sparse (BM25) index over book chunks, with optional re-ranker."""

    def __init__(
        self,
        chunks: List[Chunk],
        cfg: RAGConfig,
        embeddings: Optional[np.ndarray] = None,
        embedder=None,
        reranker=None,
        show_progress: bool = True,
    ):
        import faiss
        from rank_bm25 import BM25Okapi

        self.cfg = cfg
        self.chunks = chunks
        self.device = _auto_device(cfg.device)
        self.embedder = embedder
        if self.embedder is None and cfg.retrieval_mode in ("dense", "hybrid"):
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(cfg.embedding_model, device=self.device)

        texts = [self.index_text(c) for c in chunks]
        self.embeddings = None
        self.faiss_index = None
        if cfg.retrieval_mode in ("dense", "hybrid"):
            if embeddings is None:
                embeddings = self.embedder.encode(
                    texts,
                    batch_size=64,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=show_progress,
                )
            self.embeddings = np.ascontiguousarray(embeddings, dtype="float32")
            # normalized vectors + inner product == cosine similarity
            self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.faiss_index.add(self.embeddings)

        self.bm25 = None
        if cfg.retrieval_mode in ("bm25", "hybrid"):
            self.bm25 = BM25Okapi([tokenize(t) for t in texts])

        self.reranker = reranker
        if self.reranker is None and cfg.reranker_model:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(cfg.reranker_model, device=self.device)

        self.book_rows: Dict[int, np.ndarray] = {}
        self.books_meta: Dict[int, dict] = {}
        for i, c in enumerate(chunks):
            self.book_rows.setdefault(c.book_id, []).append(i)
            self.books_meta.setdefault(c.book_id, dict(title=c.title, author=c.author))
        self.book_rows = {k: np.array(v) for k, v in self.book_rows.items()}
        self._alias_patterns = [
            (bid, re.compile(r"\b" + re.escape(alias) + r"\b"))
            for bid, meta in self.books_meta.items()
            for alias in _title_aliases(meta["title"])
        ]

    # ------------------------------------------------------------------ helpers
    def index_text(self, c: Chunk) -> str:
        return f"{c.header()}\n\n{c.text}" if self.cfg.contextual_headers else c.text

    def route(self, question: str) -> List[int]:
        if not self.cfg.title_routing:
            return []
        q = " ".join(question.lower().replace("’", "'").split())
        return sorted({bid for bid, pat in self._alias_patterns if pat.search(q)})

    def _dense(self, question: str, n: int, rows: Optional[np.ndarray]):
        qv = self.embedder.encode(
            [self.cfg.query_prefix + question], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        if rows is None:
            scores, idx = self.faiss_index.search(qv, n)
            return list(zip(idx[0].tolist(), scores[0].tolist()))
        sims = self.embeddings[rows] @ qv[0]
        order = np.argsort(-sims)[:n]
        return [(int(rows[j]), float(sims[j])) for j in order]

    def _bm25(self, question: str, n: int, rows: Optional[np.ndarray]):
        toks = tokenize(question)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        if rows is not None:
            masked = np.full_like(scores, -np.inf)
            masked[rows] = scores[rows]
            scores = masked
        order = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in order if np.isfinite(scores[i]) and scores[i] > 0]

    # ------------------------------------------------------------------ search
    def search(self, question: str, top_k: Optional[int] = None) -> RetrievalResult:
        cfg = self.cfg
        top_k = top_k or cfg.top_k
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()

        routed = self.route(question)
        rows = np.concatenate([self.book_rows[b] for b in routed]) if routed else None
        n = max(cfg.n_candidates, top_k)

        dense = self._dense(question, n, rows) if cfg.retrieval_mode in ("dense", "hybrid") else []
        sparse = self._bm25(question, n, rows) if cfg.retrieval_mode in ("bm25", "hybrid") else []
        dense_map, sparse_map = dict(dense), dict(sparse)

        if cfg.retrieval_mode == "hybrid":
            fused: Dict[int, float] = {}
            for ranked in (dense, sparse):
                for r, (i, _) in enumerate(ranked):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (cfg.rrf_k + r + 1)
            candidates = sorted(fused.items(), key=lambda x: -x[1])
        else:
            candidates = dense or sparse
        timings["retrieve_s"] = time.perf_counter() - t0

        rerank_map: Dict[int, float] = {}
        if self.reranker is not None and candidates:
            t1 = time.perf_counter()
            cand_ids = [i for i, _ in candidates[: cfg.n_candidates]]
            pairs = [(question, self.index_text(self.chunks[i])) for i in cand_ids]
            scores = np.asarray(self.reranker.predict(pairs, batch_size=32, show_progress_bar=False), dtype=float)
            rerank_map = dict(zip(cand_ids, scores.tolist()))
            candidates = sorted(rerank_map.items(), key=lambda x: -x[1])
            timings["rerank_s"] = time.perf_counter() - t1

        hits = [
            Hit(
                chunk=self.chunks[i],
                rank=r + 1,
                score=float(s),
                dense=dense_map.get(i),
                bm25=sparse_map.get(i),
                rerank=rerank_map.get(i),
            )
            for r, (i, s) in enumerate(candidates[:top_k])
        ]

        if rerank_map:
            gate = max(rerank_map.values())
        elif self.embedder is not None:
            gate = max(dense_map.values()) if dense_map else self._dense(question, 1, rows)[0][1]
        else:
            gate = hits[0].score if hits else 0.0
        return RetrievalResult(hits=hits, routed_book_ids=routed, gate_score=float(gate), timings=timings)

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "chunks.json"), "w") as f:
            json.dump([asdict(c) for c in self.chunks], f)
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)
        if self.embeddings is not None:
            np.save(os.path.join(directory, "embeddings.npy"), self.embeddings)

    @classmethod
    def load(cls, directory: str, cfg: Optional[RAGConfig] = None, **kwargs) -> "BookIndex":
        with open(os.path.join(directory, "chunks.json")) as f:
            chunks = [Chunk(**c) for c in json.load(f)]
        if cfg is None:
            with open(os.path.join(directory, "config.json")) as f:
                cfg = RAGConfig(**json.load(f))
        emb_path = os.path.join(directory, "embeddings.npy")
        emb = np.load(emb_path) if os.path.exists(emb_path) else None
        return cls(chunks, cfg, embeddings=emb, **kwargs)


# --------------------------------------------------------------------------------------
# LLM wrapper (used for generation, synthetic test-set creation and LLM-as-judge)
# --------------------------------------------------------------------------------------

ChatFn = Callable[[List[dict]], str]


class HFChatLLM:
    """Callable `llm(messages) -> str` around a Hugging Face causal LM.

    Applies the model's chat template and returns only the newly generated text.
    Pass `model`/`tokenizer` to reuse weights that are already loaded."""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        model=None,
        tokenizer=None,
        load_in_4bit: bool = False,
        merge_system_into_user: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        if model is None:
            kwargs = dict(device_map="auto", low_cpu_mem_usage=True)
            if load_in_4bit:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
                )
            else:
                kwargs["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model = model.eval()
        # Mistral's template only accepts alternating user/assistant turns.
        self.merge_system_into_user = merge_system_into_user
        self.last_output_tokens = 0

    def __call__(self, messages: List[dict], max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        import torch

        msgs = list(messages)
        if self.merge_system_into_user and msgs and msgs[0]["role"] == "system":
            sys_msg, msgs = msgs[0]["content"], msgs[1:]
            msgs[0] = {"role": msgs[0]["role"], "content": f"{sys_msg}\n\n{msgs[0]['content']}"}
        enc = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None)
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        new_tokens = out[0, enc["input_ids"].shape[1]:]
        self.last_output_tokens = int(new_tokens.shape[0])
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You answer questions about a library of books, using ONLY the numbered excerpts below.

Rules:
- Every statement must be supported by the excerpts. Cite the excerpt numbers you used in square brackets, e.g. [1] or [2][3].
- If the excerpts do not contain the answer, reply exactly: "{ABSTAIN_ANSWER}"
- If the question contains a false assumption, say so instead of going along with it.
- Answer in 1-4 sentences. Do not repeat the excerpts or these instructions."""

_ABSTAIN_PATTERNS = re.compile(
    r"i don'?t know|i do not know|cannot answer|can'?t answer|not (?:mentioned|contained|provided|stated|found)"
    r"|no information|do(?:es)? not (?:contain|mention|provide|say)|unable to (?:answer|find|determine)"
    r"|there is no mention|not enough information",
    re.IGNORECASE,
)


def is_abstention(answer: str) -> bool:
    return bool(_ABSTAIN_PATTERNS.search(answer or ""))


def strip_citations(answer: str) -> str:
    return re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", answer or "").strip()


def parse_citations(answer: str) -> List[int]:
    nums = []
    for grp in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer or ""):
        nums.extend(int(n) for n in re.split(r"\s*,\s*", grp))
    return sorted(set(nums))


def format_context(hits: Sequence[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] ({h.chunk.title} - {h.chunk.author})\n{h.chunk.text.strip()}" for i, h in enumerate(hits, 1)
    )


class RAGPipeline:
    def __init__(self, index: BookIndex, llm: Optional[ChatFn] = None, cfg: Optional[RAGConfig] = None):
        self.index = index
        self.llm = llm
        self.cfg = cfg or index.cfg

    def retrieve(self, question: str, top_k: Optional[int] = None) -> RetrievalResult:
        return self.index.search(question, top_k=top_k or self.cfg.top_k)

    def should_abstain(self, retrieval: RetrievalResult) -> bool:
        thr = self.cfg.abstain_threshold
        return thr is not None and retrieval.gate_score < thr

    def answer(self, question: str) -> dict:
        t0 = time.perf_counter()
        ret = self.retrieve(question)
        timings = dict(ret.timings)
        result = dict(
            question=question,
            hits=ret.hits,
            routed_book_ids=ret.routed_book_ids,
            gate_score=ret.gate_score,
            abstained_by_gate=False,
            output_tokens=0,
        )
        if self.should_abstain(ret) or not ret.hits:
            result.update(answer=ABSTAIN_ANSWER, abstained_by_gate=True)
        else:
            if self.llm is None:
                raise ValueError("RAGPipeline needs an `llm` to generate answers")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Excerpts:\n{format_context(ret.hits)}\n\nQuestion: {question}"},
            ]
            t1 = time.perf_counter()
            try:
                text = self.llm(messages, max_new_tokens=self.cfg.max_new_tokens)
            except TypeError:  # plain callables that only accept messages
                text = self.llm(messages)
            timings["generate_s"] = time.perf_counter() - t1
            result.update(answer=text, output_tokens=getattr(self.llm, "last_output_tokens", 0))
        timings["total_s"] = time.perf_counter() - t0
        result["timings"] = timings
        result["abstained"] = result["abstained_by_gate"] or is_abstention(result["answer"])
        result["citations"] = parse_citations(result["answer"])
        return result


class GraphState(TypedDict, total=False):
    # Module level so LangGraph can resolve the (string) annotations below.
    question: str
    retrieval: RetrievalResult
    answer: str
    abstained: bool


def build_langgraph(pipeline: RAGPipeline):
    """Same pipeline expressed as a LangGraph: retrieve -> (gate) -> generate | abstain."""
    from langgraph.graph import END, START, StateGraph


    def retrieve(state: GraphState):
        return {"retrieval": pipeline.retrieve(state["question"])}

    def route(state: GraphState):
        return "abstain" if pipeline.should_abstain(state["retrieval"]) else "generate"

    def generate(state: GraphState):
        hits = state["retrieval"].hits
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n{format_context(hits)}\n\nQuestion: {state['question']}"},
        ]
        text = pipeline.llm(messages)
        return {"answer": text, "abstained": is_abstention(text)}

    def abstain(state: GraphState):
        return {"answer": ABSTAIN_ANSWER, "abstained": True}

    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("abstain", abstain)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route, {"generate": "generate", "abstain": "abstain"})
    g.add_edge("generate", END)
    g.add_edge("abstain", END)
    return g.compile()
