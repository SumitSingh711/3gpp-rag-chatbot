"""
retriever.py
Hybrid dense + sparse retrieval with Reciprocal Rank Fusion (RRF) for
*ranking*, and a separate confidence gate based on *raw* similarity signals
(not RRF rank score) for the refuse-or-answer decision.

WHY THE GATE IS SEPARATE FROM RRF RANKING (this is the fix from the
previous version): RRF fused scores are a function of *rank position*
across two lists, not of how relevant the top result actually is. Chroma
always returns its k-nearest neighbours even when nothing is close, and
BM25's get_scores() always returns a full score vector even when every
score is 0 (no token overlap at all). That means a chunk can land rank #1
in one of the two systems — and therefore clear a rank-based threshold —
purely because it's "the least bad option," with no real relevance. The
previous version gated on the fused RRF score, so it accepted (almost)
everything: the top RRF score is essentially always >= 1/(rrf_k+1) as long
as the corpus is non-empty, which made the "confidence gate" a near no-op.

This version gates on the raw dense cosine similarity and/or the raw BM25
score of the top hit, in addition to computing RRF purely for ordering
the final ranked list. These raw thresholds still need calibration against
hallucination_eval.py on your actual corpus — the defaults below are a
starting point, not a validated cutoff. Run the eval, look at the score
distribution for known answerable vs. known out-of-corpus questions, and
set the thresholds where they actually separate the two populations.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA_DIR / "processed" / "chroma_db"
BM25_FILE = DATA_DIR / "processed" / "bm25.pkl"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# --- Confidence gate thresholds (raw signals, NOT RRF score) ---
# Dense: collection is created with hnsw:space="cosine" in build_index.py,
# so chroma distances are cosine distances and similarity = 1 - distance,
# in [-1, 1] (practically [0, 1] for this embedding model). A genuinely
# on-topic 3GPP passage for a 3GPP question with bge-small typically lands
# well above 0.5; unrelated text (e.g. "iPhone release date") typically
# lands well below that against a spec corpus. CALIBRATE THIS on your data.
DENSE_SIM_THRESHOLD = 0.55

# BM25 (Okapi) raw score, unbounded and corpus-size dependent. A hit on an
# exact spec identifier (e.g. "PDU Session", "RRC_INACTIVE") tends to score
# several points higher than a query with zero real term overlap. This
# number is corpus-specific — recompute after you finalize the ingested
# spec set. CALIBRATE THIS on your data.
BM25_SCORE_THRESHOLD = 6.0

RRF_K = 60


@dataclass
class RetrievedChunk:
    text: str
    spec_number: str
    doc_type: str
    version: str
    clause_number: str
    clause_title: str
    source_file: str
    fused_score: float      # RRF score — for ranking/display only, NOT gating
    dense_sim: float        # raw cosine similarity of this chunk to the query
    bm25_score: float       # raw BM25 score of this chunk to the query

    def citation(self) -> str:
        # Kept in sync with the exact bracketed form the generation prompt
        # is instructed to emit: "[TS 23.501 clause 5.1]". Do not change one
        # without the other — hallucination_eval.py parses this format to
        # validate citations against retrieved chunks.
        return f"[{self.doc_type} {self.spec_number} clause {self.clause_number}]"

    def display_citation(self) -> str:
        """Human-friendly version for the UI (includes the clause title)."""
        return f"{self.doc_type} {self.spec_number} clause {self.clause_number} ({self.clause_title})"


class HybridRetriever:
    def __init__(self):
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        # NOTE: hnsw:space is fixed at collection *creation* time. If your
        # existing chroma_db was built before this fix (default L2 space),
        # you must rebuild the index (build_index.py) for dense_sim below
        # to be a meaningful cosine similarity rather than a distance proxy.
        self.collection = client.get_collection(name="tgpp_specs", embedding_function=ef)

        with open(BM25_FILE, "rb") as f:
            saved = pickle.load(f)
        self.bm25 = saved["bm25"]
        self.chunks = saved["chunks"]
        self.id_to_idx = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}

    def _dense_search(self, query: str, k: int):
        res = self.collection.query(query_texts=[query], n_results=k)
        ids = res["ids"][0]
        dists = res["distances"][0]  # cosine distance, given cosine hnsw:space
        # cosine similarity = 1 - cosine distance
        return [(cid, 1.0 - d) for cid, d in zip(ids, dists)]

    def _sparse_search(self, query: str, k: int):
        tokens = query.lower().replace("/", " ").replace("-", " ").split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i]["chunk_id"], scores[i]) for i in ranked]

    def retrieve(self, query: str, k: int = 6, rrf_k: int = RRF_K) -> list[RetrievedChunk]:
        dense = self._dense_search(query, k=20)
        sparse = self._sparse_search(query, k=20)

        dense_raw = {cid: sim for cid, sim in dense}
        bm25_raw = {cid: score for cid, score in sparse}

        # RRF for ranking only — this fusion is fine for deciding *order*,
        # it's just not a valid signal for deciding *whether to answer at
        # all*, which is why gating now uses dense_raw/bm25_raw instead.
        fused: dict[str, float] = {}
        for rank, (cid, _) in enumerate(dense):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, (cid, _) in enumerate(sparse):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        ranked_ids = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]

        out = []
        for cid, fscore in ranked_ids:
            c = self.chunks[self.id_to_idx[cid]]
            out.append(
                RetrievedChunk(
                    text=c["text"],
                    spec_number=c["spec_number"],
                    doc_type=c["doc_type"],
                    version=c["version"],
                    clause_number=c["clause_number"],
                    clause_title=c["clause_title"],
                    source_file=c["source_file"],
                    fused_score=fscore,
                    dense_sim=dense_raw.get(cid, -1.0),
                    bm25_score=bm25_raw.get(cid, 0.0),
                )
            )
        return out

    def retrieve_with_gate(self, query: str, k: int = 6):
        """Returns (chunks, is_confident). is_confident=False means the
        caller should refuse to answer rather than fall back to the LLM's
        own (unverifiable) knowledge.

        Confidence is judged on the RAW dense similarity or RAW BM25 score
        of the top-ranked chunk, not on the RRF fused score (see module
        docstring for why that distinction matters)."""
        results = self.retrieve(query, k=k)
        if not results:
            return results, False

        top = results[0]
        is_confident = (
            top.dense_sim >= DENSE_SIM_THRESHOLD
            or top.bm25_score >= BM25_SCORE_THRESHOLD
        )
        return results, is_confident
