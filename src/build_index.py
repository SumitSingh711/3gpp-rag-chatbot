"""
build_index.py
Builds two complementary indexes over the chunked 3GPP corpus:

1. Dense vector index (Chroma) using a local sentence-transformers model
   (BAAI/bge-small-en-v1.5) — no API key required, no per-token cost, no data
   leaves your machine. Swap in a hosted embedding API later (see README)
   if you need higher multilingual quality.
2. Sparse BM25 index (rank_bm25) — lexical matching is essential for specs
   full of exact identifiers (e.g. "5QI", "RRC_INACTIVE", "38.331") that
   dense embeddings alone often blur together.

Hybrid retrieval (see retriever.py) fuses both so the system doesn't miss a
chunk just because the embedding model didn't capture an exact acronym.
"""

import json
import pickle
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_FILE = DATA_DIR / "processed" / "chunks.jsonl"
CHROMA_DIR = DATA_DIR / "processed" / "chroma_db"
BM25_FILE = DATA_DIR / "processed" / "bm25.pkl"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # local, free, 384-dim, strong for technical English


def load_chunks() -> list[dict]:
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize(text: str) -> list[str]:
    return text.lower().replace("/", " ").replace("-", " ").split()


def build(batch_size: int = 64):
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("No chunks found — run ingest.py first.")

    # ---- Dense index ----
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    
    collection = client.get_or_create_collection(
        name="tgpp_specs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding + indexing"):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "spec_number": c["spec_number"],
                    "doc_type": c["doc_type"],
                    "version": c["version"],
                    "clause_number": c["clause_number"],
                    "clause_title": c["clause_title"],
                    "source_file": c["source_file"],
                }
                for c in batch
            ],
        )

    # ---- Sparse index ----
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"Indexed {len(chunks)} chunks into Chroma ({CHROMA_DIR}) and BM25 ({BM25_FILE})")


if __name__ == "__main__":
    build()
