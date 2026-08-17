"""
pipeline.py
One-command orchestrator for the whole project, so you don't have to run
ingest.py, build_index.py, etc. one by one every time you add a spec.

Usage:
    python src/pipeline.py                 # ingest (incremental) + build index
    python src/pipeline.py --full          # force full re-ingest + re-index
    python src/pipeline.py --chat          # ...then drop into the CLI chatbot
    python src/pipeline.py --serve         # ...then launch the Streamlit UI
    python src/pipeline.py --eval          # ...then run the hallucination eval
    python src/pipeline.py --skip-index    # only run ingest, skip indexing
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def step(name: str):
    """Small decorator-ish context manager for consistent step logging."""
    class _Step:
        def __enter__(self):
            print(f"\n{'='*60}\n▶ {name}\n{'='*60}")
            self.t0 = time.time()
            return self

        def __exit__(self, exc_type, exc, tb):
            elapsed = time.time() - self.t0
            if exc_type is None:
                print(f"✔ {name} done in {elapsed:.1f}s")
            else:
                print(f"✘ {name} FAILED after {elapsed:.1f}s: {exc}")
            return False  # don't swallow exceptions

    return _Step()


def run_ingest(full: bool):
    from ingest import process_directory

    raw_dir = ROOT / "data" / "raw"
    out_path = ROOT / "data" / "processed" / "chunks.jsonl"

    if not raw_dir.exists() or not any(raw_dir.glob("**/*")):
        print(f"⚠ No files found in {raw_dir} — nothing to ingest.")
        print("  Drop 3GPP PDFs/TXT files there first (see README section 1).")
        sys.exit(1)

    n = process_directory(raw_dir, out_path, incremental=not full)
    if n == 0:
        print("⚠ Zero chunks produced. Check that your files aren't empty/corrupted.")
        sys.exit(1)


def run_build_index():
    from build_index import build

    build()


def run_chat():
    subprocess.run([sys.executable, str(SRC / "rag_chatbot.py")])


def run_serve():
    subprocess.run(["streamlit", "run", str(SRC / "app.py")])


def run_eval():
    subprocess.run([sys.executable, str(ROOT / "eval" / "hallucination_eval.py")])


def main():
    parser = argparse.ArgumentParser(description="3GPP RAG pipeline orchestrator")
    parser.add_argument("--full", action="store_true",
                         help="Force full re-ingest + re-index instead of incremental")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip the ingest step")
    parser.add_argument("--skip-index", action="store_true", help="Skip the indexing step")
    parser.add_argument("--chat", action="store_true", help="Launch CLI chatbot after pipeline")
    parser.add_argument("--serve", action="store_true", help="Launch Streamlit UI after pipeline")
    parser.add_argument("--eval", action="store_true", help="Run hallucination eval after pipeline")
    args = parser.parse_args()

    overall_start = time.time()

    if not args.skip_ingest:
        with step("Ingest: chunking raw specs"):
            run_ingest(full=args.full)
    else:
        print("Skipping ingest step (--skip-ingest)")

    if not args.skip_index:
        with step("Build index: dense (Chroma) + sparse (BM25)"):
            run_build_index()
    else:
        print("Skipping index step (--skip-index)")

    print(f"\nPipeline finished in {time.time() - overall_start:.1f}s total.")

    if args.eval:
        with step("Hallucination eval"):
            run_eval()

    if args.chat:
        run_chat()
    elif args.serve:
        run_serve()


if __name__ == "__main__":
    main()
