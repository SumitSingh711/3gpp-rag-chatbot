"""
ingest.py
Parses 3GPP Technical Specification documents (PDF from
the official .doc/.docx releases) into clause-aware chunks.
"""

import re
import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader

# Matches 3GPP clause numbering like "4", "4.2", "5.3.1.2" at line start,
CLAUSE_RE = re.compile(r"^(?P<num>\d+(?:\.\d+){0,5})\s+(?P<title>[A-Z][^\n]{2,120})$")

# 3GPP spec number pattern e.g. "TS 38.331", "TS 23.501", "TR 21.905"
SPEC_RE = re.compile(r"\b(TS|TR)\s?(\d{2}\.\d{3})\b")

# 3GPP official filename convention: <specdigits>[-<part>[-<subpart>]]-<version>
# version is 3 base-36 chars (e.g. "j30") or, for specs with >35 technical
FILENAME_VERSION_RE = re.compile(r"-([0-9a-zA-Z]{3}|\d{6})(?:_[^.]*)?$")
FILENAME_SPEC_RE = re.compile(r"^(\d{5})(?:-(\d{1,2}))?(?:-(\d{1,2}))?")

_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def _decode_base36_digit(ch: str) -> int:
    return _BASE36.index(ch.lower())


def decode_filename_version(filename: str) -> str | None:
    """Decodes the version encoded in a 3GPP filename, e.g.
    '36331-j30.zip' -> '19.3.0', '24229-083700.doc' -> '8.37.0'.
    Returns None if the filename doesn't match the convention."""
    stem = Path(filename).stem
    # strip multi-file suffixes like "_cover", "_s00-s11" if present
    stem = re.sub(r"_.*$", "", stem)
    m = FILENAME_VERSION_RE.search(stem)
    if not m:
        return None
    code = m.group(1)
    if len(code) == 3:
        major, tech, edit = (_decode_base36_digit(c) for c in code)
    else:  # extended 6-digit decimal form, 2 digits per field
        major, tech, edit = int(code[0:2]), int(code[2:4]), int(code[4:6])
    return f"{major}.{tech}.{edit}"


@dataclass
class Chunk:
    chunk_id: str
    spec_number: str      # e.g. "38.331"
    doc_type: str         # "TS" or "TR"
    version: str          # e.g. "18.2.0" if detectable, else "unknown"
    clause_number: str    # e.g. "5.3.1.2"
    clause_title: str
    text: str
    source_file: str
    page_hint: int | None = None


def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        pages.append(t)
    return "\n".join(pages)


def detect_spec_metadata(text: str, filename: str) -> tuple[str, str, str]:
    """Extracts spec number / doc type / version.

    doc_type defaults to "TS" since the indexed corpus here is all Technical
    Specifications (21.905, 23.501, 23.502, 24.301, 36.331, 38.331, 38.300,
    38.211, 38.214). If you later add a Technical Report (TR) to the corpus
    — e.g. TR 21.900 — either rename its file with a "TR" hint or extend the
    boilerplate-text check below, since TS/TR can't be told apart from the
    filename or spec number alone (they're interleaved within a series).
    """
    m = SPEC_RE.search(text[:3000])
    if m:
        doc_type, spec_number = m.group(1), m.group(2)
    else:
        fm = FILENAME_SPEC_RE.match(Path(filename).stem)
        if fm:
            digits = fm.group(1)
            spec_number = f"{digits[:2]}.{digits[2:]}"
            if fm.group(2):  # multi-part spec, e.g. 29.198-08
                spec_number += f"-{fm.group(2)}"
            if fm.group(3):
                spec_number += f"-{fm.group(3)}"
        else:
            spec_number = "unknown"

        # Prefer the literal boilerplate phrase 3GPP puts on every cover
        # page when it's present; otherwise default to TS.
        if re.search(r"Technical Report", text[:3000], re.IGNORECASE):
            doc_type = "TR"
        else:
            doc_type = "TS"

    version = decode_filename_version(filename)
    if version is None:
        vm = re.search(r"[Vv]ersion\s*(\d+\.\d+\.\d+)", text[:3000])
        version = vm.group(1) if vm else "unknown"

    return spec_number, doc_type, version


# Known ETSI/3GPP cover-page and footer boilerplate that should never be
# treated as clause body content, even when it survives PDF text extraction
# sitting next to a number.
BOILERPLATE_RE = re.compile(
    r"(Sophia Antipolis|Route des Lucioles|Valbonne|ETSI Secretariat|"
    r"All rights reserved|European Telecommunications Standards Institute)",
    re.IGNORECASE,
)


def split_into_clauses(text: str) -> Iterator[tuple[str, str, str]]:
    """Yields (clause_number, clause_title, clause_body) by scanning line by
    line for clause headers. Body accumulates until the next header.

    Guards against front-matter false positives (cover page, Table of
    Contents, ETSI copyright/address block) that share the same
    "number + capitalized text" shape as a real clause header:
      - TOC lines have dot-leaders ("....") before a trailing page number —
        real clause bodies never do, so we skip those lines entirely.
      - The first segment of a real 3GPP clause number is small (1-99).
        A 4-digit first segment (e.g. "2017.12") is almost always a date
        from a header/footer, not a clause number, so it's rejected.
    """
    TOC_LINE_RE = re.compile(r"\.{4,}\s*\d+\s*$")  # e.g. "....... 205"

    lines = text.splitlines()
    current_num, current_title, buf = "0", "Preamble", []

    for line in lines:
        stripped = line.strip()

        if TOC_LINE_RE.search(stripped) or BOILERPLATE_RE.search(stripped):
            continue  # TOC entry or ETSI cover-page boilerplate, not real clause content

        m = CLAUSE_RE.match(stripped)
        if m:
            num = m.group("num")
            title = m.group("title").strip()
            first_segment = int(num.split(".")[0])
            looks_like_date = 1900 <= first_segment <= 2100
            looks_like_toc_entry = bool(re.search(r"\.{2,}", title))

            if len(num.split(".")) <= 6 and first_segment < 100 and not looks_like_date and not looks_like_toc_entry:
                if buf:
                    yield current_num, current_title, "\n".join(buf).strip()
                current_num, current_title = num, title
                buf = []
                continue

        buf.append(line)
    if buf:
        yield current_num, current_title, "\n".join(buf).strip()


def chunk_clause_body(body: str, max_chars: int = 1400, overlap: int = 150) -> list[str]:
    """Further splits an oversized clause body on sentence boundaries so no
    chunk exceeds max_chars, keeping a small overlap for continuity."""
    if len(body) <= max_chars:
        return [body] if body.strip() else []

    sentences = re.split(r"(?<=[.;:])\s+", body)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 > max_chars and cur:
            chunks.append(cur.strip())
            cur = cur[-overlap:] + " " + s
        else:
            cur = (cur + " " + s).strip()
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def make_chunk_id(spec_number: str, clause_number: str, idx: int) -> str:
    raw = f"{spec_number}-{clause_number}-{idx}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def process_document(path: Path) -> list[Chunk]:
    if path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    spec_number, doc_type, version = detect_spec_metadata(text, path.name)

    chunks: list[Chunk] = []
    for clause_num, clause_title, body in split_into_clauses(text):
        if len(body.strip()) < 40:
            continue  # skip empty/near-empty clauses (headers, blank pages)
        for i, piece in enumerate(chunk_clause_body(body)):
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(spec_number, clause_num, len(chunks)),
                    spec_number=spec_number,
                    doc_type=doc_type,
                    version=version,
                    clause_number=clause_num,
                    clause_title=clause_title,
                    text=piece,
                    source_file=path.name,
                )
            )
    return chunks


def process_directory(raw_dir: Path, out_path: Path, incremental: bool = True) -> int:
    """Chunks every PDF/TXT under raw_dir into out_path (JSONL).

    When incremental=True (default), keeps a manifest of {filename: sha1}
    next to out_path and skips any file whose hash hasn't changed since the
    last run — so adding one new spec doesn't force re-chunking the whole
    corpus. Pass incremental=False to force a full rebuild.
    """
    manifest_path = out_path.parent / "ingest_manifest.json"
    manifest: dict[str, str] = {}
    existing_chunks: list[dict] = []

    if incremental and manifest_path.exists() and out_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with open(out_path, encoding="utf-8") as f:
            existing_chunks = [json.loads(line) for line in f]

    def file_hash(path: Path) -> str:
        return hashlib.sha1(path.read_bytes()).hexdigest()

    all_chunks: list[dict] = []
    new_manifest: dict[str, str] = {}
    changed_any = False

    for path in sorted(raw_dir.glob("**/*")):
        if path.suffix.lower() not in {".pdf", ".txt"}:
            continue

        h = file_hash(path)
        new_manifest[path.name] = h

        if incremental and manifest.get(path.name) == h:
            # unchanged since last run — reuse existing chunks for this file
            reused = [c for c in existing_chunks if c["source_file"] == path.name]
            all_chunks.extend(reused)
            print(f"Skipping {path.name} (unchanged, {len(reused)} chunks reused)")
            continue

        print(f"Processing {path.name} ...")
        all_chunks.extend(asdict(c) for c in process_document(path))
        changed_any = True

    # Detect deletions: a file that was in the old manifest but no longer exists
    removed = set(manifest) - set(new_manifest)
    if removed:
        print(f"Removed from corpus: {', '.join(sorted(removed))}")
        changed_any = True

    if not incremental or changed_any or not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        manifest_path.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
        print(f"Wrote {len(all_chunks)} chunks to {out_path}")
    else:
        print("No changes detected — chunks.jsonl left untouched")

    return len(all_chunks)


if __name__ == "__main__":
    RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
    OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "chunks.jsonl"
    process_directory(RAW, OUT)
