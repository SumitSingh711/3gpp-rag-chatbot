"""
hallucination_eval.py
A minimal eval harness to quantify hallucination rate.

Metrics reported:
- Refusal rate on out-of-corpus questions (should be HIGH — refusing is correct)
- Answer rate on in-corpus questions (should be HIGH)
- Citation validity rate: fraction of citation tags in answers that match a
  real (doc_type, spec_number, clause_number) triple present in the
  retrieved context (should be 100%). Previously documented in this
  docstring but never actually computed — that gap is fixed below.
- Unsupported-sentence rate from the verifier pass (should be near 0)

NOTE ON RELIABILITY: this harness measures the pipeline's *behavior*
(refuse/answer, citation match), not ground-truth factual correctness of
answered content — you still need a human (or a stronger separate judge
model) to spot-check that answered questions are actually right, not just
"cites something that exists." A confident wrong citation still counts as
"valid" here if the (spec, clause) tuple happens to be one that was
retrieved.
"""

import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from rag_chatbot import RAGChatbot  # noqa: E402

EVAL_FILE = Path(__file__).parent / "eval_set.jsonl"

# Matches the exact bracket format the generation system prompt is
# instructed to emit and that RetrievedChunk.citation() renders, e.g.
# "[TS 23.501 clause 5.1]". Keep this in sync with both of those.
CITATION_RE = re.compile(r"\[(TS|TR)\s+(\d{2}\.\d{3}(?:-\d+)?)\s+clause\s+([\d.]+)\]")


def extract_citations(answer: str) -> list[tuple[str, str, str]]:
    return [(m.group(1), m.group(2), m.group(3)) for m in CITATION_RE.finditer(answer)]


def run_eval():
    bot = RAGChatbot()
    rows = [json.loads(l) for l in open(EVAL_FILE)]

    total = len(rows)
    correct_refusals = 0
    correct_answers = 0
    total_out_of_corpus = 0
    total_in_corpus = 0
    unsupported_count = 0

    total_citations = 0
    valid_citations = 0
    answers_with_zero_citations = 0  # an answered (non-refused) RAG turn with no citation at all is itself a red flag

    for row in rows:
        q, expect_answerable = row["question"], row["answerable"]
        r = bot.ask(q)

        if expect_answerable:
            total_in_corpus += 1
            if not r.refused:
                correct_answers += 1
        else:
            total_out_of_corpus += 1
            if r.refused:
                correct_refusals += 1

        unsupported_count += len(r.unsupported_sentences)

        citation_note = ""
        if r.mode == "rag" and not r.refused:
            cites = extract_citations(r.answer)
            valid_keys = {(s.doc_type, s.spec_number, s.clause_number) for s in r.sources}
            n_valid = sum(1 for c in cites if c in valid_keys)
            total_citations += len(cites)
            valid_citations += n_valid
            if not cites:
                answers_with_zero_citations += 1
            citation_note = f", citations={len(cites)} (valid={n_valid})"

        print(f"[{'ANSWERABLE' if expect_answerable else 'OUT-OF-CORPUS'}] {q}")
        print(f"  -> mode={r.mode}, refused={r.refused}, unsupported_flags={len(r.unsupported_sentences)}{citation_note}\n")

    print("=" * 60)
    print(f"Total questions: {total}")
    if total_in_corpus:
        print(f"Answer rate on in-corpus questions: {correct_answers}/{total_in_corpus} "
              f"({100*correct_answers/total_in_corpus:.1f}%)")
    if total_out_of_corpus:
        print(f"Correct refusal rate on out-of-corpus questions: {correct_refusals}/{total_out_of_corpus} "
              f"({100*correct_refusals/total_out_of_corpus:.1f}%)")
    print(f"Total sentences flagged unsupported by verifier: {unsupported_count}")
    if total_citations:
        print(f"Citation validity rate: {valid_citations}/{total_citations} "
              f"({100*valid_citations/total_citations:.1f}%)")
    else:
        print("Citation validity rate: n/a (no citations were emitted on any answered question — "
              "check the generation prompt/format if this is unexpected)")
    if answers_with_zero_citations:
        print(f"WARNING: {answers_with_zero_citations} answered RAG response(s) had zero citation tags at all.")


if __name__ == "__main__":
    run_eval()
