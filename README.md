# 3GPP Standards RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot designed to answer technical questions using 3GPP Telecom standards as the primary knowledge source.

The main objective of this project is not simply to generate answers, but to minimize hallucinations by grounding responses in retrieved 3GPP evidence and refusing to answer when sufficient evidence is unavailable.

## Live Demo

🔗 **Live Demo:** [[Deployed App URL](https://3gpp-rag-chatbot-web.streamlit.app/)]

## Repository

🔗 **GitHub:** [[GITHUB URL](https://github.com/SumitSingh711/3gpp-rag-chatbot)]

---

# How to run locally with custom 3gpp pdf

# clone the project
```bash
git clone YOUR_GITHUB_REPO_URL
cd 3gpp-rag-chatbot
```

# Automatic environment creation and Libraries installation
run setup.bat file

# Adding a new PDF (command line)

1. Convert the spec to PDF and drop it into `data/raw/`.
2. Re-run the pipeline — it's incremental, so only the new file gets processed:
```bash
python src/pipeline.py
```
   or, to force a full rebuild of everything:
```bash
python src/pipeline.py --full
```
3. Chat immediately from the terminal:
```bash
python src/pipeline.py --chat 

or 

# for streamlit UI
streamlit run src/app.py 
```

---

# Evaluation

```bash
python eval/hallucination_eval.py
```
Reports: answer rate on in-corpus questions, correct-refusal rate on out-of-corpus questions, and count of sentences flagged unsupported by the verifier. See `eval/eval_set.jsonl` to add more test questions.

# Overview


Technical standards contain large amounts of structured and highly specific information distributed across multiple specifications and clauses.

A general-purpose LLM may:

- generate technically plausible but incorrect information
- mix information from different specifications
- confuse similar telecom concepts
- accept false premises in questions
- provide information that is not present in the source documents

This project addresses these problems using a retrieval-grounded architecture.

The chatbot retrieves relevant passages from 3GPP specifications before generating an answer. If the retrieved evidence is insufficient, or if any generated claim can't be verified against that evidence, the system abstains instead of guessing.

---

# Architecture

```text
                 3GPP Standards
                       │
                       ▼
              ┌──────────────────┐
              │ Document Ingest  │
              │                  │
              │ PDF extraction   │         
              │ Chunking         │
              │ Metadata         │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Indexing         │
              │                  │
              │ Embeddings       │
              │ Vector Index     │
              │ BM25 Index       │
              └────────┬─────────┘
                       │
                       ▼
                  User Query
                       │
                       ▼
              ┌──────────────────┐
              │ Intent Router    │
              │                  │
              │ CHAT vs RAG      │
              └────────┬─────────┘
                       │
              ┌────────┴─────────┐
              │                  │
           CHAT mode         RAG mode
       (small talk, no          │
        retrieval needed)       ▼
              │        ┌──────────────────┐
              │        │ Hybrid Retrieval │
              │        │                  │
              │        │ Dense Retrieval  │
              │        │ BM25             │
              │        │ RRF Fusion       │
              │        │ Confidence Gate  │
              │        └────────┬─────────┘
              │                 │
              │          Retrieved Evidence
              │                 │
              │                 ▼
              │        ┌──────────────────┐
              │        │ LLM Generation   │
              │        │                  │
              │        │ Evidence-grounded│
              │        │ response +       │
              │        │ citations        │
              │        └────────┬─────────┘
              │                 │
              │                 ▼
              │        ┌──────────────────┐
              │        │ Claim Verification│
              │        │                  │
              │        │ Verify generated │
              │        │ claims against    │
              │        │ retrieved context │
              │        └────────┬─────────┘
              │                 │
              │          ┌──────┴──────┐
              │          │             │
              │       Supported    Unsupported
              │          │             │
              ▼          ▼             ▼
         Direct Reply  Final Answer   Abstain
                        + Citations
```

---

# Anti-hallucination design

1. **Confidence gate** — if hybrid retrieval doesn't find anything confidently relevant, the bot refuses instead of guessing.
2. **Grounded-generation prompt** — temperature 0, must cite every sentence as `[SPEC clause X.Y.Z]`, must say "The provided context does not specify this" instead of extrapolating, and must not assume uplink/downlink symmetry or borrow from unrelated releases.
3. **Post-hoc claim verification** — a second LLM call checks every sentence in the draft answer against the retrieved context. If any claim can't be verified, the whole answer is withheld rather than shown with a warning label — abstaining is the default, not a fallback.
4. **Intent routing** — small talk and meta questions ("hi", "what can you do") are answered conversationally without going through retrieval at all, so the bot behaves like a normal chatbot instead of refusing greetings.

No RAG system can guarantee zero hallucination — this design fails toward "I don't know" rather than toward a confident wrong answer, which is the correct behavior for a standards-compliance assistant.

---

# Documents indexed

| Spec | Title |
|---|---|
| TS 21.905 | Vocabulary for 3GPP Specifications |
| TS 23.501 | 5G System Architecture |
| TS 23.502 | 5G System Procedures |
| TS 24.301 | NAS Protocol for EPS |
| TS 36.331 | LTE RRC Protocol |
| TS 38.300 | NR Overall Description |
| TS 38.214 | NR Physical Layer Procedures for Data |

Downloaded from: `https://www.3gpp.org/ftp/Specs/latest` (picked the Release 19 versions).

---

# ⚠️ Challenges faced

**3GPP terminology**

Exact technical terms and acronyms are important in standards documents.

Approach: Hybrid Dense + BM25 retrieval.

**Clause-level information**

Important information is often tied to a specific specification and clause.

Approach: Clause-aware chunking and metadata-based citations.

**Hallucination***

An LLM can generate plausible information that is not present in the
retrieved standards.

Approach: Retrieval grounding, claim verification and abstention.

**False premises**

Questions may contain technically incorrect assumptions.

Approach: Verify generated claims against retrieved evidence instead of
blindly accepting the premise.

# 🚧 Currently working on improvements like

Better calibration of the retrieval confidence threshold
Stronger out-of-corpus detection
More robust citation enforcement
Better multi-hop question handling
Larger hallucination evaluation dataset
Support for more 3GPP specifications and releases
Better handling of different specification versions