# 3GPP Standards RAG Chatbot

A retrieval-grounded chatbot for answering questions from **3GPP specifications** with clause-level citations, high factual accuracy, and controlled abstention.

The system is designed to reduce common LLM failure modes such as hallucination, specification/release confusion, incorrect technical assumptions, and unsupported answers. When sufficient evidence cannot be retrieved or a generated claim cannot be verified, the system **abstains instead of guessing**.

## Live Demo

🔗 **[3GPP Standards RAG Chatbot](https://3gpp-rag-chatbot-web.streamlit.app/)**

---

## Features

- **Hybrid Retrieval** using Dense Embeddings + BM25
- **RRF Fusion** for combining semantic and lexical retrieval results
- **Intent Routing** between conversational and standards-related queries
- **Clause-aware retrieval** for precise specification references
- **Confidence Gate** to reject queries with insufficient evidence
- **Grounded LLM Generation** using retrieved specification context
- **Claim Verification** against retrieved evidence
- **Clause-level Citations** for technical answers
- **Controlled Abstention** for unsupported or out-of-corpus questions
- **Release-aware retrieval** to reduce cross-release confusion
- **False-premise detection** to avoid accepting incorrect assumptions

---

## Architecture

```text
                 3GPP Specifications
                         |
                         v
              +---------------------+
              |  Document Ingestion |
              +---------------------+
                         |
                 PDF Extraction
                         |
                 Clause Chunking
                         |
                     Metadata
                         |
                         v
              +---------------------+
              |      Indexing       |
              |                     |
              | Dense Embeddings    |
              | BM25                |
              | Vector Index        |
              +---------------------+
                         |
                         v
                    User Query
                         |
                         v
              +---------------------+
              |    Intent Router    |
              +---------------------+
                    /         \
                 CHAT         RAG
                  |            |
             Direct Reply   Hybrid Retrieval
                              |
                         Dense + BM25
                              |
                             RRF
                              |
                              v
                    Confidence Gate
                         /      \
                      Fail       Pass
                       |           |
                    Abstain        v
                              LLM Generation
                                   |
                                   v
                           Claim Verification
                              /        \
                           Fail        Pass
                            |            |
                         Abstain    Answer + Citation
```

The complete pipeline consists of intent routing, hybrid retrieval, RRF fusion, confidence gating, grounded generation, claim verification, and abstention.

---

## RAG Pipeline

### 1. Intent Routing

Determines whether the user's query is:

- Conversational
- Related to 3GPP standards and requires retrieval

### 2. Hybrid Retrieval

Combines:

- **Dense semantic retrieval** for conceptual similarity
- **BM25 lexical retrieval** for exact terminology, identifiers, and specification language

### 3. RRF Fusion

Reciprocal Rank Fusion combines the ranked results from both retrieval methods.

### 4. Confidence Gate

Checks whether the retrieved evidence is strong enough to answer the query.

If the evidence is insufficient, the system abstains.

### 5. Grounded Generation

The LLM generates an answer using the retrieved specification context rather than relying on unsupported model knowledge.

### 6. Claim Verification

Generated claims are checked against the retrieved evidence.

### 7. Abstention

If claims cannot be sufficiently verified, the system refuses to provide an unsupported answer.

---

## Anti-Hallucination Design

The chatbot follows strict grounding rules:

- Temperature is set to **0** for deterministic generation.
- Technical claims must have a specification/clause citation.
- Unsupported extrapolation is not allowed.
- The system does not assume uplink/downlink symmetry.
- Behavior from unrelated releases is not borrowed.
- False premises are checked against retrieved evidence.
- Claims that cannot be verified cause the system to abstain.

### Decision Flow

```text
Question
   |
   v
Evidence Found?
   |
   +---- No ----> Abstain
   |
  Yes
   |
   v
Generate Answer
   |
   v
Claims Verified?
   |
   +---- No ----> Abstain
   |
  Yes
   |
   v
Answer + Citation
```

---

## Indexed Specifications

The current knowledge base contains the following specifications:

| Specification | Title |
|---|---|
| TS 21.905 | Vocabulary for 3GPP Specifications |
| TS 23.501 | 5G System Architecture |
| TS 23.502 | 5G System Procedures |
| TS 24.301 | NAS Protocol for EPS |
| TS 36.331 | LTE RRC Protocol |
| TS 38.300 | NR Overall Description |
| TS 38.214 | NR Physical Layer Procedures for Data |

The project currently uses **Release 19** versions of these specifications.

---

## Evaluation

The evaluation dataset contains several categories of questions:

1. Direct-answer
2. Conditional-rule
3. Example-based
4. Exact-value
5. Reasoning
6. Hallucination / unanswerable
7. Entity-mismatch / trap
8. Out-of-context

Evaluation focuses on:

- Answer accuracy
- Retrieval quality
- Citation correctness
- Groundedness
- Abstention behavior
- False-answer rate
- False-premise rejection

### Run Evaluation

```bash
python eval/hallucination_eval.py
```

The evaluation reports the answer rate on in-corpus questions, correct-refusal rate on out-of-corpus questions, and unsupported claims identified by the verifier.

Additional evaluation questions can be added to:

```text
eval/eval_set.jsonl
```



---

## Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/SumitSingh711/3gpp-rag-chatbot

cd 3gpp-rag-chatbot
```

### 2. Setup Environment

Run:

```text
setup.bat
```

The setup script automatically creates the required environment and installs the project dependencies.

### 3. Add Custom 3GPP PDFs

Place custom 3GPP PDF specifications inside:

```text
data/raw/
```

The preprocessing pipeline is designed specifically for 3GPP-format PDFs and extracts the required metadata.

Then run:

```bash
python src/pipeline.py
```

The pipeline is **incremental**, so only newly added PDF files are processed.

### 4. Start the Streamlit Application

```bash
streamlit run src/app.py
```

The application will start locally and provide a URL for accessing the chatbot.

---

## Key Challenges & Solutions

| Challenge | Solution |
|---|---|
| 3GPP terminology | Dense + BM25 retrieval |
| Exact clause information | Clause-aware chunking |
| Hallucination | Grounding + claim verification |
| False premises | Evidence-based validation |
| Release confusion | Specification/release metadata |
| Out-of-corpus questions | Confidence gate |
| Multi-hop questions | Improved retrieval planned |



---

## Current Improvements

Planned and ongoing improvements include:

- Better retrieval confidence calibration
- Stronger out-of-corpus detection
- More robust citation enforcement
- Better multi-hop retrieval
- Larger hallucination evaluation dataset
- Support for additional specifications and releases
- Improved version-aware retrieval

---

## Core Design Principle

```text
Retrieve Evidence
       ↓
Generate From Evidence
       ↓
Verify Claims
       ↓
Answer Only If Supported
       ↓
Otherwise Abstain
```

The chatbot prioritizes **groundedness over answer completion**: it should provide an answer only when the available specification evidence supports it.

---

## Project Structure

```text
3gpp-rag-chatbot/
│
├── data/
│   └── raw/
│       └── 3GPP PDF specifications
│
├── src/
│   ├── pipeline.py
│   └── app.py
│
├── eval/
│   ├── hallucination_eval.py
│   └── eval_set.jsonl
│
├── setup.bat
└── README.md
```

---