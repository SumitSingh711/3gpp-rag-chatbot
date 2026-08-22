# 3GPP Standards RAG Chatbot

## Overview

A retrieval-grounded chatbot for answering questions from 3GPP specifications with **high factual accuracy, clause-level citations, and controlled abstention**.

The system is designed to prevent common LLM failures such as:

- hallucinating unsupported technical information
- mixing concepts across specifications
- confusing similar telecom procedures
- accepting false premises
- mixing different releases

**Core principle:** if the retrieved evidence is insufficient, the system abstains instead of guessing.

## Architecture

```text
3GPP Specifications
        |
        v
 Document Ingestion
 PDF Extraction
 Clause Chunking
 Metadata
        |
        v
     Indexing
 Dense Embeddings
 BM25
 Vector Index
        |
        v
    User Query
        |
        v
   Intent Router
      /        CHAT      RAG
    |         |
 Direct    Hybrid Retrieval
 Reply     Dense + BM25
              |
            RRF
              |
       Confidence Gate
          /              Fail       Pass
        |           |
     Abstain        v
                LLM Generation
                     |
                     v
              Claim Verification
                 /                     Fail       Pass
               |           |
            Abstain   Answer + Citation
```

## RAG Pipeline

1. **Intent Routing** — separates conversational queries from standards-related questions.
2. **Hybrid Retrieval** — combines dense semantic search with BM25 lexical search.
3. **RRF Fusion** — merges both ranked result sets.
4. **Confidence Gate** — rejects queries when relevant evidence is insufficient.
5. **Grounded Generation** — LLM answers only from retrieved context.
6. **Claim Verification** — every generated claim is checked against the retrieved evidence.
7. **Abstention** — unsupported answers are withheld.

## Anti-Hallucination Design

- Temperature 0 for deterministic generation.
- Every technical claim must have a specification/clause citation.
- No unsupported extrapolation.
- No assumptions about uplink/downlink symmetry.
- No borrowing behavior from unrelated releases.
- False premises are checked against evidence.
- If a claim cannot be verified, the system abstains.

Example:

```text
Question
   |
   v
Evidence found?
   |
  No ------> Abstain
   |
  Yes
   |
   v
Generate
   |
   v
Claims verified?
   |
  No ------> Abstain
   |
  Yes
   |
   v
Answer + Citation
```

## Indexed Specifications

| Specification | Title |
|---|---|
| TS 21.905 | Vocabulary for 3GPP Specifications |
| TS 23.501 | 5G System Architecture |
| TS 23.502 | 5G System Procedures |
| TS 24.301 | NAS Protocol for EPS |
| TS 36.331 | LTE RRC Protocol |
| TS 38.300 | NR Overall Description |
| TS 38.214 | NR Physical Layer Procedures for Data |

The project currently uses Release 19 versions.

## Evaluation Dataset

Questions are categorized into:

1. Direct-answer
2. Conditional-rule
3. Example-based
4. Exact-value
5. Reasoning
6. Hallucination / unanswerable
7. Entity-mismatch / trap
8. Out-of-context

The evaluation focuses on:

- answer accuracy
- retrieval quality
- citation correctness
- groundedness
- abstention behavior
- false-answer rate
- false-premise rejection

## Key Challenges

| Challenge | Solution |
|---|---|
| 3GPP terminology | Dense + BM25 retrieval |
| Exact clause information | Clause-aware chunking |
| Hallucination | Grounding + claim verification |
| False premises | Evidence-based validation |
| Release confusion | Specification/release metadata |
| Out-of-corpus questions | Confidence gate |
| Multi-hop questions | Improved retrieval planned |

## Current Improvements

- Better retrieval confidence calibration
- Stronger out-of-corpus detection
- More robust citation enforcement
- Better multi-hop retrieval
- Larger hallucination evaluation dataset
- Support for more specifications and releases
- Improved version-aware retrieval

## Core Design Principle

```text
Retrieve evidence
      ↓
Generate from evidence
      ↓
Verify claims
      ↓
Answer only if supported
      ↓
Otherwise abstain
```
