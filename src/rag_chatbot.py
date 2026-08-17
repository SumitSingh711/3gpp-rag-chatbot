"""
rag_chatbot.py
The user-facing chatbot. Routes every message through a small LLM-based
intent classifier first, then branches into one of two modes:

 - RAG mode: grounded generation over retrieved 3GPP spec chunks, with the
   full anti-hallucination stack (confidence gate, forced citations,
   post-hoc verification).
 - CHAT mode: normal conversational reply (greetings, thanks, "what can you
   do", follow-up chit-chat) — no retrieval, no citation requirement, just
   a helpful assistant reply. This is what makes it behave like an actual
   chatbot instead of a form that only accepts spec questions.
"""

import json
import os
import re
from dataclasses import dataclass
from dotenv import load_dotenv
from groq import Groq

from retriever import HybridRetriever, RetrievedChunk

load_dotenv()  

GEN_MODEL = os.environ.get("GEN_MODEL")
FAST_MODEL = os.environ.get("VERIFY_MODEL")

ROUTER_PROMPT = """You are a routing classifier for a 3GPP telecom-standards RAG chatbot.
Decide whether the user's LATEST message is small talk / meta conversation about the
assistant itself (CHAT), or an actual factual question (RAG).

Route to CHAT ONLY for: greetings, thanks, goodbyes, small talk with no factual content,
meta questions about the assistant itself ("what can you do", "who are you", "how do you
work"), or requests to rephrase/shorten/clarify a PREVIOUS answer already given earlier
in this conversation.

Route everything else to RAG — including questions that are NOT about telecom or 3GPP.
Do not try to judge whether a question is in-scope or answerable; that is not this
classifier's job. The RAG pipeline has its own retrieval-confidence gate that is
responsible for refusing questions the corpus can't support (e.g. "what's the release
date of the next iPhone" or "Nokia's 2023 stock price") — those must still be routed to
RAG so that gate can do its job and refuse properly, rather than being answered here
ungrounded. When in doubt, choose RAG.

Respond with exactly one word: RAG or CHAT. No punctuation, no explanation.
"""

CHAT_SYSTEM_PROMPT = """You are a friendly assistant for a 3GPP telecom-standards RAG chatbot.
The router has classified the user's current message as small talk or a meta question
about you (the assistant) — NOT a factual question. Reply naturally and briefly, the way
a normal chatbot would, for greetings/thanks/"what can you do"/"rephrase that" type
messages. If relevant, mention you can answer 5G/LTE questions grounded in indexed 3GPP
specs with clause-level citations, but don't force this into every reply.

Do NOT answer any factual question in this mode, even a simple-sounding one, even about
telecom, even if you're confident you know the answer. If the user's message actually
contains a real factual question, say so plainly and tell them you can only answer
questions that are checked against the indexed 3GPP specs, not from general knowledge —
do not just answer it anyway.
"""

SYSTEM_PROMPT = """You are a 3GPP telecom standards assistant. You answer ONLY using the
CONTEXT chunks provided below, which are extracted verbatim from 3GPP Technical
Specifications (TS) and Technical Reports (TR).

Rules (follow strictly):
1. Do not use any knowledge outside the provided CONTEXT, even if you believe you know
   the answer from training data. 3GPP specs are versioned and change between releases;
   only the attached context is authoritative for this answer.
2. Every factual sentence in your answer must end with a citation tag in the form
   [SPEC clause CLAUSE_NUMBER], copied exactly from the context chunk it came from.
3. If the context does not contain enough information to answer the question, say
   exactly: "The provided context does not specify this." Do not guess or extrapolate.
4. If different context chunks conflict (e.g. different spec versions), prefer the latest one
5. Keep answers precise and technical. Do not pad with generic telecom background that
   isn't in the context.
6. Do not infer information that is not explicitly stated in the retrieved
   3GPP context. In particular, do not assume that a property stated for
   downlink also applies to uplink, or vice versa.
"""

VERIFY_PROMPT = """You are a strict fact-checker. You will be given CONTEXT passages and a DRAFT
ANSWER that claims to be grounded in that context. For each sentence in the draft answer,
check whether it is directly supported by the CONTEXT.

Respond with a JSON list of objects: [{"sentence": "...", "supported": true/false}]
Do not add commentary. If a sentence is a citation tag only or a refusal statement, mark it
supported: true.
"""


@dataclass
class ChatResponse:
    answer: str
    sources: list[RetrievedChunk]
    refused: bool
    unsupported_sentences: list[str]
    mode: str = "rag"  # "rag" or "chat"

@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str


class RAGChatbot:
    def __init__(self, groq_api_key: str | None = None):
        self.client = Groq(api_key=groq_api_key or os.environ["GROQ_API_KEY"])
        self.retriever = HybridRetriever()

    # ---------- routing ----------

    def _classify_intent(self, question: str, history: list[Turn]) -> str:
       
        recent = history[-4:] if history else []
        history_text = "\n".join(f"{t.role}: {t.content}" for t in recent)

        try:
            resp = self.client.chat.completions.create(
                model=FAST_MODEL,
                temperature=0.0,
                max_tokens=50,
                messages=[
                    {"role": "system", "content": ROUTER_PROMPT},
                    {
                        "role": "user",
                        "content": f"RECENT CONVERSATION:\n{history_text}\n\nLATEST MESSAGE: {question}",
                    },
                ],
            )
            label = resp.choices[0].message.content.strip().upper()

            return "chat" if label == "CHAT" else "rag"
        
        except Exception as e:
            return "rag"

    # ---------- RAG mode ----------

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(f"--- {c.citation()} ---\n{c.text}" for c in chunks)

    def _generate(self, question: str, context: str) -> str:
        resp = self.client.chat.completions.create(
            model=GEN_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer using only the context above.",
                },
            ],
        )
        return resp.choices[0].message.content.strip()

    def _verify(self, context: str, draft_answer: str) -> list[str]:
        """Returns list of sentences flagged as unsupported by context."""
        try:
            resp = self.client.chat.completions.create(
                model=FAST_MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": VERIFY_PROMPT},
                    {
                        "role": "user",
                        "content": f"CONTEXT:\n{context}\n\nDRAFT ANSWER:\n{draft_answer}",
                    },
                ],
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
            checks = json.loads(raw)
            return [c["sentence"] for c in checks if not c.get("supported", True)]
        except Exception:
            return ["[verification step failed — treat this answer with caution]"]

    def _answer_rag(self, question: str, k: int) -> ChatResponse:
        chunks, confident = self.retriever.retrieve_with_gate(question, k=k)

        if not confident:
            return ChatResponse(
                answer=(
                    "I don't have a confidently relevant passage in the indexed 3GPP "
                    "specifications for this question. Rather than guess, I'm refusing "
                    "to answer. Try rephrasing with the specific spec number/clause, or "
                    "confirm the relevant TS/TR is included in the corpus."
                ),
                sources=[],
                refused=True,
                unsupported_sentences=[],
                mode="rag",
            )

        context = self._format_context(chunks)
        draft = self._generate(question, context)
        unsupported = self._verify(context, draft)

        if unsupported:
            return ChatResponse(
                answer=(
                    "I cannot provide a reliable answer from the retrieved "
                    "3GPP context because one or more generated claims could "
                    "not be verified against the source."
                ),
                sources=chunks,
                refused=True,
                unsupported_sentences=unsupported,
                mode="rag",
            )

        return ChatResponse(
            answer=draft,
            sources=chunks,
            refused=False,
            unsupported_sentences=[],
            mode="rag",
        )

    # ---------- CHAT mode ----------

    def _answer_chat(self, question: str, history: list[Turn]) -> ChatResponse:
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        for t in history[-6:]:
            messages.append({"role": t.role, "content": t.content})
        messages.append({"role": "user", "content": question})

        resp = self.client.chat.completions.create(
            model=FAST_MODEL, 
            temperature=0.4,
            messages=messages,
        )
        return ChatResponse(
            answer=resp.choices[0].message.content.strip(),
            sources=[],
            refused=False,
            unsupported_sentences=[],
            mode="chat",
        )

    # ---------- public entry point ----------

    def ask(self, question: str, k: int = 6, history: list[Turn] | None = None) -> ChatResponse:
        history = history or []
        intent = self._classify_intent(question, history)

        if intent == "chat":
            return self._answer_chat(question, history)
        return self._answer_rag(question, k=k)


def main():
    bot = RAGChatbot()
    history: list[Turn] = []
    print("3GPP RAG Chatbot (type 'exit' to quit)\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        r = bot.ask(q, history=history)
        print(f"\nBot [{r.mode}]: {r.answer}\n")

        if r.unsupported_sentences:
            print("⚠️  Flagged as possibly unsupported by context:")
            for s in r.unsupported_sentences:
                print(f"   - {s}")
        if r.sources:
            print("\nSources:")
            for s in r.sources:
                print(f"   - {s.citation()}  (fused score {s.score:.4f})")
        print()

        history.append(Turn(role="user", content=q))
        history.append(Turn(role="assistant", content=r.answer))


if __name__ == "__main__":
    main()
