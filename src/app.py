"""
app.py — Streamlit UI.  Run with: streamlit run src/app.py
"""
import streamlit as st
from rag_chatbot import RAGChatbot, Turn

st.set_page_config(page_title="3GPP RAG Assistant", page_icon="📡")
st.title("📡 3GPP Standards RAG Assistant")
st.caption("Technical questions are answered strictly from indexed 3GPP TS/TR documents, with per-clause citations. Everything else gets a normal chat reply.")

if "bot" not in st.session_state:
    with st.spinner("Loading indexes..."):
        st.session_state.bot = RAGChatbot()
if "turns" not in st.session_state:
    st.session_state.turns: list[Turn] = []

for t in st.session_state.turns:
    with st.chat_message(t.role):
        st.markdown(t.content)

question = st.chat_input("Ask about 5G NR, RRC, NAS, QoS, handover procedures, etc. — or just say hi.")
if question:
    st.session_state.turns.append(Turn(role="user", content=question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            r = st.session_state.bot.ask(question, history=st.session_state.turns[:-1])

        st.markdown(r.answer)

        if r.mode == "rag" and not r.refused:
            st.caption("🔎 answered from retrieved 3GPP context")

        if r.unsupported_sentences:
            st.warning("Flagged as possibly unsupported by retrieved context:\n\n" +
                       "\n".join(f"- {s}" for s in r.unsupported_sentences))

        if r.sources:
            with st.expander("Sources"):
                for s in r.sources:
                    st.markdown(
                        f"**{s.display_citation()}** — "
                        f"dense sim `{s.dense_sim:.3f}`, bm25 `{s.bm25_score:.2f}`, "
                        f"fused (rank) `{s.fused_score:.4f}`"
                    )
                    st.text(s.text[:400] + ("..." if len(s.text) > 400 else ""))

    st.session_state.turns.append(Turn(role="assistant", content=r.answer))
