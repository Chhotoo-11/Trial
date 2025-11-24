import streamlit as st
import asyncio
from agent import run_agent   # the agent we defined above

st.title("🤖 AI-Powered Knowledge Hub (Agent + MCP)")

mode = st.sidebar.selectbox(
    "Mode (just for UI hint, agent still decides tools)",
    ["General Chat", "Chat with PDF", "URL/YouTube Summarizer", "Web Search"],
)

user_input = st.text_area("Ask your question or paste a URL…")

# For PDF mode you can pass pdf_path info in the prompt itself
uploaded_file = None
if mode == "Chat with PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if st.button("Ask"):
    query = user_input
    if uploaded_file:
        pdf_path = f"temp_{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # hint the agent that a pdf exists and how to use it
        query = (
            f"You have access to a tool 'get_pdf_context' that can read this "
            f"PDF at path: {pdf_path}. Use it to answer my question.\n\n"
            f"Question: {user_input}"
        )

    answer = asyncio.run(run_agent(query))
    st.write(answer)
