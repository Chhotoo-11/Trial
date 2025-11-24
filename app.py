import streamlit as st
import asyncio
from agent import run_agent   # your agent logic

st.set_page_config(
    page_title="AI-Powered Knowledge Hub",
    page_icon="🤖",
    layout="wide",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': 'https://github.com/Chhotoo-11/RAG-Applications/issues',
        'About': "This app was created by Chhotoo Solanki."
    }
)

st.title("🤖 AI-Powered Knowledge Hub (Agent + MCP)")

# Sidebar
with st.sidebar:
    st.header("🛠️ App Mode")
    mode = st.selectbox(
        "Choose a mode (for UI only — agent decides tools automatically)",
        ["General Chat", "Chat with PDF", "URL Summarizer", "Web Search"],
    )

# ------------------- INPUT AREA -------------------
user_input = st.text_area("💬 Ask your question or paste a URL…")

uploaded_file = None
pdf_path = None

# If user selects PDF mode
if mode == "Chat with PDF":
    uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

# ------------------- PROCESS BUTTON -------------------
if st.button("Ask"):
    if not user_input and not uploaded_file:
        st.warning("Please enter a query or upload a PDF.")
        st.stop()

    query = user_input

    # If PDF uploaded — save it and instruct the agent
    if uploaded_file:
        pdf_path = f"temp_{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        query = (
            f"You have access to a tool called 'get_pdf_context' which can read PDF files.\n"
            f"The user has uploaded this PDF: {pdf_path}\n"
            f"Use the tool to extract relevant information.\n\n"
            f"User question: {user_input}"
        )

    # Run the agent
    with st.spinner("Thinking…"):
        answer = asyncio.run(run_agent(query))

    st.success(answer)
