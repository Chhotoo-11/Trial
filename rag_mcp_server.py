import asyncio
import os
import re

from mcp.server.fastmcp import FastMCP
from langchain_community.document_loaders import (
    PyPDFLoader, YoutubeLoader, UnstructuredURLLoader
)
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP("RAG-MCP-Server")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def normalize_youtube_url(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return f"https://www.youtube.com/watch?v={match.group(1)}" if match else url


# 1) TOOL: get relevant context from a PDF (NO LLM)
@mcp.tool()
async def get_pdf_context(pdf_path: str, query: str, k: int = 4) -> str:
    """
    Returns relevant context chunks from the given PDF for the user query.
    No LLM is called here – only retrieval.
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200
    )
    splits = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(d.page_content for d in relevant_docs)
    return context


# 2) TOOL: web search context (Arxiv + Wiki + DDG) – NO LLM
@mcp.tool()
async def web_search_context(query: str) -> str:
    """
    Returns combined raw context from Arxiv, Wikipedia and DuckDuckGo.
    No summarization or LLM here.
    """
    arxiv = ArxivAPIWrapper(top_k_results=1)
    wiki = WikipediaAPIWrapper(top_k_results=1)
    ddg = DuckDuckGoSearchRun()

    parts = []
    try:
        parts.append(arxiv.run(query))
    except Exception:
        pass
    try:
        parts.append(wiki.run(query))
    except Exception:
        pass
    try:
        parts.append(ddg.run(query))
    except Exception:
        pass

    combined = "\n\n".join(p for p in parts if p.strip())
    return combined or "NO_CONTEXT_FOUND"


# 3) TOOL: load raw text from URL / YouTube – NO summarization here
@mcp.tool()
async def load_url_text(url: str) -> str:
    """
    Loads raw text from a URL or YouTube video (transcript/description).
    Returns text only – summarization is done by the LLM agent.
    """
    if any(d in url for d in ["youtube.com", "youtu.be"]):
        url = normalize_youtube_url(url)
        try:
            loader = YoutubeLoader.from_youtube_url(
                url, add_video_info=True, language=["en"]
            )
            docs = loader.load()
        except Exception:
            docs = [Document(page_content=f"Video metadata only: {url}")]
    else:
        loader = UnstructuredURLLoader(
            urls=[url],
            ssl_verify=True,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=200
    )
    splits = splitter.split_documents(docs)
    text = "\n\n".join(d.page_content for d in splits)
    return text


if __name__ == "__main__":
    asyncio.run(mcp.run())
