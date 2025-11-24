import re
from langchain_community.document_loaders import (
    PyPDFLoader,
    YoutubeLoader,
    UnstructuredURLLoader
)
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Embeddings for RAG
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def normalize_youtube_url(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return f"https://www.youtube.com/watch?v={match.group(1)}" if match else url


# -------------------------------
# PURE TOOL FUNCTIONS (NO LLM)
# -------------------------------

def get_pdf_context(pdf_path: str, query: str, k: int = 4) -> str:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    results = retriever.get_relevant_documents(query)
    return "\n\n".join([d.page_content for d in results])


def web_search_context(query: str) -> str:
    arxiv = ArxivAPIWrapper(top_k_results=1)
    wiki = WikipediaAPIWrapper(top_k_results=1)
    ddg = DuckDuckGoSearchRun()

    parts = []
    try: parts.append(arxiv.run(query))
    except: pass
    try: parts.append(wiki.run(query))
    except: pass
    try: parts.append(ddg.run(query))
    except: pass

    combined = "\n\n".join([p for p in parts if p.strip()])
    return combined or "NO_CONTEXT_FOUND"


def load_url_text(url: str) -> str:
    if "youtube.com" in url or "youtu.be" in url:
        url = normalize_youtube_url(url)
        try:
            loader = YoutubeLoader.from_youtube_url(
                url, add_video_info=True, language=["en"]
            )
            docs = loader.load()
        except:
            docs = [Document(page_content=f"Unable to load YouTube: {url}")]
    else:
        loader = UnstructuredURLLoader(urls=[url])
        docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    return "\n\n".join([d.page_content for d in splits])
