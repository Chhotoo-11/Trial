import asyncio
from langchain.agents import AgentType, initialize_agent
from langchain_groq import ChatGroq
from langchain.tools import Tool   # available in LC 0.1.x

# MCP server tools
from rag_mcp_server import (
    get_pdf_context,
    web_search_context,
    load_url_text
)


def get_llm():
    return ChatGroq(model_name="Gemma2-9b-It")

llm = get_llm()


# -------------------- TOOL WRAPPERS --------------------

# Since LC old agent accepts only ONE string input,
# We encode PDF path + query together using delimiter "|||"

def pdf_wrapper(input_str: str) -> str:
    """Input format: 'pdf_path|||question'"""
    pdf_path, query = input_str.split("|||")
    return get_pdf_context(pdf_path, query)

tools = [
    Tool(
        name="get_pdf_context",
        func=pdf_wrapper,
        description="Fetch PDF context. Format: 'PDF_PATH|||QUERY'"
    ),
    Tool(
        name="web_search_context",
        func=web_search_context,
        description="Search Arxiv, Wikipedia, and DuckDuckGo."
    ),
    Tool(
        name="load_url_text",
        func=load_url_text,
        description="Load and return raw text from URL or YouTube link."
    ),
]


# -------------------- AGENT --------------------

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


# -------------------- RUN FUNCTION --------------------

async def run_agent(user_input: str):
    result = await agent.ainvoke(user_input)
    return result["output"]
