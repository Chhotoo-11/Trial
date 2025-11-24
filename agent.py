import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_groq import ChatGroq
from langchain_core.tools import tool

# MCP server tools (pure functions)
from rag_mcp_server import (
    get_pdf_context,
    web_search_context,
    load_url_text
)

def get_llm():
    return ChatGroq(model_name="Gemma2-9b-It")

llm = get_llm()


# -------------------- WRAPPED TOOLS --------------------

@tool
def pdf_tool(pdf_path: str, query: str) -> str:
    """Fetch relevant PDF text based on user query."""
    return get_pdf_context(pdf_path, query)

@tool
def web_search_tool(query: str) -> str:
    """Fetch raw context from Arxiv, Wikipedia, DuckDuckGo."""
    return web_search_context(query)

@tool
def url_loader_tool(url: str) -> str:
    """Fetch raw text from a URL or YouTube link."""
    return load_url_text(url)

tools = [pdf_tool, web_search_tool, url_loader_tool]


# ---------------------- AGENT --------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an intelligent agent. "
         "Use tools when needed and then generate a final answer."),
        ("human", "{input}")
    ]
)

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

async def run_agent(user_input: str):
    result = await agent_executor.ainvoke({"input": user_input})
    return result["output"]
