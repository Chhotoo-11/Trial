import asyncio
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool

# Import MCP server tools
from rag_mcp_server import (
    get_pdf_context,
    web_search_context,
    load_url_text
)

# ------------ LLM -------------
def get_llm():
    return ChatGroq(model_name="Gemma2-9b-It")

llm = get_llm()


# ------------ TOOLS -------------
@tool
def pdf_tool(pdf_path: str, query: str) -> str:
    """Fetch relevant PDF context."""
    return get_pdf_context(pdf_path, query)

@tool
def web_tool(query: str) -> str:
    """Web search using Arxiv, Wikipedia, DuckDuckGo."""
    return web_search_context(query)

@tool
def url_tool(url: str) -> str:
    """Load raw text from URL or YouTube."""
    return load_url_text(url)


tools = {
    "get_pdf_context": pdf_tool,
    "web_search_context": web_tool,
    "load_url_text": url_tool
}


# ------------ AGENT LOGIC -------------

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant with access to tools. "
     "When needed, call a tool using this format:\n\n"
     "<tool_name>[argument]\n\n"
     "After receiving tool result, continue your reasoning."),
    ("human", "{input}")
])

def call_tools(query: str):
    """Check if input matches tool call pattern."""
    if query.startswith("get_pdf_context"):
        _, pdf_path, q = query.split("|||")
        return tools["get_pdf_context"].invoke({"pdf_path": pdf_path, "query": q})
    if query.startswith("web_search_context"):
        return tools["web_search_context"].invoke({"query": query.replace("web_search_context ", "")})
    if query.startswith("load_url_text"):
        return tools["load_url_text"].invoke({"url": query.replace("load_url_text ", "")})
    return None


# Runnable pipeline (simple agent)
agent = (
    {"input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


async def run_agent(user_input: str):
    # Step 1: Ask LLM how to answer
    action = agent.invoke(user_input)

    # Step 2: Check for tool call
    tool_result = call_tools(action)

    if tool_result:
        # Step 3: Pass tool result back to LLM
        followup = agent.invoke(
            f"Tool result:\n{tool_result}\n\nNow answer the question."
        )
        return followup

    # No tool needed → direct answer
    return action
