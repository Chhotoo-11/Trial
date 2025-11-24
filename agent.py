import asyncio
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_groq import ChatGroq

# Import MCP server tools (pure functions, no LLM inside)
from rag_mcp_server import (
    get_pdf_context,
    web_search_context,
    load_url_text
)

# ----------------------- LLM -------------------------
def get_llm():
    return ChatGroq(model_name="Gemma2-9b-It")

llm = get_llm()


# ----------------------- TOOLS -------------------------
tools = [
    Tool.from_function(
        func=get_pdf_context,
        name="get_pdf_context",
        description=(
            "Use this tool to fetch relevant text from a PDF. "
            "Inputs: pdf_path (str), query (str)."
        ),
    ),
    Tool.from_function(
        func=web_search_context,
        name="web_search_context",
        description=(
            "Use this tool to gather up-to-date information from "
            "Arxiv, Wikipedia, and DuckDuckGo based on a query."
        ),
    ),
    Tool.from_function(
        func=load_url_text,
        name="load_url_text",
        description=(
            "Use this tool when the user provides a URL or YouTube link. "
            "It loads and returns the raw text content."
        ),
    ),
]


# ----------------------- PROMPT -------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an intelligent AI assistant with access to tools. "
            "Think step-by-step and decide which tool to call when needed. "
            "After using tools, synthesize a final, clear answer."
        ),
        ("human", "{input}")
    ]
)


# ----------------------- AGENT -------------------------
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


# ----------------------- RUN FUNCTION -------------------------
async def run_agent(user_input: str):
    result = await agent_executor.ainvoke({"input": user_input})
    return result["output"]
