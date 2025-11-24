import asyncio
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_groq import ChatGroq

# Import the same functions that are MCP tools
from rag_mcp_server import get_pdf_context, web_search_context, load_url_text


def get_llm():
    return ChatGroq(model_name="Gemma2-9b-It")


llm = get_llm()

tools = [
    Tool.from_function(
        func=get_pdf_context,
        name="get_pdf_context",
        description=(
            "Use this to fetch relevant text from a PDF when the user asks "
            "questions about an uploaded document. "
            "Inputs: pdf_path (str), query (str)."
        ),
    ),
    Tool.from_function(
        func=web_search_context,
        name="web_search_context",
        description=(
            "Use this to gather up-to-date information from Arxiv, Wikipedia "
            "and the web for general knowledge or research queries."
        ),
    ),
    Tool.from_function(
        func=load_url_text,
        name="load_url_text",
        description=(
            "Use this when the user provides a URL or YouTube link and wants "
            "a summary or analysis of its content."
        ),
    ),
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant that can use tools to answer questions. "
            "Decide which tool(s) to call when needed. Always explain your final answer clearly.",
        ),
        ("human", "{input}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


async def run_agent(user_input: str):
    result = await agent_executor.ainvoke({"input": user_input})
    return result["output"]
