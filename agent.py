import asyncio
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# MCP tools
from rag_mcp_server import (
    get_pdf_context,
    web_search_context,
    load_url_text
)

# LLM
def get_llm():
    return ChatGroq(model_name="llama-3.3-70b-versatile")

llm = get_llm()

# TOOL MAP
TOOLS = {
    "get_pdf_context": get_pdf_context,
    "web_search_context": web_search_context,
    "load_url_text": load_url_text,
}

# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. "
        "If a tool is needed, output ONLY in this exact format:\n"
        "<tool> [arg1] [arg2]\n"
        "Example: get_pdf_context[temp.pdf|||what is abstract]\n"
        "Otherwise, answer directly."
    ),
    ("human", "{input}")
])

# LLM pipeline
agent = (
    {"input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# UTILITY — detect tool call
TOOL_PATTERN = r"(\w+)\[(.*?)\]"

def detect_tool_call(text):
    match = re.search(TOOL_PATTERN, text)
    if not match:
        return None, None
    tool_name, arg_string = match.group(1), match.group(2)
    return tool_name, arg_string


# MAIN EXECUTION
async def run_agent(user_input: str):
    # Step 1: LLM decides whether to call a tool
    model_output = agent.invoke(user_input)

    tool_name, args = detect_tool_call(model_output)

    if tool_name and tool_name in TOOLS:
        # PDF calls use "path|||query"
        if tool_name == "get_pdf_context":
            pdf_path, query = args.split("|||")
            tool_result = TOOLS[tool_name](pdf_path.strip(), query.strip())
        else:
            tool_result = TOOLS[tool_name](args.strip())

        # Step 2: Send tool result back to LLM for final answer
        final_answer = agent.invoke(
            f"Tool result:\n{tool_result}\n\nNow answer the question."
        )
        return final_answer

    # If no tool call → return LLM response directly
    return model_output
