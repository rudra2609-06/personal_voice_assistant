from langchain.agents import create_agent
from voice_notion_agent.config import CHAT_MODEL, GEMINI_API_KEY
from voice_notion_agent.mcp_client import mcp_client

SYSTEM_PROMPT = """
You are an email assistant with tools connected to the user's Gmail account.
Use the available Gmail tools to search for and retrieve requested messages.
Never say that you cannot access email without trying the relevant tool.
Only report messages and details returned by the tools. If a tool reports an
authentication or access error, explain that email access needs to be configured.
"""


async def build_email_expert_agent():
    """Build an email agent with the connected Gmail tools."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = ChatGoogleGenerativeAI(
        model=CHAT_MODEL, api_key=GEMINI_API_KEY, temperature=0.3
    )
    mcp_tools = await mcp_client.get_tools(server_name="gmail")
    agent = create_agent(model=model, tools=mcp_tools, system_prompt=SYSTEM_PROMPT)
    return agent
