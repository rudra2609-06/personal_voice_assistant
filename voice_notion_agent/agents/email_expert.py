from voice_notion_agent.mcp_client import mcp_client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from voice_notion_agent.config import CHAT_MODEL, GEMINI_API_KEY

model = ChatGoogleGenerativeAI(
    model=CHAT_MODEL, temperature=0.3, api_key=GEMINI_API_KEY
)


SYSTEM_PROMPT = """
You are a Email expert agent that can perform various tasks using mcp tools.
You have access to email tools that allow you to interact with your Email account to retrieve and manage emails.
"""


async def build_email_expert_agent():
    """
    This agent has access to all the mcp_tools from gmail
    """
    model = ChatGoogleGenerativeAI(
        model=CHAT_MODEL, api_key=GEMINI_API_KEY, temperature=0.3
    )
    mcp_tools = await mcp_client.get_tools(server_name="gmail")
    agent = create_agent(model=model, tools=mcp_tools, system_prompt=SYSTEM_PROMPT)
    return agent
