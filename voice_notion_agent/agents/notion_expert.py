from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from voice_notion_agent.config import CHAT_MODEL, GEMINI_API_KEY
from voice_notion_agent.mcp_client import mcp_client

import logging
import asyncio

_logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a Notion expert agent that can perform various tasks using mcp tools.
You have access to notion tools that allow you to interact with your Notion account to retrieve and manage pages."""


async def build_notion_agent():
    """
    This agent has access to all the mcp_tools from notion
    """
    model = ChatGoogleGenerativeAI(
        model=CHAT_MODEL, api_key=GEMINI_API_KEY, temperature=0.3
    )
    mcp_tools = await mcp_client.get_tools(server_name="notion")
    agent = create_agent(system_prompt=SYSTEM_PROMPT, model=model, tools=mcp_tools)
    return agent
