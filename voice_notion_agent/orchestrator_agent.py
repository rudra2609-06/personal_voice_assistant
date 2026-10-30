import asyncio
import logging

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from voice_notion_agent import config
from voice_notion_agent.tools.researcher_tool import research_tool
from voice_notion_agent.tools.notion_expert_tool import notion_expert_tool
from voice_notion_agent.tools.emal_expert_tool import email_expert_tool

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an orchestrator voice assistant.
Tools:
- research: searches the web and summarizes the results.
- notion_expert: handles any Notion task, such as searching, creating or updating pages.
For tasks that need these tools, use them. For everything else, answer on your own.

Your replies are read aloud, so keep them short and plain: no markdown, lists, emojis, or URLs.
Only report what the tools return, and never make up results."""


async def build_agent():
    model = ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL, api_key=config.GEMINI_API_KEY, temperature=0.3
    )
    return create_agent(
        model=model,
        tools=[research_tool, notion_expert_tool, email_expert_tool],
        system_prompt=SYSTEM_PROMPT,
    )


async def main():
    agent = await build_agent()
    user_input = "fetch 3 emails from email account"
    res = await agent.ainvoke({"messages": [HumanMessage(content=user_input)]})
    print(res["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())
