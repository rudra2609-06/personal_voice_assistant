from langchain_core.tools import tool
from voice_notion_agent.agents.notion_expert import build_notion_agent
from langchain_core.messages import HumanMessage


@tool
async def notion_expert_tool(request: str) -> str:
    """Handle any Notion task. Pass the user's full request as plain text."""
    notion_agent = await build_notion_agent()
    res = await notion_agent.ainvoke({"messages": [HumanMessage(content=request)]})
    return res

