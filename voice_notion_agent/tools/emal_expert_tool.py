from langchain_core.tools import tool
from voice_notion_agent.agents.email_expert import build_email_expert_agent
from langchain_core.messages import HumanMessage


@tool
async def email_expert_tool(request: str) -> str:
    """Use Gmail tools to search, read, or manage the user's email as requested."""
    email_agent = await build_email_expert_agent()
    result = await email_agent.ainvoke(
        {"messages": [HumanMessage(content=request)]}
    )
    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError("The Gmail agent returned no messages.")

    reply = getattr(messages[-1], "text", "")
    if not isinstance(reply, str) or not reply.strip():
        raise RuntimeError("The Gmail agent returned an empty reply.")
    return reply.strip()