from langchain_core.tools import tool
from voice_notion_agent.agents.email_expert import build_email_expert_agent
from langchain_core.messages import HumanMessage

@tool
async def email_expert_tool(req:str) -> str:
	"""
	Recives User Request/Query and returns the output 
	Related to Email Tasks
	"""
	email_agent = await build_email_expert_agent()
	res = await email_agent.ainvoke({"messages" : HumanMessage(content=req)})
	return res	