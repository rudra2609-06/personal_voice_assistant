from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from voice_notion_agent.config import CHAT_MODEL, TAVILY_API_KEY, GEMINI_API_KEY
import logging

_model = ChatGoogleGenerativeAI(
    model=CHAT_MODEL, temperature=0.3, api_key=GEMINI_API_KEY
)

_tavily = TavilySearch(api_key=TAVILY_API_KEY, max_results=5)

_logger = logging.getLogger(__name__)


@tool
def research_tool(topic: str) -> str:
    """Research a topic on the web and return a short summary."""
    _logger.info("Researching the topic")
    search_results = _tavily.invoke({"query": topic})
    _logger.info(f"Raw Search Results: {search_results}")

    prompt = (
        f"Summarize these search results about '{topic}' in 3 to 4 short sentences. Generate results quickly"
        f"Use plain text only.\n\nResults:\n{search_results}"
    )
    response = _model.invoke(prompt)
    return response.text

if __name__ == '__main__':
    res = research.invoke({'topic':'python programming'})
    print(res)	