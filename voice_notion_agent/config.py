import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
CHAT_MODEL = os.getenv('CHAT_MODEL','gemini-3.1-flash-lite')
STT_MODEL = os.getenv('STT_MODEL','whisper-large-v3-turbo')
TTS_MODEL = os.getenv('TTS_MODEL','gemini-3.8-flash-tts')



def validate() -> None:
	if not GROQ_API_KEY or TAVILY_API_KEY or LANGSMITH_API_KEY or GEMINI_API_KEY:
		raise ValueError('Some of ENV Variables are not set')