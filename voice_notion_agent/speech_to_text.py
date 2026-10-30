import os
from groq import Groq
from config import STT_MODEL, GROQ_API_KEY
import logging


_logger = logging.getLogger(__name__)

client = Groq(api_key=GROQ_API_KEY)

def transcribe_audio(file_path: str) -> str:
    _logger.info(f"Transcribing the audio file: {file_path}")
    try:
        with open(file_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), f.read()),
                model=STT_MODEL,
            )
    except Exception:
        _logger.exception(f"Transcription failed for file: {file_path}")
        raise

    text = transcription.text
    if not text.strip():
        _logger.warning(f"Empty transcription for file: {file_path}")
    _logger.info(f"Transcription complete ({len(text)} characters)")
    _logger.debug(f"Transcript: {text}")
    return text

res = transcribe_audio(file_path=r'D:\projects\voice_agent\sample\sample_audio_file.mp3')
print(res)