import asyncio
import base64
import logging
import os
import tempfile
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from google import genai
from groq import Groq
from langchain.messages import HumanMessage
from pydantic import BaseModel, Field

from voice_notion_agent import config
from voice_notion_agent.orchestrator_agent import build_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
MAX_AUDIO_SIZE = 25 * 1024 * 1024
MAX_SESSIONS = 100
MAX_HISTORY_MESSAGES = 40

sessions: OrderedDict[str, list[Any]] = OrderedDict()
session_locks: dict[str, asyncio.Lock] = {}
sessions_guard = asyncio.Lock()

groq_client = Groq(api_key=config.GROQ_API_KEY)
gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)


class TextRequest(BaseModel):
    text: str
    session_id: str = Field(min_length=1, max_length=128)


def _trim_history(messages: list[Any]) -> list[Any]:
    """Keep recent complete turns, starting at a user message."""
    turns: list[list[Any]] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            turns.append([])
        if turns:
            turns[-1].append(message)

    while len(turns) > 1 and sum(map(len, turns)) > MAX_HISTORY_MESSAGES:
        turns.pop(0)
    return [message for turn in turns for message in turn]


async def _get_session_lock(session_id: str) -> asyncio.Lock:
    async with sessions_guard:
        if session_id not in session_locks:
            sessions[session_id] = []
            session_locks[session_id] = asyncio.Lock()
        else:
            sessions.move_to_end(session_id)

        while len(sessions) > MAX_SESSIONS:
            oldest = next(
                (key for key, lock in session_locks.items()
                 if key != session_id and not lock.locked()),
                None,
            )
            if oldest is None:
                break
            sessions.pop(oldest)
            session_locks.pop(oldest)
        return session_locks[session_id]


async def _run_agent_turn(session_id: str, text: str) -> str:
    lock = await _get_session_lock(session_id)
    async with lock:
        try:
            history = sessions[session_id]
            result = await app.state.agent.ainvoke(
                {"messages": [*history, HumanMessage(content=text)]}
            )
            messages = result["messages"]
            if not messages:
                raise ValueError("Agent returned no messages")
            sessions[session_id] = _trim_history(messages)
            reply = getattr(messages[-1], "text", "")
            return reply.strip() if isinstance(reply, str) and reply.strip() else "Done."
        except Exception:
            logger.exception("Assistant agent failed.")
            raise HTTPException(
                status_code=500,
                detail="The assistant ran into a problem. Please try again.",
            ) from None


def _audio_suffix(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix in {".webm", ".ogg", ".mp4", ".wav", ".mp3", ".m4a"}:
        return suffix
    return {
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/mp4": ".mp4",
        "audio/wav": ".wav",
        "audio/mpeg": ".mp3",
    }.get((upload.content_type or "").split(";", 1)[0], ".webm")


async def _save_upload(upload: UploadFile) -> Path:
    temporary = tempfile.NamedTemporaryFile(
        prefix="voice-agent-", suffix=_audio_suffix(upload), delete=False
    )
    path = Path(temporary.name)
    total_size = 0
    try:
        with temporary:
            while chunk := await upload.read(1024 * 1024):
                total_size += len(chunk)
                if total_size > MAX_AUDIO_SIZE:
                    raise HTTPException(status_code=413, detail="Audio must be 25 MB or smaller.")
                temporary.write(chunk)
        if total_size == 0:
            raise HTTPException(status_code=400, detail="The uploaded audio is empty.")
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _transcribe(path: Path) -> str:
    with path.open("rb") as audio_file:
        result = groq_client.audio.transcriptions.create(
            file=(path.name, audio_file.read()),
            model=config.STT_MODEL,
        )
    return result.text


def _synthesize_sync(text: str) -> bytes:
    result = gemini_client.interactions.create(
        model=config.TTS_MODEL,
        input=[{
            "type": "user_input",
            "content": [{
                "type": "text",
                "text": text,
                "annotations": [{
                    "type": "speech_metadata",
                    "style": "cheerful and friendly",
                }],
            }],
        }],
        response_format={"type": "audio"},
        generation_config={"speech_config": [{"voice": "Kore"}]},
    )
    return base64.b64decode(result.output_audio.data)


async def _synthesize(text: str) -> bytes | None:
    try:
        audio = await asyncio.to_thread(_synthesize_sync, text)
        return audio or None
    except Exception:
        logger.exception("Speech synthesis failed.")
        return None


async def _make_response(session_id: str, transcript: str) -> dict[str, str | None]:
    reply = await _run_agent_turn(session_id, transcript)
    audio = await _synthesize(reply)
    return {
        "transcript": transcript,
        "reply": reply,
        "audio_base64": base64.b64encode(audio).decode("ascii") if audio else None,
        "audio_mime": "audio/wav",
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Voice Notion Agent.")
    app.state.agent = await build_agent()
    try:
        yield
    finally:
        app.state.agent = None
        sessions.clear()
        session_locks.clear()
        logger.info("Voice Notion Agent stopped.")


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exception: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request. Please check the submitted data."},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exception: Exception) -> JSONResponse:
    logger.exception("Unexpected server error.")
    return JSONResponse(
        status_code=500,
        content={"detail": "The server ran into a problem. Please try again."},
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/text")
async def text_message(payload: TextRequest) -> dict[str, str | None]:
    transcript = payload.text.strip()
    session_id = payload.session_id.strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")
    if not session_id:
        raise HTTPException(status_code=422, detail="A valid session_id is required.")
    return await _make_response(session_id, transcript)


@app.post("/api/voice")
async def voice(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
) -> dict[str, str | None]:
    session_id = session_id.strip()
    if not session_id or len(session_id) > 128:
        raise HTTPException(status_code=422, detail="A valid session_id is required.")

    audio_path: Path | None = None
    try:
        audio_path = await _save_upload(audio)
        try:
            transcript = (await asyncio.to_thread(_transcribe, audio_path)).strip()
        except Exception:
            logger.exception("Audio transcription failed.")
            raise HTTPException(
                status_code=500,
                detail="Audio transcription failed. Please try again.",
            ) from None
        if not transcript:
            raise HTTPException(
                status_code=422,
                detail="I couldn't hear any speech, please try again.",
            )
        return await _make_response(session_id, transcript)
    finally:
        await audio.close()
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)



