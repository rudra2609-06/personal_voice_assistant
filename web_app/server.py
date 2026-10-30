import asyncio
import base64
import importlib
import logging
import mimetypes
import os
import secrets
import sys
import tempfile
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from groq import Groq
from langchain.messages import HumanMessage
from pydantic import BaseModel, Field
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from voice_notion_agent import config as project_config
from voice_notion_agent.orchestrator_agent import build_agent

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_SESSIONS = 100
MAX_HISTORY_MESSAGES = 40

_session_histories: OrderedDict[str, list[Any]] = OrderedDict()
_session_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
_sessions_guard = asyncio.Lock()
_stt_client: Groq | None = None
_stt_client_guard = threading.Lock()
_tts_function: Callable[..., str] | None = None
_tts_function_guard = threading.Lock()
_basic_auth = HTTPBasic(auto_error=False)


class TextRequest(BaseModel):
    text: str
    session_id: str = Field(min_length=1, max_length=128)


def _validate_session_id(session_id: str) -> str:
    value = session_id.strip()
    if not value or len(value) > 128:
        raise HTTPException(status_code=422, detail="A valid session_id is required.")
    return value


def _require_auth(
    credentials: HTTPBasicCredentials | None = Depends(_basic_auth),
) -> None:
    username = os.getenv("VOICE_AGENT_USERNAME")
    password = os.getenv("VOICE_AGENT_PASSWORD")

    # No credentials configured: auth is off. Fine for local use on 127.0.0.1.
    if not username or not password:
        return

    if credentials is None or not (
        secrets.compare_digest(credentials.username, username)
        and secrets.compare_digest(credentials.password, password)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )


async def _evict_sessions(keep: str | None = None) -> None:
    async with _sessions_guard:
        while len(_session_locks) > MAX_SESSIONS:
            evicted = False
            for session_id, lock in _session_locks.items():
                if session_id != keep and not lock.locked():
                    del _session_locks[session_id]
                    _session_histories.pop(session_id, None)
                    evicted = True
                    break
            if not evicted:
                break


async def _get_session_lock(session_id: str) -> asyncio.Lock:
    async with _sessions_guard:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            _session_locks[session_id] = lock
            _session_histories[session_id] = []
        else:
            _session_locks.move_to_end(session_id)
        await _evict_sessions_locked(session_id)
        return lock


async def _evict_sessions_locked(keep: str) -> None:
    while len(_session_locks) > MAX_SESSIONS:
        evicted = False
        for session_id, lock in _session_locks.items():
            if session_id != keep and not lock.locked():
                del _session_locks[session_id]
                _session_histories.pop(session_id, None)
                evicted = True
                break
        if not evicted:
            break


def _trim_history(messages: list[Any]) -> list[Any]:
    turns: list[list[Any]] = []
    current_turn: list[Any] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            if current_turn:
                turns.append(current_turn)
            current_turn = [message]
        elif current_turn:
            current_turn.append(message)
    if current_turn:
        turns.append(current_turn)

    message_count = sum(len(turn) for turn in turns)
    while len(turns) > 1 and message_count > MAX_HISTORY_MESSAGES:
        message_count -= len(turns.pop(0))
    return [message for turn in turns for message in turn]


async def _run_agent_turn(session_id: str, text: str) -> str:
    session_lock = await _get_session_lock(session_id)
    try:
        async with session_lock:
            history = _session_histories[session_id]
            try:
                agent = app.state.agent
                result = await agent.ainvoke({"messages": [*history, HumanMessage(content=text)]})
            except Exception:
                _logger.exception("The assistant agent failed for a session.")
                raise HTTPException(
                    status_code=500,
                    detail="The assistant ran into a problem. Please try again.",
                ) from None

            result_messages = result.get("messages") if isinstance(result, dict) else None
            if not isinstance(result_messages, list) or not result_messages:
                _logger.error("The assistant returned no messages.")
                raise HTTPException(
                    status_code=500,
                    detail="The assistant ran into a problem. Please try again.",
                )

            _session_histories[session_id] = _trim_history(result_messages)
            last_text = getattr(result_messages[-1], "text", "")
            return last_text if isinstance(last_text, str) and last_text.strip() else "Done."
    finally:
        await _evict_sessions(keep=session_id)


def _get_stt_client() -> Groq:
    global _stt_client
    if _stt_client is None:
        with _stt_client_guard:
            if _stt_client is None:
                _stt_client = Groq(api_key=project_config.GROQ_API_KEY)
    return _stt_client


def _transcribe_audio(file_path: str) -> str:
    # The existing module runs a sample transcription at import time, so use its
    # same synchronous Groq call here without triggering that demo side effect.
    with open(file_path, "rb") as audio_file:
        transcription = _get_stt_client().audio.transcriptions.create(
            file=(os.path.basename(file_path), audio_file.read()),
            model=project_config.STT_MODEL,
        )
    return transcription.text


def _upload_suffix(upload: UploadFile) -> str:
    allowed_suffixes = {".webm", ".ogg", ".mp4", ".wav", ".mp3", ".m4a"}
    filename_suffix = Path(upload.filename or "").suffix.lower()
    if filename_suffix in allowed_suffixes:
        return filename_suffix

    media_type = (upload.content_type or "").split(";", 1)[0].strip().lower()
    suffix_by_type = {
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/mp4": ".mp4",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp4a-latm": ".m4a",
    }
    return suffix_by_type.get(media_type, mimetypes.guess_extension(media_type) or ".webm")


def _remove_temp_file(file_path: Path) -> None:
    try:
        file_path.unlink(missing_ok=True)
    except OSError:
        _logger.warning("Could not remove a temporary file.", exc_info=True)


async def _save_upload(upload: UploadFile) -> Path:
    temporary_file = tempfile.NamedTemporaryFile(
        prefix="voice-agent-",
        suffix=_upload_suffix(upload),
        delete=False,
    )
    file_path = Path(temporary_file.name)
    try:
        total_bytes = 0
        with temporary_file:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Audio must be 25 MB or smaller.")
                temporary_file.write(chunk)
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="The uploaded audio is empty.")
        return file_path
    except BaseException:
        _remove_temp_file(file_path)
        raise


def _load_synthesizer() -> Callable[..., str]:
    global _tts_function
    if _tts_function is not None:
        return _tts_function

    with _tts_function_guard:
        if _tts_function is not None:
            return _tts_function

        # text_to_speech.py imports config as a top-level module; temporarily map
        # the existing package config while loading it, then restore sys.modules.
        previous_config = sys.modules.get("config")
        sys.modules["config"] = project_config
        try:
            module = importlib.import_module("voice_notion_agent.text_to_speech")
            _tts_function = module.synthesize_speech
        finally:
            if previous_config is None:
                sys.modules.pop("config", None)
            else:
                sys.modules["config"] = previous_config
    return _tts_function


async def _synthesize(text: str) -> tuple[bytes | None, str]:
    file_path: Path | None = None
    try:
        temporary_file = tempfile.NamedTemporaryFile(
            prefix="voice-agent-reply-",
            suffix=".wav",
            delete=False,
        )
        file_path = Path(temporary_file.name)
        temporary_file.close()

        synthesize_speech = _load_synthesizer()
        await asyncio.to_thread(synthesize_speech, text, str(file_path))
        audio_bytes = await asyncio.to_thread(file_path.read_bytes)
        if not audio_bytes:
            _logger.warning("Speech synthesis returned an empty audio file.")
            return None, "audio/wav"
        return audio_bytes, "audio/wav"
    except Exception:
        _logger.exception("Speech synthesis failed.")
        return None, "audio/wav"
    finally:
        if file_path is not None:
            _remove_temp_file(file_path)


@asynccontextmanager
async def lifespan(application: FastAPI):
    _logger.info("Starting Voice Notion Agent.")
    application.state.agent = await build_agent()
    _logger.info("Voice Notion Agent is ready.")
    try:
        yield
    finally:
        application.state.agent = None
        async with _sessions_guard:
            _session_histories.clear()
            _session_locks.clear()
        _logger.info("Voice Notion Agent stopped.")


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request,
    exception: RequestValidationError,
) -> JSONResponse:
    _logger.info("Rejected an invalid request to %s.", request.url.path)
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request. Please check the submitted data."},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exception: Exception) -> JSONResponse:
    _logger.exception("Unexpected server error for %s.", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "The server ran into a problem. Please try again."},
    )


@app.get("/", dependencies=[Depends(_require_auth)])
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/api/health", dependencies=[Depends(_require_auth)])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/voice", dependencies=[Depends(_require_auth)])
async def voice(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
) -> dict[str, str | None]:
    temporary_path: Path | None = None
    try:
        session_id = _validate_session_id(session_id)
        temporary_path = await _save_upload(audio)
        try:
            transcript = await asyncio.to_thread(_transcribe_audio, str(temporary_path))
        except Exception:
            _logger.exception("Audio transcription failed.")
            raise HTTPException(
                status_code=500,
                detail="Audio transcription failed. Please try again.",
            ) from None

        if not transcript or not transcript.strip():
            raise HTTPException(
                status_code=422,
                detail="I couldn't hear any speech, please try again.",
            )

        reply = await _run_agent_turn(session_id, transcript)
        audio_bytes, audio_mime = await _synthesize(reply)
        audio_base64 = base64.b64encode(audio_bytes).decode("ascii") if audio_bytes else None
        return {
            "transcript": transcript,
            "reply": reply,
            "audio_base64": audio_base64,
            "audio_mime": audio_mime,
        }
    finally:
        await audio.close()
        if temporary_path is not None:
            await asyncio.to_thread(_remove_temp_file, temporary_path)


@app.post("/api/text", dependencies=[Depends(_require_auth)])
async def text_message(payload: TextRequest) -> dict[str, str | None]:
    if not payload.text.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    session_id = _validate_session_id(payload.session_id)
    reply = await _run_agent_turn(session_id, payload.text)
    audio_bytes, audio_mime = await _synthesize(reply)
    audio_base64 = base64.b64encode(audio_bytes).decode("ascii") if audio_bytes else None
    return {
        "transcript": payload.text,
        "reply": reply,
        "audio_base64": audio_base64,
        "audio_mime": audio_mime,
    }


# Run from the project root: py -m uvicorn web_app.server:app --host 127.0.0.1 --port 8000
