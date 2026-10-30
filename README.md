# Voice Notion Agent

A browser-based voice and text assistant for web research, Notion, and Gmail. The
FastAPI service serves the single-page web app and its API from the same origin.

## Features

- Record audio in the browser or send a typed message.
- Transcribe audio with Groq Whisper.
- Use a LangChain orchestrator backed by Gemini, with research, Notion, and Gmail tools.
- Return the assistant reply as text and, when speech synthesis succeeds, audio.
- Keep conversation history per browser session while the server process is running.

## Project layout

```text
voice_agent/
├── voice_notion_agent/
│   ├── agents/                 # Notion and Gmail specialist agents
│   ├── tools/                  # Research and specialist tools
│   ├── orchestrator_agent.py   # Builds the main agent
│   ├── speech_to_text.py       # Groq transcription
│   └── text_to_speech.py       # Gemini speech synthesis
├── web_app/
│   ├── static/index.html       # Browser interface
│   └── server.py               # FastAPI app and API routes
├── sample/
├── .env.example
└── requirements.txt
```

## Requirements

- Python version compatible with the packages pinned in `requirements.txt`.
- API keys for Gemini, Groq, and Tavily.
- Node.js and npm/npx for the Gmail and Notion MCP integrations.
- Gmail/Notion MCP authentication configured for the environment where the server runs.

## Local setup (Windows PowerShell)

From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and add your API keys. Then start the server:

```powershell
py -m uvicorn web_app.server:app --host 127.0.0.1 --port 8000
```

Open [http://localhost:8000](http://localhost:8000). Allow microphone access in
the browser to use voice input; the text input is available as a fallback.

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | Yes | Gemini chat model and speech generation |
| `GROQ_API_KEY` | Yes | Groq audio transcription |
| `TAVILY_API_KEY` | Yes | Web research tool |
| `VOICE_AGENT_USERNAME` | No | Optional HTTP Basic Auth username |
| `VOICE_AGENT_PASSWORD` | No | Optional HTTP Basic Auth password |
| `CHAT_MODEL` | No | Chat model; defaults to `gemini-3.1-flash-lite` |
| `STT_MODEL` | No | Transcription model; defaults to `whisper-large-v3-turbo` |
| `TTS_MODEL` | No | Example/config model setting; defaults to `gemini-3.8-flash-tts` |
| `LANGSMITH_TRACING` | No | Enable LangSmith tracing when configured |
| `LANGSMITH_ENDPOINT` | No | LangSmith endpoint |
| `LANGSMITH_API_KEY` | No | LangSmith API key |
| `LANGSMITH_PROJECT` | No | LangSmith project name |

`TTS_MODEL` is documented as a config setting; the current synthesis function
specifies its model directly. Copy `.env.example` to `.env` for local use, and
never commit `.env` or paste secret values into source files.

HTTP Basic Auth is disabled when either `VOICE_AGENT_USERNAME` or
`VOICE_AGENT_PASSWORD` is unset. When both are set, the app page and API
endpoints require those credentials.

## API

- `GET /` serves the web interface.
- `GET /api/health` returns `{"status":"ok"}`.
- `POST /api/text` accepts JSON containing `text` and `session_id`.
- `POST /api/voice` accepts multipart fields `audio` and `session_id`; audio
  uploads are limited to 25 MB.

Successful message responses include `transcript`, `reply`, `audio_base64`, and
`audio_mime`. `audio_base64` is `null` when synthesis fails; the text reply is
still returned. Errors use a JSON `detail` field. Session history is held in
memory, capped, and is cleared when the server restarts.

## Deploying to Render

Create a **Web Service** connected to this repository. Leave the root directory
blank when deploying from the repository root.

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn web_app.server:app --host 0.0.0.0 --port $PORT`

Add the API keys and optional Basic Auth credentials in the Render service's
Environment settings. Do not upload `.env`; enter secret values in the Render
dashboard. Use HTTPS for the deployed site, especially when using Basic Auth or
microphone recording.

The MCP configuration launches Gmail and Notion integrations with `npx`, so
Node.js/npm and working MCP authentication must also be available in the
deployment runtime. Render's standard Python runtime may not include Node.js;
verify this before deploying, or use a Docker image that provides both Python
and Node.js. `pywin32` is Windows-only and is marked as such in
`requirements.txt`, so pip skips it on Render's Linux host.

## Troubleshooting

- **401 / invalid credentials:** use the exact `VOICE_AGENT_USERNAME` and
  `VOICE_AGENT_PASSWORD` configured for the running server.
- **Slow first reply:** transcription, model/tool calls, and speech synthesis
  run in sequence. Research and MCP operations depend on external services.
- **Text reply but no audio:** synthesis is optional; check server logs for
  speech synthesis errors. The interface provides a Play control when audio is
  returned.
- **Microphone unavailable:** allow microphone permission and use localhost or a
  secure HTTPS origin.
- **MCP tools unavailable after deployment:** confirm Node.js/npm and the
  required Gmail/Notion credentials or authorization are available to the
  deployed process.