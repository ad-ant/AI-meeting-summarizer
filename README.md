# AI Meeting Summarizer

Python project that transcribes meeting audio, creates a short summary, and extracts action items.

## What this project does

The app runs a simple local pipeline:
1. Reads one audio file from `data/input`
2. Transcribes speech with faster-whisper
3. Splits transcript into chunks
4. Summarizes each chunk with Ollama (Llama 3.2)
5. Merges chunk summaries into one coherent summary
6. Tries to extract action items and save them to markdown

## Why I built it

I wanted a practical project that combines:
- file processing
- API communication
- basic prompt-based LLM workflow
- clean Python structure (config, logging, tests)

## Tech stack
- Python 3.10+
- faster-whisper 1.0.1
- Ollama (Llama 3.2)
- requests 2.31.0

## Setup

### Prerequisites:

- Python 3.10+
- Ollama running locally (default: port 11434)
- Audio file in `data/input` (mp3, mp4, wav)

### Installation:

1. Clone repository
2. Copy `.env.example` to `.env`
3. Install dependencies: 
```
pip install -r requirements.txt
```

For tests:
```
pip install -r requirements-dev.txt
```

Run:
```
python main.py
```

Test:
```
python -m pytest tests/ -v
```

## Output files
After running, files are written to `data/output`:
- `transcript.txt`
- `summary.txt`
- `action_items.md` (only if tasks are detected)

## Configuration
Main settings are in `.env` and loaded from `config.py`:
- `WHISPER_MODEL`
- `WHISPER_DEVICE`
- `CHUNK_MAX_WORDS`
- `OLLAMA_MODEL`
- `REQUEST_TIMEOUT`
- `SUMMARY_MAX_RETRIES`
- `AGENT_MAX_RETRIES`

## Current limitations
- Works locally with Ollama, no hosted API
- No web UI yet
- Chunk order is preserved, but there's no speaker separation

## Future improvements
- Add REST API
- Add Docker setup
- Add CI pipeline