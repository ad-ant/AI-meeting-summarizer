"""App settings loaded from environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# directories
INPUT_DIR = Path(os.getenv("INPUT_DIR", "data/input"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/output"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", "logs"))

# ollama (llm) configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# whisper (transcription) configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # "cpu" or "cuda" or "mps"
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "pl")

# text chunking configuration
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "500"))

# agent configuration
AGENT_MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", "2"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))

# logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"