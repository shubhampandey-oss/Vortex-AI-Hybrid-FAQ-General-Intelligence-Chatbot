"""
config.py
=============================================================================
Central Configuration Module for CodeAlpha Hybrid AI FAQ Chatbot
-----------------------------------------------------------------------------
Loads and validates settings from environment variables (.env) with safe defaults.
Never hardcodes secret keys.
=============================================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAQS_FILE = DATA_DIR / "faqs.json"

# Load .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

# LLM Provider Configuration
# Supported providers: "gemini", "openai", "auto"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower().strip()

# Google Gemini Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()

# OpenAI Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# FAQ & Retrieval Defaults
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "8"))

# Server Configuration
PORT = int(os.getenv("PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
