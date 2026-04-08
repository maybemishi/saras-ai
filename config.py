"""
config.py — Central configuration for SARAS.
Supports both .env (local) and Streamlit secrets (cloud).
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_env(key: str, default=None):
    # Try Streamlit secrets first (for deployment)
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


class Settings:
    # API keys
    GOOGLE_API_KEY: str = get_env("GOOGLE_API_KEY", "")
    TAVILY_API_KEY: str = get_env("TAVILY_API_KEY", "")

    # Model
    GEMINI_MODEL: str = get_env("GEMINI_MODEL", "gemini-2.5-flash")

    # ChromaDB
    CHROMA_PERSIST_DIR: str = get_env("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME: str = "saras_session"

    # Research pipeline
    MAX_SEARCH_RESULTS: int = int(get_env("MAX_SEARCH_RESULTS", 8))
    MAX_RETRY_LOOPS: int = int(get_env("MAX_RETRY_LOOPS", 3))

    # PDF chunking
    CHUNK_SIZE: int = int(get_env("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int = int(get_env("CHUNK_OVERLAP", 100))

    def validate(self):
        missing = []
        if not self.GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
        if not self.TAVILY_API_KEY:
            missing.append("TAVILY_API_KEY")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Add them to Streamlit Secrets or .env file."
            )


settings = Settings()