from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    anthropic_api_key: str | None = None
    api_token: str
    # Where phrase/document embeddings come from. "ollama" (default) calls a local Ollama server;
    # "native" loads the GGUF model in-process through libargus (native), with no
    # Ollama needed. "native" requires ARGUS_MODEL_PATH; ARGUS_LIB_PATH defaults to the repo's release build.
    embeddings_backend: Literal["ollama", "native"] = "ollama"
    argus_model_path: str | None = None
    argus_lib_path: str | None = None

    # Absolute, so the MCP server (which a client launches from any directory) finds the same .env.
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
