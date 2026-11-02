from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    anthropic_api_key: str | None = None
    api_token: str
    # Where phrase/document embeddings come from. "native" (default) loads the GGUF model in-process
    # through libargus (native/) and persists vectors in an index file; it raises, naming the build and
    # model-download commands, if the library or model is missing. "ollama" calls a local Ollama server.
    # There is no automatic switch between them.
    embeddings_backend: Literal["ollama", "native"] = "native"
    # Defaults (see app/native): the model native/tools/fetch_model.py downloads, the release build of
    # libargus under native/, and an index file next to the model directory.
    argus_model_path: str | None = None
    argus_lib_path: str | None = None
    argus_index_path: str | None = None

    # Absolute, so the MCP server (which a client launches from any directory) finds the same .env.
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()  # type: ignore[call-arg]  # required fields are filled from env/.env by pydantic-settings
