# src/deep_research_agent/config/settings.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_ROOT = Path(__file__).resolve().parents[3]



class Secrets(BaseSettings):
    """Credential and sensitive configuration layer."""

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    google_cse_id: str | None = Field(default=None, alias="GOOGLE_CSE_ID")
    serpapi_api_key: str | None = Field(default=None, alias="SERPAPI_API_KEY")
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    llama_model_path: Path | None = Field(default=None, alias="LLAMA_MODEL_PATH")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class RuntimeSettings(BaseSettings):
    """Non-secret application settings."""

    environment: Literal["local", "dev", "staging", "prod"] = Field(default="local")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO", alias="LOG_LEVEL")
    request_timeout_seconds: int = Field(default=30, alias="REQUEST_TIMEOUT")
    max_concurrent_requests: int = Field(default=5, alias="MAX_CONCURRENT_REQUESTS")
    default_search_engine: Literal["tavily", "serpapi", "google", "web"] = Field(default="web")
    enable_trace_export: bool = Field(default=False)
    trace_endpoint: HttpUrl | None = Field(default=None)
    data_dir: Path = Field(default=APP_ROOT / "data")
    reports_dir: Path = Field(default=APP_ROOT / "data" / "reports")
    logs_dir: Path = Field(default=APP_ROOT / "data" / "logs")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class Settings:
    """Centralized settings facade accessible throughout the application."""

    def __init__(self) -> None:
        self.secrets = Secrets()
        self.runtime = RuntimeSettings()

    def as_dict(self) -> dict[str, Any]:
        return {
            "environment": self.runtime.environment,
            "log_level": self.runtime.log_level,
            "request_timeout_seconds": self.runtime.request_timeout_seconds,
            "max_concurrent_requests": self.runtime.max_concurrent_requests,
            "default_search_engine": self.runtime.default_search_engine,
            "enable_trace_export": self.runtime.enable_trace_export,
            "trace_endpoint": self.runtime.trace_endpoint,
            "data_dir": str(self.runtime.data_dir),
            "reports_dir": str(self.runtime.reports_dir),
            "logs_dir": str(self.runtime.logs_dir),
            "openai_configured": self.secrets.openai_api_key is not None,
            "google_configured": self.secrets.google_api_key is not None,
            "serpapi_configured": self.secrets.serpapi_api_key is not None,
            "tavily_configured": self.secrets.tavily_api_key is not None,
            "llama_configured": self.secrets.llama_model_path is not None,
        }


settings = Settings()
