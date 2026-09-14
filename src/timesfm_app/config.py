"""Native application configuration; credentials never enter browser settings.

Settings are read once from ``TIMESFM_``-prefixed environment variables and
cached by ``get_settings()``. See api.py, worker.py, and native.py for the
processes that read this configuration; store.py and artifacts.py are
constructed from the values here.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  model_config = SettingsConfigDict(env_prefix="TIMESFM_", extra="ignore")

  database_url: str = "postgresql+psycopg://timesfm@127.0.0.1:55432/timesfm"
  redis_url: str = "redis://127.0.0.1:56379/0"
  artifact_root: Path = Path("data/product/artifacts")
  storage_backend: Literal["local", "s3"] = "local"
  s3_bucket: str = "timesfm"
  s3_endpoint_url: str | None = None
  device: Literal["cuda", "cpu"] = "cuda"
  queued_job_limit: int = Field(default=10, ge=1, le=1000)
  heartbeat_seconds: int = Field(default=5, ge=1)
  lease_seconds: int = Field(default=60, ge=15)
  cancellation_grace_seconds: int = Field(default=30, ge=1)
  worker_metrics_enabled: bool = True
  api_port: int = 8001
  frontend_port: int = 3000
  otlp_endpoint: str | None = None


@lru_cache
def get_settings() -> Settings:
  # Memoized: once called, later environment-variable changes in this process
  # have no effect. Tests must construct Settings()/Store() directly rather
  # than mutating the environment and expecting get_settings() to pick it up.
  return Settings()
