"""Runtime settings from environment variables (and an optional .env file). See .env.example."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LLMProviderKind = Literal["fake", "openrouter", "compat"]
ASRKind = Literal["whisper", "sidecar"]

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"

# Chosen from the smoke test before the real run (results/smoke.json).
DEFAULT_FREE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_FREE_FALLBACKS = ["nvidia/nemotron-3-ultra-550b-a55b:free"]
DEFAULT_JUDGE_MODEL = "google/gemma-4-31b-it:free"
DEFAULT_JUDGE_FALLBACKS = ["dots-studio/dots-3-note-preview:free"]
DEFAULT_SCRIPT_MODEL = "google/gemma-4-31b-it:free"
"""The synthetic test scripts are drafted by a different model family than the one that picks clips."""
DEFAULT_MODELS: dict[str, str] = {
    "fake": "fake-rules",
    "openrouter": DEFAULT_FREE_MODEL,
    "compat": "",  # any OpenAI-compatible endpoint: the model id must be given (RECUT_LLM_MODEL)
}


def _split(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RECUT_", extra="ignore")

    # ---- storage
    data_dir: Path = Path("data/projects")
    """One folder per project: project.json (atomic writes), the extracted audio, renders."""
    samples_dir: Path = REPO_ROOT / "data" / "samples"
    seed_demo: bool = True
    """On API startup: import the bundled sample talk as a demo project when there are no projects."""
    max_upload_mb: int = 2048
    allowed_source_dirs: Annotated[list[str], NoDecode] = Field(default_factory=list)
    """Local folders that `POST /api/projects` with a `path` may read from (empty = the samples and test-set
    folders only). Uploads are always allowed."""

    # ---- speech recognition (local, CPU)
    asr: ASRKind = "whisper"
    """whisper = faster-whisper on CPU; sidecar = read <video>.words.json next to the file (CI, tests, demo)."""
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    whisper_threads: int = 0
    """0 = all cores."""
    whisper_language: str = "en"
    whisper_beam_size: int = 1
    transcript_cache_dir: Path | None = Path(".cache/transcripts")
    """Transcripts keyed by the source's SHA-256 and the model, so re-processing a file does not transcribe it again
    (the original transcription time is kept in the transcript). Empty = no cache."""
    diarize: bool = True
    max_speakers: int = 4

    # ---- clips
    clip_min_seconds: float = 20.0
    clip_max_seconds: float = 60.0
    clips_per_10_minutes: int = 3
    max_clips: int = 12
    window_minutes: float = 14.0
    """Long transcripts are sent to the model in windows of this many minutes (with one minute of overlap)."""
    pad_in_seconds: float = 0.12
    pad_out_seconds: float = 0.30

    # ---- rendering
    render_width: int = 1080
    render_height: int = 1920
    render_preset: str = "veryfast"
    render_crf: int = 21
    render_threads: int = 0
    loudnorm: bool = True
    face_model: Path = ASSETS_DIR / "models" / "face_detection_yunet_2023mar.onnx"
    fonts_dir: Path = ASSETS_DIR / "fonts"
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"

    # ---- LLM
    llm_provider: LLMProviderKind = "fake"
    llm_model: str = ""
    """Empty = the provider's default (DEFAULT_MODELS)."""
    llm_fallback_models: Annotated[list[str], NoDecode] = Field(default_factory=list)
    judge_model: str = DEFAULT_JUDGE_MODEL
    """The clip judge in the evaluation: a different free model than the one that picked the clips."""
    llm_timeout_seconds: float = 240.0
    llm_max_tokens: int = 12000
    """A ceiling, not a target: free reasoning models spend thousands of tokens thinking before the JSON."""
    llm_temperature: float = 0.2
    llm_reasoning_effort: str = "low"
    require_free_models: bool = True
    """Refuse any OpenRouter model id that does not end in ":free" (requests, fallbacks and the model that answered)."""
    allow_paid_models: bool = False
    """The explicit opt-in for paid models: with RECUT_ALLOW_PAID_MODELS=true the free-only guard is lifted for the
    `openrouter` and `compat` providers. Off by default, so a fresh clone refuses every non-":free" id."""
    llm_base_url: str = ""
    """Endpoint of the `compat` provider (any OpenAI-compatible /chat/completions server)."""
    llm_api_key: str | None = None
    llm_send_temperature: bool = True

    openrouter_api_key: str | None = Field(default=None, validation_alias=AliasChoices("OPENROUTER_API_KEY"))
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", validation_alias=AliasChoices("OPENROUTER_BASE_URL")
    )

    llm_cache: bool = True
    llm_cache_dir: Path = Path(".cache/llm")
    llm_ledger: Path | None = Path("results/calls.jsonl")
    """Every real request (retries included) is appended here: model ids, status, latency, tokens. Never prompts."""
    llm_max_calls: int = 1500
    """Hard ceiling on real requests recorded in the ledger (the free tier allows about 1,000 a day)."""
    llm_min_seconds_between_requests: float = 3.2
    """Free OpenRouter models allow about 20 requests a minute."""
    llm_max_retries: int = 4

    # ---- API
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    log_level: str = "INFO"
    log_format: str = Field(default="console", validation_alias=AliasChoices("LOG_FORMAT", "RECUT_LOG_FORMAT"))

    # ---- MCP
    mcp_max_output_chars: int = 24000

    _split_lists = field_validator("llm_fallback_models", "cors_origins", "allowed_source_dirs", mode="before")(
        classmethod(lambda cls, v: _split(v))
    )

    @field_validator("llm_ledger", "transcript_cache_dir", mode="before")
    @classmethod
    def _empty_ledger(cls, value: object) -> object:
        return None if value in ("", None) else value

    @property
    def free_only(self) -> bool:
        return self.require_free_models and not self.allow_paid_models

    @property
    def model_name(self) -> str:
        return self.llm_model or DEFAULT_MODELS[self.llm_provider]

    @property
    def fallbacks(self) -> list[str]:
        if self.llm_fallback_models:
            return self.llm_fallback_models
        return DEFAULT_FREE_FALLBACKS if self.llm_provider == "openrouter" and not self.llm_model else []


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
