"""Robot Video Segmentor - Configuration management."""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from pathlib import Path
import json
import os
import yaml
from pydantic import BaseModel, Field, field_validator


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    root: str = Field(..., description="Path to data root directory")
    subset: str = Field(..., description="Subset/directory name")


class RunConfig(BaseModel):
    """Run/output configuration."""
    base_dir: str = Field(default="./runs", description="Base directory for outputs")
    run_id: str = Field(default="default", description="Run identifier")


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8099, description="Server port")
    max_queue: int = Field(default=32, description="Maximum job queue size")
    inflight_timeout_sec: float = Field(default=300.0, description="Timeout for in-flight jobs")
    max_retries_per_job: int = Field(default=5, description="Maximum retries per job")
    max_empty_retries_per_job: int = Field(
        default=0,
        description="Maximum retries after an empty VLM JSON (<= 0 means unlimited)",
    )
    auto_exit_after_all_done: bool = Field(default=False, description="Auto exit when all done")


class Qwen3VLConfig(BaseModel):
    """Qwen3VL-specific configuration."""
    model_path: str = Field(
        default="Qwen/Qwen3-VL-32B-Instruct",
        description="Model path or HuggingFace model name"
    )
    device_map: str = Field(default="balanced", description="Device map strategy")


class RemoteAPIConfig(BaseModel):
    """Remote API backend configuration."""
    api_url: str = Field(default="http://127.0.0.1:8080/infer", description="Remote API URL")
    api_key: str = Field(default="", description="API key for remote API")
    timeout_sec: float = Field(default=60.0, description="Request timeout in seconds")
    headers: dict = Field(default_factory=dict, description="Extra headers for remote API")


class OpenAIConfig(BaseModel):
    """OpenAI Responses API backend configuration."""
    api_key: str = Field(default="", description="OpenAI API key")
    model: str = Field(default="gpt-5.2", description="OpenAI model name")
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL"
    )
    timeout_sec: float = Field(default=60.0, description="Request timeout in seconds")
    organization: str = Field(default="", description="OpenAI organization header")
    project: str = Field(default="", description="OpenAI project header")
    reasoning_effort: str = Field(default="low", description="Reasoning effort")
    max_output_tokens: int = Field(default=512, description="Maximum output tokens")
    jpeg_quality: int = Field(default=85, description="JPEG quality for uploaded images")

    @field_validator("reasoning_effort")
    @classmethod
    def validate_reasoning_effort(cls, v: str) -> str:
        allowed = ["low", "medium", "high"]
        if v not in allowed:
            raise ValueError(f"reasoning_effort must be one of {allowed}, got {v}")
        return v


class GeminiConfig(BaseModel):
    """Gemini API backend configuration."""
    api_key: str = Field(default="", description="Gemini API key")
    model: str = Field(default="gemini-3-flash-preview", description="Gemini model name")
    api_mode: str = Field(default="native", description="Gemini API mode")
    base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        description="Gemini API base URL"
    )
    timeout_sec: float = Field(default=60.0, description="Request timeout in seconds")
    max_output_tokens: int = Field(default=512, description="Maximum output tokens")
    jpeg_quality: int = Field(default=85, description="JPEG quality for uploaded images")

    @field_validator("api_mode")
    @classmethod
    def validate_api_mode(cls, v: str) -> str:
        allowed = ["native", "openai_compatible"]
        if v not in allowed:
            raise ValueError(f"api_mode must be one of {allowed}, got {v}")
        return v


class WorkerConfig(BaseModel):
    """Worker configuration."""
    count: int = Field(default=7, ge=1, description="Number of worker processes")
    server_url: str = Field(default="http://127.0.0.1:8099", description="Server URL")
    backend: str = Field(default="dummy", description="VLM backend type")
    qwen3vl: Qwen3VLConfig = Field(default_factory=Qwen3VLConfig)
    remote_api: RemoteAPIConfig = Field(default_factory=RemoteAPIConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed = ["dummy", "qwen3vl", "remote_api", "openai", "gemini"]
        if v not in allowed:
            raise ValueError(f"backend must be one of {allowed}, got {v}")
        return v


class WindowingConfig(BaseModel):
    """Video windowing configuration."""
    window_sec: float = Field(default=12.0, description="Window duration in seconds")
    step_sec: float = Field(default=6.0, description="Step size in seconds")
    frames_per_window: int = Field(default=24, description="Frames per window")
    boundary_prompt_mode: str = Field(
        default="freeform",
        description="Boundary prompting mode: whole-window freeform output or center-focused local boundary judgment",
    )
    segment_labeling_mode: str = Field(
        default="inline",
        description="Final segment labeling mode: keep inline window-derived labels or defer to a second labeling pass after boundaries are finalized",
    )
    enable_refinement_pass: bool = Field(
        default=False,
        description="Run a selective second pass on ambiguous windows using shorter refinement subwindows",
    )
    enable_boundary_refinement: bool = Field(
        default=False,
        description="Run a local second pass around provisional boundaries to refine their positions",
    )
    boundary_refinement_window_sec: float = Field(
        default=4.0,
        description="Short clip duration in seconds for local boundary refinement",
    )
    boundary_refinement_frames_per_window: int = Field(
        default=0,
        description="Frames per local boundary refinement clip (0 = reuse frames_per_window)",
    )
    boundary_refinement_abstain_merge_max_support: float = Field(
        default=-1.0,
        description="If >= 0, allow boundary refinement abstentions to merge adjacent segments only when the coarse boundary support is at or below this value",
    )
    refinement_frames_per_window: int = Field(
        default=0,
        description="Frames per refinement window (0 = reuse frames_per_window)",
    )
    target_width: int = Field(default=720, description="Target frame width")
    target_height: int = Field(default=480, description="Target frame height")
    png_compression: int = Field(default=0, description="PNG compression level (0-9)")
    use_contact_sheets: bool = Field(
        default=False,
        description="Pack multiple logical frames into each uploaded image before VLM inference",
    )
    contact_sheet_rows: int = Field(default=4, description="Rows per uploaded contact sheet")
    contact_sheet_cols: int = Field(default=4, description="Columns per uploaded contact sheet")
    adaptive_merge_guard: bool = Field(
        default=True,
        description="Fallback to a lighter cleanup pass when semantic merging collapses segments too aggressively",
    )
    adaptive_merge_min_segments: int = Field(
        default=8,
        description="Minimum pre-merge segment count before adaptive merge guard can trigger",
    )
    adaptive_merge_collapse_ratio: float = Field(
        default=0.6,
        description="Trigger fallback when merged segment count drops below this fraction of the light-cleaned count",
    )
    boundary_support_threshold: float = Field(
        default=0.9,
        description="Treat clustered cut support at or above this value as a strong boundary when adaptive local merge decisions are available",
    )
    refine_final_instructions: bool = Field(
        default=True,
        description="Refine final segment instructions from contributing sub-segment instructions",
    )

    @field_validator("boundary_prompt_mode")
    @classmethod
    def validate_boundary_prompt_mode(cls, v: str) -> str:
        allowed = ["freeform", "center_scan", "multi_probe_scan", "candidate_scan"]
        if v not in allowed:
            raise ValueError(f"boundary_prompt_mode must be one of {allowed}, got {v}")
        return v

    @field_validator("segment_labeling_mode")
    @classmethod
    def validate_segment_labeling_mode(cls, v: str) -> str:
        allowed = ["inline", "deferred"]
        if v not in allowed:
            raise ValueError(f"segment_labeling_mode must be one of {allowed}, got {v}")
        return v


class ProgressConfig(BaseModel):
    """Progress tracking configuration."""
    total_override: int = Field(default=0, description="Override total count (0=auto)")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO", description="Log level")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"level must be one of {allowed}, got {v}")
        return v_upper


class Config(BaseModel):
    """Main application configuration."""
    datasets: List[DatasetConfig] = Field(default_factory=list)
    run: RunConfig = Field(default_factory=RunConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    windowing: WindowingConfig = Field(default_factory=WindowingConfig)
    progress: ProgressConfig = Field(default_factory=ProgressConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        config = cls(**data)
        _apply_env_overrides(config)
        return config
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls()
        _apply_env_overrides(config)
        return config
    
    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "Config":
        """Load configuration with priority: file > env > defaults."""
        if path:
            return cls.from_yaml(path)
        
        # Try to find config.yaml in current directory
        default_path = Path("config.yaml")
        if default_path.exists():
            return cls.from_yaml(default_path)
        
        # Fall back to environment variables
        return cls.from_env()


def _apply_env_overrides(config: Config) -> None:
    """Apply environment variable overrides onto an existing config."""
    if "DATASETS" in os.environ:
        config.datasets = _parse_datasets_env(os.environ["DATASETS"])
    if "RUN_BASE" in os.environ:
        config.run.base_dir = os.environ["RUN_BASE"]
    if "RUN_ID" in os.environ:
        config.run.run_id = os.environ["RUN_ID"]
    if "PORT" in os.environ:
        config.server.port = int(os.environ["PORT"])
    if "MAX_RETRIES_PER_JOB" in os.environ:
        config.server.max_retries_per_job = int(os.environ["MAX_RETRIES_PER_JOB"])
    if "MAX_EMPTY_RETRIES_PER_JOB" in os.environ:
        config.server.max_empty_retries_per_job = int(os.environ["MAX_EMPTY_RETRIES_PER_JOB"])
    if "SERVER_URL" in os.environ:
        config.worker.server_url = os.environ["SERVER_URL"]
    if "WORKER_COUNT" in os.environ:
        config.worker.count = int(os.environ["WORKER_COUNT"])
    if "MODEL_PATH" in os.environ:
        config.worker.qwen3vl.model_path = os.environ["MODEL_PATH"]
    if "BACKEND" in os.environ:
        config.worker.backend = os.environ["BACKEND"]
    if "REMOTE_API_URL" in os.environ:
        config.worker.remote_api.api_url = os.environ["REMOTE_API_URL"]
    if "REMOTE_API_KEY" in os.environ:
        config.worker.remote_api.api_key = os.environ["REMOTE_API_KEY"]
    if "REMOTE_API_TIMEOUT" in os.environ:
        config.worker.remote_api.timeout_sec = float(os.environ["REMOTE_API_TIMEOUT"])
    if "REMOTE_API_HEADERS" in os.environ:
        headers_raw = os.environ["REMOTE_API_HEADERS"]
        headers = json.loads(headers_raw)
        if not isinstance(headers, dict):
            raise ValueError("REMOTE_API_HEADERS must be a JSON object")
        config.worker.remote_api.headers = headers
    if "OPENAI_API_KEY" in os.environ:
        config.worker.openai.api_key = os.environ["OPENAI_API_KEY"]
    if "OPENAI_MODEL" in os.environ:
        config.worker.openai.model = os.environ["OPENAI_MODEL"]
    if "OPENAI_BASE_URL" in os.environ:
        config.worker.openai.base_url = os.environ["OPENAI_BASE_URL"]
    if "OPENAI_TIMEOUT" in os.environ:
        config.worker.openai.timeout_sec = float(os.environ["OPENAI_TIMEOUT"])
    if "OPENAI_ORGANIZATION" in os.environ:
        config.worker.openai.organization = os.environ["OPENAI_ORGANIZATION"]
    if "OPENAI_PROJECT" in os.environ:
        config.worker.openai.project = os.environ["OPENAI_PROJECT"]
    if "OPENAI_REASONING_EFFORT" in os.environ:
        config.worker.openai.reasoning_effort = os.environ["OPENAI_REASONING_EFFORT"]
    if "OPENAI_MAX_OUTPUT_TOKENS" in os.environ:
        config.worker.openai.max_output_tokens = int(os.environ["OPENAI_MAX_OUTPUT_TOKENS"])
    if "OPENAI_JPEG_QUALITY" in os.environ:
        config.worker.openai.jpeg_quality = int(os.environ["OPENAI_JPEG_QUALITY"])
    if "GEMINI_API_KEY" in os.environ:
        config.worker.gemini.api_key = os.environ["GEMINI_API_KEY"]
    if "GEMINI_MODEL" in os.environ:
        config.worker.gemini.model = os.environ["GEMINI_MODEL"]
    if "GEMINI_API_MODE" in os.environ:
        config.worker.gemini.api_mode = os.environ["GEMINI_API_MODE"]
    gemini_base_url = os.environ.get("GEMINI_BASE_URL") or os.environ.get("GOOGLE_GEMINI_BASE_URL")
    if gemini_base_url:
        config.worker.gemini.base_url = gemini_base_url
    if "GEMINI_TIMEOUT" in os.environ:
        config.worker.gemini.timeout_sec = float(os.environ["GEMINI_TIMEOUT"])
    if "GEMINI_MAX_OUTPUT_TOKENS" in os.environ:
        config.worker.gemini.max_output_tokens = int(os.environ["GEMINI_MAX_OUTPUT_TOKENS"])
    if "GEMINI_JPEG_QUALITY" in os.environ:
        config.worker.gemini.jpeg_quality = int(os.environ["GEMINI_JPEG_QUALITY"])


def _parse_datasets_env(spec: str) -> List[DatasetConfig]:
    """Parse DATASETS environment variable."""
    configs = []
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    for p in parts:
        if ":" in p:
            root, subset = p.split(":", 1)
            configs.append(DatasetConfig(root=root.strip(), subset=subset.strip()))
        else:
            data_dir = Path(p.rstrip("/"))
            root = str(data_dir.parent)
            subset = data_dir.name
            configs.append(DatasetConfig(root=root, subset=subset))
    return configs
