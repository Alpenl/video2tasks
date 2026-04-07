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


class SubtitleConfig(BaseModel):
    """Subtitle overlay configuration for exported videos."""
    enabled: bool = Field(default=True, description="Burn instruction subtitles into exported videos")
    language: str = Field(
        default="zh",
        description="Subtitle language for exported videos: zh/en. Only export subtitles change language; source instructions remain unchanged.",
    )
    position: str = Field(
        default="top_right",
        description="Subtitle position: top_right/top_left/bottom_right/bottom_left",
    )
    font_file: str = Field(default="", description="Optional subtitle font file path")
    font_size: int = Field(default=28, ge=1, description="Subtitle font size in pixels")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        normalized = str(v).strip().lower()
        aliases = {
            "zh": "zh",
            "zh-cn": "zh",
            "cn": "zh",
            "中文": "zh",
            "chinese": "zh",
            "en": "en",
            "english": "en",
        }
        if normalized not in aliases:
            raise ValueError("subtitle language must be one of ['zh', 'en']")
        return aliases[normalized]

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: str) -> str:
        allowed = ["top_right", "top_left", "bottom_right", "bottom_left"]
        if v not in allowed:
            raise ValueError(f"subtitle position must be one of {allowed}, got {v}")
        return v


class ExportConfig(BaseModel):
    """Optional exported video artifacts."""
    enabled: bool = Field(default=False, description="Export annotated and/or per-segment videos")
    mode: str = Field(
        default="annotated",
        description="Export mode: annotated/clips/both",
    )
    clips_dirname: str = Field(default="clips", description="Directory name for exported segment clips")
    manifest_name: str = Field(default="manifest.json", description="Manifest filename for exported clips")
    annotated_dirname: str = Field(default="exports", description="Directory name for exported annotated videos")
    annotated_name: str = Field(default="annotated.mp4", description="Filename for exported annotated videos")
    subtitles: SubtitleConfig = Field(default_factory=SubtitleConfig)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = ["annotated", "clips", "both"]
        if v not in allowed:
            raise ValueError(f"export.mode must be one of {allowed}, got {v}")
        return v


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
    window_repeat_count: int = Field(
        default=1,
        ge=1,
        description="Repeat each boundary window this many times and vote across repeats",
    )
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


class LLMMergeConfig(BaseModel):
    """Optional text-only LLM pass for merging obvious over-segmentation."""
    enabled: bool = Field(
        default=False,
        description="Run an optional text-only LLM pass to merge obviously over-split adjacent segments",
    )
    backend: str = Field(default="openai", description="Backend type for the merge pass")
    api_key: str = Field(default="", description="API key for the merge pass backend")
    model: str = Field(default="gpt-5.2", description="Model name for the merge pass")
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="API base URL for the merge pass backend",
    )
    timeout_sec: float = Field(default=60.0, description="Request timeout in seconds")
    organization: str = Field(default="", description="Optional organization header")
    project: str = Field(default="", description="Optional project header")
    reasoning_effort: str = Field(default="low", description="Reasoning effort for the merge pass")
    max_output_tokens: int = Field(default=2048, description="Maximum output tokens for the merge pass")
    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of merge-pass request attempts when the backend returns an empty payload or transient error",
    )
    summary_levels: List[int] = Field(
        default_factory=lambda: [1, 1, 1],
        description="Three-level summary switches [coarse, medium, fine], where 1 enables the level and 0 disables it",
    )
    repeat_count: int = Field(
        default=1,
        ge=1,
        description="Number of successful coarse merge samples to collect before optional boundary-level consensus",
    )
    boundary_vote_threshold: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Strict vote threshold used for coarse boundary consensus across repeated successful samples",
    )
    granularity: str = Field(
        default="guarded",
        description="Merge objective: guarded preserves fine boundaries; coarse aims for broader task-level steps",
    )
    min_input_segments: int = Field(
        default=12,
        ge=2,
        description="Skip the merge pass when the finalized segment count is below this threshold",
    )
    max_input_segments: int = Field(
        default=256,
        ge=2,
        description="Skip the merge pass when the finalized segment count exceeds this threshold",
    )
    min_output_ratio: float = Field(
        default=0.35,
        gt=0.0,
        le=1.0,
        description="Reject LLM merge results that collapse too aggressively below this output/input ratio",
    )
    coarse_min_output_ratio: float = Field(
        default=0.18,
        gt=0.0,
        le=1.0,
        description="Lower output/input ratio floor used when granularity=coarse",
    )
    coarse_max_supported_anchors_per_range: int = Field(
        default=1,
        ge=0,
        description="When granularity=coarse, preserve at most this many top-ranked guard-derived anchor boundaries inside each requested merged range",
    )
    coarse_anchor_min_spacing_segments: int = Field(
        default=3,
        ge=1,
        description="When granularity=coarse, keep selected internal anchor boundaries at least this many segment indices apart inside each requested merged range",
    )
    coarse_anchor_min_side_segments: int = Field(
        default=2,
        ge=1,
        description="When granularity=coarse, only reinsert an internal anchor if at least this many original segments remain on both sides of that anchor inside the requested merged range",
    )
    coarse_anchor_min_score: float = Field(
        default=1.03,
        ge=0.0,
        description="When granularity=coarse, only reinsert an internal anchor boundary if its coarse anchor score reaches this threshold",
    )
    protect_supported_boundaries: bool = Field(
        default=True,
        description="Keep internal boundaries that already have strong support from the visual segmentation pass",
    )
    protected_boundary_support_threshold: float = Field(
        default=0.45,
        ge=0.0,
        description="Do not let the merge pass swallow an internal boundary at or above this support value",
    )
    protect_distinct_sequence_markers: bool = Field(
        default=True,
        description="Keep boundaries when adjacent instructions use distinct order markers such as first/second/third",
    )
    protect_instruction_drift: bool = Field(
        default=True,
        description="Keep boundaries when adjacent instructions drift across action family or object focus",
    )
    protect_duplicate_tail_anchor: bool = Field(
        default=True,
        description="Keep a duplicate-label tail anchor before a strongly protected next external boundary",
    )
    duplicate_tail_anchor_min_frames: int = Field(
        default=5,
        ge=1,
        description="Minimum duration of the duplicate trailing segment to preserve its anchor boundary",
    )

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed = ["openai"]
        if v not in allowed:
            raise ValueError(f"llm_merge.backend must be one of {allowed}, got {v}")
        return v

    @field_validator("granularity")
    @classmethod
    def validate_granularity(cls, v: str) -> str:
        allowed = ["guarded", "coarse"]
        if v not in allowed:
            raise ValueError(f"llm_merge.granularity must be one of {allowed}, got {v}")
        return v

    @field_validator("reasoning_effort")
    @classmethod
    def validate_reasoning_effort(cls, v: str) -> str:
        allowed = ["low", "medium", "high"]
        if v not in allowed:
            raise ValueError(f"llm_merge.reasoning_effort must be one of {allowed}, got {v}")
        return v

    @field_validator("summary_levels")
    @classmethod
    def validate_summary_levels(cls, v: List[int]) -> List[int]:
        if len(v) != 3:
            raise ValueError("llm_merge.summary_levels must contain exactly 3 integers: [coarse, medium, fine]")
        normalized = []
        for item in v:
            value = int(item)
            if value not in {0, 1}:
                raise ValueError("llm_merge.summary_levels values must be 0 or 1")
            normalized.append(value)
        return normalized


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
    export: ExportConfig = Field(default_factory=ExportConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    windowing: WindowingConfig = Field(default_factory=WindowingConfig)
    llm_merge: LLMMergeConfig = Field(default_factory=LLMMergeConfig)
    progress: ProgressConfig = Field(default_factory=ProgressConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return _build_config_with_env_overrides(cls, data)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return _build_config_with_env_overrides(cls, {})

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "Config":
        """Load configuration with priority: file > env > defaults."""
        if path:
            return cls.from_yaml(path)

        default_path = Path("config.yaml")
        if default_path.exists():
            return cls.from_yaml(default_path)

        return cls.from_env()


def _set_nested_value(target: dict, path: List[str], value) -> None:
    cursor = target
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _collect_env_override_data() -> dict:
    override: dict = {}

    if "DATASETS" in os.environ:
        _set_nested_value(override, ["datasets"], [cfg.model_dump() for cfg in _parse_datasets_env(os.environ["DATASETS"])])
    if "RUN_BASE" in os.environ:
        _set_nested_value(override, ["run", "base_dir"], os.environ["RUN_BASE"])
    if "RUN_ID" in os.environ:
        _set_nested_value(override, ["run", "run_id"], os.environ["RUN_ID"])
    if "EXPORT_ENABLED" in os.environ:
        _set_nested_value(override, ["export", "enabled"], _parse_env_bool(os.environ["EXPORT_ENABLED"]))
    if "EXPORT_MODE" in os.environ:
        _set_nested_value(override, ["export", "mode"], os.environ["EXPORT_MODE"])
    if "EXPORT_CLIPS_DIRNAME" in os.environ:
        _set_nested_value(override, ["export", "clips_dirname"], os.environ["EXPORT_CLIPS_DIRNAME"])
    if "EXPORT_MANIFEST_NAME" in os.environ:
        _set_nested_value(override, ["export", "manifest_name"], os.environ["EXPORT_MANIFEST_NAME"])
    if "EXPORT_ANNOTATED_DIRNAME" in os.environ:
        _set_nested_value(override, ["export", "annotated_dirname"], os.environ["EXPORT_ANNOTATED_DIRNAME"])
    if "EXPORT_ANNOTATED_NAME" in os.environ:
        _set_nested_value(override, ["export", "annotated_name"], os.environ["EXPORT_ANNOTATED_NAME"])
    if "EXPORT_SUBTITLES_ENABLED" in os.environ:
        _set_nested_value(
            override,
            ["export", "subtitles", "enabled"],
            _parse_env_bool(os.environ["EXPORT_SUBTITLES_ENABLED"]),
        )
    if "EXPORT_SUBTITLE_POSITION" in os.environ:
        _set_nested_value(override, ["export", "subtitles", "position"], os.environ["EXPORT_SUBTITLE_POSITION"])
    if "EXPORT_SUBTITLE_LANGUAGE" in os.environ:
        _set_nested_value(override, ["export", "subtitles", "language"], os.environ["EXPORT_SUBTITLE_LANGUAGE"])
    if "EXPORT_SUBTITLE_FONT_FILE" in os.environ:
        _set_nested_value(override, ["export", "subtitles", "font_file"], os.environ["EXPORT_SUBTITLE_FONT_FILE"])
    if "EXPORT_SUBTITLE_FONT_SIZE" in os.environ:
        _set_nested_value(override, ["export", "subtitles", "font_size"], int(os.environ["EXPORT_SUBTITLE_FONT_SIZE"]))
    if "PORT" in os.environ:
        _set_nested_value(override, ["server", "port"], int(os.environ["PORT"]))
    if "MAX_RETRIES_PER_JOB" in os.environ:
        _set_nested_value(override, ["server", "max_retries_per_job"], int(os.environ["MAX_RETRIES_PER_JOB"]))
    if "MAX_EMPTY_RETRIES_PER_JOB" in os.environ:
        _set_nested_value(override, ["server", "max_empty_retries_per_job"], int(os.environ["MAX_EMPTY_RETRIES_PER_JOB"]))
    if "SERVER_URL" in os.environ:
        _set_nested_value(override, ["worker", "server_url"], os.environ["SERVER_URL"])
    if "WORKER_COUNT" in os.environ:
        _set_nested_value(override, ["worker", "count"], int(os.environ["WORKER_COUNT"]))
    if "MODEL_PATH" in os.environ:
        _set_nested_value(override, ["worker", "qwen3vl", "model_path"], os.environ["MODEL_PATH"])
    if "BACKEND" in os.environ:
        _set_nested_value(override, ["worker", "backend"], os.environ["BACKEND"])
    if "REMOTE_API_URL" in os.environ:
        _set_nested_value(override, ["worker", "remote_api", "api_url"], os.environ["REMOTE_API_URL"])
    if "REMOTE_API_KEY" in os.environ:
        _set_nested_value(override, ["worker", "remote_api", "api_key"], os.environ["REMOTE_API_KEY"])
    if "REMOTE_API_TIMEOUT" in os.environ:
        _set_nested_value(override, ["worker", "remote_api", "timeout_sec"], float(os.environ["REMOTE_API_TIMEOUT"]))
    if "REMOTE_API_HEADERS" in os.environ:
        headers_raw = os.environ["REMOTE_API_HEADERS"]
        headers = json.loads(headers_raw)
        if not isinstance(headers, dict):
            raise ValueError("REMOTE_API_HEADERS must be a JSON object")
        _set_nested_value(override, ["worker", "remote_api", "headers"], headers)
    if "OPENAI_API_KEY" in os.environ:
        _set_nested_value(override, ["worker", "openai", "api_key"], os.environ["OPENAI_API_KEY"])
    if "OPENAI_MODEL" in os.environ:
        _set_nested_value(override, ["worker", "openai", "model"], os.environ["OPENAI_MODEL"])
    if "OPENAI_BASE_URL" in os.environ:
        _set_nested_value(override, ["worker", "openai", "base_url"], os.environ["OPENAI_BASE_URL"])
    if "OPENAI_TIMEOUT" in os.environ:
        _set_nested_value(override, ["worker", "openai", "timeout_sec"], float(os.environ["OPENAI_TIMEOUT"]))
    if "OPENAI_ORGANIZATION" in os.environ:
        _set_nested_value(override, ["worker", "openai", "organization"], os.environ["OPENAI_ORGANIZATION"])
    if "OPENAI_PROJECT" in os.environ:
        _set_nested_value(override, ["worker", "openai", "project"], os.environ["OPENAI_PROJECT"])
    if "OPENAI_REASONING_EFFORT" in os.environ:
        _set_nested_value(override, ["worker", "openai", "reasoning_effort"], os.environ["OPENAI_REASONING_EFFORT"])
    if "OPENAI_MAX_OUTPUT_TOKENS" in os.environ:
        _set_nested_value(override, ["worker", "openai", "max_output_tokens"], int(os.environ["OPENAI_MAX_OUTPUT_TOKENS"]))
    if "OPENAI_JPEG_QUALITY" in os.environ:
        _set_nested_value(override, ["worker", "openai", "jpeg_quality"], int(os.environ["OPENAI_JPEG_QUALITY"]))
    if "GEMINI_API_KEY" in os.environ:
        _set_nested_value(override, ["worker", "gemini", "api_key"], os.environ["GEMINI_API_KEY"])
    if "GEMINI_MODEL" in os.environ:
        _set_nested_value(override, ["worker", "gemini", "model"], os.environ["GEMINI_MODEL"])
    if "GEMINI_API_MODE" in os.environ:
        _set_nested_value(override, ["worker", "gemini", "api_mode"], os.environ["GEMINI_API_MODE"])
    gemini_base_url = os.environ.get("GEMINI_BASE_URL") or os.environ.get("GOOGLE_GEMINI_BASE_URL")
    if gemini_base_url:
        _set_nested_value(override, ["worker", "gemini", "base_url"], gemini_base_url)
    if "GEMINI_TIMEOUT" in os.environ:
        _set_nested_value(override, ["worker", "gemini", "timeout_sec"], float(os.environ["GEMINI_TIMEOUT"]))
    if "GEMINI_MAX_OUTPUT_TOKENS" in os.environ:
        _set_nested_value(override, ["worker", "gemini", "max_output_tokens"], int(os.environ["GEMINI_MAX_OUTPUT_TOKENS"]))
    if "GEMINI_JPEG_QUALITY" in os.environ:
        _set_nested_value(override, ["worker", "gemini", "jpeg_quality"], int(os.environ["GEMINI_JPEG_QUALITY"]))
    if "LLM_MERGE_ENABLED" in os.environ:
        _set_nested_value(override, ["llm_merge", "enabled"], _parse_env_bool(os.environ["LLM_MERGE_ENABLED"]))
    if "LLM_MERGE_BACKEND" in os.environ:
        _set_nested_value(override, ["llm_merge", "backend"], os.environ["LLM_MERGE_BACKEND"])
    if "LLM_MERGE_API_KEY" in os.environ:
        _set_nested_value(override, ["llm_merge", "api_key"], os.environ["LLM_MERGE_API_KEY"])
    if "LLM_MERGE_MODEL" in os.environ:
        _set_nested_value(override, ["llm_merge", "model"], os.environ["LLM_MERGE_MODEL"])
    if "LLM_MERGE_BASE_URL" in os.environ:
        _set_nested_value(override, ["llm_merge", "base_url"], os.environ["LLM_MERGE_BASE_URL"])
    if "LLM_MERGE_TIMEOUT" in os.environ:
        _set_nested_value(override, ["llm_merge", "timeout_sec"], float(os.environ["LLM_MERGE_TIMEOUT"]))
    if "LLM_MERGE_ORGANIZATION" in os.environ:
        _set_nested_value(override, ["llm_merge", "organization"], os.environ["LLM_MERGE_ORGANIZATION"])
    if "LLM_MERGE_PROJECT" in os.environ:
        _set_nested_value(override, ["llm_merge", "project"], os.environ["LLM_MERGE_PROJECT"])
    if "LLM_MERGE_REASONING_EFFORT" in os.environ:
        _set_nested_value(override, ["llm_merge", "reasoning_effort"], os.environ["LLM_MERGE_REASONING_EFFORT"])
    if "LLM_MERGE_MAX_OUTPUT_TOKENS" in os.environ:
        _set_nested_value(override, ["llm_merge", "max_output_tokens"], int(os.environ["LLM_MERGE_MAX_OUTPUT_TOKENS"]))
    if "LLM_MERGE_MAX_ATTEMPTS" in os.environ:
        _set_nested_value(override, ["llm_merge", "max_attempts"], int(os.environ["LLM_MERGE_MAX_ATTEMPTS"]))
    if "LLM_MERGE_SUMMARY_LEVELS" in os.environ:
        _set_nested_value(override, ["llm_merge", "summary_levels"], _parse_env_int_list(os.environ["LLM_MERGE_SUMMARY_LEVELS"]))
    if "LLM_MERGE_REPEAT_COUNT" in os.environ:
        _set_nested_value(override, ["llm_merge", "repeat_count"], int(os.environ["LLM_MERGE_REPEAT_COUNT"]))
    if "LLM_MERGE_BOUNDARY_VOTE_THRESHOLD" in os.environ:
        _set_nested_value(
            override,
            ["llm_merge", "boundary_vote_threshold"],
            float(os.environ["LLM_MERGE_BOUNDARY_VOTE_THRESHOLD"]),
        )
    if "LLM_MERGE_GRANULARITY" in os.environ:
        _set_nested_value(override, ["llm_merge", "granularity"], os.environ["LLM_MERGE_GRANULARITY"])
    if "LLM_MERGE_MIN_INPUT_SEGMENTS" in os.environ:
        _set_nested_value(override, ["llm_merge", "min_input_segments"], int(os.environ["LLM_MERGE_MIN_INPUT_SEGMENTS"]))
    if "LLM_MERGE_MAX_INPUT_SEGMENTS" in os.environ:
        _set_nested_value(override, ["llm_merge", "max_input_segments"], int(os.environ["LLM_MERGE_MAX_INPUT_SEGMENTS"]))
    if "LLM_MERGE_MIN_OUTPUT_RATIO" in os.environ:
        _set_nested_value(override, ["llm_merge", "min_output_ratio"], float(os.environ["LLM_MERGE_MIN_OUTPUT_RATIO"]))
    if "LLM_MERGE_COARSE_MIN_OUTPUT_RATIO" in os.environ:
        _set_nested_value(override, ["llm_merge", "coarse_min_output_ratio"], float(os.environ["LLM_MERGE_COARSE_MIN_OUTPUT_RATIO"]))
    if "LLM_MERGE_COARSE_MAX_SUPPORTED_ANCHORS_PER_RANGE" in os.environ:
        _set_nested_value(
            override,
            ["llm_merge", "coarse_max_supported_anchors_per_range"],
            int(os.environ["LLM_MERGE_COARSE_MAX_SUPPORTED_ANCHORS_PER_RANGE"]),
        )
    if "LLM_MERGE_COARSE_ANCHOR_MIN_SPACING_SEGMENTS" in os.environ:
        _set_nested_value(
            override,
            ["llm_merge", "coarse_anchor_min_spacing_segments"],
            int(os.environ["LLM_MERGE_COARSE_ANCHOR_MIN_SPACING_SEGMENTS"]),
        )
    if "LLM_MERGE_COARSE_ANCHOR_MIN_SIDE_SEGMENTS" in os.environ:
        _set_nested_value(
            override,
            ["llm_merge", "coarse_anchor_min_side_segments"],
            int(os.environ["LLM_MERGE_COARSE_ANCHOR_MIN_SIDE_SEGMENTS"]),
        )
    if "LLM_MERGE_COARSE_ANCHOR_MIN_SCORE" in os.environ:
        _set_nested_value(
            override,
            ["llm_merge", "coarse_anchor_min_score"],
            float(os.environ["LLM_MERGE_COARSE_ANCHOR_MIN_SCORE"]),
        )
    if "LLM_MERGE_PROTECT_SUPPORTED_BOUNDARIES" in os.environ:
        _set_nested_value(
            override,
            ["llm_merge", "protect_supported_boundaries"],
            _parse_env_bool(os.environ["LLM_MERGE_PROTECT_SUPPORTED_BOUNDARIES"]),
        )
    if "LLM_MERGE_PROTECTED_BOUNDARY_SUPPORT_THRESHOLD" in os.environ:
        _set_nested_value(
            override,
            ["llm_merge", "protected_boundary_support_threshold"],
            float(os.environ["LLM_MERGE_PROTECTED_BOUNDARY_SUPPORT_THRESHOLD"]),
        )
    if "LLM_MERGE_PROTECT_DISTINCT_SEQUENCE_MARKERS" in os.environ:
        _set_nested_value(
            override,
            ["llm_merge", "protect_distinct_sequence_markers"],
            _parse_env_bool(os.environ["LLM_MERGE_PROTECT_DISTINCT_SEQUENCE_MARKERS"]),
        )
    if "LLM_MERGE_PROTECT_INSTRUCTION_DRIFT" in os.environ:
        _set_nested_value(
            override,
            ["llm_merge", "protect_instruction_drift"],
            _parse_env_bool(os.environ["LLM_MERGE_PROTECT_INSTRUCTION_DRIFT"]),
        )
    if "LLM_MERGE_PROTECT_DUPLICATE_TAIL_ANCHOR" in os.environ:
        _set_nested_value(
            override,
            ["llm_merge", "protect_duplicate_tail_anchor"],
            _parse_env_bool(os.environ["LLM_MERGE_PROTECT_DUPLICATE_TAIL_ANCHOR"]),
        )
    if "LLM_MERGE_DUPLICATE_TAIL_ANCHOR_MIN_FRAMES" in os.environ:
        _set_nested_value(
            override,
            ["llm_merge", "duplicate_tail_anchor_min_frames"],
            int(os.environ["LLM_MERGE_DUPLICATE_TAIL_ANCHOR_MIN_FRAMES"]),
        )

    return override


def _build_config_with_env_overrides(config_cls, base_data: Optional[dict] = None) -> Config:
    merged = _deep_merge_dicts(base_data or {}, _collect_env_override_data())
    return config_cls.model_validate(merged)


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


def _parse_env_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean environment value: {value}")


def _parse_env_int_list(value: str) -> List[int]:
    text = str(value).strip()
    if not text:
        raise ValueError("Invalid integer list environment value: empty string")
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"Invalid integer list environment value: {value}")
    return [int(part) for part in parts]
