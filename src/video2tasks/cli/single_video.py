"""Single-video CLI entrypoint."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import click
from pydantic import ValidationError

from ..config import Config, DatasetConfig
from .cluster import run_cluster


_INPUT_ROOTNAME = ".v2t_single_input"


def _load_config(config: Path | None = None) -> Config:
    try:
        if config is None:
            return Config.load()
        return Config.load(config)
    except (FileNotFoundError, ValidationError, ValueError) as exc:
        raise click.UsageError(str(exc)) from exc


def _sanitize_token(value: str, fallback: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value).strip())
    text = text.strip("._-")
    return text or fallback


def _materialize_input_video(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.is_symlink():
            try:
                if target.resolve() == source.resolve():
                    return
            except OSError:
                pass
        target.unlink()

    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def _prepare_single_video_config(cfg: Config, video_path: Path, output_dir: Path) -> Config:
    output_dir.mkdir(parents=True, exist_ok=True)

    subset = _sanitize_token(video_path.stem, "video")
    sample_id = subset
    input_root = output_dir / _INPUT_ROOTNAME
    sample_dir = input_root / subset / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    wrapped_video_path = sample_dir / f"Frame_{sample_id}.mp4"
    _materialize_input_video(video_path, wrapped_video_path)

    next_cfg = cfg.model_copy(deep=True)
    next_cfg.datasets = [DatasetConfig(root=str(input_root), subset=subset)]
    next_cfg.run.base_dir = str(output_dir)
    return next_cfg


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.argument(
    "video_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "output_dir",
    required=False,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
def main(config: Path | None, video_path: Path, output_dir: Path | None) -> None:
    """Run the pipeline on a single video path."""
    cfg = _load_config(config)

    resolved_video_path = video_path.expanduser().resolve()
    resolved_output_dir = (output_dir or resolved_video_path.parent).expanduser().resolve()
    single_cfg = _prepare_single_video_config(cfg, resolved_video_path, resolved_output_dir)

    click.echo(f"Input video: {resolved_video_path}")
    click.echo(f"Output base dir: {resolved_output_dir}")
    run_cluster(single_cfg)


if __name__ == "__main__":
    main()
