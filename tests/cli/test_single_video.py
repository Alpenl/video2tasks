from pathlib import Path

from click.testing import CliRunner

from video2tasks.config import Config


def test_single_video_cli_passes_explicit_config_path_to_config_load(
    monkeypatch, tmp_path: Path
) -> None:
    from video2tasks.cli.single_video import main

    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("datasets: []\n", encoding="utf-8")

    captured = {}

    def fake_load(config_path_arg: Path | None = None) -> Config:
        captured["config_path"] = config_path_arg
        return Config(datasets=[], worker={"backend": "dummy"})

    monkeypatch.setattr("video2tasks.cli.single_video.Config.load", fake_load)
    monkeypatch.setattr("video2tasks.cli.single_video.run_cluster", lambda _cfg: None)

    result = CliRunner().invoke(main, ["--config", str(config_path), str(video_path)])

    assert result.exit_code == 0
    assert captured["config_path"] == config_path


def test_single_video_cli_uses_default_config_load_when_config_not_provided(
    monkeypatch, tmp_path: Path
) -> None:
    from video2tasks.cli.single_video import main

    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")

    captured = {}

    def fake_load(*args, **kwargs) -> Config:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return Config(datasets=[], worker={"backend": "dummy"})

    monkeypatch.setattr("video2tasks.cli.single_video.Config.load", fake_load)
    monkeypatch.setattr("video2tasks.cli.single_video.run_cluster", lambda _cfg: None)

    result = CliRunner().invoke(main, [str(video_path)])

    assert result.exit_code == 0
    assert captured["args"] == ()
    assert captured["kwargs"] == {}


def test_single_video_cli_defaults_output_dir_to_video_parent(monkeypatch, tmp_path: Path) -> None:
    from video2tasks.cli.single_video import main

    video_dir = tmp_path / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / "demo.mp4"
    video_path.write_bytes(b"video")

    captured = {}

    monkeypatch.setattr(
        "video2tasks.cli.single_video.Config.load",
        lambda: Config(datasets=[], worker={"backend": "dummy"}),
    )

    def fake_run_cluster(cfg: Config) -> None:
        captured["config"] = cfg

    monkeypatch.setattr("video2tasks.cli.single_video.run_cluster", fake_run_cluster)

    result = CliRunner().invoke(main, [str(video_path)])

    assert result.exit_code == 0
    cfg = captured["config"]
    assert cfg.run.base_dir == str(video_dir)
    assert len(cfg.datasets) == 1
    assert cfg.datasets[0].root == str(video_dir / ".v2t_single_input")
    assert cfg.datasets[0].subset == "demo"

    wrapped_video = video_dir / ".v2t_single_input" / "demo" / "demo" / "Frame_demo.mp4"
    assert wrapped_video.exists()


def test_single_video_cli_uses_explicit_output_dir(monkeypatch, tmp_path: Path) -> None:
    from video2tasks.cli.single_video import main

    video_dir = tmp_path / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / "demo.mp4"
    video_path.write_bytes(b"video")
    output_dir = tmp_path / "outputs"

    captured = {}

    monkeypatch.setattr(
        "video2tasks.cli.single_video.Config.load",
        lambda: Config(datasets=[], worker={"backend": "dummy"}),
    )

    def fake_run_cluster(cfg: Config) -> None:
        captured["config"] = cfg

    monkeypatch.setattr("video2tasks.cli.single_video.run_cluster", fake_run_cluster)

    result = CliRunner().invoke(main, [str(video_path), str(output_dir)])

    assert result.exit_code == 0
    cfg = captured["config"]
    assert cfg.run.base_dir == str(output_dir)
    assert cfg.datasets[0].root == str(output_dir / ".v2t_single_input")
    assert (output_dir / ".v2t_single_input" / "demo" / "demo" / "Frame_demo.mp4").exists()


def test_single_video_cli_wraps_invalid_configuration_as_usage_error(monkeypatch, tmp_path: Path) -> None:
    from video2tasks.cli.single_video import main

    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")

    def fake_load() -> Config:
        raise ValueError("invalid env configuration")

    monkeypatch.setattr("video2tasks.cli.single_video.Config.load", fake_load)

    result = CliRunner().invoke(main, [str(video_path)])

    assert result.exit_code != 0
    assert "invalid env configuration" in result.output


def test_single_video_help_includes_config_option() -> None:
    from video2tasks.cli.single_video import main

    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "-c, --config PATH" in result.output
