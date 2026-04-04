from click.testing import CliRunner

from video2tasks.cli.worker import main


def test_worker_cli_supports_env_only_configuration(monkeypatch, tmp_path) -> None:
    captured = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUN_ID", "env-only-run")
    monkeypatch.setenv("WORKER_COUNT", "3")

    def fake_run_worker(cfg):
        captured["run_id"] = cfg.run.run_id
        captured["worker_count"] = cfg.worker.count

    monkeypatch.setattr("video2tasks.cli.worker.run_worker", fake_run_worker)

    result = CliRunner().invoke(main, [])

    assert result.exit_code == 0
    assert captured == {"run_id": "env-only-run", "worker_count": 3}


def test_worker_cli_wraps_invalid_configuration_as_usage_error(monkeypatch) -> None:
    def fake_load(_config):
        raise ValueError("invalid env configuration")

    monkeypatch.setattr("video2tasks.cli.worker.Config.load", fake_load)

    result = CliRunner().invoke(main, [])

    assert result.exit_code != 0
    assert "invalid env configuration" in result.output
