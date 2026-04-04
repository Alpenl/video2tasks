from click.testing import CliRunner

from video2tasks.cli.server import main


def test_server_cli_supports_env_only_configuration(monkeypatch, tmp_path) -> None:
    captured = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUN_ID", "env-only-run")
    monkeypatch.setenv("WORKER_COUNT", "2")

    def fake_run_server(cfg):
        captured["run_id"] = cfg.run.run_id
        captured["worker_count"] = cfg.worker.count

    monkeypatch.setattr("video2tasks.cli.server.run_server", fake_run_server)

    result = CliRunner().invoke(main, [])

    assert result.exit_code == 0
    assert captured == {"run_id": "env-only-run", "worker_count": 2}


def test_server_cli_wraps_invalid_configuration_as_usage_error(monkeypatch) -> None:
    def fake_load(_config):
        raise ValueError("invalid env configuration")

    monkeypatch.setattr("video2tasks.cli.server.Config.load", fake_load)

    result = CliRunner().invoke(main, [])

    assert result.exit_code != 0
    assert "invalid env configuration" in result.output
