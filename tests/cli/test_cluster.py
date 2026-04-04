from pathlib import Path

from click.testing import CliRunner

from video2tasks.cli.cluster import main


class FakeProcess:
    created = []

    def __init__(self, target, args=(), name=None):
        self.target = target
        self.args = args
        self.name = name
        self._alive = False
        type(self).created.append(self)

    def start(self):
        self._alive = True

    def is_alive(self):
        if self.name == "v2t-server":
            return False
        return self._alive

    def terminate(self):
        self._alive = False

    def join(self, timeout=None):
        self._alive = False


def _reset_fake_process_state() -> None:
    FakeProcess.created = []


def test_cluster_cli_uses_default_worker_count_when_not_set(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("datasets: []\n", encoding="utf-8")

    _reset_fake_process_state()
    monkeypatch.setattr("video2tasks.cli.cluster.multiprocessing.Process", FakeProcess)
    monkeypatch.setattr("video2tasks.cli.cluster.time.sleep", lambda _secs: None)

    runner = CliRunner()
    result = runner.invoke(main, ["--config", str(config_path)])

    assert result.exit_code == 0
    assert len(FakeProcess.created) == 8

    server = [proc for proc in FakeProcess.created if proc.name == "v2t-server"]
    workers = [proc for proc in FakeProcess.created if str(proc.name).startswith("v2t-worker-")]
    assert len(server) == 1
    assert len(workers) == 7


def test_cluster_cli_uses_worker_count_from_config(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "datasets: []\nworker:\n  count: 3\n",
        encoding="utf-8",
    )

    _reset_fake_process_state()
    monkeypatch.setattr("video2tasks.cli.cluster.multiprocessing.Process", FakeProcess)
    monkeypatch.setattr("video2tasks.cli.cluster.time.sleep", lambda _secs: None)

    runner = CliRunner()
    result = runner.invoke(main, ["--config", str(config_path)])

    assert result.exit_code == 0
    assert len(FakeProcess.created) == 4

    server = [proc for proc in FakeProcess.created if proc.name == "v2t-server"]
    workers = [proc for proc in FakeProcess.created if str(proc.name).startswith("v2t-worker-")]
    assert len(server) == 1
    assert len(workers) == 3


def test_cluster_cli_supports_env_only_configuration(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORKER_COUNT", "2")

    _reset_fake_process_state()
    monkeypatch.setattr("video2tasks.cli.cluster.multiprocessing.Process", FakeProcess)
    monkeypatch.setattr("video2tasks.cli.cluster.time.sleep", lambda _secs: None)

    runner = CliRunner()
    result = runner.invoke(main, [])

    assert result.exit_code == 0
    assert len(FakeProcess.created) == 3


def test_cluster_cli_wraps_invalid_configuration_as_usage_error(monkeypatch) -> None:
    def fake_load(_config):
        raise ValueError("invalid env configuration")

    monkeypatch.setattr("video2tasks.cli.cluster.Config.load", fake_load)

    result = CliRunner().invoke(main, [])

    assert result.exit_code != 0
    assert "invalid env configuration" in result.output
