import video2tasks.server.app as server_app_module
import video2tasks.vlm.openai_api as openai_api_module
import video2tasks.worker.runner as worker_runner_module
from video2tasks.config import Config
from video2tasks.logging_utils import PACKAGE_LOGGER_NAME


def test_server_worker_and_vlm_loggers_share_package_namespace(tmp_path) -> None:
    config = Config(
        run={"base_dir": str(tmp_path), "run_id": "testrun"},
        server={"auto_exit_after_all_done": False},
    )

    server_app_module.create_app(config)

    assert server_app_module.logger.name == "video2tasks.server.app"
    assert worker_runner_module.logger.name == "video2tasks.worker.runner"
    assert openai_api_module.logger.name == "video2tasks.vlm.openai_api"
    assert server_app_module.logger.parent.name == PACKAGE_LOGGER_NAME
    assert worker_runner_module.logger.parent.name == PACKAGE_LOGGER_NAME
    assert openai_api_module.logger.parent.name == PACKAGE_LOGGER_NAME


def test_config_logging_level_suppresses_info_output_across_package(tmp_path, capsys) -> None:
    config = Config(
        run={"base_dir": str(tmp_path), "run_id": "testrun"},
        server={"auto_exit_after_all_done": False},
        logging={"level": "WARNING"},
    )

    server_app_module.create_app(config)
    server_app_module.logger.info("[Server] hidden info")
    worker_runner_module.logger.info("[Worker] hidden info")
    openai_api_module.logger.warning("[OpenAI] visible warning")

    out = capsys.readouterr().out

    assert "[Server] hidden info" not in out
    assert "[Worker] hidden info" not in out
    assert "[OpenAI] visible warning" in out
