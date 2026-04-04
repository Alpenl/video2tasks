"""Cluster CLI entrypoint for starting server and workers together."""

import multiprocessing
import time
from pathlib import Path

import click
from pydantic import ValidationError

from ..config import Config
from ..server.app import run_server
from ..worker.runner import run_worker


def _load_config(config: Path | None) -> Config:
    try:
        return Config.load(config)
    except (FileNotFoundError, ValidationError, ValueError) as exc:
        raise click.UsageError(str(exc)) from exc


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
def main(config: Path | None) -> None:
    """Start one server process plus N worker processes."""
    cfg = _load_config(config)

    worker_count = int(cfg.worker.count)
    server_proc = multiprocessing.Process(
        target=run_server,
        args=(cfg,),
        name="v2t-server",
    )
    worker_procs = [
        multiprocessing.Process(
            target=run_worker,
            args=(cfg,),
            name=f"v2t-worker-{index + 1}",
        )
        for index in range(worker_count)
    ]

    procs = [server_proc, *worker_procs]
    for proc in procs:
        proc.start()

    click.echo(f"Started v2t-server + {worker_count} workers")

    try:
        while True:
            if not server_proc.is_alive():
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        click.echo("Shutting down server/workers...")
    finally:
        for proc in reversed(procs):
            if proc.is_alive():
                proc.terminate()
        for proc in reversed(procs):
            proc.join(timeout=5)


if __name__ == "__main__":
    main()
