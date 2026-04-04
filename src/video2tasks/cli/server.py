"""Server CLI entrypoint."""

import click
from pydantic import ValidationError
from pathlib import Path
from ..config import Config
from ..server.app import run_server


@click.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file"
)
def main(config: Path | None) -> None:
    """Start the Video2Tasks server."""
    try:
        cfg = Config.load(config)
    except (FileNotFoundError, ValidationError, ValueError) as exc:
        raise click.UsageError(str(exc)) from exc

    run_server(cfg)


if __name__ == "__main__":
    main()
