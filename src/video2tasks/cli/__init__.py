"""CLI module."""

from .cluster import main as cluster_main
from .server import main as server_main
from .single_video import main as single_video_main
from .validate_config import main as validate_main
from .worker import main as worker_main

__all__ = [
    "cluster_main",
    "server_main",
    "single_video_main",
    "validate_main",
    "worker_main",
]
