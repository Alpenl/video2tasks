"""Runtime lifecycle handles for server background workers."""

from __future__ import annotations

import threading
from typing import Callable


RuntimeTarget = Callable[[threading.Event], None]


class ThreadRuntime:
    """Manage an explicitly started thread-backed runtime."""

    def __init__(
        self,
        *,
        name: str,
        target: RuntimeTarget,
        daemon: bool = True,
    ) -> None:
        self._name = str(name).strip() or "runtime-thread"
        self._target = target
        self._daemon = bool(daemon)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=self._name,
                daemon=self._daemon,
            )
            thread = self._thread

        thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def _run(self) -> None:
        self._target(self._stop_event)
