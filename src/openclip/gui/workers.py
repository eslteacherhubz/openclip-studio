"""Background workers so the UI never blocks on engine work."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from openclip.engine.ffmpeg import MediaInfo, probe  # noqa: F401 (re-export convenience)
from openclip.engine.pipeline import run_pipeline
from openclip.engine.project import Project
from openclip.engine.render import RenderOptions, render_project


@dataclass
class AnalysisRequest:
    project: Project
    media: MediaInfo
    transcriber: str = "auto"
    language: str | None = None


class AnalysisWorker(QObject):
    """Runs align+plan off the UI thread."""

    finished = Signal(object)  # PlanOutcome
    failed = Signal(str)
    note = Signal(str)

    def __init__(self, request: AnalysisRequest) -> None:
        super().__init__()
        self._request = request

    def run(self) -> None:  # slot, executed on the worker thread
        try:
            outcome = run_pipeline(
                self._request.project,
                media=self._request.media,
                transcriber=self._request.transcriber,
                language=self._request.language,
            )
            for note in outcome.notes:
                self.note.emit(note)
            self.finished.emit(outcome)
        except Exception as exc:  # engine errors are surfaced, not fatal to the app
            self.failed.emit(str(exc))


@dataclass
class RenderRequest:
    project: Project
    media: MediaInfo
    out_path: Path
    options: RenderOptions


class RenderWorker(QObject):
    finished = Signal(object)  # RenderResult
    failed = Signal(str)
    progress = Signal(str, int, int)  # stage, i, n

    def __init__(self, request: RenderRequest) -> None:
        super().__init__()
        self._request = request

    def run(self) -> None:  # slot, executed on the worker thread
        try:
            result = render_project(
                self._request.project,
                self._request.media,
                self._request.out_path,
                self._request.options,
                progress=lambda stage, i, n: self.progress.emit(stage, i, n),
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class ThreadedRunner(QObject):
    """Owns a QThread + worker pair; keeps references alive; cleans up."""

    def __init__(self) -> None:
        super().__init__()
        self._thread: QThread | None = None
        self._worker: QObject | None = None

    def start(self, worker: QObject) -> None:
        """Connect UI slots to worker signals BEFORE calling start()."""
        assert self._thread is None, "runner busy"
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)  # type: ignore[attr-defined]
        worker.finished.connect(self._cleanup)  # type: ignore[attr-defined]
        worker.failed.connect(self._cleanup)  # type: ignore[attr-defined]
        self._thread = thread
        self._worker = worker
        thread.start()

    def _cleanup(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
        self._thread = None
        self._worker = None

    @property
    def busy(self) -> bool:
        return self._thread is not None
