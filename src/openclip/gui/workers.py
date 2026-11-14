"""Background workers so the UI never blocks on engine work."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from openclip.engine.ffmpeg import MediaInfo, probe
from openclip.engine.pipeline import run_pipeline
from openclip.engine.project import Project
from openclip.engine.render import RenderOptions, render_project


@dataclass
class AIPlanRequest:
    media_path: Path
    prompt: str
    transcript: str
    provider: str = "local_mock"
    api_key: str | None = None  # BYOK: passed via env override, never logged/stored


def provider_env_overrides(provider: str, api_key: str | None) -> dict[str, str]:
    """Map a BYOK key from the UI to the provider's env var (in-memory only)."""
    if not api_key:
        return {}
    from openclip.ai.config import key_env_var_for

    var = key_env_var_for(provider)
    return {var: api_key} if var else {}


class AIPlanWorker(QObject):
    """Runs the AI prompt-to-plan pipeline off the UI thread. BYOK: the key
    from the request is injected as an env override for this call only (never
    logged, never persisted); tests use local_mock with no key."""

    finished = Signal(object)  # AIDirectorPlan
    failed = Signal(str)

    def __init__(self, request: AIPlanRequest) -> None:
        super().__init__()
        self._request = request

    def run(self) -> None:  # slot, executed on the worker thread
        try:
            import os

            from openclip.ai import create_provider, generate_plan, load_provider_config

            env: dict[str, str] | None = None
            overrides = provider_env_overrides(self._request.provider, self._request.api_key)
            if overrides:
                env = {**os.environ, **overrides}
            media = probe(self._request.media_path)
            provider = create_provider(load_provider_config(self._request.provider, env=env))
            plan = generate_plan(
                provider,
                self._request.prompt,
                self._request.transcript,
                media.duration_s,
            )
            self.finished.emit(plan)
        except Exception as exc:  # surfaced, not fatal to the app
            self.failed.emit(str(exc))


@dataclass
class AIRenderRequest:
    project: Project
    media: MediaInfo
    edl: object  # EditDecisionList — typed loosely to avoid a hard import here
    out_path: Path
    burn_captions: bool = True
    crf: int = 20
    transcript_captions: bool = False  # burn transcript-derived bilingual captions
    caption_style: str | None = None  # STYLE_TEMPLATES name


class AIRenderWorker(QObject):
    """Renders an AI EDL (only accepted decisions execute) off the UI thread."""

    finished = Signal(object)  # RenderResult
    failed = Signal(str)
    progress = Signal(str, int, int)  # stage, i, n

    def __init__(self, request: AIRenderRequest) -> None:
        super().__init__()
        self._request = request

    def run(self) -> None:  # slot, executed on the worker thread
        try:
            from openclip.director.executor import render_edl
            from openclip.director.models import EditDecisionList

            assert isinstance(self._request.edl, EditDecisionList)
            result = render_edl(
                self._request.project,
                self._request.media,
                self._request.edl,
                self._request.out_path,
                RenderOptions(
                    burn_captions=self._request.burn_captions,
                    crf=self._request.crf,
                    transcript_captions=self._request.transcript_captions,
                    caption_style=self._request.caption_style,
                ),
                progress=lambda stage, i, n: self.progress.emit(stage, i, n),
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


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
