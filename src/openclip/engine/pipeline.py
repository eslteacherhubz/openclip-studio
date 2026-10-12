"""High-level pipeline: alignment + cut planning over a Project.

Shared by the CLI and the GUI so both exercise identical engine paths.
``run_pipeline`` transcribes at most once per invocation and feeds the same
word hypotheses to both the aligner and the cut planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from openclip.engine.align import (
    Alignment,
    WhisperTranscriber,
    WordHyp,
    align_script,
    alignment_from_dict,
    even_alignment,
)
from openclip.engine.cutplan import CutOptions, CutPlan, plan_cuts
from openclip.engine.ffmpeg import MediaInfo, detect_silences, probe
from openclip.engine.project import Project
from openclip.engine.synth import WordSpan


def available_transcribers() -> list[str]:
    out = ["even"]
    try:
        import faster_whisper  # noqa: F401

        out.append("whisper")
    except ImportError:
        pass
    return out


def spans_to_hyps(spans: list[WordSpan]) -> list[WordHyp]:
    return [WordHyp(s.word, s.start_s, s.end_s) for s in spans]


@dataclass
class PlanOutcome:
    alignment: Alignment
    cut_plan: CutPlan
    transcriber: str
    silences_found: int
    words_found: int = 0
    notes: list[str] = field(default_factory=list)


def _transcribe(
    project: Project,
    transcriber: str,
    language: str | None,
    media: MediaInfo,
    hyps: list[WordHyp] | None,
) -> tuple[str, list[WordHyp]]:
    if transcriber == "auto":
        transcriber = "whisper" if "whisper" in available_transcribers() else "even"
    if transcriber not in ("even", "whisper", "mock"):
        raise ValueError(f"Unknown transcriber {transcriber!r}")
    if transcriber == "even":
        return "even", []
    if hyps is not None:
        return transcriber, hyps
    return transcriber, WhisperTranscriber().transcribe(Path(project.source), language)


def run_pipeline(
    project: Project,
    media: MediaInfo | None = None,
    transcriber: str = "auto",
    language: str | None = None,
    options: CutOptions | None = None,
    hyps: list[WordHyp] | None = None,
    noise_db: float = -35.0,
    silence_min_s: float = 0.5,
) -> PlanOutcome:
    """Align + plan in one pass; caches results into the project."""
    media = media or probe(project.source)
    language = language or project.align_language
    opts = options or project.cut_options
    notes: list[str] = []

    name, hyps_used = _transcribe(project, transcriber, language, media, hyps)
    if name == "even":
        if len(project.script) == 0:
            notes.append("No script provided: silence-only cuts.")
        else:
            notes.append(
                "No ASR available: even alignment estimate; cuts are silence-only."
            )
        alignment = even_alignment(project.script, media.duration_s)
    else:
        alignment = align_script(project.script, hyps_used)

    alignment.transcriber = name
    project.alignment = alignment.to_dict()

    silences = detect_silences(
        project.source, noise_db=noise_db, min_duration_s=silence_min_s
    )
    cut_plan = plan_cuts(alignment, hyps_used, media.duration_s, silences, opts)
    project.cut_plan = cut_plan
    return PlanOutcome(
        alignment=alignment,
        cut_plan=cut_plan,
        transcriber=name,
        silences_found=len(silences),
        words_found=len(hyps_used),
        notes=notes,
    )


def replan(
    project: Project,
    media: MediaInfo | None = None,
    options: CutOptions | None = None,
    noise_db: float = -35.0,
    silence_min_s: float = 0.5,
) -> CutPlan:
    """Re-plan from a cached alignment without re-transcribing.

    Filler/off-script cuts need word hypotheses, so when the cached
    alignment did not carry them (or no ASR ran), this produces
    silence-only cuts. Full re-analysis uses ``run_pipeline``.
    """
    media = media or probe(project.source)
    opts = options or project.cut_options
    if project.alignment is None:
        raise ValueError("Project has no cached alignment; call run_pipeline first")
    alignment = alignment_from_dict(project.alignment)
    silences = detect_silences(
        project.source, noise_db=noise_db, min_duration_s=silence_min_s
    )
    hyps: list[WordHyp] = []
    cut_plan = plan_cuts(alignment, hyps, media.duration_s, silences, opts)
    project.cut_plan = cut_plan
    return cut_plan
