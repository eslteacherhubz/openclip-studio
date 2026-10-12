"""Engine core: headless editing pipeline (no GUI or gaze imports)."""

from openclip.engine.captions import CaptionStyle, build_ass
from openclip.engine.cutplan import CutDecision, CutOptions, CutPlan, kept_intervals, plan_cuts
from openclip.engine.ffmpeg import FFmpegError, MediaInfo, probe, run_ffmpeg
from openclip.engine.project import Project
from openclip.engine.render import RenderOptions, RenderResult, render_project
from openclip.engine.script import LessonScript, parse_script
from openclip.engine.synth import SynthLesson, generate_lesson

__all__ = [
    "CaptionStyle",
    "CutDecision",
    "CutOptions",
    "CutPlan",
    "FFmpegError",
    "LessonScript",
    "MediaInfo",
    "Project",
    "RenderOptions",
    "RenderResult",
    "SynthLesson",
    "build_ass",
    "generate_lesson",
    "kept_intervals",
    "parse_script",
    "plan_cuts",
    "probe",
    "render_project",
    "run_ffmpeg",
]
