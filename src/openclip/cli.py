"""Headless CLI: every engine capability is reachable without the GUI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openclip import __version__


def _cmd_probe(args: argparse.Namespace) -> int:
    from openclip.engine.ffmpeg import probe

    info = probe(args.media)
    print(json.dumps(info.to_dict(), indent=2))
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    from openclip.engine.synth import generate_lesson

    lesson = generate_lesson(args.out)
    sidecar = Path(args.out).with_suffix(".lesson.json")
    sidecar.write_text(lesson.ground_truth_json(), encoding="utf-8")
    print(f"Synthetic lesson: {lesson.path}")
    print(f"Ground truth:     {sidecar}")
    print(f"Duration:         {lesson.duration_s:.2f}s, {len(lesson.sentences)} sentences")
    return 0


def _cmd_new(args: argparse.Namespace) -> int:
    from openclip.engine.ffmpeg import probe
    from openclip.engine.project import Project

    script_text = Path(args.script).read_text(encoding="utf-8")
    project = Project.create(source=args.media, script_text=script_text)
    project.save(args.project)
    info = probe(args.media)
    print(
        f"Project created: {args.project}\n"
        f"Media: {info.width}x{info.height} @ {info.fps:.3f} fps, "
        f"{info.duration_s:.2f}s, audio={info.has_audio}"
    )
    return 0


def _cmd_align(args: argparse.Namespace) -> int:
    from openclip.engine.ffmpeg import probe
    from openclip.engine.pipeline import available_transcribers, run_pipeline
    from openclip.engine.project import Project

    project = Project.load(args.project)
    if args.transcriber == "whisper" and "whisper" not in available_transcribers():
        print(
            "Error: faster-whisper not installed. pip install eterna-openclip-studio[asr]",
            file=sys.stderr,
        )
        return 2
    outcome = run_pipeline(
        project,
        media=probe(project.source),
        transcriber=args.transcriber,
        language=args.language,
    )
    project.save(args.project)
    print(f"Transcriber: {outcome.transcriber}, words: {outcome.words_found}")
    for ln in outcome.alignment.lines:
        if ln.confidence > 0:
            print(
                f"  line {ln.index}: {ln.start_s:7.2f} -> {ln.end_s:7.2f} "
                f"(conf {ln.confidence:.2f})"
            )
        else:
            print(f"  line {ln.index}: UNMATCHED")
    for note in outcome.notes:
        print(f"  note: {note}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    from openclip.engine.ffmpeg import probe
    from openclip.engine.pipeline import run_pipeline
    from openclip.engine.project import Project

    project = Project.load(args.project)
    outcome = run_pipeline(
        project,
        media=probe(project.source),
        transcriber=args.transcriber,
        language=args.language,
    )
    project.save(args.project)
    print(f"Silences found: {outcome.silences_found}")
    for cut in outcome.cut_plan.cuts:
        print(
            f"  [{cut.kind:>9}] {cut.start_s:7.2f} -> {cut.end_s:7.2f} "
            f"(conf {cut.confidence:.2f}) {cut.reason}"
        )
    from openclip.engine.cutplan import kept_intervals

    kept = kept_intervals(outcome.cut_plan.cuts, probe(project.source).duration_s)
    total = sum(e - s for s, e in kept)
    print(f"Kept: {len(kept)} intervals, {total:.2f}s")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    from openclip.engine.ffmpeg import probe
    from openclip.engine.pipeline import run_pipeline
    from openclip.engine.project import Project
    from openclip.engine.render import RenderOptions, render_project

    project = Project.load(args.project)
    media = probe(project.source)
    if project.alignment is None or args.replan:
        run_pipeline(project, media=media, transcriber=args.transcriber)
        project.save(args.project)
    result = render_project(
        project,
        media,
        args.out,
        RenderOptions(
            burn_captions=not args.no_captions,
            crf=args.crf,
        ),
        progress=lambda stage, i, n: print(f"  [{stage} {i}/{n}]", file=sys.stderr),
    )
    print(
        f"Exported: {result.out}\n"
        f"  kept {result.kept_duration_s:.2f}s from {media.duration_s:.2f}s "
        f"({result.cuts_applied} cuts, {result.segments} segments)\n"
        f"  captions={'yes' if result.captions_burned else 'no'}, "
        f"audio_preset={args.audio_preset or project.audio_preset}"
    )
    return 0


def _cmd_auto(args: argparse.Namespace) -> int:
    from openclip.engine.ffmpeg import probe
    from openclip.engine.pipeline import run_pipeline
    from openclip.engine.project import Project
    from openclip.engine.render import RenderOptions, render_project

    project = Project.load(args.project)
    media = probe(project.source)
    outcome = run_pipeline(
        project, media=media, transcriber=args.transcriber, language=args.language
    )
    project.save(args.project)
    for note in outcome.notes:
        print(f"note: {note}", file=sys.stderr)
    result = render_project(
        project,
        media,
        args.out,
        RenderOptions(
            burn_captions=not args.no_captions,
            crf=args.crf,
            audio_preset=args.audio_preset,
        ),
    )
    print(
        f"Exported: {result.out}\n"
        f"  transcriber={outcome.transcriber}, silences={outcome.silences_found}, "
        f"cuts={result.cuts_applied}, kept {result.kept_duration_s:.2f}s "
        f"of {media.duration_s:.2f}s"
    )
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    try:
        from openclip.gui.app import main as gui_main
    except ImportError as exc:
        print(
            "GUI requires the 'gui' extra: pip install eterna-openclip-studio[gui]",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return gui_main(args.argv)


def _require_gaze() -> bool:
    from openclip.gaze import available

    if not available():
        print(
            "Gaze tools need the 'gaze' extra: pip install eterna-openclip-studio[gaze]",
            file=sys.stderr,
        )
        return False
    return True


def _cmd_gaze_analyze(args: argparse.Namespace) -> int:
    if not _require_gaze():
        return 2
    from openclip.gaze.apply import analyze_to_json

    print(analyze_to_json(args.media, max_frames=args.max_frames))
    return 0


def _cmd_gaze_apply(args: argparse.Namespace) -> int:
    if not _require_gaze():
        return 2
    from openclip.gaze.apply import apply_to_video
    from openclip.gaze.evaluate import render_report, run_all

    stats = apply_to_video(
        args.media,
        args.out,
        strength=args.strength,
        max_frames=args.max_frames,
        mode=args.mode,
    )
    print(f"Corrected video: {args.out}")
    print(f"Frames: {stats.frames} (eyes detected in {stats.frames_with_eyes})")
    print(render_report(run_all(strength=args.strength)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="openclip",
        description="Eterna OpenClip Studio — bilingual teaching-video editor",
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("probe", help="inspect a media file")
    sp.add_argument("media")
    sp.set_defaults(func=_cmd_probe)

    sp = sub.add_parser("synth", help="generate a synthetic bilingual lesson (testing/demo)")
    sp.add_argument("--out", required=True, help="output .mp4 path")
    sp.set_defaults(func=_cmd_synth)

    sp = sub.add_parser("new", help="create a project from media + script")
    sp.add_argument("--media", required=True)
    sp.add_argument("--script", required=True, help="script text file")
    sp.add_argument("-o", "--project", required=True, help="project .json path")
    sp.set_defaults(func=_cmd_new)

    def add_transcriber(sp: argparse.ArgumentParser, default: str = "auto") -> None:
        sp.add_argument(
            "-t",
            "--transcriber",
            choices=["auto", "even", "whisper", "mock"],
            default=default,
            help="alignment source (default: whisper if installed, else even)",
        )
        sp.add_argument("--language", default=None, help="BCP-47 hint for ASR (e.g. 'en')")

    sp = sub.add_parser("align", help="align script to media and cache into project")
    sp.add_argument("project")
    add_transcriber(sp)
    sp.set_defaults(func=_cmd_align)

    sp = sub.add_parser("plan", help="align + build the cut plan and cache into project")
    sp.add_argument("project")
    add_transcriber(sp)
    sp.set_defaults(func=_cmd_plan)

    sp = sub.add_parser("render", help="export the project to MP4")
    sp.add_argument("project")
    sp.add_argument("-o", "--out", required=True)
    sp.add_argument("--crf", type=int, default=20)
    sp.add_argument("--no-captions", action="store_true")
    sp.add_argument("--audio-preset", default=None)
    sp.add_argument("--replan", action="store_true", help="re-run alignment+planning first")
    add_transcriber(sp)
    sp.set_defaults(func=_cmd_render)

    sp = sub.add_parser("auto", help="one-shot: align + plan + render")
    sp.add_argument("project")
    sp.add_argument("-o", "--out", required=True)
    sp.add_argument("--crf", type=int, default=20)
    sp.add_argument("--no-captions", action="store_true")
    sp.add_argument("--audio-preset", default=None)
    add_transcriber(sp)
    sp.set_defaults(func=_cmd_auto)

    sp = sub.add_parser("gui", help="launch the desktop app (needs the 'gui' extra)")
    sp.add_argument("argv", nargs="*", help="arguments passed to Qt")
    sp.set_defaults(func=_cmd_gui)

    sp = sub.add_parser(
        "gaze-analyze", help="report detected gaze offsets (research; 'gaze' extra)"
    )
    sp.add_argument("media")
    sp.add_argument("--max-frames", type=int, default=None)
    sp.set_defaults(func=_cmd_gaze_analyze)

    sp = sub.add_parser(
        "gaze-apply", help="write a gaze-corrected copy (research; 'gaze' extra)"
    )
    sp.add_argument("media")
    sp.add_argument("-o", "--out", required=True)
    sp.add_argument("--strength", type=float, default=0.8)
    sp.add_argument("--max-frames", type=int, default=None)
    sp.add_argument("--mode", choices=["frame", "patch"], default="frame")
    sp.set_defaults(func=_cmd_gaze_apply)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
