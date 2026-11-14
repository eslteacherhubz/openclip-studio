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


def _cmd_ai_edit(args: argparse.Namespace) -> int:
    from openclip.ai import (
        ai_plan_to_director_decisions,
        create_provider,
        generate_plan,
        load_provider_config,
        plan_to_review_items,
    )
    from openclip.director.executor import render_edl
    from openclip.director.models import EditDecisionList
    from openclip.engine.ffmpeg import probe
    from openclip.engine.project import Project
    from openclip.engine.render import RenderOptions

    media = probe(args.media)
    if args.transcript:
        transcript = Path(args.transcript).read_text(encoding="utf-8")
    else:
        transcript = f"(no transcript provided; source duration {media.duration_s:.1f}s)"
    config_path = Path(args.config) if args.config else None
    provider = create_provider(load_provider_config(args.provider, config_path=config_path))
    plan = generate_plan(provider, args.prompt, transcript, media.duration_s)
    items = plan_to_review_items(plan)
    decisions = ai_plan_to_director_decisions(plan)

    if not args.auto_accept:
        print(
            json.dumps(
                {
                    "provider": plan.provider,
                    "prompt": plan.prompt,
                    "plan": plan.to_dict(),
                    "review": [item.to_dict() for item in items],
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        print("Review required: re-run with --auto-accept to render.", file=sys.stderr)
        return 0

    for decision in decisions:
        decision.status = "accepted"
    edl = EditDecisionList(project_source=str(media.path), decisions=decisions)
    project = Project.create(source=media.path, script_text="")
    result = render_edl(
        project,
        media,
        edl,
        args.out,
        RenderOptions(burn_captions=False, transcript_captions=args.captions),
    )
    print(
        f"Exported: {result.out}\n"
        f"  provider={plan.provider}, decisions={len(decisions)} (all auto-accepted), "
        f"cuts={result.cuts_applied}, kept {result.kept_duration_s:.2f}s "
        f"of {media.duration_s:.2f}s"
    )
    return 0


def _cmd_ai_edit_transcript(args: argparse.Namespace) -> int:
    """Transcript-based editing: diff edited transcript vs original → cut EDL → render."""
    import json as _json

    from openclip.ai.transcript_editing import decisions_to_json, transcript_to_edits
    from openclip.director.executor import kept_intervals_from_spans, render_edl
    from openclip.director.models import EditDecisionList
    from openclip.engine.ffmpeg import probe
    from openclip.engine.project import Project
    from openclip.engine.render import RenderOptions, normalize_cut_spans

    media = probe(args.media)
    try:
        original_raw = Path(args.original).read_text(encoding="utf-8")
        original = _json.loads(original_raw)
    except (OSError, ValueError):
        original = original_raw  # plain text (no timestamps)
    edited_raw = Path(args.transcript).read_text(encoding="utf-8")
    try:
        edited: object = _json.loads(edited_raw)
    except ValueError:
        edited = edited_raw  # plain text

    decisions = transcript_to_edits(original, edited)
    if not decisions:
        print("No deleted transcript blocks detected; nothing to cut.")
        return 0
    for decision in decisions:
        decision.status = "accepted"  # the user's edit IS the review decision
    edl = EditDecisionList(project_source=str(media.path), decisions=decisions)

    spans = [d.source_span for d in decisions if d.source_span is not None]
    cut_spans = normalize_cut_spans(spans)
    kept = kept_intervals_from_spans(cut_spans, media.duration_s)
    if not kept:
        print(
            "Error: the edited transcript deletes every timestamped block — "
            "nothing would remain to render. Aborting without rendering.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        kept_total = sum(e - s for s, e in kept)
        print(decisions_to_json(decisions))
        print(f"Would keep {len(kept)} interval(s), {kept_total:.2f}s", file=sys.stderr)
        return 0

    project = Project.create(source=media.path, script_text="")
    result = render_edl(project, media, edl, args.out, RenderOptions(burn_captions=False))
    print(
        f"Exported: {result.out}\n"
        f"  transcript cuts applied: {result.cuts_applied}, "
        f"kept {result.kept_duration_s:.2f}s of {media.duration_s:.2f}s"
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

    sp = sub.add_parser(
        "ai-edit",
        help="AI Director: prompt → plan → review JSON (or render with --auto-accept)",
    )
    sp.add_argument("--media", required=True, help="input video file")
    sp.add_argument("--prompt", required=True, help="natural-language editing request")
    sp.add_argument("--transcript", default=None, help="optional transcript text file")
    sp.add_argument("-o", "--out", required=True, help="output .mp4 path")
    sp.add_argument("--provider", default="local_mock", help="AI provider (default: local_mock)")
    sp.add_argument(
        "--config", default=None, help="optional local AI config JSON (never contains the API key)"
    )
    sp.add_argument(
        "--auto-accept",
        action="store_true",
        help="accept all AI proposals and render; without it, prints review JSON only",
    )
    sp.add_argument(
        "--captions",
        action="store_true",
        help="with --auto-accept: burn the plan's bilingual captions (inline drawtext)",
    )
    sp.set_defaults(func=_cmd_ai_edit)

    sp = sub.add_parser(
        "ai-edit-transcript",
        help="edit video by editing its transcript: diff → cut EDL → render",
    )
    sp.add_argument("--media", "--input", dest="media", required=True, help="input video file")
    sp.add_argument(
        "--transcript", required=True, help="edited transcript (JSON with segments, or plain text)"
    )
    sp.add_argument(
        "--original", required=True, help="original timestamped transcript (JSON with segments)"
    )
    sp.add_argument("-o", "--out", required=True, help="output .mp4 path")
    sp.add_argument(
        "--dry-run", action="store_true", help="print the generated EDL JSON without rendering"
    )
    sp.set_defaults(func=_cmd_ai_edit_transcript)

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
