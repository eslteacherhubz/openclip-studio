"""Headless CLI: every engine capability is reachable without the GUI."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from openclip import __version__
from openclip.hardening import (
    handle_errors,
    validate_media,
)


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Check system dependencies and report status."""
    import platform

    from openclip.hardening import get_missing_dependencies

    print("OpenClip System Check")
    print("=" * 50)
    print(f"Python: {platform.python_version()} ({platform.python_implementation()})")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print()

    # Check ffmpeg/ffprobe
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    print("Core Dependencies:")
    print(f"  ffmpeg:  {'✅ ' + ffmpeg if ffmpeg else '❌ NOT FOUND'}")
    print(f"  ffprobe: {'✅ ' + ffprobe if ffprobe else '❌ NOT FOUND'}")
    print()

    # Check optional extras
    deps = get_missing_dependencies()
    print("Optional Extras:")
    for extra, status in deps.items():
        print(f"  {extra}: {status}")
    print()

    # Overall status
    all_ok = ffmpeg and ffprobe and all("✅" in s for s in deps.values())
    if all_ok:
        print("✅ All checks passed — ready to use!")
        return 0
    else:
        print("⚠ Some dependencies missing — see above for install commands")
        return 1


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
    from openclip.ai.platform_presets import get_preset
    from openclip.director.executor import render_edl
    from openclip.director.models import EditDecisionList
    from openclip.engine.ffmpeg import probe
    from openclip.engine.project import Project
    from openclip.engine.render import RenderOptions

    validate_media(args.media)
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
    beauty = None
    if args.beauty:
        from openclip.ai.beauty import BeautyConfig

        beauty = BeautyConfig(
            skin_smooth=args.beauty_smooth,
            jaw_slim=args.beauty_jaw,
            neck_tighten=args.beauty_neck,
        )

    # Handle platform preset(s)
    platforms = [p.strip() for p in args.platform.split(",")] if args.platform else [None]
    out_base = Path(args.out)
    results = []
    for platform_name in platforms:
        platform = get_preset(platform_name) if platform_name else None
        out_path = out_base
        if platform:
            out_path = out_base.with_name(f"{out_base.stem}_{platform.name}{out_base.suffix}")
        # Guard: refuse to overwrite the source recording
        if out_path.resolve() == Path(media.path).resolve():
            print(
                "Error: Output path is the same as the source recording "
                "— refusing to overwrite it.",
                file=sys.stderr,
            )
            return 1
        result = render_edl(
            project,
            media,
            edl,
            out_path,
            RenderOptions(
                burn_captions=False,
                transcript_captions=args.captions,
                caption_style=args.caption_style,
                smart_reframe=args.auto_reframe,
                beauty=beauty,
                platform=platform,
            ),
        )
        results.append((platform_name or "source", result))

    # Print summary
    if len(results) == 1:
        print(
            f"Exported: {results[0][1].out}\n"
            f"  provider={plan.provider}, decisions={len(decisions)} (all auto-accepted), "
            f"cuts={results[0][1].cuts_applied}, kept {results[0][1].kept_duration_s:.2f}s "
            f"of {media.duration_s:.2f}s"
        )
    else:
        lines = [f"Exported {len(results)} platform variant(s):"]
        for name, res in results:
            lines.append(f"  {name}: {res.out}")
            lines.append(
                f"    cuts={res.cuts_applied}, "
                f"kept {res.kept_duration_s:.2f}s of {media.duration_s:.2f}s"
            )
        print("\n".join(lines))
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

    # Guard: refuse to overwrite the source recording
    out_path = Path(args.out)
    if out_path.resolve() == Path(media.path).resolve():
        print(
            "Error: Output path is the same as the source recording "
            "— refusing to overwrite it.",
            file=sys.stderr,
        )
        return 1

    project = Project.create(source=media.path, script_text="")
    result = render_edl(project, media, edl, args.out, RenderOptions(burn_captions=False))
    print(
        f"Exported: {result.out}\n"
        f"  transcript cuts applied: {result.cuts_applied}, "
        f"kept {result.kept_duration_s:.2f}s of {media.duration_s:.2f}s"
    )
    return 0


def _cmd_ai_clips(args: argparse.Namespace) -> int:
    """Detect viral hook moments from a transcript → JSON clip proposals."""
    from openclip.ai.hook_detection import HookDetector
    from openclip.ai.transcript_editing import parse_transcript
    from openclip.engine.ffmpeg import probe

    raw = Path(args.transcript).read_text(encoding="utf-8")
    try:
        data: object = json.loads(raw)
    except ValueError:
        data = raw  # plain text (no timestamps)
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        transcript: list[dict[str, object]] = [
            {"text": str(item.get("text", "")), "start": item.get("start"), "end": item.get("end")}
            for item in data["segments"]
            if isinstance(item, dict)
        ]
    elif isinstance(data, list):
        transcript = [item for item in data if isinstance(item, dict)]
    else:
        segments = parse_transcript(data)
        transcript = [
            {"text": s.text, "start": s.start_s, "end": s.end_s} for s in segments
        ]

    media = probe(args.media)
    detector = HookDetector(transcript)
    hooks = detector.find_hooks(
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        top_n=args.top,
    )
    clips: list[dict[str, Any]] = []
    for hook in hooks:
        end = min(hook.end, media.duration_s) if media.duration_s > 0 else hook.end
        clips.append(
            {
                **hook.to_dict(),
                "end": round(end, 3),
                "duration": round(end - hook.start, 3),
            }
        )

    payload: dict[str, object] = {
        "media": str(media.path),
        "duration_s": media.duration_s,
        "top": args.top,
        "clips": clips,
    }

    if args.render:
        from openclip.engine.ffmpeg import run_ffmpeg

        render_dir = (
            Path(args.render_dir)
            if args.render_dir
            else (Path(args.out).with_suffix("") if args.out else Path("clips")) / "clips"
        )
        render_dir.mkdir(parents=True, exist_ok=True)
        rendered: list[str] = []
        for index, clip in enumerate(clips):
            start = float(clip["start"])
            end = float(clip["end"])
            hook_type = str(clip["hook_type"])
            out_path = render_dir / f"clip_{index + 1:02d}_{hook_type}.mp4"
            run_ffmpeg(
                [
                    "-y",
                    "-ss", f"{start:.3f}",
                    "-to", f"{end:.3f}",
                    "-i", str(media.path),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k",
                    str(out_path),
                ]
            )
            rendered.append(str(out_path))
            print(f"Rendered clip: {out_path}", file=sys.stderr)
        payload["rendered"] = rendered

    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _cmd_ai_broll(args: argparse.Namespace) -> int:
    """Detect B-roll opportunities from a transcript → JSON proposals with asset sourcing."""
    from openclip.ai.broll_detection import BrollDetector, broll_to_director_plan
    from openclip.ai.broll_sourcing import BrollSourceResolver
    from openclip.ai.transcript_editing import TranscriptSegment, parse_transcript
    from openclip.engine.ffmpeg import probe

    raw = Path(args.transcript).read_text(encoding="utf-8")
    try:
        data: object = json.loads(raw)
    except ValueError:
        data = raw  # plain text (no timestamps)
    segments = parse_transcript(data)
    blocks: list[TranscriptSegment] = [
        TranscriptSegment(text=s.text, start_s=s.start_s, end_s=s.end_s)
        for s in segments
        if s.start_s is not None and s.end_s is not None
    ]

    media = probe(args.media)
    detector = BrollDetector(blocks)

    # Get hook windows to exclude (Face-First rule)
    hook_windows: tuple[tuple[float, float], ...] = ()
    if args.exclude_hooks:
        hook_data = Path(args.exclude_hooks).read_text(encoding="utf-8")
        import json as _json
        hook_json = _json.loads(hook_data)
        if "clips" in hook_json:
            hook_windows = tuple((float(c["start"]), float(c["end"])) for c in hook_json["clips"])

    opportunities = detector.find_opportunities(exclude_hooks=hook_windows)

    # Resolve sources
    library_path = Path(args.library) if args.library else None
    resolver = BrollSourceResolver(library_path)
    sources = [resolver.resolve(opp, library_path=library_path) for opp in opportunities]

    # Build output payload
    broll_items: list[dict[str, object]] = []
    for index, (opp, src) in enumerate(zip(opportunities, sources, strict=True)):
        broll_items.append({
            **opp.to_dict(),
            "source": src.to_dict(),
            "id": f"broll-{index:03d}",
        })

    payload: dict[str, object] = {
        "media": str(media.path),
        "duration_s": media.duration_s,
        "broll": broll_items,
    }

    # Generate AI Director plan if requested
    if args.plan:
        plan = broll_to_director_plan(opportunities, sources, media.duration_s)
        payload["plan"] = plan.to_dict()

    # Render if requested
    if args.render:
        from openclip.ai import ai_plan_to_director_decisions
        from openclip.director.executor import render_edl
        from openclip.director.models import EditDecisionList
        from openclip.engine.render import RenderOptions

        plan = broll_to_director_plan(opportunities, sources, media.duration_s)
        decisions = ai_plan_to_director_decisions(plan)
        for decision in decisions:
            decision.status = "accepted"
        edl = EditDecisionList(project_source=str(media.path), decisions=decisions)

        from openclip.engine.project import Project
        project = Project.create(source=media.path, script_text="")

        out_path = Path(args.out) if args.out else Path("broll_output.mp4")
        result = render_edl(
            project,
            media,
            edl,
            out_path,
            RenderOptions(
                burn_captions=False,
                transcript_captions=args.captions,
                caption_style=args.caption_style,
            ),
        )
        payload["rendered"] = str(result.out)
        print(f"Rendered B-roll output: {result.out}", file=sys.stderr)

    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
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
    sp.add_argument(
        "--caption-style",
        default=None,
        choices=["Classic", "Social", "Cinema", "Neon"],
        help="caption style template (with --captions)",
    )
    sp.add_argument(
        "--auto-reframe",
        action="store_true",
        help="with --auto-accept: smart 9:16 subject-tracking crop (mock tracking data)",
    )
    sp.add_argument(
        "--beauty",
        action="store_true",
        help="with --auto-accept: apply the beauty/retouching pre-pass to the source",
    )
    sp.add_argument(
        "--beauty-smooth",
        type=float,
        default=0.15,
        help="skin smoothness intensity [0,1] (default: %(default)s)",
    )
    sp.add_argument(
        "--beauty-jaw",
        type=float,
        default=0.10,
        help="jaw slim intensity [0,1] (default: %(default)s)",
    )
    sp.add_argument(
        "--beauty-neck",
        type=float,
        default=0.10,
        help="neck tighten intensity [0,1] (default: %(default)s)",
    )
    sp.add_argument(
        "--platform",
        default=None,
        help=(
            "export platform preset (tiktok, reels, shorts, youtube, "
            "instagram, linkedin, twitter). "
            "Comma-separated for multi-export (e.g., 'tiktok,youtube'). "
            "Sets dimensions, aspect ratio, duration limits, and enables "
            "smart reframe for vertical."
        ),
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

    sp = sub.add_parser(
        "ai-clips",
        help="detect viral hook moments from a transcript → JSON clip proposals",
    )
    sp.add_argument("--media", required=True, help="input video file")
    sp.add_argument(
        "--transcript", required=True, help="timestamped transcript JSON (segments) or text"
    )
    sp.add_argument("--top", type=int, default=5, help="number of clips to propose (default: 5)")
    sp.add_argument("-o", "--out", default=None, help="write the JSON proposals to this file")
    sp.add_argument("--min-duration", type=float, default=15.0, help="min clip length in s")
    sp.add_argument("--max-duration", type=float, default=90.0, help="max clip length in s")
    sp.add_argument(
        "--render", action="store_true", help="render each proposed clip to its own MP4"
    )
    sp.add_argument(
        "--render-dir", default=None, help="directory for rendered clips (with --render)"
    )
    sp.set_defaults(func=_cmd_ai_clips)

    sp = sub.add_parser(
        "ai-broll",
        help="detect B-roll opportunities from a transcript → JSON proposals with asset sourcing",
    )
    sp.add_argument("--media", required=True, help="input video file")
    sp.add_argument(
        "--transcript", required=True, help="timestamped transcript JSON (segments) or text"
    )
    sp.add_argument("--library", default=None, help="path to broll_library.json manifest")
    sp.add_argument(
        "--exclude-hooks",
        default=None,
        help="path to ai-clips JSON output to exclude hook windows (Face-First)",
    )
    sp.add_argument("-o", "--out", default=None, help="write the JSON proposals to this file")
    sp.add_argument(
        "--plan", action="store_true", help="include AI Director plan in output"
    )
    sp.add_argument(
        "--render", action="store_true", help="render the video with accepted B-roll overlays"
    )
    sp.add_argument(
        "--captions", action="store_true", help="with --render: burn transcript captions"
    )
    sp.add_argument(
        "--caption-style",
        default=None,
        choices=["Classic", "Social", "Cinema", "Neon"],
        help="caption style template (with --captions)",
    )
    sp.set_defaults(func=_cmd_ai_broll)

    sp = sub.add_parser("gui", help="launch the desktop app (needs the 'gui' extra)")
    sp.add_argument("argv", nargs="*", help="arguments passed to Qt")
    sp.set_defaults(func=_cmd_gui)

    sp = sub.add_parser(
        "gaze-analyze", help="report detected gaze offsets (research; 'gaze' extra)"
    )
    sp.add_argument("media")
    sp.add_argument("--max-frames", type=int, default=None)
    sp.set_defaults(func=_cmd_gaze_analyze)

    sp = sub.add_parser("gaze-apply", help="write a gaze-corrected copy (research; 'gaze' extra)")
    sp.add_argument("media")
    sp.add_argument("-o", "--out", required=True)
    sp.add_argument("--strength", type=float, default=0.8)
    sp.add_argument("--max-frames", type=int, default=None)
    sp.add_argument("--mode", choices=["frame", "patch"], default="frame")
    sp.set_defaults(func=_cmd_gaze_apply)

    sp = sub.add_parser("doctor", help="check system dependencies and report status")
    sp.set_defaults(func=_cmd_doctor)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


@handle_errors
def _main_entry(argv: list[str] | None = None) -> int:
    import logging

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
        handlers=[logging.NullHandler()],
    )
    return main(argv)


if __name__ == "__main__":
    sys.exit(_main_entry())
