"""Tests for AI render primitives: zoom filter strings, audio chains, executor wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from openclip.ai import (
    STYLE_TEMPLATES,
    CaptionBlock,
    CaptionStyle,
    audio_chain_for_enhancement,
    build_broll_filter,
    build_caption_filter,
    build_export_config_args,
    build_export_vf,
    build_zoom_vf,
    get_caption_style,
    hex_to_ffmpeg_color,
    to_ffmpeg_color,
)
from openclip.director.executor import render_edl
from openclip.director.models import Decision, EditDecisionList
from openclip.engine.ffmpeg import MediaInfo, probe, run_ffmpeg
from openclip.engine.project import Project
from openclip.engine.render import RenderResult
from openclip.engine.synth import generate_lesson


@pytest.fixture(scope="module")
def lesson(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return generate_lesson(tmp_path_factory.mktemp("ai-render") / "lesson.mp4").path


@pytest.fixture(scope="module")
def broll_asset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("ai-assets") / "broll.png"
    run_ffmpeg(
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=320x180:duration=1",
            "-frames:v",
            "1",
            str(out),
        ],
        timeout=60,
    )
    return out


class TestBuildZoomVf:
    def test_span_converted_to_frame_range(self) -> None:
        vf = build_zoom_vf((2.0, 3.0), 320, 180, 0.5, fps=15.0)
        assert vf.startswith("zoompan=")
        assert "between(on,30,45)" in vf  # 2s–3s at 15 fps
        assert "s=320x180" in vf  # output resolution unchanged

    def test_strength_controls_zoom_factor(self) -> None:
        # strength 0.5 → zoom factor 2x punch-in
        assert ",2.0000,1)" in build_zoom_vf((0.0, 1.0), 320, 180, 0.5)

    def test_default_strength(self) -> None:
        # default 0.2 → zoom factor 1.25x
        assert ",1.2500,1)" in build_zoom_vf((0.0, 1.0), 320, 180, None)

    def test_strength_clamped(self) -> None:
        # strength 5.0 clamped to 0.9 → zoom factor 10x
        assert ",10.0000,1)" in build_zoom_vf((0.0, 1.0), 320, 180, 5.0)

    def test_centered_pan(self) -> None:
        vf = build_zoom_vf((0.0, 1.0), 320, 180, 0.3)
        assert "x='iw/2-(iw/zoom)/2'" in vf
        assert "y='ih/2-(ih/zoom)/2'" in vf

    def test_deterministic(self) -> None:
        assert build_zoom_vf((1.0, 2.0), 320, 180, 0.4) == build_zoom_vf((1.0, 2.0), 320, 180, 0.4)


class TestAudioChainForEnhancement:
    def test_denoise_and_normalize(self) -> None:
        chain = audio_chain_for_enhancement(True, True, True)
        assert chain is not None
        assert "afftdn" in chain
        assert "loudnorm" in chain

    def test_denoise_only(self) -> None:
        chain = audio_chain_for_enhancement(True, True, None)
        assert chain is not None and "afftdn" in chain and "loudnorm" not in chain

    def test_normalize_only(self) -> None:
        chain = audio_chain_for_enhancement(True, None, True)
        assert chain is not None and "loudnorm" in chain and "afftdn" not in chain

    def test_disabled_returns_none(self) -> None:
        assert audio_chain_for_enhancement(False, True, True) is None

    def test_no_flags_returns_none(self) -> None:
        assert audio_chain_for_enhancement(True, None, None) is None


def _project_media(tmp_path: Path) -> tuple[Project, MediaInfo]:
    project = Project.create(source=tmp_path / "in.mp4", script_text="Hello :: Hola\n")
    media = MediaInfo(
        path=tmp_path / "in.mp4",
        duration_s=14.0,
        has_video=True,
        has_audio=True,
        width=320,
        height=180,
    )
    return project, media


class TestExecutorWiring:
    def test_zoom_and_audio_passed_to_render(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from openclip.director import executor

        captured: dict[str, object] = {}

        def _fake(**kwargs: object) -> RenderResult:
            captured.update(kwargs)
            return RenderResult(
                out=Path(str(kwargs["out_path"])),
                kept_duration_s=12.0,
                segments=1,
                cuts_applied=1,
                captions_burned=False,
                audio_enhanced=True,
            )

        monkeypatch.setattr(executor, "render_export", _fake)
        project, media = _project_media(tmp_path)
        decisions = [
            Decision(
                id="c1",
                kind="cut",
                source_span=(2.0, 3.0),
                rationale="silence",
                status="accepted",
            ),
            Decision(
                id="z1",
                kind="zoom",
                source_span=(4.0, 6.0),
                params={"strength": 0.5},
                rationale="punch in",
                status="accepted",
            ),
            Decision(
                id="a1",
                kind="audio_enhance",
                params={"enabled": True, "denoise": True, "loudness_normalize": True},
                rationale="clean audio",
                status="accepted",
            ),
        ]
        edl = EditDecisionList(project_source="in.mp4", decisions=decisions)
        render_edl(project, media, edl, tmp_path / "out.mp4")

        filters = captured["video_filters"]
        assert isinstance(filters, list) and len(filters) == 1
        zoom_vf = str(filters[0])
        # source span 4–6s with cut 2–3s before it → export timeline 3–5s;
        # MediaInfo.fps is 0 → default 25 fps → frames 75–125
        assert "between(on,75,125)" in zoom_vf
        assert "s=320x180" in zoom_vf
        assert ",2.0000,1)" in zoom_vf  # strength 0.5 → 2x
        chain = str(captured["audio_chain"])
        assert "afftdn" in chain and "loudnorm" in chain

    def test_zoom_fully_inside_cut_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from openclip.director import executor

        captured: dict[str, object] = {}

        def _fake(**kwargs: object) -> RenderResult:
            captured.update(kwargs)
            return RenderResult(
                out=Path(str(kwargs["out_path"])),
                kept_duration_s=12.0,
                segments=1,
                cuts_applied=1,
                captions_burned=False,
                audio_enhanced=False,
            )

        monkeypatch.setattr(executor, "render_export", _fake)
        project, media = _project_media(tmp_path)
        decisions = [
            Decision(
                id="c1",
                kind="cut",
                source_span=(2.0, 6.0),
                rationale="silence",
                status="accepted",
            ),
            Decision(
                id="z1",
                kind="zoom",
                source_span=(3.0, 4.0),
                rationale="inside cut",
                status="accepted",
            ),
        ]
        edl = EditDecisionList(project_source="in.mp4", decisions=decisions)
        render_edl(project, media, edl, tmp_path / "out.mp4")
        assert captured["video_filters"] is None

    def test_broll_and_export_config_passed_to_render(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from openclip.director import executor

        captured: dict[str, object] = {}

        def _fake(**kwargs: object) -> RenderResult:
            captured.update(kwargs)
            return RenderResult(
                out=Path(str(kwargs["out_path"])),
                kept_duration_s=12.0,
                segments=1,
                cuts_applied=1,
                captions_burned=False,
                audio_enhanced=False,
            )

        monkeypatch.setattr(executor, "render_export", _fake)
        asset = tmp_path / "b.png"
        asset.write_bytes(b"fake")  # exists() is all the executor checks
        project, media = _project_media(tmp_path)
        decisions = [
            Decision(
                id="c1",
                kind="cut",
                source_span=(2.0, 3.0),
                rationale="silence",
                status="accepted",
            ),
            Decision(
                id="b1",
                kind="broll",
                source_span=(4.0, 6.0),
                params={"asset": str(asset)},
                rationale="b-roll",
                status="accepted",
            ),
            Decision(
                id="e1",
                kind="export_config",
                params={"width": 640, "height": 360, "fps": 30.0, "vertical": False},
                rationale="hd",
                status="accepted",
            ),
        ]
        edl = EditDecisionList(project_source="in.mp4", decisions=decisions)
        render_edl(project, media, edl, tmp_path / "out.mp4")

        overlays = captured["broll"]
        assert isinstance(overlays, list) and len(overlays) == 1
        overlay = overlays[0]
        # source span 4–6s with cut 2–3s → export timeline 3–5s
        assert "between(t,3.000,5.000)" in overlay.segment
        # broll pads to the export_config TARGET resolution (640x360), not source
        assert "scale=640:360" in overlay.segment
        assert "pad=640:360" in overlay.segment
        filters = captured["video_filters"]
        assert isinstance(filters, list) and filters[-1] == "scale=640:360"
        assert captured["out_fps"] == 30.0

    def test_clip_reshapes_timeline(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from openclip.director import executor

        captured: dict[str, object] = {}

        def _fake(**kwargs: object) -> RenderResult:
            captured.update(kwargs)
            return RenderResult(
                out=Path(str(kwargs["out_path"])),
                kept_duration_s=4.0,
                segments=1,
                cuts_applied=2,
                captions_burned=False,
                audio_enhanced=False,
            )

        monkeypatch.setattr(executor, "render_export", _fake)
        project, media = _project_media(tmp_path)
        decisions = [
            Decision(
                id="keep1",
                kind="clip",
                source_span=(2.0, 6.0),
                rationale="keep this",
                status="accepted",
            ),
            Decision(
                id="c1",
                kind="cut",
                source_span=(3.0, 4.0),
                rationale="filler inside clip",
                status="accepted",
            ),
            Decision(
                id="z1",
                kind="zoom",
                source_span=(5.0, 6.0),
                params={"strength": 0.5},
                rationale="zoom in clip",
                status="accepted",
            ),
            Decision(
                id="z2",
                kind="zoom",
                source_span=(10.0, 12.0),
                params={"strength": 0.5},
                rationale="zoom outside clip",
                status="accepted",
            ),
        ]
        edl = EditDecisionList(project_source="in.mp4", decisions=decisions)
        render_edl(project, media, edl, tmp_path / "out.mp4")

        # kept = clip span minus inner cut; everything outside the clip removed
        assert captured["kept"] == [(2.0, 3.0), (4.0, 6.0)]
        assert captured["cut_spans"] == [(0.0, 2.0), (3.0, 4.0), (6.0, 14.0)]
        # zoom inside the clip: source 5–6s, removed before it = (0,2)+(3,4) = 3s
        # → export span 2–3s → frames 50–75 at default 25 fps;
        # zoom outside the clip is dropped entirely
        filters = captured["video_filters"]
        assert isinstance(filters, list) and len(filters) == 1
        assert "between(on,50,75)" in str(filters[0])

    def test_unimplemented_ai_kinds_still_skipped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from openclip.director import executor

        captured: dict[str, object] = {}

        def _fake(**kwargs: object) -> RenderResult:
            captured.update(kwargs)
            return RenderResult(
                out=Path(str(kwargs["out_path"])),
                kept_duration_s=14.0,
                segments=1,
                cuts_applied=0,
                captions_burned=False,
                audio_enhanced=False,
            )

        monkeypatch.setattr(executor, "render_export", _fake)
        project, media = _project_media(tmp_path)
        decisions = [
            Decision(
                id="b1",
                kind="broll",
                source_span=(1.0, 2.0),
                rationale="b-roll",
                status="accepted",
            ),
            Decision(id="e1", kind="export_config", rationale="mp4", status="accepted"),
        ]
        edl = EditDecisionList(project_source="in.mp4", decisions=decisions)
        render_edl(project, media, edl, tmp_path / "out.mp4")  # must not raise
        assert captured["video_filters"] is None


class TestBuildBrollFilter:
    def _decision(self, **params: object) -> Decision:
        base: dict[str, object] = {"asset": "assets/x.png"}
        base.update(params)
        return Decision(
            id="b1", kind="broll", source_span=(2.0, 4.0), params=base, rationale="b-roll"
        )

    def test_exact_filter_string(self) -> None:
        segment, out_label = build_broll_filter(self._decision(), (320, 180))
        assert segment == (
            "[1:v]scale=320:180:force_original_aspect_ratio=decrease,"
            "pad=320:180:(ow-iw)/2:(oh-ih)/2[bo1];"
            "[base0][bo1]overlay=0:0:enable='between(t,2.000,4.000)'[bm1]"
        )
        assert out_label == "bm1"

    def test_custom_input_index_and_prev_label(self) -> None:
        segment, out_label = build_broll_filter(
            self._decision(), (1920, 1080), input_index=3, prev_label="bm1"
        )
        assert "[3:v]scale=1920:1080" in segment
        assert "[bm1][bo3]overlay=0:0" in segment
        assert out_label == "bm3"

    def test_missing_asset_raises(self) -> None:
        with pytest.raises(ValueError, match="asset"):
            build_broll_filter(self._decision(asset=""), (320, 180))

    def test_missing_span_raises(self) -> None:
        d = Decision(id="b2", kind="broll", params={"asset": "a.png"}, rationale="x")
        with pytest.raises(ValueError, match="span"):
            build_broll_filter(d, (320, 180))

    def test_deterministic(self) -> None:
        assert build_broll_filter(self._decision(), (320, 180)) == build_broll_filter(
            self._decision(), (320, 180)
        )


class TestBuildExportConfig:
    def _decision(self, **params: object) -> Decision:
        return Decision(id="e1", kind="export_config", params=params, rationale="export")

    def test_landscape_scale(self) -> None:
        cfg = build_export_config_args(
            self._decision(format="mp4", width=1920, height=1080, fps=30.0, vertical=False)
        )
        assert cfg.vf == "scale=1920:1080"
        assert cfg.fps == 30.0
        assert cfg.vertical is False

    def test_vertical_center_crop(self) -> None:
        cfg = build_export_config_args(self._decision(width=1080, height=1920, vertical=True))
        assert cfg.vf is not None
        assert cfg.vf.startswith("crop=w=trunc(ih*0.562500/2)*2:h=ih:x=(iw-ow)/2:y=0,")
        assert cfg.vf.endswith(",scale=1080:1920")
        assert cfg.fps is None

    def test_missing_dimensions_no_vf(self) -> None:
        cfg = build_export_config_args(self._decision(format="mp4", vertical=False))
        assert cfg.vf is None
        assert cfg.fps is None

    def test_invalid_values_ignored(self) -> None:
        cfg = build_export_config_args(self._decision(width=-5, height="x", fps=0, vertical="yes"))
        assert cfg.vf is None and cfg.fps is None and cfg.vertical is False

    def test_build_export_vf_directly(self) -> None:
        assert build_export_vf(None, None, False) is None
        assert build_export_vf(640, 360, False) == "scale=640:360"


class TestBuildCaptionFilter:
    def test_single_block_exact_string(self) -> None:
        blocks = [CaptionBlock(text="Hello", start_s=1.0, end_s=3.0)]
        vf = build_caption_filter(blocks, (320, 180))
        assert vf is not None
        fs = max(12, int(180 * 0.06))  # 12
        margin = max(8, int(180 * 0.05))  # 9
        assert vf == (
            f"drawtext=font='Sans':text='Hello':fontsize={fs}:fontcolor=white"
            f":borderw=2:bordercolor=black:x=(w-text_w)/2"
            f":y={180 - fs - margin}:enable='between(t,1.000,3.000)'"
        )

    def test_bilingual_stacked(self) -> None:
        blocks = [CaptionBlock(text="Hello", translation="Hola", start_s=0.0, end_s=2.0)]
        vf = build_caption_filter(blocks, (320, 180))
        assert vf is not None
        assert vf.count("drawtext=") == 2
        assert "text='Hello'" in vf and "text='Hola'" in vf
        fs, margin = 12, 9
        assert f":y={180 - 2 * fs - margin}" in vf  # original on top
        assert f":y={180 - fs - margin}" in vf  # translation below
        assert "fontcolor=white" in vf and "bordercolor=black" in vf

    def test_escaping(self) -> None:
        blocks = [CaptionBlock(text="it's 100%: a,b;c[d]", start_s=0.0, end_s=1.0)]
        vf = build_caption_filter(blocks, (320, 180))
        assert vf is not None
        assert "it\\'s 100\\%\\: a\\,b\\;c\\[d\\]" in vf

    def test_newlines_flattened(self) -> None:
        block = CaptionBlock(text="line1\nline2", start_s=0.0, end_s=1.0)
        vf = build_caption_filter([block], (320, 180))
        assert vf is not None and "line1 line2" in vf

    def test_invalid_blocks_skipped_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        blocks: list[dict[str, object]] = [
            {"text": "", "start": 0.0, "end": 1.0},  # empty text
            {"text": "ok", "start": 5.0, "end": 1.0},  # end <= start
            {"text": "good", "start": 0.0, "end": 1.0},
        ]
        with caplog.at_level("WARNING"):
            vf = build_caption_filter(blocks, (320, 180))
        assert vf is not None and vf.count("drawtext=") == 1 and "good" in vf
        assert len(caplog.records) == 2

    def test_empty_returns_none(self) -> None:
        assert build_caption_filter([], (320, 180)) is None

    def test_deterministic_and_resolution_dependent(self) -> None:
        blocks = [CaptionBlock(text="Hi", start_s=0.0, end_s=1.0)]
        assert build_caption_filter(blocks, (320, 180)) == build_caption_filter(blocks, (320, 180))
        vf_1080 = build_caption_filter(blocks, (1920, 1080))
        assert vf_1080 is not None and "fontsize=64" in vf_1080  # 1080*0.06


class TestCaptionStyle:
    def test_hex_to_ffmpeg_color(self) -> None:
        assert hex_to_ffmpeg_color("#FFFFFF") == "0xFFFFFF"
        assert hex_to_ffmpeg_color("ffd400") == "0xFFD400"
        assert hex_to_ffmpeg_color("#abc") == "0xAABBCC"
        with pytest.raises(ValueError):
            hex_to_ffmpeg_color("red")
        with pytest.raises(ValueError):
            hex_to_ffmpeg_color("#12345")

    def test_to_ffmpeg_color_passthrough_names(self) -> None:
        assert to_ffmpeg_color("white") == "white"
        assert to_ffmpeg_color("#39FF14") == "0x39FF14"
        with pytest.raises(ValueError):
            to_ffmpeg_color("bad color;drop")

    def test_default_style_byte_identical(self) -> None:
        # Classic/default must reproduce the historical filter exactly
        blocks = [CaptionBlock(text="Hello", start_s=1.0, end_s=3.0)]
        vf = build_caption_filter(blocks, (320, 180))
        assert vf == build_caption_filter(blocks, (320, 180), CaptionStyle())
        assert vf == build_caption_filter(blocks, (320, 180), get_caption_style(None))
        assert vf == build_caption_filter(blocks, (320, 180), get_caption_style("Classic"))
        assert "fontcolor=white" in vf and "bordercolor=black" in vf

    def test_templates_resolve(self) -> None:
        assert set(STYLE_TEMPLATES) == {"Classic", "Social", "Cinema", "Neon"}
        for name, style in STYLE_TEMPLATES.items():
            assert get_caption_style(name) == style
        assert get_caption_style("nope") == STYLE_TEMPLATES["Classic"]

    def test_style_changes_filter_string(self) -> None:
        blocks = [CaptionBlock(text="Hi", start_s=0.0, end_s=1.0)]
        social = get_caption_style("Social")
        vf = build_caption_filter(blocks, (320, 180), social)
        assert vf is not None
        assert "fontcolor=0xFFD400" in vf and "borderw=4" in vf
        assert "shadowcolor=black@0.6" in vf
        center = get_caption_style("Neon")
        vf2 = build_caption_filter(blocks, (320, 180), center)
        assert vf2 is not None and "font='Monospace'" in vf2 and "y=(h-" in vf2

    def test_invalid_style_rejected(self) -> None:
        with pytest.raises(ValueError):
            CaptionStyle(primary_color="not a color;")
        with pytest.raises(ValueError):
            CaptionStyle(position="middle")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            CaptionStyle(outline_width=-1)


class TestRealRender:
    def _edl(self, zoom_strength: float) -> EditDecisionList:
        decisions = [
            Decision(
                id="c1",
                kind="cut",
                source_span=(2.0, 3.0),
                rationale="silence",
                status="accepted",
            ),
            Decision(
                id="z1",
                kind="zoom",
                source_span=(4.0, 8.0),
                params={"strength": zoom_strength},
                rationale="punch in on key point",
                status="accepted",
            ),
            Decision(
                id="a1",
                kind="audio_enhance",
                params={"enabled": True, "denoise": True, "loudness_normalize": True},
                rationale="clean audio",
                status="accepted",
            ),
        ]
        return EditDecisionList(project_source="lesson.mp4", decisions=decisions)

    def _frame(self, video: Path, t: float, out: Path) -> bytes:
        run_ffmpeg(
            ["-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", str(out)],
            timeout=120,
        )
        return out.read_bytes()

    def test_zoom_changes_pixels_and_cut_changes_duration(
        self, lesson: Path, tmp_path: Path
    ) -> None:
        project = Project.create(source=lesson, script_text="")
        media = probe(lesson)

        out_zoom = tmp_path / "zoom.mp4"
        result = render_edl(project, media, self._edl(0.6), out_zoom)
        assert result.out == out_zoom
        info = probe(out_zoom)
        assert (info.width, info.height) == (320, 180)  # zoom restores resolution
        assert abs(info.duration_s - 13.0) < 0.5  # 14s lesson minus 1s cut

        # Control render: identical pipeline but zoom strength 0 (zoom factor 1),
        # so both outputs are re-encoded the same way and frames are comparable.
        out_flat = tmp_path / "flat.mp4"
        render_edl(project, media, self._edl(0.0), out_flat)

        # export-timeline t=3.5s is inside the mapped zoom span (source 4–8s,
        # shifted by the 2–3s cut → 3–7s); the punch-in must change the frame.
        frame_zoom = self._frame(out_zoom, 3.5, tmp_path / "f_zoom.png")
        frame_flat = self._frame(out_flat, 3.5, tmp_path / "f_flat.png")
        assert frame_zoom != frame_flat
        # outside the zoom span the frames must be identical (zoom factor 1)
        frame_zoom_early = self._frame(out_zoom, 1.0, tmp_path / "f_zoom_early.png")
        frame_flat_early = self._frame(out_flat, 1.0, tmp_path / "f_flat_early.png")
        assert frame_zoom_early == frame_flat_early

    def _edl_with(self, *decisions: Decision) -> EditDecisionList:
        return EditDecisionList(project_source="lesson.mp4", decisions=list(decisions))

    def test_broll_overlay_renders_and_missing_asset_is_skipped(
        self, lesson: Path, broll_asset: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        project = Project.create(source=lesson, script_text="")
        media = probe(lesson)
        good = Decision(
            id="b1",
            kind="broll",
            source_span=(2.0, 6.0),
            params={"asset": str(broll_asset)},
            rationale="red board",
            status="accepted",
        )
        missing = Decision(
            id="b2",
            kind="broll",
            source_span=(7.0, 9.0),
            params={"asset": str(tmp_path / "nope.png")},
            rationale="gone",
            status="accepted",
        )
        out = tmp_path / "broll.mp4"
        with caplog.at_level("WARNING"):
            result = render_edl(project, media, self._edl_with(good, missing), out)
        assert result.out == out
        assert any("nope.png" in rec.message for rec in caplog.records)
        # frame inside the broll span is (mostly) the red asset; outside it is not
        inside = self._frame(out, 3.0, tmp_path / "br_in.png")
        outside = self._frame(out, 12.0, tmp_path / "br_out.png")
        assert inside != outside

    def test_export_config_rescales_and_sets_fps(self, lesson: Path, tmp_path: Path) -> None:
        project = Project.create(source=lesson, script_text="")
        media = probe(lesson)
        decision = Decision(
            id="e1",
            kind="export_config",
            params={"format": "mp4", "width": 160, "height": 90, "fps": 30.0},
            rationale="small preview",
            status="accepted",
        )
        out = tmp_path / "small.mp4"
        render_edl(project, media, self._edl_with(decision), out)
        info = probe(out)
        assert (info.width, info.height) == (160, 90)
        assert abs(info.fps - 30.0) < 0.5

    def test_clip_only_renders_clip_span(self, lesson: Path, tmp_path: Path) -> None:
        project = Project.create(source=lesson, script_text="")
        media = probe(lesson)
        decision = Decision(
            id="keep1",
            kind="clip",
            source_span=(4.0, 8.0),
            rationale="keep only the key example",
            status="accepted",
        )
        out = tmp_path / "clip.mp4"
        result = render_edl(project, media, self._edl_with(decision), out)
        info = probe(out)
        assert abs(info.duration_s - 4.0) < 0.5  # only the 4–8s span survives
        assert (info.width, info.height) == (320, 180)
        assert result.cuts_applied == 2  # [0,4) and [8,14] removed

    def test_executor_builds_caption_filter_from_kept_captions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from openclip.director import executor
        from openclip.engine.render import RenderOptions

        captured: dict[str, object] = {}

        def _fake(**kwargs: object) -> RenderResult:
            captured.update(kwargs)
            return RenderResult(
                out=Path(str(kwargs["out_path"])),
                kept_duration_s=12.0,
                segments=1,
                cuts_applied=1,
                captions_burned=True,
                audio_enhanced=False,
            )

        monkeypatch.setattr(executor, "render_export", _fake)
        project, media = _project_media(tmp_path)
        decisions = [
            Decision(
                id="c1",
                kind="cut",
                source_span=(2.0, 3.0),
                rationale="silence",
                status="accepted",
            ),
            Decision(
                id="cap1",
                kind="caption",
                source_span=(4.0, 6.0),
                params={"primary": "Hello", "support": "Hola", "origin": "ai"},
                rationale="Hello",
                status="accepted",
            ),
            Decision(
                id="cap2",
                kind="caption",
                source_span=(2.0, 3.0),  # inside the cut
                params={"primary": "Removed", "origin": "ai"},
                rationale="Removed",
                status="accepted",
            ),
        ]
        edl = EditDecisionList(project_source="in.mp4", decisions=decisions)
        render_edl(
            project,
            media,
            edl,
            tmp_path / "out.mp4",
            RenderOptions(transcript_captions=True),
        )
        vf = str(captured["caption_filter"])
        assert "text='Hello'" in vf and "text='Hola'" in vf
        assert "between(t,3.000,5.000)" in vf  # 4–6s shifted by the 2–3s cut
        assert "Removed" not in vf  # caption inside a cut is dropped

    def test_caption_filter_not_built_unless_requested(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from openclip.director import executor

        captured: dict[str, object] = {}

        def _fake(**kwargs: object) -> RenderResult:
            captured.update(kwargs)
            return RenderResult(
                out=Path(str(kwargs["out_path"])),
                kept_duration_s=14.0,
                segments=1,
                cuts_applied=0,
                captions_burned=False,
                audio_enhanced=False,
            )

        monkeypatch.setattr(executor, "render_export", _fake)
        project, media = _project_media(tmp_path)
        decisions = [
            Decision(
                id="cap1",
                kind="caption",
                source_span=(1.0, 2.0),
                params={"primary": "Hi", "origin": "ai"},
                rationale="Hi",
                status="accepted",
            )
        ]
        edl = EditDecisionList(project_source="in.mp4", decisions=decisions)
        render_edl(project, media, edl, tmp_path / "out.mp4")  # default opts
        assert captured["caption_filter"] is None

    def test_transcript_captions_burned_in_real_render(self, tmp_path: Path) -> None:
        from openclip.engine.render import RenderOptions

        # solid black 10s video: burned white text is unambiguous in pixels
        black = tmp_path / "black.mp4"
        run_ffmpeg(
            [
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:size=320x180:rate=15:duration=10",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                str(black),
            ],
            timeout=120,
        )
        project = Project.create(source=black, script_text="")
        media = probe(black)
        decision = Decision(
            id="cap1",
            kind="caption",
            source_span=(2.0, 6.0),
            params={"primary": "BURNED CAPTION", "support": "Subtitulo", "origin": "ai"},
            rationale="caption",
            status="accepted",
        )
        edl = EditDecisionList(project_source="black.mp4", decisions=[decision])
        out_cap = tmp_path / "cap.mp4"
        render_edl(project, media, edl, out_cap, RenderOptions(transcript_captions=True))

        inside = self._frame(out_cap, 4.0, tmp_path / "c_in.png")  # caption visible
        outside = self._frame(out_cap, 8.0, tmp_path / "c_out.png")  # black, no caption
        assert inside != outside
        # the inside frame has bright (white text) pixels, the outside is ~black
        assert sum(inside) > sum(outside) * 2

    def test_export_config_vertical_916(self, lesson: Path, tmp_path: Path) -> None:
        project = Project.create(source=lesson, script_text="")
        media = probe(lesson)
        decision = Decision(
            id="e1",
            kind="export_config",
            params={"width": 180, "height": 320, "vertical": True},
            rationale="shorts",
            status="accepted",
        )
        out = tmp_path / "short.mp4"
        render_edl(project, media, self._edl_with(decision), out)
        info = probe(out)
        assert (info.width, info.height) == (180, 320)
