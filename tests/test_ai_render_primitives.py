"""Tests for AI render primitives: zoom filter strings, audio chains, executor wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from openclip.ai import audio_chain_for_enhancement, build_zoom_vf
from openclip.director.executor import render_edl
from openclip.director.models import Decision, EditDecisionList
from openclip.engine.ffmpeg import MediaInfo, probe, run_ffmpeg
from openclip.engine.project import Project
from openclip.engine.render import RenderResult
from openclip.engine.synth import generate_lesson


@pytest.fixture(scope="module")
def lesson(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return generate_lesson(tmp_path_factory.mktemp("ai-render") / "lesson.mp4").path


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


class TestExecutorWiring:
    def _project_media(self, tmp_path: Path) -> tuple[Project, MediaInfo]:
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
        project, media = self._project_media(tmp_path)
        decisions = [
            Decision(
                id="c1", kind="cut", source_span=(2.0, 3.0), rationale="silence",
                status="accepted",
            ),
            Decision(
                id="z1", kind="zoom", source_span=(4.0, 6.0),
                params={"strength": 0.5}, rationale="punch in", status="accepted",
            ),
            Decision(
                id="a1", kind="audio_enhance",
                params={"enabled": True, "denoise": True, "loudness_normalize": True},
                rationale="clean audio", status="accepted",
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
                out=Path(str(kwargs["out_path"])), kept_duration_s=12.0, segments=1,
                cuts_applied=1, captions_burned=False, audio_enhanced=False,
            )

        monkeypatch.setattr(executor, "render_export", _fake)
        project, media = self._project_media(tmp_path)
        decisions = [
            Decision(
                id="c1", kind="cut", source_span=(2.0, 6.0), rationale="silence",
                status="accepted",
            ),
            Decision(
                id="z1", kind="zoom", source_span=(3.0, 4.0), rationale="inside cut",
                status="accepted",
            ),
        ]
        edl = EditDecisionList(project_source="in.mp4", decisions=decisions)
        render_edl(project, media, edl, tmp_path / "out.mp4")
        assert captured["video_filters"] is None

    def test_unimplemented_ai_kinds_still_skipped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from openclip.director import executor

        captured: dict[str, object] = {}

        def _fake(**kwargs: object) -> RenderResult:
            captured.update(kwargs)
            return RenderResult(
                out=Path(str(kwargs["out_path"])), kept_duration_s=14.0, segments=1,
                cuts_applied=0, captions_burned=False, audio_enhanced=False,
            )

        monkeypatch.setattr(executor, "render_export", _fake)
        project, media = self._project_media(tmp_path)
        decisions = [
            Decision(
                id="b1", kind="broll", source_span=(1.0, 2.0), rationale="b-roll",
                status="accepted",
            ),
            Decision(id="e1", kind="export_config", rationale="mp4", status="accepted"),
        ]
        edl = EditDecisionList(project_source="in.mp4", decisions=decisions)
        render_edl(project, media, edl, tmp_path / "out.mp4")  # must not raise
        assert captured["video_filters"] is None


class TestRealRender:
    def _edl(self, zoom_strength: float) -> EditDecisionList:
        decisions = [
            Decision(
                id="c1", kind="cut", source_span=(2.0, 3.0), rationale="silence",
                status="accepted",
            ),
            Decision(
                id="z1", kind="zoom", source_span=(4.0, 8.0),
                params={"strength": zoom_strength}, rationale="punch in on key point",
                status="accepted",
            ),
            Decision(
                id="a1", kind="audio_enhance",
                params={"enabled": True, "denoise": True, "loudness_normalize": True},
                rationale="clean audio", status="accepted",
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
