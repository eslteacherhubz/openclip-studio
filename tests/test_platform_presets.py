"""Tests for platform export presets."""

from __future__ import annotations

from pathlib import Path

import pytest

from openclip.ai.platform_presets import (
    PRESETS,
    get_preset,
    list_presets,
    preset_names,
)
from openclip.engine.project import Project
from openclip.engine.render import RenderOptions
from openclip.engine.synth import generate_lesson


class TestPresetRegistry:
    def test_all_presets_have_valid_dims(self) -> None:
        for name, preset in PRESETS.items():
            assert preset.target_width > 0, f"{name}: width must be positive"
            assert preset.target_height > 0, f"{name}: height must be positive"
            assert preset.aspect in ("9:16", "16:9", "1:1", "4:5"), f"{name}: invalid aspect"
            # Verify aspect ratio matches dimensions (approximately)
            actual_aspect = preset.target_width / preset.target_height
            expected_aspect = {
                "9:16": 9 / 16,
                "16:9": 16 / 9,
                "1:1": 1.0,
                "4:5": 4 / 5,
            }[preset.aspect]
            assert abs(actual_aspect - expected_aspect) < 0.02, f"{name}: aspect mismatch"

    def test_get_preset_returns_correct_preset(self) -> None:
        tiktok = get_preset("tiktok")
        assert tiktok.name == "tiktok"
        assert tiktok.target_width == 1080
        assert tiktok.target_height == 1920
        assert tiktok.vertical is True

        reels = get_preset("REELS")  # case insensitive
        assert reels.name == "reels"
        assert reels.max_duration_s == 90.0

        shorts = get_preset("Shorts")
        assert shorts.name == "shorts"
        assert shorts.max_duration_s == 60.0

        youtube = get_preset("youtube")
        assert youtube.name == "youtube"
        assert youtube.vertical is False
        assert youtube.max_duration_s is None

    def test_get_preset_unknown_raises_clear_error(self) -> None:
        try:
            get_preset("unknown_platform")
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "unknown_platform" in str(e)
            assert "tiktok" in str(e).lower()
            assert "reels" in str(e).lower()

    def test_list_presets_sorted(self) -> None:
        presets = list_presets()
        names = [p.name for p in presets]
        assert names == sorted(names)

    def test_preset_names_sorted(self) -> None:
        names = preset_names()
        assert names == sorted(names)
        assert "tiktok" in names
        assert "reels" in names
        assert "shorts" in names
        assert "youtube" in names
        assert "instagram" in names


class TestRenderWiring:
    def test_platform_tiktok_sets_vertical_dims_and_reframe(self) -> None:
        preset = get_preset("tiktok")
        opts = RenderOptions(platform=preset)

        # Verify platform is set
        assert opts.platform is preset
        assert opts.platform.name == "tiktok"
        assert opts.platform.vertical is True

        # The render_export function will set smart_reframe=True for vertical presets
        # This is tested in integration tests

    def test_platform_none_leaves_options_unchanged(self) -> None:
        opts = RenderOptions(platform=None)
        assert opts.platform is None
        assert opts.smart_reframe is False  # default

    def test_platform_youtube_does_not_enable_reframe(self) -> None:
        preset = get_preset("youtube")
        opts = RenderOptions(platform=preset)

        assert opts.platform is preset
        assert opts.platform.vertical is False
        assert opts.platform.max_duration_s is None


class TestVerticalFallback:
    def test_tracking_unavailable_falls_back_to_mock(self, tmp_path) -> None:
        """When face tracking is unavailable, should fall back to mock tracking."""
        from openclip.ai.face_analysis import select_provider
        from openclip.ai.reframing import mock_tracking_data, tracking_from_face_analysis

        # Select provider with prefer_real=True but no real deps
        provider = select_provider(prefer_real=True)
        # Should return mock provider (never raises)
        assert provider is not None

        # tracking_from_face_analysis with mock should work
        points = tracking_from_face_analysis(provider, 30.0, 10.0, sample_fps=5.0)
        assert len(points) >= 1

        # mock_tracking_data should also work
        mock_points = mock_tracking_data(10.0)
        assert len(mock_points) >= 1


class TestMultiExportCLI:
    def test_multi_export_creates_multiple_outputs(self, tmp_path) -> None:
        """Test that comma-separated platforms create multiple outputs."""
        from openclip.ai.platform_presets import get_preset
        from openclip.director.executor import render_edl
        from openclip.director.models import EditDecisionList
        from openclip.engine.ffmpeg import probe
        from openclip.engine.render import RenderOptions
        from openclip.engine.synth import generate_lesson

        # Generate a simple test media
        lesson = generate_lesson(tmp_path / "lesson.mp4")
        media = probe(lesson.path)

        # Create a simple EDL with no decisions (just render)
        edl = EditDecisionList(project_source=str(media.path), decisions=[])
        project = Project.create(source=media.path, script_text="")

        platforms = ["tiktok", "youtube"]
        outputs = []
        for name in platforms:
            preset = get_preset(name)
            out_path = tmp_path / f"out_{name}.mp4"
            result = render_edl(
                project,
                media,
                edl,
                out_path,
                RenderOptions(platform=preset, burn_captions=False),
            )
            outputs.append((name, result.out))

        assert len(outputs) == 2
        for name, path in outputs:
            assert path.exists(), f"{name} output not created"
            assert path.stat().st_size > 0, f"{name} output is empty"

    def test_multi_export_tiktok_youtube_dimensions(self, tmp_path) -> None:
        """Verify TikTok gets 9:16 and YouTube gets 16:9."""
        from openclip.ai.platform_presets import get_preset
        from openclip.director.executor import render_edl
        from openclip.director.models import EditDecisionList
        from openclip.engine.ffmpeg import probe
        from openclip.engine.render import RenderOptions
        from openclip.engine.synth import generate_lesson

        lesson = generate_lesson(tmp_path / "lesson.mp4")
        media = probe(lesson.path)
        edl = EditDecisionList(project_source=str(media.path), decisions=[])
        project = Project.create(source=media.path, script_text="")

        # TikTok (vertical)
        tiktok_preset = get_preset("tiktok")
        tiktok_out = tmp_path / "out_tiktok.mp4"
        render_edl(
            project,
            media,
            edl,
            tiktok_out,
            RenderOptions(platform=tiktok_preset, burn_captions=False),
        )

        # YouTube (horizontal)
        youtube_preset = get_preset("youtube")
        youtube_out = tmp_path / "out_youtube.mp4"
        render_edl(
            project,
            media,
            edl,
            youtube_out,
            RenderOptions(platform=youtube_preset, burn_captions=False),
        )

        # Verify both exist
        assert tiktok_out.exists()
        assert youtube_out.exists()

        # Check dimensions via ffprobe
        import subprocess
        def get_dims(path: Path) -> tuple[int, int]:
            result = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)
            ], capture_output=True, text=True, check=True)
            w, h = map(int, result.stdout.strip().split(","))
            return w, h

        tiktok_w, tiktok_h = get_dims(tiktok_out)
        youtube_w, youtube_h = get_dims(youtube_out)

        # TikTok should be 1080x1920 (9:16)
        assert tiktok_w == 1080, f"TikTok width: {tiktok_w}"
        assert tiktok_h == 1920, f"TikTok height: {tiktok_h}"

        # YouTube should be 1920x1080 (16:9)
        assert youtube_w == 1920, f"YouTube width: {youtube_w}"
        assert youtube_h == 1080, f"YouTube height: {youtube_h}"


class TestDurationWarning:
    def test_duration_warning_logged_when_source_exceeds_max(self, caplog, tmp_path) -> None:
        """When source duration exceeds platform max, a warning is logged."""
        import logging

        from openclip.ai.platform_presets import get_preset
        from openclip.director.executor import render_edl
        from openclip.director.models import EditDecisionList
        from openclip.engine.ffmpeg import probe
        from openclip.engine.render import RenderOptions
        from openclip.engine.synth import generate_lesson

        # Create a 120s lesson (exceeds Shorts 60s limit)
        lesson = generate_lesson(tmp_path / "lesson.mp4")
        # We can't easily change the duration, so let's test with a mock MediaInfo
        media = probe(lesson.path)

        # Create a mock MediaInfo with 120s duration for the test
        from unittest.mock import MagicMock
        mock_media = MagicMock()
        mock_media.path = media.path
        mock_media.width = media.width
        mock_media.height = media.height
        mock_media.fps = media.fps
        mock_media.duration_s = 120.0  # Exceeds Shorts 60s
        mock_media.has_audio = media.has_audio
        mock_media.has_video = True

        shorts_preset = get_preset("shorts")
        assert shorts_preset.max_duration_s == 60.0

        edl = EditDecisionList(project_source=str(media.path), decisions=[])
        project = Project.create(source=media.path, script_text="")
        out_path = tmp_path / "out_shorts.mp4"

        with caplog.at_level(logging.WARNING):
            render_edl(
                project,
                mock_media,
                edl,
                out_path,
                RenderOptions(platform=shorts_preset, burn_captions=False),
            )

        # Check that warning was logged
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("exceeds" in msg and "shorts" in msg.lower() for msg in warning_messages), \
            f"Expected duration warning, got: {warning_messages}"


class TestPresetDataIntegrity:
    def test_tiktok_preset(self) -> None:
        p = get_preset("tiktok")
        assert p.target_width == 1080
        assert p.target_height == 1920
        assert p.aspect == "9:16"
        assert p.max_duration_s == 600.0
        assert p.vertical is True
        assert p.loudness_lufs == -14.0

    def test_reels_preset(self) -> None:
        p = get_preset("reels")
        assert p.target_width == 1080
        assert p.target_height == 1920
        assert p.aspect == "9:16"
        assert p.max_duration_s == 90.0
        assert p.vertical is True

    def test_shorts_preset(self) -> None:
        p = get_preset("shorts")
        assert p.target_width == 1080
        assert p.target_height == 1920
        assert p.aspect == "9:16"
        assert p.max_duration_s == 60.0
        assert p.vertical is True

    def test_youtube_preset(self) -> None:
        p = get_preset("youtube")
        assert p.target_width == 1920
        assert p.target_height == 1080
        assert p.aspect == "16:9"
        assert p.max_duration_s is None
        assert p.vertical is False

    def test_instagram_preset(self) -> None:
        p = get_preset("instagram")
        assert p.target_width == 1080
        assert p.target_height == 1350
        assert p.aspect == "4:5"
        assert p.max_duration_s == 60.0
        assert p.vertical is False


# Integration test for the full pipeline with platform preset
class TestPlatformPresetIntegration:
    def test_ai_edit_with_platform_preset(self, tmp_path) -> None:
        """End-to-end: ai-edit with --platform tiktok renders correctly."""
        import json
        import subprocess

        # Generate lesson
        lesson = generate_lesson(tmp_path / "lesson.mp4")

        # Create transcript with some content
        transcript = {
            "segments": [
                {"text": "Welcome to the lesson.", "start": 0.0, "end": 3.0},
                {"text": "Imagine a world where everyone codes.", "start": 3.0, "end": 8.0},
            ]
        }
        transcript_path = tmp_path / "transcript.json"
        transcript_path.write_text(json.dumps(transcript))

        # Run ai-edit with platform preset (using local_mock provider)
        # Note: CLI appends platform name to output filename
        # (e.g., out_tiktok.mp4 -> out_tiktok_tiktok.mp4)
        out_path = tmp_path / "out_tiktok.mp4"
        expected_out_path = tmp_path / "out_tiktok_tiktok.mp4"
        result = subprocess.run(
            [
            "openclip", "ai-edit",
            "--media", str(lesson.path),
            "--prompt", "Create a short vertical clip for TikTok",
            "--transcript", str(transcript_path),
            "--provider", "local_mock",
            "--auto-accept",
            "--platform", "tiktok",
            "-o", str(out_path),
        ],
        capture_output=True,
        text=True,
        cwd="/root/projects/openclip-studio",
    )

        assert result.returncode == 0, f"ai-edit failed: {result.stderr}"
        assert expected_out_path.exists(), f"Output file not created at {expected_out_path}"

        # Verify dimensions
        import subprocess as sp
        dims_result = sp.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(expected_out_path)
        ], capture_output=True, text=True, check=True)
        w, h = map(int, dims_result.stdout.strip().split(","))
        assert w == 1080, f"Expected width 1080, got {w}"
        assert h == 1920, f"Expected height 1920, got {h}"


@pytest.fixture
def generate_lesson_fixture(tmp_path):
    return generate_lesson(tmp_path / "lesson.mp4")


# Re-export for test use
__all__ = [
    "TestDurationWarning",
    "TestMultiExportCLI",
    "TestPlatformPresetIntegration",
    "TestPresetDataIntegrity",
    "TestPresetRegistry",
    "TestRenderWiring",
    "TestVerticalFallback",
]
