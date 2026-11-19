"""Tests for hardening: error handling, input validation, edge-case safety."""

from __future__ import annotations

import subprocess
import sys

import pytest

from openclip.engine.ffmpeg import probe
from openclip.engine.synth import generate_lesson
from openclip.hardening import (
    CancellationError,
    DependencyError,
    InputError,
    MediaError,
    OpenClipError,
    RenderError,
    get_missing_dependencies,
    validate_dependencies,
    validate_media,
)


class TestValidateMedia:
    def test_valid_media_passes(self, tmp_path_factory):
        lesson = generate_lesson(tmp_path_factory.mktemp("valid") / "lesson.mp4")
        validate_media(lesson.path)  # should not raise

    def test_missing_file_raises_media_error(self, tmp_path):
        missing = tmp_path / "nonexistent.mp4"
        with pytest.raises(MediaError) as exc_info:
            validate_media(missing)
        assert "not found" in str(exc_info.value)

    def test_directory_raises_media_error(self, tmp_path):
        with pytest.raises(MediaError) as exc_info:
            validate_media(tmp_path)
        assert "not a file" in str(exc_info.value)

    def test_non_video_raises_media_error(self, tmp_path):
        # Create a text file and try to validate as media
        text_file = tmp_path / "not_video.txt"
        text_file.write_text("hello")
        with pytest.raises(MediaError) as exc_info:
            validate_media(text_file)
        # ffprobe on text file fails with various messages
        error_msg = str(exc_info.value).lower()
        assert any(
            keyword in error_msg
            for keyword in ("video", "cannot read", "duration")
        )

    def test_too_short_raises_media_error(self, tmp_path):
        # Create a very short synthetic video (0.1s)
        short = tmp_path / "short.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            "testsrc=duration=0.1:size=320x240:rate=30",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", str(short)
        ], check=True, capture_output=True)
        with pytest.raises(MediaError) as exc_info:
            validate_media(short)
        assert "too short" in str(exc_info.value)


class TestValidateDependencies:
    def test_available_dependencies_pass(self):
        # vision (cv2, numpy) should be available in test env
        validate_dependencies({"vision"})

    def test_missing_dependency_raises(self):
        with pytest.raises(DependencyError) as exc_info:
            validate_dependencies({"this_extra_does_not_exist_12345"})
        assert "this_extra_does_not_exist_12345" in str(exc_info.value)

    def test_get_missing_dependencies_returns_dict(self):
        deps = get_missing_dependencies()
        assert isinstance(deps, dict)
        assert "vision" in deps
        assert "gui" in deps
        assert "asr" in deps
        assert "gaze" in deps
        for status in deps.values():
            assert status.startswith("✅") or status.startswith("❌")


class TestExceptionHierarchy:
    def test_openclip_error_base(self):
        exc = OpenClipError("test message", exit_code=2)
        assert str(exc) == "test message"
        assert exc.exit_code == 2
        assert isinstance(exc, Exception)

    def test_input_error(self):
        exc = InputError("bad input")
        assert isinstance(exc, OpenClipError)

    def test_dependency_error(self):
        exc = DependencyError("missing dep")
        assert isinstance(exc, OpenClipError)

    def test_media_error(self):
        exc = MediaError("bad media")
        assert isinstance(exc, OpenClipError)

    def test_render_error(self):
        exc = RenderError("render failed")
        assert isinstance(exc, OpenClipError)

    def test_cancellation_error(self):
        exc = CancellationError("cancelled")
        assert isinstance(exc, OpenClipError)


class TestErrorSurfacingCLI:
    """Test that CLI commands surface friendly errors, not tracebacks."""

    def test_ai_edit_missing_media_shows_friendly_error(self, tmp_path):
        out = tmp_path / "out.mp4"
        result = subprocess.run([
            sys.executable, "-m", "openclip.cli", "ai-edit",
            "--media", str(tmp_path / "nonexistent.mp4"),
            "--prompt", "test",
            "--provider", "local_mock",
            "--auto-accept",
            "-o", str(out),
        ], capture_output=True, text=True, cwd="/root/projects/openclip-studio")

        assert result.returncode != 0
        # Should NOT contain traceback
        assert "Traceback" not in result.stderr
        assert "Traceback" not in result.stdout
        # Should contain friendly error message
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_ai_edit_missing_prompt_shows_friendly_error(self, tmp_path_factory):
        lesson = generate_lesson(tmp_path_factory.mktemp("lesson") / "lesson.mp4")
        out = tmp_path_factory.mktemp("out") / "out.mp4"
        result = subprocess.run([
            "openclip", "ai-edit",
            "--media", str(lesson.path),
            # --prompt omitted
            "--provider", "local_mock",
            "--auto-accept",
            "-o", str(out),
        ], capture_output=True, text=True, cwd="/root/projects/openclip-studio")

        assert result.returncode != 0
        assert "Traceback" not in result.stderr
        assert "Traceback" not in result.stdout

    def test_doctor_runs_without_crashing(self):
        result = subprocess.run([
            sys.executable, "-m", "openclip.cli", "doctor"
        ], capture_output=True, text=True, cwd="/root/projects/openclip-studio")

        assert result.returncode in (0, 1)  # 0 = all ok, 1 = some missing
        assert "OpenClip System Check" in result.stdout
        assert "ffmpeg" in result.stdout
        assert "ffprobe" in result.stdout
        assert "Traceback" not in result.stdout
        assert "Traceback" not in result.stderr


class TestEdgeCases:
    def test_no_face_beauty_is_noop(self, tmp_path_factory):
        """Beauty pre-pass with no face detected should pass frames through unchanged."""
        from openclip.ai.beauty import BeautyConfig, beauty_available, beauty_prepass
        from openclip.ai.face_analysis import CachedFaceAnalysis, MockFaceAnalysisProvider

        if not beauty_available():
            pytest.skip("Vision dependencies not available")

        lesson = generate_lesson(tmp_path_factory.mktemp("lesson") / "lesson.mp4")
        media = probe(lesson.path)

        # Use a mock provider that always returns None (no face)
        class NoFaceProvider(MockFaceAnalysisProvider):
            def analyze_frame(self, frame_index, timestamp, frame=None):
                return None

        face_analysis = CachedFaceAnalysis(NoFaceProvider())
        config = BeautyConfig(skin_smooth=0.5, jaw_slim=0.3, neck_tighten=0.2)

        out = tmp_path_factory.mktemp("out") / "beauty_noface.mp4"
        beauty_prepass(
            str(lesson.path), str(out), config, face_analysis,
            media.fps if media.fps > 0 else 25.0,
            media.width or 1280, media.height or 720,
        )

        # Output should exist and be valid
        assert out.exists()
        out_info = probe(out)
        assert out_info.duration_s == media.duration_s
        assert out_info.width == media.width
        assert out_info.height == media.height

    def test_audio_less_source_muxes_fine(self, tmp_path):
        """Render a source with no audio track should work."""
        from openclip.director.executor import render_edl
        from openclip.director.models import EditDecisionList
        from openclip.engine.project import Project
        from openclip.engine.render import RenderOptions

        # Create a video with no audio
        no_audio = tmp_path / "no_audio.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            "testsrc=duration=2:size=320x240:rate=30",
            "-an",  # no audio
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", str(no_audio)
        ], check=True, capture_output=True)

        media = probe(no_audio)
        assert not media.has_audio

        edl = EditDecisionList(project_source=str(media.path), decisions=[])
        project = Project.create(source=media.path, script_text="")
        out = tmp_path / "out.mp4"

        result = render_edl(
            project, media, edl, out,
            RenderOptions(burn_captions=False),
        )

        assert result.out.exists()
        out_media = probe(result.out)
        assert not out_media.has_audio

    def test_empty_hook_results_valid_empty_plan(self, tmp_path_factory):
        """Hook detection with no results produces valid empty plan."""
        from openclip.ai.hook_detection import HookDetector, hooks_to_director_plan
        from openclip.ai.transcript_editing import parse_transcript

        # Transcript with no hooks (just filler)
        transcript = {
            "segments": [
                {"text": "The cat sat on the mat.", "start": 0.0, "end": 2.0},
                {"text": "It was a sunny day.", "start": 2.0, "end": 4.0},
            ]
        }
        segments = parse_transcript(transcript)
        transcript_list = [
            {"text": s.text, "start": s.start_s, "end": s.end_s}
            for s in segments if s.start_s is not None and s.end_s is not None
        ]
        detector = HookDetector(transcript_list)
        hooks = detector.find_hooks(top_n=5)

        # Should find 0 hooks
        assert len(hooks) == 0

        # Plan should be valid (empty clips)
        lesson = generate_lesson(tmp_path_factory.mktemp("lesson") / "lesson.mp4")
        media = probe(lesson.path)
        plan = hooks_to_director_plan(hooks, source_duration_seconds=media.duration_s)

        assert plan.provider == "hook_detection"
        assert len(plan.clips) == 0

    def test_empty_broll_results_valid_empty_plan(self, tmp_path_factory):
        """B-roll detection with no results produces valid empty plan."""
        from openclip.ai.broll_detection import BrollDetector, broll_to_director_plan
        from openclip.ai.transcript_editing import TranscriptSegment, parse_transcript

        # Transcript with no visual language
        transcript = {
            "segments": [
                {"text": "The cat sat on the mat.", "start": 0.0, "end": 2.0},
            ]
        }
        segments = parse_transcript(transcript)
        blocks = [
            TranscriptSegment(text=s.text, start_s=s.start_s, end_s=s.end_s)
            for s in segments if s.start_s is not None and s.end_s is not None
        ]
        detector = BrollDetector(blocks)
        opportunities = detector.find_opportunities()

        assert len(opportunities) == 0

        lesson = generate_lesson(tmp_path_factory.mktemp("lesson") / "lesson.mp4")
        media = probe(lesson.path)
        plan = broll_to_director_plan([], [], source_duration_seconds=media.duration_s)

        assert plan.provider == "broll_detection"
        assert len(plan.broll) == 0

    def test_very_short_clip_rejected_by_validation(self, tmp_path_factory):
        """Clips shorter than 0.5s should be rejected by validation."""
        from openclip.director.executor import render_edl
        from openclip.director.models import EditDecisionList
        from openclip.engine.project import Project
        from openclip.engine.render import RenderOptions

        # Create a 0.3s clip
        short = tmp_path_factory.mktemp("short") / "short.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            "testsrc=duration=0.3:size=320x240:rate=30",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", str(short)
        ], check=True, capture_output=True)

        media = probe(short)
        edl = EditDecisionList(project_source=str(media.path), decisions=[])
        project = Project.create(source=media.path, script_text="")
        out = tmp_path_factory.mktemp("out") / "out.mp4"

        with pytest.raises(ValueError) as exc_info:
            render_edl(
                project, media, edl, out,
                RenderOptions(burn_captions=False),
            )
        assert "too short" in str(exc_info.value).lower()


class TestDoctorOutput:
    def test_doctor_output_format(self):
        result = subprocess.run([
            "openclip", "doctor"
        ], capture_output=True, text=True, cwd="/root/projects/openclip-studio")

        assert "OpenClip System Check" in result.stdout
        assert "Python:" in result.stdout
        assert "Platform:" in result.stdout
        assert "Core Dependencies:" in result.stdout
        assert "Optional Extras:" in result.stdout
        assert "ffmpeg" in result.stdout
        assert "ffprobe" in result.stdout
        assert "vision:" in result.stdout
        assert "gui:" in result.stdout
        assert "asr:" in result.stdout
        assert "gaze:" in result.stdout
