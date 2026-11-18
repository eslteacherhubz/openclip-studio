"""Tests for B-roll detection and sourcing logic."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from openclip.ai.broll_detection import BrollDetector, BrollOpportunity, broll_to_director_plan
from openclip.ai.broll_sourcing import BrollSource, BrollSourceResolver
from openclip.ai.transcript_editing import TranscriptSegment

# GUI test setup
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import Qt

    from openclip.gui.main_window import MainWindow
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    Qt = None
    MainWindow = None


def _block(text: str, start: float, end: float) -> TranscriptSegment:
    return TranscriptSegment(text=text, start_s=start, end_s=end)


class TestBrollDetector:
    def test_finds_visual_language_opportunity(self) -> None:
        blocks = [
            _block("Welcome to the lesson.", 0.0, 5.0),
            _block("Imagine a world where everyone codes.", 5.0, 12.0),
            _block("Let's continue with the basics.", 12.0, 18.0),
        ]
        detector = BrollDetector(blocks)
        opportunities = detector.find_opportunities()

        assert len(opportunities) >= 1
        visual_ops = [o for o in opportunities if o.reason == "visual_language"]
        assert len(visual_ops) >= 1
        op = visual_ops[0]
        assert op.start >= 5.0
        assert op.end <= 12.0 + 5.0  # extended to min duration
        assert "imagine" in op.keywords
        assert op.priority > 0.5

    def test_finds_multiple_visual_keywords(self) -> None:
        blocks = [
            _block("Look at this picture, for example, visualize the result.", 10.0, 20.0),
        ]
        detector = BrollDetector(blocks)
        opportunities = detector.find_opportunities()

        visual_ops = [o for o in opportunities if o.reason == "visual_language"]
        assert len(visual_ops) == 1
        op = visual_ops[0]
        assert len(op.keywords) >= 3
        assert op.priority >= 0.8

    def test_exclude_hooks_prevents_detection_in_hook_windows(self) -> None:
        blocks = [
            _block("Welcome to the lesson.", 0.0, 5.0),
            _block("Imagine a world where everyone codes.", 5.0, 12.0),
            _block("This is unbelievable! A secret nobody knows!", 12.0, 20.0),
            _block("Let's continue with the basics.", 20.0, 28.0),
        ]
        detector = BrollDetector(blocks)

        # Hook window covers the "imagine" block (5-12)
        hook_windows = ((4.0, 13.0),)
        opportunities = detector.find_opportunities(exclude_hooks=hook_windows)

        # Should not find visual_language opportunity in the excluded window
        visual_ops = [o for o in opportunities if o.reason == "visual_language"]
        assert len(visual_ops) == 0

        # But should still find opportunities outside hook window
        # (topic_shift between "unbelievable" and "continue" blocks)
        all_ops = detector.find_opportunities(exclude_hooks=hook_windows)
        # The topic shift at 20s should still be detected
        topic_ops = [o for o in all_ops if o.reason == "topic_shift"]
        assert len(topic_ops) >= 1
        assert topic_ops[0].start >= 18.0  # after the hook window

    def test_exclude_hooks_with_multiple_windows(self) -> None:
        blocks = [
            _block("Imagine the first concept.", 0.0, 8.0),
            _block("Now picture the second idea.", 8.0, 16.0),
            _block("For example, visualize the third.", 16.0, 24.0),
        ]
        detector = BrollDetector(blocks)

        # Exclude first two blocks
        hook_windows = ((0.0, 16.0),)
        opportunities = detector.find_opportunities(exclude_hooks=hook_windows)

        visual_ops = [o for o in opportunities if o.reason == "visual_language"]
        # Only the third block (16-24) should remain
        assert len(visual_ops) == 1
        assert visual_ops[0].start >= 16.0

    def test_concept_intro_detection(self) -> None:
        blocks = [
            _block("Welcome everyone.", 0.0, 3.0),
            _block("What is the concept of recursion? Let me define it clearly.", 3.0, 12.0),
            _block("The key principle is base case and recursive step.", 12.0, 20.0),
        ]
        detector = BrollDetector(blocks)
        opportunities = detector.find_opportunities()

        concept_ops = [o for o in opportunities if o.reason == "concept_intro"]
        assert len(concept_ops) >= 1
        op = concept_ops[0]
        keywords_lower = " ".join(op.keywords).lower()
        assert "concept" in keywords_lower or "principle" in keywords_lower
        assert op.priority > 0.6

    def test_topic_shift_detection(self) -> None:
        blocks = [
            _block("Let's talk about Python syntax and variables.", 0.0, 10.0),
            _block("Now moving to machine learning and neural networks.", 10.0, 20.0),
        ]
        detector = BrollDetector(blocks)
        opportunities = detector.find_opportunities()

        topic_ops = [o for o in opportunities if o.reason == "topic_shift"]
        assert len(topic_ops) >= 1
        op = topic_ops[0]
        assert op.start <= 10.0
        assert op.end >= 10.0
        assert op.priority > 0.4

    def test_no_topic_shift_for_similar_content(self) -> None:
        blocks = [
            _block("Python variables store values in memory.", 0.0, 10.0),
            _block("Variables in Python can hold different data types.", 10.0, 20.0),
        ]
        detector = BrollDetector(blocks)
        opportunities = detector.find_opportunities()

        topic_ops = [o for o in opportunities if o.reason == "topic_shift"]
        # High word overlap should not trigger topic shift
        assert len(topic_ops) == 0

    def test_non_overlapping_opportunities(self) -> None:
        blocks = [
            _block("Imagine the first scene.", 0.0, 8.0),
            _block("Look at this picture now.", 8.0, 16.0),
            _block("For example, visualize this.", 16.0, 24.0),
        ]
        detector = BrollDetector(blocks)
        opportunities = detector.find_opportunities()

        # Should not have overlapping opportunities
        for i in range(len(opportunities) - 1):
            assert opportunities[i].end <= opportunities[i + 1].start

    def test_sorted_by_start_time(self) -> None:
        blocks = [
            _block("For example, visualize this.", 16.0, 24.0),
            _block("Imagine the first scene.", 0.0, 8.0),
            _block("Look at this picture now.", 8.0, 16.0),
        ]
        detector = BrollDetector(blocks)
        opportunities = detector.find_opportunities()

        starts = [o.start for o in opportunities]
        assert starts == sorted(starts)

    def test_untimed_blocks_ignored(self) -> None:
        blocks = [
            _block("Imagine a world.", 0.0, 5.0),
            TranscriptSegment(text="No timestamps here.", start_s=None, end_s=None),
            _block("Look at this.", 10.0, 15.0),
        ]
        detector = BrollDetector(blocks)
        opportunities = detector.find_opportunities()

        # Should only process timed blocks
        visual_ops = [o for o in opportunities if o.reason == "visual_language"]
        assert len(visual_ops) == 2

    def test_empty_blocks_returns_empty(self) -> None:
        detector = BrollDetector([])
        assert detector.find_opportunities() == []

    def test_broll_opportunity_dataclass(self) -> None:
        op = BrollOpportunity(
            start=1.0,
            end=10.0,
            concept_query="test query",
            keywords=("test", "query"),
            priority=0.8,
            reason="visual_language",
        )
        assert op.duration == 9.0
        d = op.to_dict()
        assert d["start"] == 1.0
        assert d["end"] == 10.0
        assert d["duration"] == 9.0
        assert d["concept_query"] == "test query"
        assert d["keywords"] == ["test", "query"]
        assert d["priority"] == 0.8
        assert d["reason"] == "visual_language"

    def test_deterministic(self) -> None:
        blocks = [
            _block("Imagine the first scene.", 0.0, 8.0),
            _block("Look at this picture now.", 8.0, 16.0),
        ]
        detector = BrollDetector(blocks)
        assert detector.find_opportunities() == detector.find_opportunities()


class TestBrollSourceResolver:
    def test_prefers_local_match_over_generated(self) -> None:
        # Create a temporary library file
        library_data = {
            "assets": [
                {
                    "path": "/assets/classroom.mp4",
                    "keywords": ["classroom", "students", "teacher", "learning"],
                },
                {
                    "path": "/assets/office.mp4",
                    "keywords": ["office", "meeting", "business", "work"],
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(library_data, f)
            library_path = f.name

        try:
            resolver = BrollSourceResolver(library_path)

            opportunity = BrollOpportunity(
                start=5.0,
                end=15.0,
                concept_query="students raising hands in a classroom",
                keywords=("students", "classroom", "hands"),
                priority=0.8,
                reason="visual_language",
            )

            source = resolver.resolve(opportunity)
            assert source.match_type == "local"
            assert source.path == "/assets/classroom.mp4"
            assert source.confidence > 0.15
        finally:
            Path(library_path).unlink()

    def test_falls_back_to_generated_when_no_local_match(self) -> None:
        library_data = {
            "assets": [
                {
                    "path": "/assets/office.mp4",
                    "keywords": ["office", "meeting", "business"],
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(library_data, f)
            library_path = f.name

        try:
            resolver = BrollSourceResolver(library_path)

            opportunity = BrollOpportunity(
                start=5.0,
                end=15.0,
                concept_query="underwater coral reef diving",
                keywords=("coral", "reef", "diving"),
                priority=0.8,
                reason="visual_language",
            )

            source = resolver.resolve(opportunity)
            assert source.match_type == "generated"
            assert source.path.startswith("generated:broll/")
            assert source.confidence == 0.5
        finally:
            Path(library_path).unlink()

    def test_falls_back_to_generated_when_no_library(self) -> None:
        resolver = BrollSourceResolver()

        opportunity = BrollOpportunity(
            start=5.0,
            end=15.0,
            concept_query="any concept",
            keywords=("any", "concept"),
            priority=0.8,
            reason="visual_language",
        )

        source = resolver.resolve(opportunity)
        assert source.match_type == "generated"
        assert source.path.startswith("generated:broll/")
        assert source.confidence == 0.5

    def test_library_reload_on_different_path(self) -> None:
        library_data1 = {
            "assets": [{"path": "/assets/a.mp4", "keywords": ["alpha"]}]
        }
        library_data2 = {
            "assets": [{"path": "/assets/b.mp4", "keywords": ["beta"]}]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f1:
            json.dump(library_data1, f1)
            path1 = f1.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2:
            json.dump(library_data2, f2)
            path2 = f2.name

        try:
            resolver = BrollSourceResolver(path1)

            opportunity = BrollOpportunity(
                start=0.0,
                end=10.0,
                concept_query="alpha concept",
                keywords=("alpha",),
                priority=0.8,
                reason="visual_language",
            )

            # First resolve with path1
            source1 = resolver.resolve(opportunity, library_path=path1)
            assert source1.path == "/assets/a.mp4"

            # Resolve with path2 (should reload)
            source2 = resolver.resolve(opportunity, library_path=path2)
            assert source2.path == "/assets/b.mp4"
        finally:
            Path(path1).unlink()
            Path(path2).unlink()

    def test_broll_source_dataclass(self) -> None:
        source = BrollSource(match_type="local", path="/test.mp4", confidence=0.9)
        d = source.to_dict()
        assert d["match_type"] == "local"
        assert d["path"] == "/test.mp4"
        assert d["confidence"] == 0.9

    def test_invalid_library_file_handled_gracefully(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")
            library_path = f.name

        try:
            resolver = BrollSourceResolver(library_path)
            # Should not crash, just have empty library
            assert len(resolver._library) == 0

            opportunity = BrollOpportunity(
                start=0.0,
                end=10.0,
                concept_query="test",
                keywords=("test",),
                priority=0.5,
                reason="visual_language",
            )
            source = resolver.resolve(opportunity)
            assert source.match_type == "generated"
        finally:
            Path(library_path).unlink()

    def test_missing_library_file_handled_gracefully(self) -> None:
        resolver = BrollSourceResolver("/nonexistent/path.json")
        assert len(resolver._library) == 0


class TestBrollIntegration:
    def test_detector_and_resolver_together(self) -> None:
        library_data = {
            "assets": [
                {
                    "path": "/assets/coding.mp4",
                    "keywords": ["coding", "programming", "computer", "screen"],
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(library_data, f)
            library_path = f.name

        try:
            blocks = [
                _block("Welcome to coding class.", 0.0, 5.0),
                _block("Imagine writing your first program.", 5.0, 12.0),
                _block("Let's look at the code on screen.", 12.0, 20.0),
            ]
            detector = BrollDetector(blocks)
            opportunities = detector.find_opportunities()

            resolver = BrollSourceResolver(library_path)
            for opp in opportunities:
                source = resolver.resolve(opp)
                assert source.match_type in ("local", "generated")
                assert source.confidence > 0.0
                assert source.path
        finally:
            Path(library_path).unlink()


class TestBrollToDirectorPlan:
    def test_broll_to_director_plan_mapping(self) -> None:
        from openclip.ai.broll_sourcing import BrollSource

        opportunities = [
            BrollOpportunity(
                start=5.0,
                end=15.0,
                concept_query="students in classroom",
                keywords=("students", "classroom"),
                priority=0.8,
                reason="visual_language",
            ),
            BrollOpportunity(
                start=20.0,
                end=30.0,
                concept_query="coding on screen",
                keywords=("coding", "screen"),
                priority=0.7,
                reason="concept_intro",
            ),
        ]
        sources = [
            BrollSource(
                match_type="local", path="/assets/classroom.mp4", confidence=0.9
            ),
            BrollSource(
                match_type="generated",
                path="generated:broll/coding_20.0-30.0.mp4",
                confidence=0.5,
            ),
        ]

        plan = broll_to_director_plan(opportunities, sources, source_duration_seconds=60.0)

        assert plan.provider == "broll_detection"
        assert plan.source_duration_seconds == 60.0
        assert len(plan.broll) == 2

        broll1 = plan.broll[0]
        assert broll1.id == "broll-000"
        assert broll1.span.start_seconds == 5.0
        assert broll1.span.end_seconds == 15.0
        assert broll1.query == "students in classroom"
        assert broll1.source == "local"
        assert broll1.mode == "overlay"
        assert broll1.extra["asset"] == "/assets/classroom.mp4"
        assert broll1.extra["confidence"] == 0.9
        assert broll1.extra["reason"] == "visual_language"

        broll2 = plan.broll[1]
        assert broll2.id == "broll-001"
        assert broll2.span.start_seconds == 20.0
        assert broll2.span.end_seconds == 30.0
        assert broll2.query == "coding on screen"
        assert broll2.source == "generated"
        assert broll2.extra["asset"] == "generated:broll/coding_20.0-30.0.mp4"
        assert broll2.extra["confidence"] == 0.5
        assert broll2.extra["reason"] == "concept_intro"

    def test_broll_to_director_plan_empty(self) -> None:

        plan = broll_to_director_plan([], [], source_duration_seconds=60.0)

        assert plan.provider == "broll_detection"
        assert len(plan.broll) == 0
        assert plan.notes == ["0 B-roll opportunity(ies) detected"]


@pytest.mark.gui
class TestBrollGUI:
    """GUI tests for B-roll detection (require Qt)."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_gui(self):
        if not GUI_AVAILABLE:
            pytest.skip("PySide6 (gui extra) not installed")

    def test_find_broll_button_exists(self, qtbot) -> None:
        window = MainWindow()
        window.show()
        qtbot.waitExposed(window)

        # Check that the Find B-roll button exists in the B-roll panel
        assert hasattr(window, "btn_find_broll")
        assert window.btn_find_broll.text() == "Find B-roll"
        assert hasattr(window, "broll_table")
        assert window.broll_table.columnCount() == 5
        headers = [
            window.broll_table.horizontalHeaderItem(i).text()
            for i in range(window.broll_table.columnCount())
        ]
        assert headers == ["Accept?", "Start (s)", "End (s)", "Match Type", "Snippet"]

        window.close()

    def test_find_broll_populates_table(self, qtbot, tmp_path) -> None:
        from openclip.ai.transcript_editing import TranscriptSegment

        window = MainWindow()
        window.show()
        qtbot.waitExposed(window)

        # Create a simple transcript with visual language
        blocks = [
            TranscriptSegment(text="Welcome to the lesson.", start_s=0.0, end_s=5.0),
            TranscriptSegment(
                text="Imagine a world where everyone codes.", start_s=5.0, end_s=12.0
            ),
            TranscriptSegment(text="Let's continue with the basics.", start_s=12.0, end_s=18.0),
        ]
        window._original_transcript = {
            "segments": [
                {"text": b.text, "start": b.start_s, "end": b.end_s}
                for b in blocks
            ]
        }

        # Mock media duration
        from openclip.engine.ffmpeg import MediaInfo
        window.media = MediaInfo(
            path=tmp_path / "dummy.mp4",
            width=1920,
            height=1080,
            fps=30.0,
            duration_s=20.0,
            has_audio=False,
        )

        # Run find_broll
        window.find_broll()
        qtbot.wait(100)  # allow UI to update

        # Check that table was populated
        assert window.broll_table.rowCount() >= 1
        # First row should have a checkbox
        item = window.broll_table.item(0, 0)
        assert item is not None
        assert item.checkState() == Qt.CheckState.Checked

        window.close()


@pytest.mark.gaze
class TestBrollProof:
    """End-to-end proof test with synthetic media (requires ffmpeg)."""

    def test_broll_render_proof(self, tmp_path) -> None:
        """Generate synthetic media, detect B-roll, and verify render output."""
        import shutil
        import subprocess

        # Check for required tools
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not available")

        # 1. Generate 10s main clip (testsrc)
        main_clip = tmp_path / "main.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            "testsrc=duration=10:size=1920x1080:rate=30",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", str(main_clip)
        ], check=True, capture_output=True)

        # 2. Generate 5s synthetic B-roll clip (solid red)
        broll_clip = tmp_path / "broll_red.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            "color=c=red:duration=5:size=1920x1080:rate=30",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", str(broll_clip)
        ], check=True, capture_output=True)

        # 3. Create transcript with planted "visual_language" hook at 3s-8s
        transcript = {
            "segments": [
                {"text": "Welcome to the lesson.", "start": 0.0, "end": 3.0},
                {"text": "Imagine a world where everyone codes.", "start": 3.0, "end": 8.0},
                {"text": "Let's continue with the basics.", "start": 8.0, "end": 10.0},
            ]
        }
        transcript_path = tmp_path / "transcript.json"
        import json
        transcript_path.write_text(json.dumps(transcript))

        # 4. Create broll library manifest
        library = {
            "assets": [
                {
                    "path": str(broll_clip),
                    "keywords": ["world", "codes", "imagine", "coding"]
                }
            ]
        }
        library_path = tmp_path / "broll_library.json"
        library_path.write_text(json.dumps(library))

        # 5. Run the full pipeline: detect -> source -> render
        from openclip.ai import ai_plan_to_director_decisions
        from openclip.ai.broll_detection import BrollDetector
        from openclip.ai.broll_sourcing import BrollSourceResolver
        from openclip.ai.transcript_editing import TranscriptSegment, parse_transcript
        from openclip.director.executor import render_edl
        from openclip.director.models import EditDecisionList
        from openclip.engine.ffmpeg import probe
        from openclip.engine.project import Project
        from openclip.engine.render import RenderOptions

        # Parse transcript
        segments = parse_transcript(transcript)
        blocks = [
            TranscriptSegment(text=s.text, start_s=s.start_s, end_s=s.end_s)
            for s in segments
            if s.start_s is not None and s.end_s is not None
        ]

        # Detect opportunities
        detector = BrollDetector(blocks)
        opportunities = detector.find_opportunities()

        # Should find the "imagine" visual language opportunity
        visual_ops = [o for o in opportunities if o.reason == "visual_language"]
        assert len(visual_ops) >= 1
        opp = visual_ops[0]
        assert opp.start >= 3.0
        assert opp.end <= 8.0 + 5.0  # extended to min duration

        # Resolve sources (should match local library)
        resolver = BrollSourceResolver(library_path)
        sources = [resolver.resolve(opp, library_path=library_path) for opp in opportunities]

        # Should prefer local match
        src = sources[0]
        assert src.match_type == "local"
        assert src.path == str(broll_clip)

        # Build AI Director plan
        media = probe(main_clip)
        plan = broll_to_director_plan(opportunities, sources, media.duration_s)
        decisions = ai_plan_to_director_decisions(plan)
        for decision in decisions:
            decision.status = "accepted"
        edl = EditDecisionList(project_source=str(media.path), decisions=decisions)

        # Render
        project = Project.create(source=media.path, script_text="")
        out_path = tmp_path / "output.mp4"
        render_edl(
            project,
            media,
            edl,
            out_path,
            RenderOptions(burn_captions=False),
        )

        # 6. Verify output: extract frame at 5s (middle of B-roll window) and check for red
        frame_path = tmp_path / "frame_5s.png"
        subprocess.run([
            "ffmpeg", "-y", "-ss", "5", "-i", str(out_path),
            "-vframes", "1", str(frame_path)
        ], check=True, capture_output=True)

        # Check for red pixels in the frame
        import cv2
        import numpy as np
        frame = cv2.imread(str(frame_path))
        assert frame is not None, "Failed to read output frame"

        # Convert to HSV and check for red color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Red in HSV: hue around 0 or 180, high saturation
        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        red_ratio = np.count_nonzero(red_mask) / red_mask.size
        # Should have significant red content (B-roll overlay)
        assert red_ratio > 0.1, f"Expected red B-roll overlay, got red_ratio={red_ratio:.3f}"

