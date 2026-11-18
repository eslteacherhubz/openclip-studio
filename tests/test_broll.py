"""Tests for B-roll detection and sourcing logic."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from openclip.ai.broll_detection import BrollDetector, BrollOpportunity
from openclip.ai.broll_sourcing import BrollSource, BrollSourceResolver
from openclip.ai.transcript_editing import TranscriptSegment


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

