"""Unit tests that need no media rendering (fast path)."""

from __future__ import annotations

from openclip.engine.align import (
    LineAlignment,
    WordHyp,
    align_script,
    even_alignment,
)
from openclip.engine.captions import CaptionStyle, build_ass
from openclip.engine.cutplan import CutDecision, CutOptions, kept_intervals, plan_cuts
from openclip.engine.enhance import build_audio_filter
from openclip.engine.ffmpeg import SilenceSpan
from openclip.engine.render import map_alignment, map_span
from openclip.engine.script import parse_script


def _hyp(words: list[str], start: float, per: float = 0.4) -> list[WordHyp]:
    return [WordHyp(w, start + i * per, start + (i + 1) * per) for i, w in enumerate(words)]


def test_parse_script_bilingual_pairs() -> None:
    text = (
        "Hello class :: 大家好\n"
        "\n"
        "Today we learn words.\n"
        "今天我们学单词。\n"
    )
    script = parse_script(text)
    assert len(script) == 2
    assert script.lines[0].primary == "Hello class"
    assert script.lines[0].support == "大家好"
    assert script.lines[1].primary == "Today we learn words."
    assert script.lines[1].support == "今天我们学单词。"


def test_parse_script_single_language() -> None:
    script = parse_script("One line only")
    assert len(script) == 1
    assert script.lines[0].support == ""


def test_aligner_recovers_sentence_spans() -> None:
    script = parse_script(
        "Hello everyone welcome\n\nSecond sentence here\n\nThird one"
    )
    hyps = (
        _hyp(["hello", "everyone", "welcome"], 0.5, 0.3)
        + _hyp(["second", "sentence", "here"], 3.0, 0.4)
        + _hyp(["third", "one"], 6.0, 0.5)
    )
    alignment = align_script(script, hyps)
    spans = [(ln.start_s, ln.end_s) for ln in alignment.lines]
    assert abs(spans[0][0] - 0.5) < 0.35
    assert abs(spans[1][0] - 3.0) < 0.45
    assert abs(spans[2][0] - 6.0) < 0.55
    assert all(ln.confidence > 0.5 for ln in alignment.lines)


def test_aligner_marks_offscript_as_unmatched_then_cut() -> None:
    script = parse_script("Say the target\n\nSay it again")
    hyps = (
        _hyp(["say", "the", "target"], 1.0, 0.4)
        + _hyp(["oops", "wrong", "take"], 4.0, 0.4)
        + _hyp(["say", "it", "again"], 8.0, 0.4)
    )
    alignment = align_script(script, hyps)
    assert len(alignment.lines) == 2
    opts = CutOptions()
    plan = plan_cuts(alignment, hyps, 12.0, [], opts)
    offscript = [c for c in plan.cuts if c.kind == "offscript"]
    assert offscript, f"expected offscript cut, got {[c.kind for c in plan.cuts]}"
    assert offscript[0].start_s >= 3.5
    assert offscript[0].end_s <= 6.0


def test_filler_cut() -> None:
    script = parse_script("Big idea")
    hyps = _hyp(["big", "um", "idea"], 1.0, 0.4)
    alignment = align_script(script, hyps)
    plan = plan_cuts(alignment, hyps, 5.0, [])
    fillers = [c for c in plan.cuts if c.kind == "filler"]
    assert fillers
    assert fillers[0].start_s >= 1.3
    assert fillers[0].end_s <= 2.0


def test_silence_cut_keeps_breathing_room() -> None:
    script = parse_script("Sentence")
    alignment = even_alignment(script, 10.0)
    silences = [SilenceSpan(2.0, 4.0)]
    plan = plan_cuts(alignment, [], 10.0, silences, CutOptions(silence_keep_s=0.15))
    assert len(plan.cuts) == 1
    c = plan.cuts[0]
    assert abs(c.start_s - 2.15) < 1e-6
    assert abs(c.end_s - 3.85) < 1e-6
    kept = kept_intervals(plan.cuts, 10.0)
    assert kept == [(0.0, 2.15), (3.85, 10.0)]


def test_min_kept_absorbs_tiny_islands() -> None:
    # Cut sandwich: keep (0,0.2) tiny island between media start and a cut.
    cuts = [CutDecision("silence", 0.2, 3.0, "r", 1.0)]
    kept = kept_intervals(cuts, 5.0)
    tiny = [(s, e) for s, e in kept if e - s < 0.6]
    assert tiny  # precondition: the tiny island exists
    plan = plan_cuts(
        even_alignment(parse_script("x"), 5.0),
        [],
        5.0,
        [SilenceSpan(0.25, 2.95)],
        CutOptions(),
    )
    kept_after = kept_intervals(plan.cuts, 5.0)
    assert all(e - s >= 0.6 - 1e-9 for s, e in kept_after)


def test_build_ass_has_bilingual_events_and_animation() -> None:
    script = parse_script("Hello class :: 大家好")
    alignment = align_script(
        script, _hyp(["hello", "class"], 1.0, 0.5)
    )
    text = build_ass(script.to_dict(), alignment, 1280, 720, CaptionStyle(animation="fade"))
    assert "[Script Info]" in text
    assert "PlayResX: 1280" in text
    assert "Style: Primary," in text
    assert "Style: Support," in text
    assert "Hello class" in text
    assert "大家好" in text
    assert "\\fad(160,160)" in text


def test_karaoke_tags_when_enabled() -> None:
    script = parse_script("Hello class")
    alignment = align_script(script, _hyp(["hello", "class"], 1.0, 0.5))
    text = build_ass(
        script.to_dict(), alignment, 1280, 720, CaptionStyle(karaoke=True, animation="none")
    )
    assert "\\k" in text


def test_audio_presets() -> None:
    assert build_audio_filter("none") is None
    chain = build_audio_filter("teaching")
    assert chain is not None and "loudnorm" in chain and "afftdn" in chain
    try:
        build_audio_filter("bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_map_span_and_alignment_after_cuts() -> None:
    cut_spans = [(2.0, 4.0), (6.0, 6.5)]
    assert map_span(0.0, 2.0, cut_spans) == (0.0, 2.0)
    assert map_span(4.0, 6.0, cut_spans) == (2.0, 4.0)
    assert map_span(2.5, 3.5, cut_spans) is None  # fully cut
    partial = map_span(3.0, 5.0, cut_spans)
    assert partial == (2.0, 3.0)

    from openclip.engine.align import Alignment

    a = Alignment(
        lines=[
            LineAlignment(0, 0.0, 2.0, 1.0, ["a", "b"]),
            LineAlignment(1, 2.5, 3.5, 1.0, ["cut", "line"]),
            LineAlignment(2, 4.0, 6.0, 1.0, ["c", "d"]),
        ]
    )
    mapped = map_alignment(a, cut_spans)
    assert [ln.index for ln in mapped.lines] == [0, 2]
    assert mapped.lines[1].start_s == 2.0
