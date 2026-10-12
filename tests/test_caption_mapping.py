"""Regression tests for caption↔export-timeline mapping (map_span/map_alignment).

Covers the correction assignment: interior cuts, multiple cuts, boundary
overlaps, fully-removed spans, and determinism/monotonicity invariants.
"""

from __future__ import annotations

import pytest

from openclip.engine.align import Alignment, LineAlignment
from openclip.engine.render import map_alignment, map_span, normalize_cut_spans

EPS = 1e-9


class TestMapSpanInteriorCut:
    def test_caption_containing_single_interior_cut(self) -> None:
        """THE reported case: caption 0–10s, cut 3–6s → output 0–7s."""
        assert map_span(0.0, 10.0, [(3.0, 6.0)]) == (0.0, 7.0)

    def test_caption_containing_multiple_cuts(self) -> None:
        # 3s + 1s removed → 10s becomes 6s.
        assert map_span(0.0, 10.0, [(3.0, 6.0), (7.0, 8.0)]) == (0.0, 6.0)

    def test_caption_starting_before_all_cuts(self) -> None:
        assert map_span(0.0, 9.0, [(1.0, 2.0), (5.0, 7.0)]) == (0.0, 6.0)

    def test_cut_before_span_only_shifts_start(self) -> None:
        # Cut entirely before the span: both endpoints shift by its length.
        assert map_span(6.0, 9.0, [(1.0, 4.0)]) == (3.0, 6.0)


class TestMapSpanBoundaryOverlaps:
    def test_span_start_exactly_at_cut_start(self) -> None:
        # Visible content begins when the cut ends (output time = cut end - removed).
        assert map_span(3.0, 10.0, [(3.0, 6.0)]) == (3.0, 7.0)

    def test_span_end_exactly_at_cut_start(self) -> None:
        # Caption ends where the cut begins: no shift anywhere.
        assert map_span(0.0, 3.0, [(3.0, 6.0)]) == (0.0, 3.0)

    def test_span_start_exactly_at_cut_end(self) -> None:
        # Content at source 6 appears at output 3 (3s were removed before it).
        assert map_span(6.0, 10.0, [(3.0, 6.0)]) == (3.0, 7.0)

    def test_end_inside_cut_clamps_to_cut_start(self) -> None:
        # Caption 2–4s, cut 3–6s → only [2,3) is visible → output [2,3).
        assert map_span(2.0, 4.0, [(3.0, 6.0)]) == (2.0, 3.0)

    def test_start_inside_cut_moves_to_cut_end(self) -> None:
        # Caption 4–8s, cut 3–6s → visible [6,8) → output [3,5).
        assert map_span(4.0, 8.0, [(3.0, 6.0)]) == (3.0, 5.0)

    def test_span_between_two_cuts_keeps_relative_position(self) -> None:
        # Caption 6–8s with cuts (1,3) and (9,10): one cut before it → shift 2s.
        assert map_span(6.0, 8.0, [(1.0, 3.0), (9.0, 10.0)]) == (4.0, 6.0)


class TestMapSpanRemovedSpans:
    def test_fully_contained_in_cut_returns_none(self) -> None:
        assert map_span(3.5, 5.5, [(3.0, 6.0)]) is None

    def test_span_equal_to_cut_returns_none(self) -> None:
        assert map_span(3.0, 6.0, [(3.0, 6.0)]) is None

    def test_zero_length_span_returns_none(self) -> None:
        assert map_span(4.0, 4.0, [(3.0, 6.0)]) is None

    def test_clipped_to_nothing_returns_none(self) -> None:
        # Caption 2–4s, cut (2.5, 5.5): visible part (2, 2.5] has positive
        # length, so it must survive; but (3, 4) is fully inside → None.
        assert map_span(3.0, 4.0, [(2.5, 5.5)]) is None

    def test_no_cuts_returns_span_unchanged(self) -> None:
        assert map_span(2.0, 7.0, []) == (2.0, 7.0)


class TestMapSpanInvariants:
    """Duration preservation and monotonicity under deterministic fuzzing."""

    def test_output_duration_equals_visible_source_duration(self) -> None:
        cases = [
            (0.0, 10.0, [(3.0, 6.0)]),
            (2.0, 12.0, [(1.0, 4.0), (5.0, 6.5), (11.0, 13.0)]),
            (0.0, 10.0, [(3.0, 6.0), (7.0, 8.0)]),
            (4.0, 8.0, [(3.0, 6.0)]),
            (2.0, 4.0, [(3.0, 6.0)]),
            (0.0, 10.0, [(3.0, 7.0), (5.0, 9.0)]),
            (2.0, 12.0, [(3.0, 7.0), (5.0, 9.0)]),
        ]
        for s, e, cuts in cases:
            mapped = map_span(s, e, cuts)
            if mapped is None:
                continue
            out_s, out_e = mapped
            removed = sum(
                min(e, ce) - max(s, cs)
                for cs, ce in normalize_cut_spans(cuts)
                if cs < e and ce > s
            )
            visible = e - s - removed
            assert out_e - out_s == pytest.approx(visible, abs=EPS), (s, e, cuts)

    def test_monotonic_nondecreasing_endpoints(self) -> None:
        cases = [
            (0.0, 10.0, [(3.0, 6.0)]),
            (0.0, 10.0, [(3.0, 6.0), (7.0, 8.0)]),
            (6.0, 9.0, [(1.0, 4.0)]),
            (4.0, 8.0, [(3.0, 6.0)]),
        ]
        for s, e, cuts in cases:
            mapped = map_span(s, e, cuts)
            if mapped is None:
                continue
            assert mapped[0] <= mapped[1]

    def test_unsorted_cut_list_same_result_as_sorted(self) -> None:
        cuts = [(7.0, 8.0), (3.0, 6.0), (1.0, 2.0)]
        assert map_span(0.0, 10.0, cuts) == map_span(0.0, 10.0, sorted(cuts))

    def test_adjacent_cuts_treated_compactly(self) -> None:
        # Touching cuts (3,5) and (5,6): total removed 3s.
        assert map_span(0.0, 10.0, [(3.0, 5.0), (5.0, 6.0)]) == (0.0, 7.0)


class TestNormalizeCutSpans:
    def test_merges_overlapping(self) -> None:
        assert normalize_cut_spans([(3.0, 7.0), (5.0, 9.0)]) == [(3.0, 9.0)]

    def test_merges_adjacent(self) -> None:
        assert normalize_cut_spans([(3.0, 5.0), (5.0, 7.0)]) == [(3.0, 7.0)]

    def test_deduplicates(self) -> None:
        assert normalize_cut_spans([(3.0, 6.0), (3.0, 6.0)]) == [(3.0, 6.0)]

    def test_sorts_unsorted_input(self) -> None:
        assert normalize_cut_spans([(7.0, 8.0), (1.0, 2.0), (3.0, 4.0)]) == [
            (1.0, 2.0),
            (3.0, 4.0),
            (7.0, 8.0),
        ]

    def test_drops_invalid_and_empty(self) -> None:
        assert normalize_cut_spans([(5.0, 3.0), (2.0, 2.0), (1.0, 2.0)]) == [(1.0, 2.0)]
        assert normalize_cut_spans([]) == []

    def test_chain_of_overlaps_collapses(self) -> None:
        assert normalize_cut_spans([(0, 2), (1.5, 4), (3.5, 6), (8, 9)]) == [
            (0.0, 6.0),
            (8.0, 9.0),
        ]


class TestMapSpanOverlappingCuts:
    def test_reported_case_overlapping_cuts_four_second_output(self) -> None:
        # Cuts [3,7] and [5,9] overlap: union is [3,9] = 6 s removed, so a
        # 0-10 s caption must map to a 4 s output interval.
        assert map_span(0.0, 10.0, [(3.0, 7.0), (5.0, 9.0)]) == (0.0, 4.0)

    def test_duplicate_cuts_counted_once(self) -> None:
        assert map_span(0.0, 10.0, [(3.0, 6.0), (3.0, 6.0)]) == (0.0, 7.0)

    def test_unsorted_overlapping_same_as_sorted(self) -> None:
        assert map_span(0.0, 10.0, [(5.0, 9.0), (3.0, 7.0)]) == (0.0, 4.0)

    def test_union_fully_covers_span(self) -> None:
        # No single cut contains (2,5), but [1,3] and [2.5,6] together do.
        assert map_span(2.0, 5.0, [(1.0, 3.0), (2.5, 6.0)]) is None

    def test_union_covers_whole_timeline_except_tail(self) -> None:
        # Cuts [0,4] and [3,9] merge to [0,9]: caption (0,10) → (9,10) → (0,1).
        assert map_span(0.0, 10.0, [(0.0, 4.0), (3.0, 9.0)]) == (0.0, 1.0)

    def test_overlapping_cuts_interior_to_caption(self) -> None:
        # Caption (2,12) with overlapping cuts (3,7),(5,9): removed 6s,
        # both endpoints outside the union → output (2,6).
        assert map_span(2.0, 12.0, [(3.0, 7.0), (5.0, 9.0)]) == (2.0, 6.0)

    def test_kept_complement_agrees_with_map_span(self) -> None:
        from openclip.engine.cutplan import CutDecision, kept_intervals

        cuts = [(3.0, 7.0), (5.0, 9.0), (11.0, 11.5)]
        duration = 14.0
        fake = [
            CutDecision(kind="silence", start_s=s, end_s=e, reason="", confidence=1.0)
            for s, e in cuts
        ]
        kept_total = sum(e - s for s, e in kept_intervals(fake, duration))
        mapped = map_span(0.0, duration, cuts)
        assert mapped is not None
        assert mapped[1] - mapped[0] == pytest.approx(kept_total, abs=EPS)


class TestMapAlignment:
    def _alignment(self, *spans: tuple[int, float, float]) -> Alignment:
        return Alignment(
            lines=[
                LineAlignment(index=i, start_s=s, end_s=e, confidence=0.9)
                for i, s, e in spans
            ],
            transcriber="fuzzy",
        )

    def test_lines_across_interior_cut_are_compressed_not_dropped(self) -> None:
        alignment = self._alignment((0, 0.0, 10.0))
        mapped = map_alignment(alignment, [(3.0, 6.0)])
        assert len(mapped.lines) == 1
        assert mapped.lines[0].start_s == pytest.approx(0.0)
        assert mapped.lines[0].end_s == pytest.approx(7.0)

    def test_lines_inside_cuts_are_dropped(self) -> None:
        alignment = self._alignment(
            (0, 0.0, 2.0), (1, 3.5, 5.5), (2, 7.0, 9.0)
        )
        mapped = map_alignment(alignment, [(3.0, 6.0)])
        assert [ln.index for ln in mapped.lines] == [0, 2]
        assert mapped.lines[1].start_s == pytest.approx(4.0)  # 7 - 3
        assert mapped.lines[1].end_s == pytest.approx(6.0)  # 9 - 3

    def test_multiple_cuts_shift_later_lines(self) -> None:
        alignment = self._alignment((0, 0.0, 2.0), (1, 8.0, 10.0))
        mapped = map_alignment(alignment, [(3.0, 6.0), (6.5, 7.5)])
        # Line 1: two cuts before it → shift by 3 + 1 = 4s.
        assert mapped.lines[1].start_s == pytest.approx(4.0)
        assert mapped.lines[1].end_s == pytest.approx(6.0)

    def test_unmatched_lines_stay_dropped(self) -> None:
        alignment = Alignment(
            lines=[LineAlignment(index=0, start_s=-1.0, end_s=-1.0, confidence=0.0)],
            transcriber="fuzzy",
        )
        mapped = map_alignment(alignment, [])
        assert mapped.lines == []
        assert mapped.transcriber == "fuzzy"
