"""Video-rendering regression: overlay timing in the exported MP4.

Renders a gray lesson through the Production Director with accepted cuts,
then inspects exported frames (via ffmpeg BMP dumps, parsed with stdlib) for
the callout colour at export-timeline times. This is the end-to-end proof
that overlays use export-timeline timestamps: before cuts (unchanged),
clipped by cuts (start/end inside a cut), compressed between cuts, mapped
after cuts, and fully removed when inside a cut.

Timeline design (source 12 s, cuts C1=[1,2.5], C2=[5,7]+C2b=[6,8]
overlapping -> merged [5,8], C3=[9,9.5]; removed total 5 s -> export 7 s):

  overlay   source span   expected export span   frame probe
  C         (0.2, 0.8)    (0.2, 0.8)   before cuts          0.5 -> orange
  D         (1.5, 3.0)    (1.0, 1.5)    starts inside C1     1.2 -> orange
  A         (3.0, 4.0)    (1.5, 2.5)    between C1 and C2    2.0 -> orange
  E         (4.5, 5.5)    (3.0, 3.5)    ends inside C2       3.2 -> orange
  B         (6.2, 6.6)    REMOVED       inside C2            6.4 -> no orange
  F         (9.6, 10.0)   (4.6, 5.0)    after all cuts       4.8 -> orange

Anti-regression sentinels (a source-timestamp bug would show orange where
the correct render has none): 3.8 (raw A window 3-4), 5.2 (raw E solid until
its fade-out; correct F has faded by 5.05), 6.4 (raw B window).
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import pytest

from openclip.director.executor import CALLOUT_RGB, render_edl
from openclip.director.models import Decision, EditDecisionList
from openclip.engine.ffmpeg import ffmpeg_exe, probe
from openclip.engine.project import Project
from openclip.engine.render import RenderOptions

ORANGE_PRESENT = 30  # solid callout-glyph pixels at a frame with an overlay
ORANGE_ABSENT = 10  # antialiasing noise floor; no text -> effectively 0


def _gray_media(tmp_path: Path) -> Path:
    """Flat-gray 12 s clip: any callout-colour pixel must come from libass."""
    out = tmp_path / "gray.mp4"
    subprocess.run(
        [
            ffmpeg_exe(),
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x505050:size=320x180:rate=15:duration=12",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
        timeout=300,
        capture_output=True,
    )
    return out


def _frame_rgb_rows(bmp_path: Path) -> list[list[tuple[int, int, int]]]:
    """Decode a 24bpp BMP into top-down RGB rows (stdlib only)."""
    data = bmp_path.read_bytes()
    if data[:2] != b"BM":
        raise AssertionError(f"not a BMP: {bmp_path}")
    offset = struct.unpack_from("<I", data, 10)[0]
    width = struct.unpack_from("<i", data, 18)[0]
    height = struct.unpack_from("<i", data, 22)[0]
    bpp = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]
    if bpp != 24 or compression != 0:
        raise AssertionError(f"unexpected BMP format: bpp={bpp} comp={compression}")
    bottom_up = height > 0
    h = abs(height)
    row_size = ((width * 3 + 3) // 4) * 4
    rows: list[list[tuple[int, int, int]]] = []
    for r in range(h):
        start = offset + r * row_size
        row: list[tuple[int, int, int]] = []
        for c in range(width):
            b, g, rr = data[start + c * 3 : start + c * 3 + 3]
            row.append((rr, g, b))
        rows.append(row)
    return rows if bottom_up else rows[::-1]


def _callout_pixels(media: Path, t: float, tmp: Path) -> int:
    bmp = tmp / f"frame_{t:.2f}.bmp"
    if bmp.exists():
        bmp.unlink()
    subprocess.run(
        [
            ffmpeg_exe(),
            "-hide_banner",
            "-nostdin",
            "-y",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(media),
            "-frames:v",
            "1",
            "-pix_fmt",
            "bgr24",
            str(bmp),
        ],
        check=True,
        timeout=120,
        capture_output=True,
    )
    if not bmp.exists():
        raise AssertionError(
            f"no frame at t={t}s in {media} (seek past end of file?)"
        )
    cr, cg, cb = CALLOUT_RGB
    count = 0
    for row in _frame_rgb_rows(bmp):
        for r, g, b in row:
            if abs(r - cr) <= 70 and abs(g - cg) <= 70 and abs(b - cb) <= 90:
                count += 1
    return count


def _timing_edl(source: Path) -> EditDecisionList:
    def cut(i: int, s: float, e: float) -> Decision:
        return Decision(
            id=f"cut-{i}",
            kind="cut",
            source_span=(s, e),
            rationale="regression fixture",
            status="accepted",
        )

    def overlay(i: int, term: str, s: float, e: float) -> Decision:
        return Decision(
            id=f"callout-{i}",
            kind="callout",
            source_line=i,
            source_span=(s, e),
            params={"term": term},
            rationale="regression fixture",
            status="accepted",
        )

    return EditDecisionList(
        project_source=str(source),
        decisions=[
            cut(1, 1.0, 2.5),
            cut(2, 5.0, 7.0),
            cut(3, 6.0, 8.0),  # overlaps cut-2: union [5,8] = 3 s
            cut(4, 9.0, 9.5),
            overlay(10, "OCCLUDE_C", 0.2, 0.8),
            overlay(11, "OCCLUDE_D", 1.5, 3.0),
            overlay(12, "OCCLUDE_A", 3.0, 4.0),
            overlay(13, "OCCLUDE_E", 4.5, 5.5),
            overlay(14, "OCCLUDE_B", 6.2, 6.6),
            overlay(15, "OCCLUDE_F", 9.6, 10.0),
        ],
    )


@pytest.fixture(scope="module")
def timed_export(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("timing")
    media = _gray_media(tmp)
    project = Project.create(source=media, script_text="")
    out = tmp / "directed.mp4"
    render_edl(
        project,
        probe(media),
        _timing_edl(media),
        out,
        RenderOptions(burn_captions=False, audio_preset="none"),
        require_review=False,
    )
    return out, tmp


def test_export_duration_reflects_normalized_cuts(timed_export: tuple[Path, Path]) -> None:
    out, _tmp = timed_export
    info = probe(out)
    # 12 s - (1.5 + 3.0 + 0.5) = 7.0 s; the overlapping pair counts once.
    assert abs(info.duration_s - 7.0) < 0.35
    assert info.vcodec == "h264"
    assert info.pix_fmt == "yuv420p"


@pytest.mark.parametrize(
    ("t", "expect_orange"),
    [
        (0.5, True),    # C: before all cuts (unmapped)
        (1.2, True),    # D: starts inside C1, clipped to (2.5,3.0) -> (1.0,1.5)
        (2.0, True),    # A: between cuts -> (1.5,2.5)
        (3.2, True),    # E: ends inside C2, clipped -> (3.0,3.5)
        (4.8, True),    # F: after all cuts -> (4.6,5.0)
        (3.8, False),   # sentinel: raw-A window (3-4) must be empty
        (5.2, False),   # sentinel: raw-E solid window; correct-F already faded
        (6.4, False),   # sentinel: B fully inside merged cut; raw window empty
    ],
)
def test_overlay_timing_in_exported_mp4(
    timed_export: tuple[Path, Path], t: float, expect_orange: bool
) -> None:
    out, tmp = timed_export
    pixels = _callout_pixels(out, t, tmp)
    if expect_orange:
        assert pixels >= ORANGE_PRESENT, (
            f"callout missing at export t={t}s ({pixels} px) — overlay timing "
            "does not match the export timeline"
        )
    else:
        assert pixels < ORANGE_ABSENT, (
            f"unexpected callout pixels at export t={t}s ({pixels} px) — an "
            "overlay rendered at source-timeline time or leaked through a cut"
        )
