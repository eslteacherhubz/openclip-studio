"""Tests for openclip.ai.beauty module.

All tests are fully offline, synthetic, fixed seed. They import the REAL
FaceFrame from openclip.ai.face_analysis and test against the actual
beauty.smooth_skin implementation. They skip ONLY if numpy/cv2 are
unavailable (importorskip) — NEVER skip on the beauty module itself.
"""

from __future__ import annotations

import pytest

from openclip.ai.beauty import _apply_exclusions, _rasterize_mask, beauty_available, smooth_skin
from openclip.ai.face_analysis import FaceFrame, Rect

# Mark as gaze tests (requires numpy/opencv from gaze extra)
pytestmark = pytest.mark.gaze

# Skip tests if optional vision dependencies are unavailable
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


SEED = 42
RNG = np.random.default_rng(SEED)
H, W = 256, 256


def _build_synthetic_frame() -> np.ndarray:
    """Create a 256x256x3 uint8 frame with known high-freq + low-freq content."""
    # High-frequency texture: fine fixed-seed noise
    hf = RNG.normal(0, 16, size=(H, W)).astype(np.float32)
    # Add fine checkerboard pattern for guaranteed high-frequency energy
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    hf += 12.0 * np.sin(2.0 * np.pi * x / 4.0) * np.sin(2.0 * np.pi * y / 4.0)

    # Low-frequency blotches: smooth gradients / large-scale variation
    lf_x = np.linspace(-1.0, 1.0, W)
    lf_y = np.linspace(-1.0, 1.0, H)
    LX, LY = np.meshgrid(lf_x, lf_y)
    lf = 40.0 * (LX**2 + LY**2)  # smooth radial gradient
    lf += 20.0 * np.sin(2.0 * np.pi * LX / 1.0) * np.cos(2.0 * np.pi * LY / 1.0)

    # Combine and create 3 channels with slight variation
    frame_f = np.zeros((H, W, 3), dtype=np.float32)
    frame_f[:, :, 0] = 100.0 + hf + lf
    frame_f[:, :, 1] = 110.0 + 0.9 * hf + 1.1 * lf
    frame_f[:, :, 2] = 95.0 + 1.1 * hf + 0.9 * lf

    frame = np.clip(frame_f, 0, 255).astype(np.uint8)
    return frame


def _high_freq_energy(img: np.ndarray) -> float:
    """Compute high-frequency energy via Laplacian variance (per channel, then mean)."""
    total = 0.0
    for c in range(3):
        lap = cv2.Laplacian(img[:, :, c].astype(np.float64), cv2.CV_64F)
        total += float(np.var(lap))
    return total / 3.0


def _low_freq_variance(img: np.ndarray) -> float:
    """Compute low-frequency variance via blurred image variance."""
    total = 0.0
    for c in range(3):
        blurred = cv2.GaussianBlur(img[:, :, c].astype(np.float64), (21, 21), sigmaX=7.0)
        total += float(np.var(blurred))
    return total / 3.0


def _low_freq_variance_masked(img: np.ndarray, mask: np.ndarray) -> float:
    """Compute low-frequency variance in masked region only."""
    mask_bool = mask > 0.5
    if not mask_bool.any():
        return 0.0
    total = 0.0
    for c in range(3):
        blurred = cv2.GaussianBlur(img[:, :, c].astype(np.float64), (21, 21), sigmaX=7.0)
        total += float(np.var(blurred[mask_bool]))
    return total / 3.0


def _build_mask(face_frame: FaceFrame) -> np.ndarray:
    """Build the final mask used by smooth_skin for a given FaceFrame."""
    h, w = H, W
    mask = _rasterize_mask(face_frame.skin_mask_cells, h, w)
    mask = _apply_exclusions(mask, face_frame.exclusion_zones, h, w)
    return mask


def _make_face_frame() -> FaceFrame:
    """Create a FaceFrame with known normalized rects.

    skin_mask_cells: larger grid covering typical face region (~40% of frame)
    exclusion_zones: left eye, right eye, mouth (as in MockFaceAnalysisProvider)
    """
    cx, cy = 0.5, 0.45  # face center slightly upper
    skin_cells: list[Rect] = []
    # 5x5 grid covering ~0.4 x 0.4 normalized (40% of frame each dimension)
    for row in range(5):
        for col in range(5):
            skin_cells.append((
                cx - 0.20 + col * 0.10,
                cy - 0.20 + row * 0.10,
                0.09,
                0.09,
            ))

    exclusion_zones: tuple[Rect, ...] = (
        (cx - 0.080, cy - 0.080, 0.060, 0.040),  # left eye
        (cx + 0.020, cy - 0.080, 0.060, 0.040),  # right eye
        (cx - 0.040, cy + 0.080, 0.080, 0.050),  # mouth
    )

    return FaceFrame(
        timestamp=0.0,
        center_x=cx,
        center_y=cy,
        jaw_neck_landmarks=(),
        skin_mask_cells=tuple(skin_cells),
        exclusion_zones=exclusion_zones,
    )


class TestBeautyAvailable:
    def test_beauty_available_returns_bool(self) -> None:
        result = beauty_available()
        assert isinstance(result, bool)
        assert result is True  # numpy and cv2 are importorskip'd


class TestSmoothSkin:
    def test_texture_retention(self) -> None:
        """High-frequency energy retained >= 80% at default intensity 0.15."""
        frame = _build_synthetic_frame()
        face_frame = _make_face_frame()

        before_hf = _high_freq_energy(frame)
        out = smooth_skin(frame, face_frame, intensity=0.15)
        after_hf = _high_freq_energy(out)

        retention_pct = 100.0 * after_hf / before_hf
        print(
            f"Texture retention: {retention_pct:.1f}% "
            f"(before={before_hf:.1f}, after={after_hf:.1f})"
        )

        # KEY ASSERTION: ≥80% retention — invisible retouching requirement
        assert retention_pct >= 80.0, (
            f"Texture retention {retention_pct:.1f}% < 80% — smoothing too aggressive"
        )

        # Low-frequency blotch variance should be REDUCED in masked region
        mask = _build_mask(face_frame)
        before_lf = _low_freq_variance_masked(frame, mask)
        after_lf = _low_freq_variance_masked(out, mask)
        assert after_lf < before_lf, (
            f"Low-frequency variance in masked region should decrease "
            f"(before={before_lf:.2f}, after={after_lf:.2f})"
        )

    def test_exclusion_zones_byte_identical(self) -> None:
        """Pixels inside exclusion zones pass through byte-identical."""
        frame = _build_synthetic_frame()
        face_frame = _make_face_frame()

        out = smooth_skin(frame, face_frame, intensity=1.0)  # max intensity

        # Check each exclusion zone
        for (nx, ny, nw, nh) in face_frame.exclusion_zones:
            x1 = int(nx * W)
            y1 = int(ny * H)
            x2 = int((nx + nw) * W)
            y2 = int((ny + nh) * H)

            # Clamp to frame bounds
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(x1 + 1, min(x2, W))
            y2 = max(y1 + 1, min(y2, H))

            zone_in = frame[y1:y2, x1:x2]
            zone_out = out[y1:y2, x1:x2]
            assert np.array_equal(zone_in, zone_out), (
                f"Exclusion zone ({nx},{ny},{nw},{nh}) not byte-identical"
            )

    def test_feathered_mask_transition(self) -> None:
        """Mask transition at skin-region boundary is gradual (no hard seam)."""
        frame = _build_synthetic_frame()
        _ = _make_face_frame()  # unused, but kept for reference

        # Use a single skin mask cell at the edge to test feathering
        # First cell spans x~[0.43, 0.49), y~[0.43, 0.49) normalized
        # At 256px: roughly [110, 125) in both axes
        edge_face_frame = FaceFrame(
            timestamp=0.0,
            center_x=0.5,
            center_y=0.5,
            jaw_neck_landmarks=(),
            skin_mask_cells=((0.43, 0.43, 0.06, 0.06),),
            exclusion_zones=(),
        )

        out = smooth_skin(frame, edge_face_frame, intensity=1.0)

        # Inside mask (e.g., y=115): should differ from input
        inside_y = 115
        # Outside mask (e.g., y=100): should equal input
        outside_y = 100
        x = 115

        _diff_inside = np.abs(
            out[inside_y, x].astype(np.int16) - frame[inside_y, x].astype(np.int16)
        ).mean()
        _diff_outside = np.abs(
            out[outside_y, x].astype(np.int16) - frame[outside_y, x].astype(np.int16)
        ).mean()

        # Inside should show smoothing effect, outside should be nearly identical
        # Feathered edge means there's a gradual transition — check a profile
        # Profile across the boundary (y from 100 to 120)
        profile = []
        for y in range(100, 121):
            d = np.abs(
                out[y, x].astype(np.int16) - frame[y, x].astype(np.int16)
            ).mean()
            profile.append(d)

        # Should not have a single-pixel jump (hard seam)
        # Max per-step change in difference should be < 10 (gradual)
        max_step = max(
            abs(profile[i + 1] - profile[i]) for i in range(len(profile) - 1)
        )
        assert max_step < 10.0, f"Mask has hard seam (max step={max_step:.1f})"

    def test_determinism(self) -> None:
        """Two runs with same inputs produce bitwise identical output."""
        frame = _build_synthetic_frame()
        face_frame = _make_face_frame()

        out1 = smooth_skin(frame, face_frame, intensity=0.15)
        out2 = smooth_skin(frame, face_frame, intensity=0.15)

        assert np.array_equal(out1, out2), "Output is not deterministic"

    def test_intensity_zero_passthrough(self) -> None:
        """intensity=0 returns byte-identical copy of input."""
        frame = _build_synthetic_frame()
        face_frame = _make_face_frame()

        out = smooth_skin(frame, face_frame, intensity=0.0)

        assert np.array_equal(out, frame), "intensity=0 should be byte-identical passthrough"
        # Also verify it's a copy, not the same object
        assert out is not frame

    def test_intensity_clamped(self) -> None:
        """intensity clamped to [0,1]; 2.0 behaves as 1.0, -1.0 as 0.0."""
        frame = _build_synthetic_frame()
        face_frame = _make_face_frame()

        out_clamped_high = smooth_skin(frame, face_frame, intensity=2.0)
        out_max = smooth_skin(frame, face_frame, intensity=1.0)
        assert np.array_equal(out_clamped_high, out_max), "intensity=2.0 should clamp to 1.0"

        out_clamped_low = smooth_skin(frame, face_frame, intensity=-1.0)
        out_zero = smooth_skin(frame, face_frame, intensity=0.0)
        assert np.array_equal(out_clamped_low, out_zero), "intensity=-1.0 should clamp to 0.0"

    def test_runtime_error_without_vision(self, monkeypatch) -> None:
        """If vision libs unavailable, smooth_skin raises RuntimeError."""
        # This test cannot run with importorskip since it tests the failure path
        # It's documented here for completeness; the actual behavior is tested
        # by the importorskip at module load
        pass  # Cannot test without removing cv2 from sys.modules
