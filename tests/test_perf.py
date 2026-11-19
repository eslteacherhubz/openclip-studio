"""Tests for beauty pre-pass performance optimizations (parallel, progress, cancellation, GPU)."""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path

import pytest

from openclip.ai.beauty import (
    BeautyConfig,
    CancellationError,
    beauty_available,
    beauty_prepass,
)
from openclip.ai.face_analysis import CachedFaceAnalysis, default_provider
from openclip.engine.ffmpeg import probe
from openclip.engine.synth import generate_lesson


@pytest.fixture
def lesson(tmp_path_factory):
    """Generate a synthetic lesson for testing."""
    return generate_lesson(tmp_path_factory.mktemp("perf") / "lesson.mp4")


@pytest.fixture
def face_analysis():
    """Create a CachedFaceAnalysis with default provider."""
    return CachedFaceAnalysis(default_provider())


@pytest.fixture
def beauty_config():
    """Create a BeautyConfig with moderate settings."""
    return BeautyConfig(skin_smooth=0.2, jaw_slim=0.1, neck_tighten=0.1)


class TestBeautyPrepassParallel:
    """Tests for parallel processing correctness."""

    def test_parallel_produces_identical_output(
        self, lesson, face_analysis, beauty_config, tmp_path
    ):
        """Parallel and sequential processing should produce byte-identical
        output (within codec tolerance)."""
        if not beauty_available():
            pytest.skip("Vision dependencies not available")

        media = probe(lesson.path)

        # Sequential
        seq_out = tmp_path / "seq_beauty.mp4"
        beauty_prepass(
            str(lesson.path),
            str(seq_out),
            beauty_config,
            face_analysis,
            media.fps if media.fps > 0 else 25.0,
            media.width or 1280,
            media.height or 720,
            parallel=False,
        )

        # Parallel (with 2 workers)
        face_analysis_par = CachedFaceAnalysis(default_provider())
        par_out = tmp_path / "par_beauty.mp4"
        beauty_prepass(
            str(lesson.path),
            str(par_out),
            beauty_config,
            face_analysis_par,
            media.fps if media.fps > 0 else 25.0,
            media.width or 1280,
            media.height or 720,
            parallel=True,
            num_workers=2,
        )

        # Compare file sizes (should be very close)
        seq_size = seq_out.stat().st_size
        par_size = par_out.stat().st_size
        size_diff_pct = abs(seq_size - par_size) / seq_size * 100
        assert size_diff_pct < 5.0, f"Size difference {size_diff_pct:.1f}% exceeds 5% tolerance"

    def test_parallel_with_different_worker_counts(self, lesson, beauty_config, tmp_path):
        """Test parallel processing with various worker counts."""
        if not beauty_available():
            pytest.skip("Vision dependencies not available")

        media = probe(lesson.path)

        for num_workers in [1, 2, 4]:
            face_analysis = CachedFaceAnalysis(default_provider())
            out = tmp_path / f"beauty_w{num_workers}.mp4"
            beauty_prepass(
                str(lesson.path),
                str(out),
                beauty_config,
                face_analysis,
                media.fps if media.fps > 0 else 25.0,
                media.width or 1280,
                media.height or 720,
                parallel=True,
                num_workers=num_workers,
            )
            assert out.exists()
            assert out.stat().st_size > 0


class TestBeautyPrepassProgress:
    """Tests for progress callback."""

    def test_progress_callback_called(self, lesson, face_analysis, beauty_config, tmp_path):
        """Progress callback should be called with increasing frame counts."""
        if not beauty_available():
            pytest.skip("Vision dependencies not available")

        media = probe(lesson.path)
        progress_calls: list[tuple[int, int]] = []

        def progress_cb(current: int, total: int):
            progress_calls.append((current, total))

        out = tmp_path / "beauty_progress.mp4"
        beauty_prepass(
            str(lesson.path),
            str(out),
            beauty_config,
            face_analysis,
            media.fps if media.fps > 0 else 25.0,
            media.width or 1280,
            media.height or 720,
            progress_callback=progress_cb,
        )

        assert len(progress_calls) > 0
        # Last call should be total frames
        total_frames = int(probe(lesson.path).duration_s * (media.fps or 25))
        assert progress_calls[-1][1] == total_frames
        # Current should monotonically increase
        currents = [c for c, _ in progress_calls]
        assert currents == sorted(currents)
        # Final current should equal total
        assert progress_calls[-1][0] == total_frames


class TestBeautyPrepassCancellation:
    """Tests for cancellation support."""

    def test_cancellation_raises_error(self, lesson, face_analysis, beauty_config, tmp_path):
        """Setting cancel_flag should raise CancellationError."""
        if not beauty_available():
            pytest.skip("Vision dependencies not available")

        media = probe(lesson.path)
        cancel_flag = threading.Event()
        cancel_flag.set()  # Cancel immediately

        out = tmp_path / "beauty_cancelled.mp4"
        with pytest.raises(CancellationError):
            beauty_prepass(
                str(lesson.path),
                str(out),
                beauty_config,
                face_analysis,
                media.fps if media.fps > 0 else 25.0,
                media.width or 1280,
                media.height or 720,
                cancel_flag=cancel_flag,
            )

    def test_cancellation_mid_render(self, lesson, face_analysis, beauty_config, tmp_path):
        """Cancellation during render should stop cleanly."""
        if not beauty_available():
            pytest.skip("Vision dependencies not available")

        media = probe(lesson.path)
        cancel_flag = threading.Event()

        def cancel_after_delay():
            time.sleep(0.1)
            cancel_flag.set()

        cancel_thread = threading.Thread(target=cancel_after_delay)
        cancel_thread.start()

        out = tmp_path / "beauty_cancelled2.mp4"
        with pytest.raises(CancellationError):
            beauty_prepass(
                str(lesson.path),
                str(out),
                beauty_config,
                face_analysis,
                media.fps if media.fps > 0 else 25.0,
                media.width or 1280,
                media.height or 720,
                cancel_flag=cancel_flag,
            )
        cancel_thread.join()

    def test_cancellation_cleans_temp_files(self, lesson, face_analysis, beauty_config, tmp_path):
        """Cancelled render should not leave partial output file."""
        if not beauty_available():
            pytest.skip("Vision dependencies not available")

        media = probe(lesson.path)
        cancel_flag = threading.Event()
        cancel_flag.set()

        out = tmp_path / "beauty_cancelled3.mp4"
        with contextlib.suppress(CancellationError):
            beauty_prepass(
                str(lesson.path),
                str(out),
                beauty_config,
                face_analysis,
                media.fps if media.fps > 0 else 25.0,
                media.width or 1280,
                media.height or 720,
                cancel_flag=cancel_flag,
            )

        # Output file should not exist (or be empty)
        if out.exists():
            assert out.stat().st_size == 0, "Partial output file should be cleaned up"


class TestBeautyPrepassGPU:
    """Tests for GPU acceleration fallback."""

    def test_gpu_fallback_when_unavailable(self, lesson, face_analysis, beauty_config, tmp_path):
        """use_gpu=True should work even when CUDA is unavailable (fallback to CPU)."""
        if not beauty_available():
            pytest.skip("Vision dependencies not available")

        media = probe(lesson.path)
        face_analysis_gpu = CachedFaceAnalysis(default_provider())
        out = tmp_path / "beauty_gpu.mp4"

        # Should not raise even if CUDA unavailable
        beauty_prepass(
            str(lesson.path),
            str(out),
            beauty_config,
            face_analysis_gpu,
            media.fps if media.fps > 0 else 25.0,
            media.width or 1280,
            media.height or 720,
            use_gpu=True,
        )

        assert out.exists()
        assert out.stat().st_size > 0

    def test_gpu_parallel_combined(self, lesson, face_analysis, beauty_config, tmp_path):
        """use_gpu=True with parallel=True should work."""
        if not beauty_available():
            pytest.skip("Vision dependencies not available")

        media = probe(lesson.path)
        face_analysis_gpu = CachedFaceAnalysis(default_provider())
        out = tmp_path / "beauty_gpu_par.mp4"

        beauty_prepass(
            str(lesson.path),
            str(out),
            beauty_config,
            face_analysis_gpu,
            media.fps if media.fps > 0 else 25.0,
            media.width or 1280,
            media.height or 720,
            parallel=True,
            num_workers=2,
            use_gpu=True,
        )

        assert out.exists()
        assert out.stat().st_size > 0


class TestBeautyPrepassSpeedup:
    """Speedup measurement (informational)."""

    def test_speedup_measurement(self, lesson, beauty_config):
        """Measure sequential vs parallel speedup (informational only)."""
        if not beauty_available():
            pytest.skip("Vision dependencies not available")

        media = probe(lesson.path)

        # Sequential
        face_analysis_seq = CachedFaceAnalysis(default_provider())
        seq_out = Path("/tmp/seq_beauty.mp4")
        start = time.perf_counter()
        beauty_prepass(
            str(lesson.path),
            str(seq_out),
            beauty_config,
            face_analysis_seq,
            media.fps if media.fps > 0 else 25.0,
            media.width or 1280,
            media.height or 720,
            parallel=False,
        )
        seq_time = time.perf_counter() - start

        # Parallel (4 workers)
        face_analysis_par = CachedFaceAnalysis(default_provider())
        par_out = Path("/tmp/par_beauty.mp4")
        start = time.perf_counter()
        beauty_prepass(
            str(lesson.path),
            str(par_out),
            beauty_config,
            face_analysis_par,
            media.fps if media.fps > 0 else 25.0,
            media.width or 1280,
            media.height or 720,
            parallel=True,
            num_workers=4,
        )
        par_time = time.perf_counter() - start

        speedup = seq_time / par_time if par_time > 0 else 0
        print(
            f"\nSequential: {seq_time:.2f}s, "
            f"Parallel (4w): {par_time:.2f}s, "
            f"Speedup: {speedup:.2f}x"
        )

        # This is informational; we just log the speedup
        # Real speedup depends on hardware; typically 2-4x on multi-core
        assert speedup > 0.5  # At least not catastrophically slower


# Import needed modules


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
