"""ASR integration test: real local whisper on real (synthesized) speech.

Requires the ``asr`` extra (faster-whisper) and espeak-ng for offline TTS;
skipped automatically when either is unavailable. Downloads the whisper
"tiny" model once (small, ~75 MB) — after that everything is offline.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.asr,
    pytest.mark.slow,
]


def _espeak_available() -> bool:
    return shutil.which("espeak-ng") is not None


def _whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="module")
def speech_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Real English speech synthesized offline with espeak-ng."""
    if not _espeak_available():
        pytest.skip("espeak-ng not installed")
    out = tmp_path_factory.mktemp("asr") / "speech.wav"
    text = (
        "Hello everyone, welcome to the lesson. "
        "Today we will learn five new words. "
        "Please listen and repeat after me."
    )
    subprocess.run(
        [
            "espeak-ng",
            "-v", "en-us",
            "-s", "140",
            "-w", str(out),
            text,
        ],
        check=True,
        timeout=120,
    )
    return out


def test_whisper_transcribes_espeak_speech(speech_wav: Path) -> None:
    if not _whisper_available():
        pytest.skip("faster-whisper (asr extra) not installed")
    from openclip.engine.align import WhisperTranscriber

    transcriber = WhisperTranscriber(model_size="tiny")
    hyps = transcriber.transcribe(speech_wav, language="en")
    assert len(hyps) > 5, f"expected words, got {len(hyps)}"
    text = " ".join(h.word.lower() for h in hyps)
    for expected in ("hello", "lesson", "words", "listen"):
        assert expected in text, f"whisper missed {expected!r} in: {text}"


def test_align_script_with_real_asr(speech_wav: Path) -> None:
    if not _whisper_available():
        pytest.skip("faster-whisper (asr extra) not installed")
    from openclip.engine.align import WhisperTranscriber, align_script
    from openclip.engine.script import parse_script

    script = parse_script(
        "Hello everyone, welcome to the lesson.\n"
        "\n"
        "Today we will learn five new words.\n"
        "\n"
        "Please listen and repeat after me."
    )
    hyps = WhisperTranscriber(model_size="tiny").transcribe(speech_wav, "en")
    alignment = align_script(script, hyps)
    matched = [ln for ln in alignment.lines if ln.confidence >= 0.55]
    assert len(matched) >= 2, f"too few aligned lines: {alignment.to_dict()}"
    # Monotonic, increasing times.
    starts = [ln.start_s for ln in matched]
    assert starts == sorted(starts)
