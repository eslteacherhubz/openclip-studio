# Eterna OpenClip Studio

A bilingual teaching-video editor for language educators. Record a lesson once;
OpenClip Studio aligns your script to your voice, proposes intelligent cuts
(mistakes, retakes, silences, filler words), burns in bilingual animated
captions, enhances the audio, optionally corrects off-camera gaze, and exports a
finished MP4 — all locally, on your own machine.

> **Privacy-first**: nothing leaves your computer. All analysis (speech
> recognition, alignment, gaze estimation) runs locally with open-source
> components. No cloud APIs, no uploads.

## Why

ESL teachers record talking-head lessons in two languages (target language +
learner language). Turning raw recordings into publishable lessons today means
hours of manual cutting, captioning and audio cleanup in a general-purpose
editor. OpenClip Studio automates the parts that are mechanical, and leaves the
creative decisions to the teacher.

## Status

Pre-alpha, under active development against the 72-hour roadmap in
[docs/ROADMAP.md](docs/ROADMAP.md). Live progress log:
[docs/CHECKPOINT.md](docs/CHECKPOINT.md). Not production-ready; see
[docs/ACCEPTANCE.md](docs/ACCEPTANCE.md) for the criteria that must be met
before any such claim.

## Features (target)

- **Script alignment** — paste your lesson script; local speech recognition
  (optional, faster-whisper) with fuzzy word-level alignment maps script
  sentences to the timeline.
- **Intelligent cuts** — detect and propose removal of off-script speech,
  silences and filler words; every cut is reviewable and reversible.
- **Bilingual composition** — animated burned-in captions in two languages
  (primary + support), fully configurable position, fonts, animation style.
- **Audio enhancement** — noise reduction, loudness normalization (EBU R128),
  gentle EQ and gating via FFmpeg filters.
- **Eye-contact correction** (experimental) — warp-based gaze redirection for
  talking-head footage; see [docs/EYE_CONTACT_RESEARCH.md](docs/EYE_CONTACT_RESEARCH.md).
- **MP4 export** — H.264 + AAC, widely compatible (yuv420p).

## Install (developers)

```bash
python -m pip install -e ".[dev]"
pytest
```

Requires Python 3.11+. FFmpeg/ffprobe are located from `PATH` or bundled via
`imageio-ffmpeg`.

## Usage

```bash
openclip probe lesson.mp4            # inspect media
openclip synth --out demo.mp4        # generate synthetic demo lesson (for testing)
openclip render project.json -o out.mp4
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE). We rely on and recommend these
excellent open-source editors, verified in
[docs/research/EDITOR_VERIFICATION.md](docs/research/EDITOR_VERIFICATION.md):
[Shotcut](https://shotcut.org), [Kdenlive](https://kdenlive.org),
[OpenShot](https://www.openshot.org), [LosslessCut](https://losslesscut.app).
