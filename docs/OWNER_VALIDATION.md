# OWNER VALIDATION — what to run on the real Windows machine

This project was developed in a Linux sandbox (no camera, no GPU, no Windows,
no real lesson footage). CI proves tests pass on `windows-latest`, but per
the master prompt **no production-readiness claim is made**. This checklist
is the path to real validation. Nothing here uploads footage anywhere.

## 1. Install (two options)

**A. CI-built artifact (recommended first):**
Download the `OpenClipStudio-windows-x64` artifact from the latest green
`ci` run on `main` (Actions → latest successful run → Artifacts). Unzip,
run `OpenClipStudio\OpenClipStudio.exe`. SmartScreen will warn (unsigned
build — no money is spent on certificates); choose *More info → Run anyway*.
Check `sha256.txt` against your download.

**B. From source:**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[gui,asr,gaze]"
openclip-gui
```

## 2. Real end-to-end lesson edit (the core claim to validate)

1. Record a 2–5 min real talking-head lesson (any camera/phone). You do NOT
   need to upload it anywhere — everything below is local.
2. Write the lesson script in the app (bilingual blocks:
   `English sentence :: 支持语言句子`).
3. Analyze with whisper (the `asr` extra; the `tiny`/`small` model downloads
   once on first use, offline afterwards) and review the proposed cuts.
4. Export MP4. Then report:
   - Did alignment match all sentences (check the per-line list)?
   - Were cuts correct (any speech wrongly cut / mistakes kept)?
   - Do captions stay in sync after cuts? Are both languages rendered?
   - Export time for your video length and PC specs.
5. Repeat a short clip with the audio presets and note subjective quality.

## 3. Eye-contact correction on real footage

Follow `docs/EYE_CONTACT_RESEARCH.md` §5 (record off-side-look vs control
clip, `gaze-analyze` → `gaze-apply --strength 0.8` → re-analyze → watch for
blink smearing and tracking loss). Paste the numbers and your subjective
verdict back into that document's results section.

## 4. Things known to be untested on real hardware (explicit)

- QMediaPlayer preview on Windows (QtMultimedia/WMF) — playback UI works
  offscreen but real-decode behavior needs eyes on it.
- Whisper accuracy on your mic/accent/room — expected good for English,
  verify the support language if you speak it in lessons.
- Haar-based face detection on your camera framing/angle.
- PyInstaller GUI exe on a clean Windows machine (the CI job smoke-tests
  the CLI; the GUI needs a real display).
- Long lessons (>30 min): memory and progress behavior.

## 5. When satisfied

- Merge PRs #1–#4 (or request changes first).
- Tag `v0.1.0-rc1` from `main`.
- Only then may docs describe the app as "tested end-to-end by the owner on
  real footage" — with the specifics of what was tested.
