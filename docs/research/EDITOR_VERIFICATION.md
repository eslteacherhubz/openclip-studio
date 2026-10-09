# Research: verify a suitable existing open-source Windows video editor

Priority 1 of the master prompt. Question: is there an existing open-source
video editor for Windows that can serve as the foundation for Eterna OpenClip
Studio, or as a complement to it?

Date of verification: 2026-10-09. Evidence gathered live from project sites and
the GitHub API.

## Candidates verified

| Editor | License (verified) | Windows builds (verified) | Activity (verified) | Render core | Fit for teaching pipeline |
|---|---|---|---|---|---|
| **Shotcut** | GPL-3.0 (GitHub API `license` field: gpl-3.0) | Yes — Windows 10/11 x86 **and** ARM, installer + portable zip (shotcut.org/download, v26.9.27, 2026-09-27 release) | Very active (repo pushed 2026-10-06; 15.4k stars) | MLT Framework | Excellent general NLE; no script alignment, no bilingual caption pipeline, no gaze correction; Python embedding not practical (C++/Qt; MLT python bindings not packaged for Windows/PyPI) |
| **Kdenlive** | GPL-3.0 | Yes (Windows installer via KDE binary factory) | Very active | MLT Framework | Same as Shotcut: no teaching pipeline; C++/Qt |
| **OpenShot** | GPL-3.0 (repo topics `gplv3`; GitHub license field shows "Other" due to multi-file licensing) | Yes | Active (pushed 2026-10-08) | libopenshot (C++ w/ Python bindings) | Python-native, but bindings are fragile on Windows packaging (known instability reports); no teaching pipeline |
| **Olive** | GPL-3.0 | Yes | **Stale** (last push 2024-12-05) | own engine | Not viable — stalled project |
| **LosslessCut** | GPL-2.0 | Yes (7zip, Windows 7+ dropped after v3.50) | Very active | FFmpeg | Lossless trimming only; no compositing/captions; great complement for rough cuts |

## Verdict

1. **A suitable open-source Windows editor exists — Shotcut** (GPL-3.0,
   actively maintained, official Windows x64/ARM builds, portable zip, uses the
   same FFmpeg/MLT primitives we need). Kdenlive and OpenShot are viable
   alternatives of the same class. This satisfies priority 1: we are not
   operating in a vacuum, and we adopt their proven architectural choice —
   FFmpeg as render engine.
2. **None of them implements the bilingual teaching pipeline** (script
   alignment, intelligent cuts from script divergence, animated bilingual
   burned-in captions, audio enhancement presets, gaze correction). Embedding
   this pipeline into a C++/Qt NLE fork costs more (in the 72-hour budget)
   than building a focused application around FFmpeg, with Shotcut/Kdenlive
   remaining the tools of choice for manual polish of OpenClip exports.
3. **Decision (D1 in MASTER prompt):** build OpenClip Studio as a focused,
   Python-based, script-driven editor (PySide6 GUI + FFmpeg engine), and
   interoperate with the verified editors by exporting standard cut lists
   (CSV/EDL-style JSON) rather than forking them. We document Shotcut as the
   recommended companion tool in the README.

## Evidence links

- Shotcut download page (Windows sections, version 26.9.27, free/open-source
  pledge): https://shotcut.org/download/ — fetched 2026-10-09.
- GitHub API `mltframework/shotcut`: `license: gpl-3.0`, `pushed_at:
  2026-10-06T19:36:06Z`, description "cross-platform (Qt), open-source (GPLv3)
  video editor".
- GitHub API `OpenShot/openshot-qt`: topics include `gplv3`, Python,
  `pushed_at: 2026-10-08`.
- GitHub API `olive-editor/olive`: `license: gpl-3.0`, `pushed_at:
  2024-12-05` (stale).
- LosslessCut repo page: "GPL-2.0 license", Windows downloads, active
  maintenance, FFmpeg-based lossless operations.
