# EYE_CONTACT_RESEARCH — warp-based gaze redirection

Working document for Priority 2 of the master prompt: *investigate and
experimentally test realistic eye-contact correction on the target
hardware*. All numbers below are real measurements from this repository's
evaluation harness (`openclip/gaze/evaluate.py`, tests in
`tests/test_gaze.py`), reproducible with:

```bash
pip install ".[gaze]"
pytest -m gaze -v
python -m openclip.gaze.evaluate        # prints the full report as JSON
```

## 1. Problem and approach

Talking-head lessons are often recorded while the teacher watches their own
preview (or notes) rather than the camera. For language learners, perceived
eye contact matters. Commercial "eye contact" features (NVIDIA Broadcast /
Maxine) are proprietary and cloud/GPU-oriented; we need a **local, CPU-only,
GPL-compatible** approach.

**Approach: warp-based iris redirection** (the family of Reale et al. 2014,
*A Warp-Based Approach to Gaze Correction*; cf. Ganin et al. 2016 DeepWarp
for the learned variant). Per frame:

1. Locate the eye aperture (brightest elongated region in the eye patch) and
   the iris (blobby dark region after morphological cleanup).
2. Compute the displacement that moves the iris center toward the aperture
   center (scaled by `strength`).
3. Inverse-map the patch with a displacement field: full strength inside the
   iris, cosine falloff to zero at `reach = iris_radius * reach_factor`.
   Bicubic sub-pixel remap, feathered blend back into the frame.
4. Temporal EMA smoothing (alpha 0.35) of iris positions suppresses jitter.

No generative model, no training data, fully offline, ~150 frames/s on CPU
(see E4). Known trade-off (shared with all warp methods): at large offsets
the iris "runs out of sclera" and quality degrades — measured in E5.

## 2. Experimental rig (and what it does / does not prove)

The dev sandbox has **no camera, no GPU, and no private footage may be
uploaded** — so all quantitative experiments use the deterministic synthetic
eye rig (`openclip/gaze/rig.py`): parameterized photometric eye patches
(skin gradient, lash shadow, almond aperture, shaded sclera, fibered iris,
pupil, specular highlight) with exact ground-truth iris offsets.

- ✅ Proves: the algorithm's mathematics, parameter choices, failure modes,
  throughput, and measurable iris-offset reduction with no artifacts outside
  the eye — on **controlled synthetic input**.
- ❌ Does **not** prove: performance on real webcam footage (skin texture,
  glasses, motion blur, compression, off-axis heads), detection robustness on
  real faces (Haar-based `locate_in_frame` is implemented but unvalidated on
  real footage), or subjective naturalness. **No "works on real footage"
  claim is made anywhere in this repo.**

## 3. Experiments and results

Hardware for all measurements: the dev sandbox — Linux aarch64 (Debian 12),
8 vCPU (shared, throttled), CPU-only, OpenCV 4.14.0, numpy 2.4.6. Numbers
are reproducible via the harness (fixed rig seed), so **relative** results
are stable; absolute fps varies with machine load.

### E1 — Detection accuracy (rig, 12-trial offset grid: ±5/15/25% eye width horizontal, ±2/6/10% vertical)

| metric | result |
|---|---|
| miss rate | 1/12 = 8.3% (only at the extreme −25% horizontal offset, iris at the search boundary) |
| mean iris-localization error | **1.36 px** (iris diameter 34 px) |
| max error | 5.5 px |

Detector: dark-blob selection with morphological cleanup (close 7×7 → erode
5×5 → dilate 5×5×2) + aspect/compactness filters. Failures before this
pipeline (centroid-of-darkest-pixels) localized the lash shadow with 30 px
mean error — the cleanup is what makes it work.

### E2 — Correction effectiveness (strength 1.0, reach factor 1.35, grid ±5/15/25%)

| offset band | error before (% eye width) | error after | reduction |
|---|---|---|---|
| ±5% | 10.8% | 3.3% | **75.6%** |
| ±15% | 13.5% | 3.4% | **80.0%** |
| ±25% | 21.2% | 14.6% | 34.0% |
| grid mean (12 trials) | 14.6% | 6.4% | **65.8%** |

Acceptance target: ≥60% mean reduction at ≤25% offsets — **met** (65.8%).
The 25% band is where warp methods hit their structural limit (E5).

### E3 — Artifacts outside the eye region

PSNR between original and corrected, measured strictly outside a generous
ellipse around the eye: **99 dB** (i.e., numerically untouched; remap + feathered
blend leave non-eye pixels intact). Target ≥35 dB — **met with margin**.

### E4 — Throughput (THIS hardware, CPU only)

detect+warp on 200×100 rig patches: **147 frames/s** (120-frame benchmark).
For 640×360 real frames the per-frame cost is dominated by full-frame
detection; budget ~2–4× slower — still real-time-adjacent on this CPU, and
the target hardware (owner's Windows PC) is expected to be faster. Measured
only on the sandbox; owner should re-run on the real machine.

### E5 — Parameter studies

**E5a — Why tight reach matters.** Reach factor controls the displacement
field's radius. At ±15% offsets, strength 1.0:

| reach factor | error after | reduction | interpretation |
|---|---|---|---|
| 1.15 | 2.2% | 92.8% | iris slides inside static sclera |
| 1.35 | 2.3% | 92.8% | chosen default (radius-estimate margin) |
| 1.70 | 2.7% | 88.1% | still fine |
| 2.30 | 7.2% | **15.8%** | whole sclera pattern slides with the iris → *relative* gaze unchanged |

This was the pivotal experiment of the investigation: with a wide field,
the aperture's shading pattern moves together with the iris and the
perceived gaze does not improve (15.8% ≈ nothing). The fix is a field only
~1.35× the iris radius.

**E5b — Offset magnitude limits.** Reduction falls from 80% (±15%) to 34%
(±25%): beyond ~1 iris radius of offset, the iris collides with lid/corner
geometry and the inverse warp visibly stretches sclera. **Operational
consequence:** cap automatic correction at offsets ≤ ~15% eye width
(natural reading-teleprompter drift); beyond that, warp correction should
refuse or blend to partial strength, not produce artifacts. This is
documented behavior of warp-based methods; a generative approach would be
needed for large-offset rescue (out of scope, non-local models).

**E5c — Strength.** strength=0.8 (leave 20% of the original offset, avoiding
a dead-centered stare) still achieves 60.5% mean reduction — the default for
`gaze-apply` is 0.8 for naturalness.

## 4. What shipped

- `src/openclip/gaze/` — rig, detector, warp, evaluator, video apply/analyze
  (optional `gaze` extra: numpy + opencv-python, both GPL-3-compatible).
- CLI (behind the extra, no silent behavior):
  - `openclip gaze-analyze MEDIA` — per-video offset stats as JSON (changes nothing)
  - `openclip gaze-apply MEDIA -o out.mp4 [--strength 0.8]` — corrected copy
    (H.264, original audio remuxed)
- 7 pytest cases (marked `gaze`) asserting E1–E4 against the acceptance
  thresholds in `docs/ACCEPTANCE.md`.
- The main editing pipeline does **not** run gaze correction implicitly;
  it is a separate, explicit pre/post step (`--mode frame` uses the Haar face
  pipeline for real footage).

## 5. Owner validation checklist (real target hardware)

This is what the numbers above do **not** cover. To validate on the real
Windows machine with real footage (none of it leaves the machine):

1. `pip install ".[gaze]"` on the target PC.
2. Record a 30–60 s talking-head clip while deliberately looking **off to
   the side** (e.g., at notes next to the webcam), then a control clip
   looking at the lens.
3. `openclip gaze-analyze offside.mp4` → expect `frames_with_eyes` > 80% of
   frames; note `mean_offset_pct` (expect > 8% for the off-side clip, < 4%
   for the control).
4. `openclip gaze-apply offside.mp4 -o corrected.mp4 --strength 0.8`.
5. `openclip gaze-analyze corrected.mp4` → expect `mean_offset_pct` reduced
   by roughly the E2 factors (≥50% at ≤15% offsets).
6. Watch `corrected.mp4` side-by-side with the original: check for iris
   "smearing" during blinks (known warp limitation; blinks are not yet
   detected and gated — see §6), tracking loss during head motion, and
   overall naturalness at 100% zoom.
7. Report numbers + subjective verdict back into this document's §6.

## 6. Known limitations / next steps

- **Blinks are not gated**: during a blink the dark line of the closed lid
  can be picked up as the "iris" and warped — visible artifact. Mitigation:
  detect blink frames (aperture height collapse) and bypass. Not implemented
  yet (rig has no blinks; needs real footage to tune).
- Haar face detection is frontal-only; profile/three-quarter heads lose
  detection. MediaPipe FaceMesh (Apache-2.0, local) is the robust upgrade
  path — deliberately not added as a hard dependency (wheel size, platform
  coverage); can go behind the same `gaze` extra later.
- Glasses/reflections will confuse the dark-blob heuristic on some faces.
- Per-eye `strength` capping at 15% offsets (E5b) is not yet automatic.
- Everything above is rig-validated only until §5 runs on real hardware.
