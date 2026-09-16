# M1: identifiability on the synthetic tube (rigid-only vs canonical + deformation)

_Status: written during the night of 2026-09-15/16; results sections are filled in as runs complete._

## What M0 told us to build

The rigid cross-check (docs/M0_VERIFICATION.md) showed that on a textured tube the rigid pipeline's **camera poses
are exact even while the posterior wall collapses by 76 %**: the cartilage half of the wall anchors them. What the
rigid pipeline gets wrong is the **map**: fusing depth across time smears the moving membrane into the lumen or
loses it, and the pipeline's own gated measurement then reads a normal tube (14 % obstruction for a 76 % event).

So M1 v0 keeps the part that works and replaces the part that fails:

| component | rigid pipeline | M1 v0 |
|---|---|---|
| camera poses | COLMAP SfM over the whole clip | the same poses (cartilage-anchored, verified exact) |
| depth | patch-match against many frames, fused over the whole clip | patch-match for frame k against frames k−w … k+w only, **never fused across time** |
| canonical map | one time-fused cloud | the wall where it is rigid (cartilage sector), the same in every frame |
| deformation | none (moving tissue is smeared or rejected) | the membrane's position in frame k relative to the canonical wall = deformation at t_k |
| measurement | one CSA(z) for the clip | CSA(z, t) per frame and station, plus the displacement field d(z, t) |

Two estimators are compared at each station ahead of the camera (4–18 mm), from that frame's points alone:

- **model-free**: lumen = free space inside the canonical disc bounded by the frame's wall points;
- **with the anatomical prior**: the posterior sector translates anteriorly with the known taper w(θ); a wall point
  keeps its y, so it predicts d = (x + √(R² − y²)) / w(θ₀); d is the robust median over posterior points with
  w > 0.6 that are not on the anterior wall; too few points → unknown (never zero); then a 3-frame temporal median.
  CSA follows from d through the same taper. The cartilage radius in the same frame is the rigidity check.

Everything is scored against gt.npz on the (t, z) cells the camera actually observed, next to the rigid map's
time-constant CSA(z) on the same cells.

Code: `m1/windowed_depth.py` (synthetic, with truth), `m1/interpolate_poses.py` (poses for frames rigid SfM dropped),
`m1/real_sfm_relaxed.py` and `m1/windowed_depth_real.py` (real video, no truth).

## Results: collapse (76 % transient at 3 s, plus 13 % breathing), window ±2

| quantity | rigid map | M1 prior (+3-frame median) | truth |
|---|---|---|---|
| CSA(z,t) relative error, all observed cells | median −1.6 %, p90 12.7 % | median −0.2 %, IQR −4.6 … +2.3 %, p90 14.3 % | — |
| membrane displacement d(z,t) | not represented | 0.42 mm median error, p90 1.5 mm | up to 8.4 mm |
| cartilage radius (rigidity check) | — | 0.04 mm median deviation, p90 0.10 mm | 0 |
| event station z = 30 mm: minimum CSA | 68 mm² constant (0 % event) | 18.6 mm² = 75.3 % reduction, at t = 3.0 s | 19.5 mm² = 74.2 % at t = 2.97 s |
| event station: RMSE over the frames with an estimate | 12.0 mm² | 2.8 mm² | — |
| event station: frames with an estimate inside the ±3σ event window | — | 8 of 27 | — |

The last row is the honest caveat: during the ramps, when the membrane moves fastest (≈ 0.6 mm per frame), the
±2-frame stereo loses the moving surface and the estimator reports "unknown". The peak and the recovery are
recovered; the ramps are not yet. The window-size ablation below addresses this.

Figure: `docs/figures/m1_collapse.png`.

## Results: static, breathing, malacia (window ±2)

_(filled in from `runs/m1_*/m1_result.json`)_

## Window-size ablation on collapse (±1, ±2, ±4)

_(pending)_

## Frames the rigid pipeline dropped: malacia with interpolated poses

Rigid SfM registered 129 of 180 malacia frames, dropping the two breathing peaks. `interpolate_poses.py` fills
them (linear centre, slerp rotation): 0.52 mm / 1.3° median error against the truth on the 51 filled frames.
_(M1 on the filled model: pending)_

## The prior violated: uniform contraction

_(pending: the whole circumference contracts, so the "cartilage" moves; the method must flag this through the
cartilage-radius check rather than absorb it)_

## Real video: 26-V2 across its documented posterior-wall collapse

**Can rigid SfM be carried through the event by relaxing registration?** No. Frames f1600–2100 were re-extracted
exactly as the pipeline does (sequential decode, bezel mask, CLAHE 3.0), matched sequentially (overlap 20), and
mapped with pinned intrinsics and progressively looser absolute-pose thresholds:

| thresholds (min inliers / min inlier ratio) | models |
|---|---|
| pipeline default (30 / 0.25, exhaustive matching) | f1620–1901, f1923–2065, f2073–2089 |
| relaxed (15 / 0.10) | f1600–1901; f1903 + f1921–2068; f2071–2090 |
| very relaxed (8 / 0.05) | f1600–1901; f1903 + f1920–2070; f2070–2091; **f1910–1919 on its own** |

The verified two-view inlier counts explain it: normal consecutive frames share about 166 inliers; inside the event
consecutive frames share 20–100, frames two apart share 0–30, and two frames (f1902 → 1903, f1908 → 1909) share
nothing with either neighbour. The event frames are consistent among themselves (they form a 10-frame island) but
nothing bridges them to the wall before or after. This differs from the synthetic collapse, where the cartilage half
of the wall stayed visible and textured throughout and anchored the cameras; here the view changes discontinuously.

So for the real clip the per-frame poses across f1904–1920 come from the motion prior: `interpolate_poses.py` fills
the gap inside the second model (f1903 → f1921, linear centre, slerp rotation), and M1 runs on that model.
_(M1 result on this model: pending)_
