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

## Results: all four scenarios, symmetric window ±2 (interim table, regenerated at the end of the night)

Estimator: prior-based, taper weight > 0.6, 3-frame temporal median, no velocity correction. "Rigid map" is the
time-constant CSA(z) of the fused rigid cloud evaluated on the same (t, z) cells.

| run | frames | cells | rigid map err median / p90 | M1 prior err median / p90 | M1 model-free err median / p90 | d err mm median / p90 | cartilage dev mm median / p90 | event station (z = 30 mm) |
|---|---|---|---|---|---|---|---|---|
| static | 180/180 | 2442 | −0.1 % / 1.5 % | −0.0 % / 0.4 % | −25.6 % / 35.5 % | 0.03 / 0.08 | 0.038 / 0.089 | no event; no false displacement anywhere |
| breathing 13 % | 180/180 | 2444 | −1.1 % / 12.6 % | −0.6 % / 9.1 % | −24.9 % / 36.1 % | 0.69 / 1.59 | 0.042 / 0.094 | truth 12 %, M1 14 %; RMSE M1 3.1 vs rigid 5.7 mm²; coverage 39/39 |
| collapse 74 % | 180/180 | 2372 | −1.5 % / 13.2 % | −1.4 % / 17.5 % | −24.1 % / 38.2 % | 0.47 / 1.83 | 0.042 / 0.101 | truth 74 %, M1 86 %; RMSE M1 8.9 vs rigid 14.2 mm²; coverage 14/27 |
| malacia 47 % | 129/180 | 1302 | +10.8 % / 40.7 % | −13.6 % / 54.9 % | −11.9 % / 33.7 % | 1.99 / 6.85 | 0.038 / 0.096 | truth 35 % (seen), M1 68 %; RMSE M1 15.9 vs rigid 15.1 mm²; coverage 34/40 |

Reading: the canonical part is right everywhere (cartilage within 0.04 mm in every scenario, and the static tube
shows no false displacement). The deformation part is right when the wall is slow (breathing) or at the instant it
stops (the collapse peak), and degrades with wall **velocity**: collapse ramps, and malacia throughout. The
model-free estimator is biased −25 % by the wall's apparent thickness and is kept only as an ablation.

### The velocity bias, and why exact poses are not enough

The degradation is not estimator noise. Comparing each frame's geometric depth map with the true wall surface at
neighbouring times shows the membrane is placed with an error

    Δx ≈ κ · v_out · Z / c        κ ≈ 0.30 (breathing) and 0.29 (collapse); zero at rest; κ 0.38 → 0.25 from Z = 6 to 14 mm

where v_out is the wall's outward radial velocity, Z its distance ahead of the camera and c the camera speed. This is
the identifiability problem per pixel: over a ±2-frame window the membrane's lateral image motion (f·v·Δt/Z) is the
same order as the forward-motion parallax (f·r·c·Δt/Z²), and stereo reads it as depth. It follows that **exact poses
do not make per-frame depth of a moving wall correct**; the wall's velocity has to enter the depth estimate.
Correcting from the estimate's own time derivative fails (the per-frame displacement is too noisy to differentiate;
it made even the static case worse). The remedy queued for the rest of the night uses the sign of the bias: stereo
with sources only in the past and only in the future gives opposite biases, so their mean is bias-free to first
order and their difference measures the wall velocity, with no calibration constant (`m1/combine_sides.py`).

Figures: `docs/figures/m1_collapse.png`, `m1_breathing.png`, `m1_malacia.png`.

## Window-size and slab ablations on collapse

| stereo window | d error median / p90 (mm) | event station: M1 min vs truth | event-window coverage |
|---|---|---|---|
| ±1 frame | 0.39 / 1.68 | 87 % vs 37 % *seen* (the peak itself is not observed) | 7 / 13 |
| ±2 frames | 0.47 / 1.83 | 86 % vs 74 % | 14 / 27 |
| ±4 frames | 0.53 / 1.89 | 87 % vs 75 % | 13 / 27 |

The window is a weak lever: a shorter baseline is slightly more accurate on slow motion but sees even less of the
fast event, and all three overshoot at the exact peak, where a +0.4 mm displacement error costs 25 % of area
because the lumen is nearly closed. Coverage on the ramps is limited by how many depth points survive the
consistency filter on a fast-moving surface, not by the window. Thickening the station slab from ±0.5 to ±1.0 mm
(the collapse extends over 7 mm in z, so nothing is lost) raises the event coverage from 14/27 to 21/28 at equal
accuracy on every scenario, and is now the default (±1.5 gives 24/29).

## Frames the rigid pipeline dropped: malacia with interpolated poses

Rigid SfM registered 129 of 180 malacia frames, dropping the two breathing peaks. `interpolate_poses.py` fills
them (linear centre, slerp rotation): 0.52 mm / 1.3° median error against the truth on the 51 filled frames.
M1 on the filled model runs on all 180 frames: the cartilage deviation rises from 0.04 to 0.10 mm (the price of
the interpolated poses) and the CSA error median stays at −14 %, i.e. coverage is recovered but the accuracy on
malacia is set by the velocity bias described above, not by the missing poses.

## The prior violated: uniform contraction

The whole circumference contracts periodically by 30 % in area (no rigid sector at all). Two things happen, both
predicted in the README:

1. **Rigid SfM loses its anchor.** Camera-centre error 0.82 mm (max 1.33) and 0.61° median relative-rotation error,
   against 0.06 mm and 0.1° when only the posterior half moved: with nothing rigid in view, a uniform radial
   contraction is partly read as camera translation along the axis. The rigid dense calibre is biased by −1.3 mm and
   the pipeline reads 40 % obstruction with CV 0.15, its own unreliable-geometry flag.
2. **M1 flags the violation rather than absorbing it.** The cartilage-radius check reads 0.51 mm median deviation,
   twelve times the 0.04 mm of every prior-satisfying scenario; the prior-based CSA is wrong (−18 % median), as it
   must be when the assumed rigid sector moves, and the flag says so.

| scenario | rigid pose error (centre / rel. rotation) | M1 cartilage deviation (rigidity check) |
|---|---|---|
| static | 0.005 mm / 0.01° | 0.038 mm |
| breathing (posterior) | 0.018 mm / 0.02° | 0.042 mm |
| collapse (posterior) | 0.06 mm / 0.04° | 0.042 mm |
| malacia (posterior) | 0.03 mm / 0.01° | 0.038 mm |
| **uniform (all sectors)** | **0.82 mm / 0.61°** | **0.51 mm** |

Figure: `docs/figures/m1_uniform.png`.

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

**Result (honest): the machinery runs end to end on the real clip, but no clean event signal yet.** Stations had to
be placed along the direction the first cameras look (the scope is nearly stationary during the event, so the path
tangent there is noise), after which every event frame gets 10–13 stations. At the best station the analysis reports
seven sectors "moving" and none holding still, with unphysical inward deviations of 1.5–2.7 radii, and the event
frames fail the coverage gate for CSA. Two reasons, both structural: the event frames are exactly the ones with
interpolated poses, and the wall region they see is observed by almost no other frame, so its canonical reference is
weak. Away from the event, per-frame CSA relative to the canonical wall scatters with an IQR of about 0.86–1.20,
which is the real-video noise floor of ±2-frame stereo. Conclusion for the plan: M3 needs the streaming front end
(M2) that carries pose and depth through the discontinuity itself; interpolating rigid poses across it is not enough.
Figure: `docs/figures/m1_real_26V2.png`.
