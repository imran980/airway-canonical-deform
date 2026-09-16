# M1: identifiability on the synthetic tube (rigid-only vs canonical + deformation)

_Written during the night of 2026-09-15/16. All numbers are from the final estimator unless a section says otherwise._

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

## Results: collapse (74 % transient at 3 s, plus 13 % breathing), window ±2, final estimator

| quantity | rigid map | M1 prior (+3-frame median) | truth |
|---|---|---|---|
| CSA(z,t) relative error, all observed cells | median −1.4 %, p90 13.8 % | median −1.5 %, IQR −5 … +2 %, p90 20.7 % | — |
| membrane displacement d(z,t) | not represented | 0.48 mm median error, p90 1.9 mm | up to 8.4 mm |
| cartilage radius (rigidity check) | — | 0.06 mm median deviation, p90 0.12 mm | 0 |
| event station z = 30 mm: minimum CSA | 68 mm² constant (0 % event) | 88 % reduction at t = 3.0–3.1 s (truth 74 %: near closure a +0.4 mm displacement error costs 14 % of area) | 19.5 mm² = 74 % at t = 2.97 s |
| event station: RMSE over the frames with an estimate | 16.8 mm² | 9.9 mm² (window ±4: 4.5 mm², reading 59 %) | — |
| event station: frames with an estimate inside the ±3σ event window | — | 21 of 28 | — |

The event is found at the right instant and its depth is right to within the area's sensitivity near closure; the
ramps, when the membrane moves fastest, are partly missing and partly biased. The next two sections quantify why.

Figure: `docs/figures/m1_collapse.png`.

## Results: all scenarios and ablations, final estimator (slab ±1.0 mm, taper > 0.6, 3-frame median)

"Rigid map" is the time-constant CSA(z) of the fused rigid cloud evaluated on the same (t, z) cells. The
model-free estimator is biased −30 % by the wall's apparent thickness and is kept only as an ablation.

| run | window | frames | cells answered | rigid map err median / p90 | M1 prior err median / p90 | d err mm median / p90 | cartilage dev mm median / p90 | event station z = 30 mm |
|---|---|---|---|---|---|---|---|---|
| static | ±2 | 180/180 | 100 % | −0.1 % / 1.5 % | −0.0 % / 0.4 % | 0.02 / 0.07 | 0.055 / 0.106 | no event; no false displacement |
| breathing 13 % | ±2 | 180/180 | 100 % | −1.2 % / 12.7 % | −0.6 % / 9.0 % | 0.68 / 1.58 | 0.057 / 0.109 | truth 12 %, M1 14 %; RMSE 3.1 vs rigid 5.6; coverage 39/39 |
| collapse 74 % | ±2 | 180/180 | 98 % | −1.4 % / 13.8 % | −1.5 % / 20.7 % | 0.48 / 1.88 | 0.057 / 0.117 | truth 74 %, M1 88 %; RMSE 9.9 vs rigid 16.8; coverage 21/28 |
| collapse 74 % | ±1 | 180/180 | 98 % | −1.5 % / 13.1 % | −1.9 % / 35.1 % | 0.40 / 2.89 | 0.065 / 0.155 | peak not observed; coverage 10/19 |
| collapse 74 % | ±4 | 180/180 | 98 % | −1.4 % / 13.5 % | −0.8 % / 13.6 % | 0.54 / 1.46 | 0.046 / 0.101 | truth 75 %, M1 59 %; RMSE 4.5 vs rigid 16.5; coverage 16/29 |
| collapse, past-only sources | ±2 | 180/180 | 99 % | −1.5 % / 13.1 % | −3.0 % / 60.1 % | 0.59 / 4.20 | 0.063 / 0.143 | M1 89 %; RMSE 23.1 |
| collapse, future-only sources | ±2 | 180/180 | 98 % | −1.4 % / 13.2 % | −3.1 % / 45.7 % | 0.49 / 3.30 | 0.063 / 0.146 | M1 88 %; RMSE 17.1 |
| malacia 47 % | ±2 | 129/180 | 88 % | +12.4 % / 45.6 % | −14.1 % / 55.5 % | 2.09 / 7.00 | 0.053 / 0.116 | truth 35 % seen, M1 68 %; RMSE 19.1 vs rigid 15.1 |
| malacia, interpolated poses | ±2 | 180/180 | 91 % | — | −14.5 % / 55.9 % | 2.18 / 7.29 | 0.103 / 0.201 | truth 44 %, M1 69 %; coverage 37/37 |
| **uniform 30 % (prior violated)** | ±2 | 180/180 | 100 % | −4.9 % / 40.7 % | −18.5 % / 92.2 % | 0.69 / 3.39 | **0.511 / 1.076** | M1 wrong by design; the cartilage check flags it |

Reading: the canonical part is right everywhere (cartilage within 0.06 mm in every prior-satisfying scenario, and
the static tube shows no false displacement). The deformation part is right when the wall is slow (breathing) or at the instant it
stops (the collapse peak), and degrades with wall **velocity**: collapse ramps, and malacia throughout. The
model-free estimator is biased −25 % by the wall's apparent thickness and is kept only as an ablation.

### The velocity bias, and why exact poses are not enough

The degradation is not estimator noise. Comparing each frame's geometric depth map with the true wall surface at
neighbouring times shows the membrane is placed with an error

    Δx ≈ κ · v_out · Z / c        κ ≈ 0.30 (breathing) and 0.29 (collapse); zero at rest; κ 0.38 → 0.25 from Z = 6 to 14 mm

where v_out is the wall's outward radial velocity, Z its distance ahead of the camera and c the camera speed. This is
the identifiability problem per pixel. As the camera advances, forward-motion parallax moves wall texture radially
outward in the image in later frames and inward in earlier ones; a membrane moving outward does exactly the same on
both sides. Stereo cannot tell the two apart, so the wall's image motion is read as extra (or missing) parallax:
outward motion looks like a closer, more displaced wall, inward motion like a farther, less displaced one. The
relative size of the effect is (v·Δt/Z)/(r·c·Δt/Z²) = v·Z/(r·c), which is the law measured above. Two consequences
were checked directly:

- it is **even in source side**: stereo with sources only in the past and only in the future gives the same sign
  of error (collapse, |v_out| > 6 mm/s: +4.6 mm past-only, +3.8 mm future-only, +2.4 mm symmetric), so pairing the
  two sides does not cancel it (`m1/combine_sides.py`, tried: 0.50 mm median error against 0.48 for the symmetric
  window, velocity proxy uncorrelated with the truth);
- correcting from the estimate's own time derivative fails: the per-frame displacement is too noisy to
  differentiate, and the correction made even the static case worse (0.03 → 0.41 mm).

It follows that **exact poses do not make per-frame depth of a moving wall correct**, and that the wall's velocity
must enter the depth estimate itself: sources have to be warped by the hypothesised deformation between frames
before matching (motion-compensated stereo). That is the M2 design decision this experiment settles. It also says
that a faster scope helps (the bias scales with 1/c) and that at the instant the wall is still (the collapse peak)
per-frame stereo is unbiased, which is why the peak was recovered.

### Conservative mode: declare "unknown" when the membrane is under-observed

The velocity bias appears exactly where the moving membrane leaves few surviving depth points. The rigid anterior
wall in the same slab says how many points a well-reconstructed wall gives, so requiring the membrane's point count
to reach 30 % of it (`--rel-pts 0.3`) turns the biased cells into declared unknowns:

| scenario | cells answered | M1 prior err median / p90 | d err mm median / p90 |
|---|---|---|---|
| static | 94 % | −0.0 % / 0.3 % | 0.02 |
| breathing 13 % | 66 % | −0.8 % / 8.7 % | 0.63 |
| collapse 74 % | 82 % (event window 3 of 28) | −0.8 % / 8.0 % | 0.40 / 1.05 |
| malacia 47 % | 10 % | −2.1 % / 16.7 % | 0.49 |

Where it answers it is right to within about 8 % of area; through the collapse and for almost all of malacia it
says nothing. That is the honest v0 envelope: a still or slowly moving wall is measured per frame from its own
depth; a fast-moving wall is not measurable by short-window stereo at this camera speed, and the method knows it.

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
