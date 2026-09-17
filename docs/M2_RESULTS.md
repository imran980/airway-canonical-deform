# M2: motion-compensated stereo

_Started 2026-09-16 after the M1 result that per-frame stereo of a moving wall is biased by κ·v·Z/c._

## What is being fixed

M1 (docs/M1_RESULTS.md) showed that with exact camera poses the per-frame depth of the moving membrane is still
wrong by κ·v_out·Z/c, κ ≈ 0.3: over the stereo window the wall's lateral image motion is indistinguishable from
forward-motion parallax, and the effect has the same sign for sources before and after the reference frame, so no
choice of sources removes it. The only fix is to stop assuming a static scene inside the stereo.

## The method (`m2/mc_sweep.py`)

A plane-sweep stereo in which, for reference frame k and source frame k+Δ, every hypothesised 3D point is first
displaced by the wall motion between the two instants,

    X'(k+Δ) = X(k) + [d(k+Δ, z) − d(k, z)] · w(θ) · x̂_anterior,

and only then projected into the source. Cartilage (w = 0) is untouched; the membrane's texture is looked for where
it actually is at time k+Δ. Full-resolution 1920×1080, 128 inverse-depth planes over 1.5–70 mm, 7×7 NCC against
each of the four sources (k±1, k±2), per pixel the mean of the three best source costs, parabolic sub-plane
refinement, and masks for weak NCC (< 0.6), textureless windows, range ends, disagreement between sources (three
of four must agree within two planes) and a 5×5 spatial median. Depth maps are written in COLMAP's layout so that
the M1 estimator (`m1/windowed_depth.py --dense-from`) scores them unchanged; a run takes about 3 s per frame on one
A6000.

The displacement field d(z, t) is the quantity under estimation, so it enters iteratively (`m2/iterate.sh`): the
biased M1 estimate seeds the compensation, the compensated depth is re-estimated by the M1 estimator, and so on.
`--deform oracle` uses the true field (the ceiling); `--deform none` is plain stereo and must reproduce the bias.

## Validation against the truth (same estimator and gates as M1; slab ±1 mm)

| run | compensation | cells | d error mm median / p90 | CSA(z,t) err median / p90 | cartilage dev mm |
|---|---|---|---|---|---|
| static | none | 1303 | 0.05 / 0.18 | −0.1 % / 1.0 % | 0.064 |
| collapse | none | 867 | 0.44 / 1.67 | −2.4 % / 15.3 % | 0.069 |
| collapse | **true motion** | 1116 | **0.07 / 0.26** | **+0.2 % / 2.4 %** | 0.067 |
| breathing | **true motion** | 1474 | **0.06 / 0.16** | **−0.1 % / 1.1 %** | 0.064 |
| _reference: M1 on COLMAP depth_ | none | | collapse 0.48 / 1.88, breathing 0.68 / 1.58 | | |

Plain stereo from this sweep reproduces the M1 bias on the moving wall (collapse 0.44 mm) while matching COLMAP on
the static tube; with the true motion supplied the collapse error drops six-fold and the breathing error ten-fold,
to the level of the static tube. That is the mechanism confirmed and removed. What remains at this stage:

- coverage: the tightened sweep keeps about 3 % of pixels (≈ 60k points per frame, precise to 2 % median against
  COLMAP), which is enough for the station estimator over the clip but thin at the far event station during the
  collapse; the density/precision trade-off is a tuning knob, not a limit;
- the true motion is not available on real data; whether the iteration from the biased estimate converges to it
  is the next table.

## Malacia (fast periodic wall, 47 %)

| compensation | cells | d error mm median / p90 | CSA(z,t) err median / p90 |
|---|---|---|---|
| none | 158 | 2.63 / 2.90 | −17.6 % / 20.0 % |
| **true motion** | 633 | **0.12 / 0.50** | **−0.7 % / 3.7 %** |

The fastest case, unmeasurable by plain stereo (and by COLMAP's, 2.1 mm), is recovered to 0.12 mm once the motion is
compensated. Coverage also quadruples, because compensated sources agree and survive the consistency filter.

## Estimation loop seeded by the biased M1 estimate (`m2/iterate.sh`): does not converge

| scenario | seed (M1 on COLMAP) | iteration 1 | iteration 2 | iteration 3 | ceiling (true motion) |
|---|---|---|---|---|---|
| collapse, d err median / p90 mm | 0.48 / 1.88 | 0.55 / 1.01 | 0.42 / 1.53 | 0.45 / 1.01 | 0.07 / 0.26 |
| breathing, d err median / p90 mm | 0.68 / 1.58 | 0.72 / 1.61 | 0.59 / 2.31 | 0.42 / 2.12 | 0.06 / 0.16 |

Compensation needs the increment d(k+Δ) − d(k) over the window, i.e. the wall velocity, to about 20 % to remove
most of the bias; differentiating a noisy, gappy per-frame estimate gives it to about 100 %, so the fixed point
moves little. The velocity must therefore be found in the images, not in the estimate.

## Joint estimation: velocity search inside the stereo (`m2/mc_sweep_vsearch.py`)

For each frame, a small set of wall-velocity hypotheses (0, ±1, ±2, ±4, ±8, ±12, ±20 mm/s) compensates the
sources; per station the hypothesis under which the membrane texture matches best across the sources (mean best
NCC) is kept if it beats "no motion" by a margin, and the final depth is swept with that v(z). This gives, per
frame and station, a velocity measured from the images that can be compared with the true d(d)/dt directly, and a
depth map free of the bias to the extent the velocity is right.

**Result (collapse, breathing, malacia; 13 hypotheses, search at half resolution, final sweep at full):**

| scenario | plain stereo d err (median / p90 mm) | velocity search | true motion (ceiling) | velocity decided | corr(v*, truth) | sign right when moving |
|---|---|---|---|---|---|---|
| breathing 13 % | 0.68 / 1.58 | **0.19 / 1.91** | 0.06 / 0.16 | 27 % of cells | 0.24 (0.42 at 4–8 mm ahead) | 70 % |
| collapse 74 % | 0.44 / 1.67 | **0.27 / 1.46** | 0.07 / 0.26 | 29 % | 0.12 (0.37 at 4–8 mm) | 55 % |
| malacia 47 % | 2.63 / 2.90 | 2.51 / 2.91 | 0.12 / 0.50 | 14 % | 0.49 (0.73 at 4–8 mm) | 71 % |

Speed dependence of the chosen velocity (collapse): magnitude ratio |v*| / |v| is 0.95 for true speeds of 1–3 mm/s,
0.76 for 3–6, 0.58 for 6–12 and 0.06 above 12 mm/s. Breathing-scale motion (≤ 3 mm/s) is found from the images with
the right sign and size at the near stations, and the displacement error drops 3.5-fold; the fast collapse ramps and
malacia are not found, the search falls back to "no motion", and nothing is gained there. Two reasons: at high
speed the membrane's texture is foreshortened and partly occluded, so few pixels survive to score any hypothesis
(decided fraction 14–29 %); and the hypothesis grid is coarse while the mean-NCC criterion separates neighbouring
hypotheses by margins near the noise.

## Where M2 stands and what comes next

Settled:
- the velocity bias is the mechanism, and motion compensation removes it completely when the motion is known,
  in every scenario including the fastest (malacia 2.63 → 0.12 mm);
- the wall velocity cannot be recovered by differentiating the displacement estimate, but it can be recovered from
  the images for slow motion (≤ 3 mm/s) by hypothesis search inside the stereo, giving a 3.5-fold gain on breathing;
- the cartilage stays canonical (0.06–0.07 mm) in every M2 configuration: compensation never disturbs the rigid part.

Open, and the next step: estimating fast wall velocity from the images. Candidates, in order of expected payoff:
continuous velocity optimisation per station with a temporal smoothness prior across frames (the wall velocity is a
smooth function of time; per-frame independent picks are what fail), scoring by the number of pixels made consistent
rather than mean NCC, and a coarse-to-fine schedule in velocity. The streaming pose front end for real
discontinuities (26-V2) is the other half of M2 and is untouched so far.

## Night 2: velocity search v2 (`m2/mc_sweep_vsearch2.py`)

Changes from v1: 21 hypotheses (0, ±1, ±2, ±3, ±4, ±6, ±8, ±11, ±15, ±20, ±27 mm/s); the score is the sum of best
NCC over the membrane pixels a hypothesis makes consistent (rewarding both more pixels and better matches); scores
are summed over ±2 neighbouring frames before the decision (wall velocity is smooth in time); acceptance needs a 5 %
relative gain over "no motion"; a temporal median (±2 frames) and gap filling (≤ 6 frames) give the final v(z, t);
the full-resolution depth is swept with it. Pass-1 scores are saved, so the decision rule can be re-tuned cheaply.

| scenario | plain stereo | v1 search | **v2 search** | true motion | cells (v2) |
|---|---|---|---|---|---|
| breathing 13 %, d err mm median / p90 | 0.68 / 1.58 | 0.19 / 1.91 | **0.14 / 0.42** | 0.06 / 0.16 | 707 |
| collapse 74 % | 0.44 / 1.67 | 0.27 / 1.46 | **0.17 / 0.58** | 0.07 / 0.26 | 518 |
| malacia 47 % | 2.63 / 2.90 | 2.51 / 2.91 | **0.35 / 0.90** | 0.12 / 0.50 | 176 |
| CSA(z,t) err median / p90 (v2) | | | breathing −0.3 % / 2.9 %; collapse −0.6 % / 5.2 %; malacia −2.1 % / 6.1 % | | |
| cartilage deviation (v2) | | | 0.062–0.066 mm | | |

Malacia, unmeasurable by every earlier variant (2.5–2.6 mm), is now recovered to 0.35 mm from the images alone,
seven times better, and breathing and collapse move most of the way to the oracle. The velocity field itself is
rougher than the depth it produces: sign right in 68–77 % of moving cells, magnitude over-estimated for slow motion
(chosen |v| about twice the truth at 1–3 mm/s, 1.1–1.4× above 6 mm/s), correlation with the true rate 0.2–0.5.
Over-compensating a slow wall costs little; under-compensating a fast one cost everything, which is why the depth
gains are large despite a noisy velocity.

Caveat: coverage. The strict final masks keep fewer cells than COLMAP's maps (518–707 vs ≈ 2400), and at the far
event station of the collapse only one frame survives, so the event itself is not measured there in this run. The
next table re-tunes the final gate and the aggregation window from the saved scores.

Code: `m2/mc_sweep.py`, `m2/mc_sweep_vsearch.py`, `m2/mc_sweep_vsearch2.py`, `m2/iterate.sh`, `m2/eval_velocity.py`,
`m2/real_periodicity.py`.
