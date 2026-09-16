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

## Estimation loop (seeded by the biased M1 estimate)

_(pending: `m2/iterate.sh collapse 3`, `breathing 3`)_

## Malacia (fast periodic wall)

_(pending: oracle and none)_
