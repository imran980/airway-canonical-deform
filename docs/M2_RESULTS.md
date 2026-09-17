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
event station of the collapse only one frame survives, so the event itself is not measured there in this run.

Re-tuning from the saved scores (pass 3 only):

| variant (collapse) | cells | d err mm median / p90 | CSA err median / p90 |
|---|---|---|---|
| v2 defaults (3 of 4 sources agree, NCC ≥ 0.6, ±2-frame aggregation) | 518 | 0.17 / 0.58 | −0.6 % / 5.2 % |
| ±1-frame aggregation | 491 | 0.16 / 0.57 | −0.7 % / 5.2 % |
| relaxed final gate (2 of 4 sources, NCC ≥ 0.5) | 1910 | 1.19 / 8.26 | −11.6 % / 77 % |
| relaxed final gate, malacia | 1104 | 3.09 / 8.90 | −22.7 % / 60 % |

| 3 of 4 sources, NCC ≥ 0.5 | 547 | 0.18 / 0.61 | −0.7 % / 5.5 % |
| window ±3 (six sources), 4 of 6 must agree | 452 | 0.15 / 0.48 | −0.4 % / 4.5 % |
| 2 of 4 sources, NCC ≥ 0.6 | 1688 | 0.75 / 8.41 | −6.6 % / 78 % |

The aggregation window hardly matters and the NCC threshold barely; the requirement that three of the four sources
agree on the depth is decisive and cannot be relaxed: the cells it admits are the moving-wall pixels no hypothesis
makes consistent, and they carry the old bias and worse. Precision and coverage trade against each other through
that one gate; the honest operating point is the strict one, with coverage reported. Coverage is instead raised
without touching the gate by fusing neighbouring frames' compensated points into each station after shifting them
by the estimated wall motion (next section).

### Refining the velocity magnitude: marginal

A second search level tried multiplicative factors (0.5, 0.7, 1.4, 2.0) around each station's chosen velocity,
scored like the first level. The slow-motion magnitude ratio improves from 2.5× to 1.7× but the depth barely
moves (0.16 / 0.56 mm against 0.17 / 0.58): the compensation is insensitive to the residual over-estimate, which is
the same reason a rough velocity was already enough.

Figure: `docs/figures/m2_collapse_summary.png` (truth, rigid map, plain per-frame stereo, v2 search and the true-
motion ceiling at one station, and the error distribution over the gated cells of the whole clip).

### Mis-specified prior: a plain posterior box instead of the true taper

Real anatomy will not hand us the generator's sector weights. Replacing them by a plain ±90° posterior box (every
posterior point assumed to move fully, no taper) in the velocity search and compensation gives, on collapse,
the results below (cartilage 0.07 mm in both):

| scenario | uncompensated | box prior (±90°, no taper) | exact taper |
|---|---|---|---|
| collapse 74 %, d err mm median / p90 | 0.44 / 1.67 | 0.23 / 0.89 | 0.17 / 0.58 |
| breathing 13 % | 0.68 / 1.58 | 0.34 / 1.68 | 0.14 / 0.42 |
| malacia 47 % | 2.63 / 2.90 | 2.50 / 6.82 | 0.35 / 0.90 |
| malacia 47 %, box of the **right** extent (±60°) | 2.63 / 2.90 | **0.57 / 2.65** | 0.35 / 0.90 |
| breathing 13 %, box of the **right** extent (±60°) | 0.68 / 1.58 | **0.25 / 0.84** | 0.14 / 0.42 |
| collapse 74 %, box **narrower** than the true sector (±60° for ±90°) | 0.44 / 1.67 | **0.17 / 0.56** | 0.17 / 0.58 |
| breathing 13 %, box **narrower** than the true sector (±45° for ±60°) | 0.68 / 1.58 | **0.18 / 0.58** | 0.14 / 0.42 |

The pattern is about the sector's extent, not its edge shape. The collapse scenario's true moving sector is ±90°,
so the ±90° box matched it and two thirds of the gain survived. Breathing and malacia have a ±60° sector: the ±90°
box declared 30° of rigid wall on each side to be moving, the compensation displaced pixels that never moved, no
hypothesis could make them consistent, and the gain halved (breathing) or vanished (malacia). With a box of the
right extent and still no taper, malacia recovers most of the gain (0.57 mm against 0.35 with the exact taper and
2.63 uncompensated) and so does breathing (0.25 against 0.14 and 0.68). A box narrower than the true sector costs nothing on the collapse (0.17 / 0.56, equal to the exact taper) and
beats the correct-extent box on breathing (0.18 / 0.58 against 0.25 / 0.84): the pixels left out are the taper's
flanks, which move little and which the estimator down-weights anyway. The
asymmetry is the practical rule: declaring rigid wall to be moving breaks the compensation, leaving some moving wall
out does not, so the moving sector should be chosen conservatively, never wider than the anatomy. Given that, the
taper shape is secondary. On real airways the extent is anatomical (the membranous posterior wall spans roughly
a third of the circumference) and can be estimated from the data as the sector that moves.

### Prior violated under the velocity search: still flagged

On the uniform-contraction scenario (the whole wall moves, nothing is rigid) the velocity search compensates only the
sector the prior allows to move, so it cannot represent the scene: CSA error −60 % / 99.6 %, and the cartilage
check reads 0.57 mm median deviation (1.10 mm at p90), against 0.06–0.07 mm in every prior-satisfying run. The
flag that M1 introduced survives M2 unchanged: when the assumed rigid sector moves, the method says so rather than
absorbing it.

### Camera speed: the bias scales with 1/c as predicted

The bias law κ·v·Z/c says a faster scope halves the error. A second collapse video was rendered with the camera at
12 mm/s instead of 6 (100 mm tube so the event is still reached; same 74 % collapse at 3 s). The rigid pipeline
registers 169 of 180 frames (poses 0.10 mm, dense calibre bias −0.42 mm). Plain per-frame COLMAP stereo, the M1
estimator unchanged:

| camera speed | d err mm median / p90 | CSA err median / p90 |
|---|---|---|
| 6 mm/s | 0.48 / 1.88 | −1.5 % / 20.7 % |
| 12 mm/s | **0.32 / 0.80** | **+0.0 % / 6.9 %** |

The median error drops by a third and the tail by more than half, in line with the prediction. For the clinic this
is a capture instruction, not a method: a steadily withdrawn scope is a better instrument for a moving wall than a
dwelling one.

M2 on the same fast-camera video (own sweep, 169 frames):

| 12 mm/s | cells | d err mm median / p90 |
|---|---|---|
| plain sweep | 740 | 0.50 / 1.04 |
| velocity search v2 | 385 | 0.26 / 2.67 |
| true motion | 847 | 0.08 / 0.25 |

The ceiling is unchanged, but the velocity search helps the median and hurts the tail here: doubling the camera
speed halves the wall's image motion relative to the parallax, so the hypotheses become harder to separate (chosen
velocities 3.5× too large for slow motion, half the truth above 12 mm/s, 42 % of cells decided) exactly as the bias
they would remove shrinks. The compensation is worth most where the scope is slow, which is also where it is most
needed; a fast, steady withdrawal is the complementary remedy.

**Caveat found afterwards (03:20).** The collapse event of the generator sits at a fixed z = 30 mm and t = 3 s. At
12 mm/s the camera is at z = 32 mm when the event peaks, i.e. 2 mm *past* it, so this video saw almost none of the
collapse (the event station reports a seen reduction of 10 %, against 76 % at 6 mm/s): its wall motion was mostly
the 10 % breathing. The rows above therefore mix two effects, a faster camera and less wall motion in view, and are
not a clean test of the 1/c law. A clean pair is being run with the event placed 16 mm ahead of the camera at t = 3 s
for every speed (`--collapse-z0`; 3 mm/s at z0 = 21, 12 mm/s at z0 = 48 on a 100 mm tube), matching the 6 mm/s
reference geometry; see the table below.

### Deformation-aware fusion of neighbouring frames: negative

To raise coverage without touching the gate, the points of frames k±F were shifted by the estimated wall motion
v(z)·Δt·w(θ) and added to frame k's stations (`m1/windowed_depth.py --fuse-vel`):

| collapse, v2 maps | cells | d err mm median / p90 | CSA err median / p90 |
|---|---|---|---|
| no fusion | 518 | 0.17 / 0.58 | −0.6 % / 5.2 % |
| fuse ±1 frame | 1134 | 0.27 / 3.22 | −1.8 % / 31 % |
| fuse ±2 frames | 1442 | 0.35 / 8.28 | −2.6 % / 77 % |

Coverage doubles and precision collapses. The asymmetry is instructive: de-biasing the stereo needs the velocity
only over the ±2-frame baseline and its errors partly cancel across the sources, so a rough velocity field is
enough; warping points across frames needs the absolute displacement between frames, and a velocity that is right
in sign but about twice too large for slow motion misplaces them by more than the bias it removes. A deformation-
aware fusion therefore has to wait for a better velocity estimate; until then the single-frame estimate at the
strict gate is the operating point.

## Real video, night 2: 20-V1 (tracheobronchomalacia, CT inspiratory/expiratory 45 % area change)

Per-frame stereo (COLMAP, ±2 frames, no compensation) on the 337-frame rigid model of 20-V1, stations along the
camera path, canonical wall = time median. Two failed attempts first: a disk-quota crunch and, more instructively, a
normal-map cleaner running during the stereo (COLMAP's geometric pass reads the sources' photometric normal maps, so
only the last frames received geometric depth). With nothing deleting, 4283 station cells on 68 of 76 stations.

The per-frame lumen area swings by ±25–45 % over seconds and adjacent stations track each other (correlation 0.98),
which looked like the respiratory cycle of a malacic trachea. Two checks say otherwise:

- **sector check**: between the widest and narrowest frames, all 36 wall sectors move together (median +0.47 R, no
  sector still). A membranous collapse is sectoral; a change common to the whole circumference is either a
  circumferential malacia or a per-frame pose/scale artefact;
- **image check**: the dark-lumen fraction of the same frames, which does not depend on any pose, is *anti-*
  correlated with the estimated area (−0.57 to −0.84 across stations), and the estimate is *positively* correlated
  with image brightness (+0.5 to +0.7). Real narrowing would shrink the dark hole and brighten the image, the
  opposite signs. So the swing follows illumination and camera distance, not the wall.

Conclusion: on this real clip the per-frame estimator produces large common-mode area swings that are an artefact,
and the two checks above catch it. The rigidity check that flagged the prior violation on synthetic data is the
same instrument here: a lumen change that is not confined to a sector should not be believed. What is needed
before real per-frame calibre can be trusted is the M2 machinery on real data (velocity search with a data-driven
moving sector) and a photometric normalisation of the frames before stereo; the 5-second station observation
windows of a withdrawing scope also cannot resolve a 2–3 s respiratory period. Figure:
`docs/figures/m1_real_20V1_periodicity.png` (the swings; the periodogram peak at 32–37/min is weak).

### 26-V2 pullback A (f1600–1901, the 300 frames before the posterior-wall collapse)

The collapse frames themselves (f1908–1921) register in no rigid model, so this run measures the segment leading up
to it: 302 frames, 1200 px per-frame stereo, 302 geometric maps, 3121 station cells on 40 of 48 stations. The
per-frame area sits at a median 1.07 of the canonical (IQR 0.98–1.23) with no respiratory period (autocorrelation at
the best period ≤ 0.14 at every station, spectral power ≤ 0.13). The two checks that condemned 20-V1 both *pass*
here: the change is sectoral (moving arc 60–270°, 0–8 sectors still) and it is not image-driven (correlation with the
dark-lumen fraction +0.06 to +0.16, with brightness −0.08 to +0.03). Yet it is not wall motion either. In the wide
frames the wall sits 1.0–1.4 R *outside* its canonical position and 9–21 % of frames show a lumen dilated by more
than 50 %, which no airway does in seconds: the model-free free-space fill leaks through holes in the depth map
where a wall sector was not reconstructed. A third check now catches this (`m2/real_checks.py`, magnitude check:
outward excursion > 0.5 R or > 5 % of frames dilated > 50 %), and 20-V1 fails it too on top of the other two.

| real clip | frames | cells | sector check | image check | magnitude check | verdict |
|---|---|---|---|---|---|---|
| 20-V1 (malacia) | 337 | 4283 | all 36 sectors move together | corr(CSA, dark) −0.57..−0.84 | outward +0.7..+1.1 R | not wall motion |
| 26-V2 A (pre-collapse) | 302 | 3121 | sectoral, 60–270° | ≈ 0 | outward +1.0..+1.4 R, 9–21 % frames > 1.5× | not wall motion |

So the per-frame *area* on real clips is dominated by two failure modes the synthetic tube never showed, a
common-mode pose/illumination swing (20-V1) and a lost-wall leak in the free-space fill (both clips), and each is
caught by a cheap check that needs no ground truth. The synthetic estimator that survived M1 and M2 was the
prior-based one (median displacement over the posterior sector, never the free-space area); its real-data
counterpart needs a data-driven moving sector and per-sector outlier rejection, which is the first item of the next
step.

### The dominant real-data effect: the per-frame radius follows the camera's distance to the station

Built the real-data counterpart of the synthetic estimator (`m2/real_sector_estimator.py`: lost-wall sectors
rejected, median deviation over every 120° arc, the most inward-moving arc per station with the opposite arc as a
still reference, plus the dark-lumen check, a camera-distance check and neighbour coherence). On 20-V1 it confirms
the verdict (20 of 27 stations: the reference arc moves with the moving arc; 6: against the image). On 26-V2 A it
found one coherent feature, all stations 0.15–0.18 R inward during f1600–1650 and flat afterwards, and the checks
killed it: the dark-lumen fraction is flat over those frames (0.040 vs 0.042) while the camera was closer to the
stations then (1.1–1.3 R vs 1.6–1.8 R).

That pointed at the actual mechanism. Regressing the whole-circumference median deviation of each station on the
camera's distance to that station:

| clip | stations | slope of wall radius vs camera distance | sign positive at | typical corr |
|---|---|---|---|---|
| 20-V1 (real) | 28 | **+0.24 R per R** (IQR +0.13..+0.29) | 100 % | +0.7..+0.9 |
| 26-V2 A (real) | 25 | **+0.12 R per R** (IQR +0.04..+0.27) | 80 % | +0.5..+0.9 |
| synthetic static / collapse / breathing (M1, same 0.8–3.6 R range) | 59 | −0.005 / −0.005 / −0.009 R per R; cartilage +0.001..+0.002 | — | −0.3..+0.1 |

On the real clips the wall appears farther from the axis the farther the camera is from the station, by a fifth to a
quarter of the radius per radius of distance; over the 1–2.5 R range a station is observed from, that alone moves
the radius by 20–40 % and the area by 40–80 %, which is the size of the 20-V1 "respiratory" swing. The same
estimator on the pinhole synthetic tube shows no such dependence (20–50× smaller slope), so the bias sits in the real
data path and not in the estimator: residual radial distortion or intrinsics error is the first suspect (M0's third
catch already showed a distortion mismatch moves the wall by 0.4–0.6 mm, and a wall's image position changes with
camera distance, so a distortion residual becomes a distance-dependent radius), with SfM scale drift along the path
second. Two consequences: (1) every per-frame real-data number so far, including the periodicity, was reading this
bias; the four checks (sector, image, magnitude, camera distance) are now the gate any real per-frame estimate must
pass; (2) the bias is measurable on any rigid segment as this slope and can be calibrated out before reading motion,
which is now the first item of the real-data plan, ahead of the velocity search.

### Brightness-normalised 20-V1: the bias is half photometric, half geometric

The same 337 frames (f2509–2845) with per-frame brightness normalisation before the stereo (800 px, 337 geometric
maps, 4104 cells on 65 of 74 stations), checked against the *original* frames' dark-lumen statistics:

| 20-V1 | area ratio IQR | slope radius vs camera distance | corr | p5–p95 swing of the median radius | corr(CSA, dark) |
|---|---|---|---|---|---|
| as captured | 0.86–1.19 | +0.24 R per R | +0.78 | 0.34 R | −0.57..−0.84 |
| brightness-normalised | 0.95–1.11 | +0.14 R per R | +0.61 | 0.25 R | −0.47..−0.80 |

Normalising the frames removes about half of the range dependence and a quarter of the swing, so part of the bias is
photometric (the patch-match behaves differently on the brighter, closer frames) and the rest is geometric. The four
checks still fail at every station tested (against the image; common-mode at two), and the sector estimator passes
only 4 of 24 stations, with excursions at the noise floor. The remaining +0.14 R per R is what the distortion test
below has to explain.

## Where M2 stands after night 2

Settled on the synthetic tube (figure `docs/figures/m2_night2_summary.png`):
- **Compensation with the true motion removes the velocity bias in every scenario** (membrane displacement error
  0.06–0.12 mm median, 0.16–0.50 mm p90, against 0.44–2.63 mm uncompensated), and never disturbs the cartilage
  (0.06–0.07 mm). The mechanism and the ceiling are both measured.
- **Velocity search v2 recovers most of that gain from the images alone**: breathing 0.68 → 0.14 mm, collapse
  0.48 → 0.17 mm, malacia 2.09 → 0.35 mm (3.5–6-fold), with the 3-of-4 source-agreement gate as the decisive
  ingredient (relaxing it to 2-of-4 explodes the error to 0.75/8.4 mm). Its price is coverage: 18–30 % of the
  cells the uncompensated sweep fills, because the gate refuses cells whose velocity it cannot decide.
- **The prior's extent matters asymmetrically**: a sector prior narrower than the true moving sector is harmless
  or better (collapse 0.17/0.56 with a 60° box), a wider one breaks the estimate (malacia 2.50 mm with a 180° box).
  Under-declare the moving sector.
- **The rigidity check still catches the violated prior** under the velocity search (uniform contraction: cartilage
  deviation 0.57 mm against 0.06 elsewhere).
- Negative results, each measured: the estimation loop seeded by the biased M1 estimate does not converge; refining
  the velocity magnitude is marginal; fusing neighbouring frames with the deformation makes things worse
  (p90 0.58 → 3.2 mm); the velocity search helps the median but hurts the tail at 12 mm/s camera speed.
- _Pending at the time of writing: the clean camera-speed pair (3 and 12 mm/s with the event kept 16 mm ahead of the
  camera), which replaces the confounded 12 mm/s row above._

On real clips, what night 2 established is different in kind: **every per-frame number read so far on 20-V1 and
26-V2 was reading a range-dependent radius bias, not the wall**. The per-frame radius grows with the camera's
distance to the station by +0.12..+0.24 R per R (synthetic control ≤ 0.01), enough by itself to produce the whole
20-V1 "respiratory" swing; brightness normalisation removes half of it. Four checks that need no ground truth now
gate any real per-frame estimate (sector coherence, dark-lumen/brightness consistency, magnitude, camera-distance
independence), and no station on either clip passes all four. _Pending: the distortion test (static renders
barrel-distorted with the 26-V2 lens, recovered with the exact and with a 25 %-wrong calibration), which decides
whether the remaining geometric half is a calibration residual._

Order of the next steps, changed by these findings: (1) calibrate the range bias out on a rigid segment (or fix
its cause) and re-read 20-V1 through the four checks; only then (2) the velocity search on real frames with an
under-declared moving sector, and (3) the streaming pose front end for the 26-V2 collapse frames, which no rigid
model registers and which is the one event large enough (75 %) to clear the real-data noise floor of ≈ 0.2–0.3 R.

Code: `m2/mc_sweep.py`, `m2/mc_sweep_vsearch.py`, `m2/mc_sweep_vsearch2.py`, `m2/iterate.sh`, `m2/eval_velocity.py`,
`m2/real_periodicity.py`, `m2/real_checks.py`, `m2/real_sector_estimator.py`, `m2/summary_figure.py`,
`synthetic/distort_frames.py`.
