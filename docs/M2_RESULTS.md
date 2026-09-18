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

Correcting the bias in post does not work either (`m2/real_range_correct.py`: each station's radii divided by the
fitted linear dependence on camera distance, then everything recomputed). On 20-V1 the 5–95 % area swing narrows
from 0.62–1.42 to 0.73–1.35 and the checks still fail (dark-lumen correlation −0.68 at one station, +0.23 at another;
the sector estimator's reference arc still moves at 21 of 27 stations). The bias is not a single linear function of
camera distance, so it has to be removed at its source, not fitted away.

### Distortion test: a calibration residual is a sufficient mechanism, but the real bias is mostly photometric

The static pinhole renders were barrel-distorted with the 26-V2 lens (k1 −0.087, k2 −0.094; `synthetic/distort_frames.py`)
and put through the rigid pipeline and M1 twice: with the exact coefficients and with k1, k2 underestimated by 25 %.
Already at the rigid level the wrong calibration turns the uniform tube's area profile from a 6 % into a 21 %
along-tube variation, flagged by the pipeline as a candidate narrowing. Per frame, with a robust slope (median over
stations of the Theil–Sen slope of the wall radius against the camera's distance to the station, 0.8–3.6 R):

| run | stations | slope, R per R | same sign at | far − near quartile |
|---|---|---|---|---|
| synthetic pinhole, static | 34 | −0.001 | — | −0.003 R |
| distorted, exact calibration | 34 | +0.005 | — | +0.010 R |
| distorted, k1 k2 × 0.75 | 34 | **−0.018** | 85 % | **−0.062 R** |
| real 20-V1 | 24 | **+0.196** | 100 % | **+0.248 R** |
| real 20-V1, brightness-normalised | 20 | +0.054 | 100 % | +0.122 R |
| real 26-V2 A | 13 | +0.041 | 77 % | +0.031 R |

Reading: the pinhole and the exactly-calibrated distorted tube show no range dependence, so neither the estimator nor
the distort–undistort resampling creates one; a 25 % distortion residual does create a consistent one (sign set by
the sign of the residual, here the radius reads *smaller* when the camera is far), of about a quarter of 20-V1's
size. On 20-V1 brightness normalisation removes three quarters of the effect (+0.20 → +0.05 R per R), so most of
the real bias is photometric (the patch-match on the brighter, closer frames), and the remainder is of the size a
calibration residual produces. These robust slopes supersede the least-squares slopes quoted above (+0.24 / +0.12),
which were inflated by lost-wall frames at the far end. Static errors under the wrong calibration: membrane
0.14 / 0.46 mm and cartilage 0.13 mm against 0.04 / 0.12 and 0.07 with the exact one, i.e. a 25 % distortion error
costs 0.1 mm on a 5 mm tube before any wall moves.

### Camera speed, clean pair (event 16 mm ahead of the camera at t = 3 s for every speed)

| membrane displacement error, mm median / p90 | 3 mm/s | 6 mm/s (reference) | 12 mm/s |
|---|---|---|---|
| M1, COLMAP per-frame stereo | 0.52 / 2.45 | 0.48 / 1.88 | 0.37 / 1.14 |
| M2 plain sweep | 0.48 / 2.14 | 0.44 / 1.67 | 0.50 / 1.02 |
| M2 + velocity search v2 | **0.24 / 0.94** | 0.17 / 0.58 | 0.33 / 2.87 |
| M2 + true motion | 0.09 / 0.28 | 0.07 / 0.26 | 0.07 / 0.27 |
| area error p90 (M1) | 25 % | 21 % | 11 % |

With the event now in view at every speed (seen reduction 73 / 76 / 71 %), the uncompensated tail error falls with
camera speed as 1/c predicts (2.45 → 1.88 → 1.14 mm), the medians sit near the noise floor, and the ceiling with
known motion is flat (0.07–0.09 mm). The velocity search behaves as the confounded run suggested: at 12 mm/s it
decides 43 % of the cells, over-estimates slow wall motion 3.6-fold and worsens the tail (2.87 mm), because the
wall's image motion shrinks relative to the parallax exactly as the bias it would remove shrinks. At 3 mm/s it
pays: 55 % of the cells decided, median halved and tail cut 2.3-fold against the plain sweep (0.48 / 2.14 →
0.24 / 0.94 mm), with the same slow-motion over-estimate (2.6-fold below 3 mm/s) that limits it everywhere. The
velocity search is a slow-scope instrument; a fast steady withdrawal is the alternative remedy, and the two meet
around 6 mm/s where the search gives its best absolute result (0.17 / 0.58 mm).

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
- **Camera speed, clean pair**: the uncompensated tail error falls 2.45 → 1.88 → 1.14 mm from 3 to 12 mm/s, the
  ceiling with known motion is flat (0.07–0.09 mm), and the velocity search pays off only where the scope is slow.

On real clips, what night 2 established is different in kind: **every per-frame number read so far on 20-V1 and
26-V2 was reading a range-dependent radius bias, not the wall**. The per-frame radius grows with the camera's
distance to the station by +0.12..+0.24 R per R (synthetic control ≤ 0.01), enough by itself to produce the whole
20-V1 "respiratory" swing; brightness normalisation removes three quarters of it (robust slope +0.20 → +0.05 R per
R), and the distortion test shows a 25 % calibration residual produces a range dependence of the size of the
remainder while the exact calibration produces none. Four checks that need no ground truth now gate any real
per-frame estimate (sector coherence, dark-lumen/brightness consistency, magnitude, camera-distance independence),
and no station on either clip passes all four.

Order of the next steps, changed by these findings: (1) calibrate the range bias out on a rigid segment (or fix
its cause) and re-read 20-V1 through the four checks; only then (2) the velocity search on real frames with an
under-declared moving sector, and (3) the streaming pose front end for the 26-V2 collapse frames, which no rigid
model registers and which is the one event large enough (75 %) to clear the real-data noise floor of ≈ 0.2–0.3 R.

## Day 3: the three next steps, taken in order

### 1. The range bias at its source (20-V1, 337 frames, 800 px per-frame stereo)

Robust slope of the wall radius against the camera's distance to the station (median over stations of the Theil–Sen
slope; the pinhole synthetic control is −0.001, a 25 % distortion residual gives ±0.018):

| 20-V1 frames | slope, R per R | positive at | far − near quartile | area ratio p5–p95 | stations passing all sector-estimator checks |
|---|---|---|---|---|---|
| as captured (CLAHE) | +0.196 | 100 % | +0.248 R | 0.62–1.42 | 0 / 27 |
| + per-frame gain (bright wall → fixed level) | +0.054 | 100 % | +0.122 R | 0.79–1.31 | 4 / 24 |
| + gain + local illumination flattening (σ 61 px) | +0.050 | 100 % | +0.104 R | 0.81–1.30 | 3 / 24 |
| + gain + saturated / black pixels removed | +0.068 | 100 % | +0.125 R | 0.78–1.31 | 2 / 25 |
| gain frames, k1 k2 × 0.85 | +0.063 | 95 % | +0.133 R | 0.79–1.32 | 3 / 24 |
| gain frames, k1 k2 × 1.15 | +0.079 | 100 % | +0.119 R | 0.78–1.34 | 3 / 25 |

The per-frame gain does the work (+0.20 → +0.05); flattening the illumination gradient and removing saturated
pixels change nothing more, so the photometric part is the frame's overall level, not falloff or blown highlights.
The remainder (+0.05 R per R, three times the synthetic residual test's size, at 100 % of stations) was the target of
the calibration variants, and both directions make it worse (×0.85: +0.063, ×1.15: +0.079): the measured
calibration is at the optimum of the distortion-scale channel, so the remaining bias is not a k1/k2 residual. What
it is remains open (SfM scale drift along the path, a principal-point or focal residual, or a property of
per-frame stereo on real texture at close range are the candidates left). In every variant the image check still
fails at the best stations (corr(area, dark lumen) −0.3 to −0.8) and the sector estimator's reference arc moves with
the moving arc at most stations.

### 2. The velocity search on real frames (`m2/mc_sweep_vsearch_real.py`)

Port of the synthetic search to the real station frames: stations and parallel-transported normals rebuilt from the
per-frame grid, hypotheses as radial inward motion of a declared arc (auto centre from the grid, half-width 30°,
i.e. under-declared), velocities in R per second, scores per station, compensated maps handed back to the real
per-frame scorer. On the gain-normalised 20-V1 it runs end to end (337 frames, 45 min on one GPU): 32 % of cells
decided, three quarters of them with a non-zero velocity (median 0.3 R/s). With the synthetic gate (NCC 0.6,
3-of-4) the compensated maps are too sparse to score on real texture (757 cells against 4104); with a real-frame
gate (NCC 0.5, 2-of-4) and the same sweep run without compensation as the like-for-like baseline:

| torch sweep, gain frames | cells | slope, R per R | area ratio p5–p95 | image check |
|---|---|---|---|---|
| plain (v = 0) | 3057 | +0.034 (83 % positive) | 0.92–1.29 | fails (−0.55..−0.64) |
| velocity search | 3073 | +0.034 (90 % positive) | 0.87–1.34 | fails (−0.27..−0.70) |

The machinery works; whether it should be believed is another matter: with the range bias unfixed, the search is
offered an apparent wall motion that is really camera distance, and it accepts velocities for it. The order of the
steps stands: no real velocity is credible before the bias is gone.

### 3. Carrying pose through the 26-V2 collapse

The two rigid models (f1600–1903 and f1901–2068, the collapse f1908–1921 in neither, even with a 40-frame matching
overlap) cannot be joined: their two common frames are the seam frames, and seam frames are the unreliable ones
(model 0's own step f1901 → f1903 is four times its cruising speed; the two common frames disagree by 3° in
rotation and the two scale estimates by 50 %; the point-feature Sim(3) leaves the common cameras 17–20° apart; the
registrator finds no 2D–3D matches for any post-gap frame; the database holds no verified match between a collapse
frame and any pre-gap frame). So model 0 was cut back to its last reliable frame (f1901) and the 20 following poses
were carried on a constant-velocity prior (`m1/extrapolate_poses.py`), then per-frame stereo was run across the
event.

What came out, and what it teaches:
- the image knows the collapse without any pose: the dark-lumen fraction falls from 0.071 (f1860–1901) to 0.062
  (f1902–1907) to 0.047 (f1908–1921);
- the carried frames' depths land 3–4 R too far along the path: the operator stops the scope as the airway
  collapses, so a constant-velocity prior over-states the stereo baseline and every depth scales up with it (the
  wall's median radius reads 0.3–0.5 R in f1908–1915 and jumps four-fold at f1916). A carried pose has an unknown
  speed scale, and the small-baseline stereo cannot supply it;
- the natural scale anchor, the rigid anterior arc at its canonical radius, is not available where the carried
  frames look: those stations lie in the straight-line extension of the camera path, where the real airway bends
  and even the registered pre-event frames show eccentric rings (outermost arc 1.2–3.1 R against 0.7–1.2 R
  opposite), and they were never observed before the event.

Design consequence for the streaming front end: the canonical wall has to be the map. The pose of a frame the rigid
pipeline drops is the Sim(3) (or SE(3) with the wall's known radius fixing the scale) that registers the frame's
depth on the *rigid* arc to the canonical tube, and the canonical of the collapsed segment is available from the
post-event pullback (model 1, f1922–2068, the same wall after it re-opens). Constant-velocity carrying, path
extension and seam-frame bridging are all ruled out by measurement now, which is what this day was for.

### 3b. Canonical-as-map, version 1 (26-V2, afternoon)

Model 1 alone spans both sides of the collapse (f1903–1907 before, f1921 onward after), so the 13 dropped frames were
carried by interpolation between two registered poses of the same model (`m1/interpolate_poses.py`), the per-frame
stereo run on gain-normalised frames f1903–2068 (166 frames, 800 px), and the canonical wall taken from the
post-event pullback f1935–2068 (`--canonical-frames`). `m2/real_event_report.py` then reads the event against that
map, with the whole-circumference deviation of the carried frames against their registered neighbours as the
pose-validity measure.

- **The carry is valid where the camera is close.** At the stations 0.9–1.5 R from the camera the carried frames
  agree with their registered neighbours to 0.01–0.05 R; at 1.7–2.1 R they read the whole wall 0.4–0.7 R outward
  against 0.12–0.14 for the neighbours, the range dependence once more.
- **First per-frame reading of a real collapse.** At the near stations the area falls to 0.55–0.93 of the post-event
  canonical (median during the event 0.83–1.17), the most inward 120° arcs sit at −0.06 to −0.32 R around ±180°
  (the same direction at three of four stations), and the pose-free dark-lumen fraction goes 0.059 → 0.047 → 0.060.
  The reading is not clean: the arcs opposite the inward ones move too at two stations, and the post-event frames
  are contaminated by lost-wall leakage (27–40 % of frames dilated > 1.5×, flagged by the magnitude check), so the
  area ratios right after the event (2.2–2.6) are not to be believed. Figure `runs/m1_real_26V2d/event_report.png`.
- **Depth-to-canonical pose registration does not work at this noise level** (`m2/register_to_canonical.py`, the
  rigid 240° arc against the map, per frame or jointly over a window with a time-linear correction). Unbounded, the
  fit collapses the depth (scale 0.02–0.05) onto the wall surface; bounded, it runs to every bound; with the scale
  fixed it moves *registered control frames* by 0.08–0.16 R and 4–7° per frame, and jointly by 0.35–0.44 R and 14°
  over a window, without improving their residual scatter (MAD 0.20–0.30 R before and after). The per-frame depth
  noise (0.22–0.30 R) and, above all, a systematic +0.16–0.20 R offset of the pre-event frames against a map built
  from post-event frames seen at other distances make the registration chase the range bias, not the pose.

So the canonical-as-map idea survives as a design, and its first version establishes that pose can be carried
between two registered frames of one model well enough to read a collapse at close range; but its refinement step
is blocked by the same range-dependent radius bias as everything else on real clips. That bias is now the single
blocker, and the instrument to find its mechanism is a real endoscope with ground truth: the C3VD phantom recordings
(real optics, known mesh) let the slope of radius against camera distance be measured against truth, photometric
and geometric causes separated, and a correction validated before it is applied to patients.

## Night 3: the range bias measured against truth on a real endoscope (C3VD), its mechanism, and its cure

C3VD (real colonoscope, 195° fisheye, silicone colon phantom registered to a CT mesh) gives per-frame ground-truth
depth and poses. `c3vd/prepare.py` undistorts the frames to the harness's 100° pinhole (1800 px, f = 755.19), remaps
the ground-truth depth through the same map (axial depth, verified by multi-view consistency to 0.1 %, and against
the mesh to 0.11 mm point-to-surface), and writes the ground-truth poses as a COLMAP model. `c3vd/stereo_vs_truth.py`
runs the airway pipeline's per-frame stereo (COLMAP patch-match, ±w frames, geometric consistency) and scores every
depth map against truth, binned by distance, brightness, image radius, incidence angle and local texture.

**The measurement (cecum_t1_a, 276 frames, 800 px, scope 0.29 mm per frame):**

| poses | sources | frames | rel. depth error median / MAD | pixel coverage | by distance (5–15 → 65–90 mm) |
|---|---|---|---|---|---|
| ground truth | ±2 (the airway setting) | raw | **+25.1 % / 42 %** | 74 % | +17 % → +37 % |
| ground truth | ±2 | gain-normalised | +26.3 % / 43 % | 74 % | +20 % → +40 % |
| ground truth | ±2 | raw, 1200 px | +31.1 % / 57 % | 57 % | +27 % → +60 % |
| COLMAP SfM (0.14 mm from truth) | ±2 | raw | +14.1 % / 34 % | 75 % | +12 % → +22 % |
| ground truth | ±2, texture gate ≥ 4 | raw | +8.5 % / 15.5 % | 8 % | flat |
| COLMAP SfM | ±2, texture gate ≥ 4 | raw | **+0.1 % / 9.5 %** | 8 % | flat 15–50 mm |
| ground truth | 1–5 (100 frames) | raw | +2.6 % / 13.8 % | 87 % | |
| ground truth | **3–5 only** (100 frames) | raw | **+0.1 % / 11.8 %** | **74 %** | flat (+3 % → −3 %) |
| ground truth | ±2 (same 100 frames) | raw | +29.8 % / 44.7 % | 76 % | +22 % → +34 % |
| **COLMAP SfM** | **3–5 only, all 276 frames (the production recipe)** | raw | **−1.3 % / 9.4 %** | **73 %** | +0.5 % → −1.9 %, flat in brightness, texture, radius, incidence |

With exact poses and the airway setting, per-frame stereo reads depth a quarter too far, more the farther the wall
(+17 % at 5–15 mm, +37 % at 65–90 mm) and more the darker the pixel (+39 % below 30 grey levels, +1.5 % above 180).
Inverse depth is itself 15 % too small on average and two thirds of all pixels read too far, so this is a systematic
under-reading of disparity, not the reciprocal skew of symmetric noise. Binned by local texture it is +43 % where
the 7×7 contrast is under 2 grey levels (44 % of all pixels), +21 % at 2–4, +8 % at 4–6 and +2 % above 40.

**Mechanism: the zero-disparity attractor of small-baseline stereo.** Consecutive frames are nearly identical
images, so the source patch at the *same* pixel, which is the hypothesis "infinitely far", always scores well in
NCC; wherever the true-depth peak is weak (dark, textureless, far, or a slow scope) the cost surface tips toward
large depth. This one mechanism gives the sign, the growth with distance, the brightness dependence, the texture
dependence, the synthetic tube's immunity (strong texture everywhere), the patient clips' radius that grows with
camera distance, and the gain normalisation's partial help. It also predicts the cure, which the last two rows
confirm: **sources 3–5 frames away instead of 1–2 remove the bias completely (+0.1 %) at full coverage**, and
COLMAP's own poses remove a further share that exact poses cannot (bundle adjustment absorbs the residual of the
fisheye-to-pinhole undistortion: +25 % → +14 %, the calibration-residual channel measured at about ten points).

**Two channels, separated by texture** (bias of the median relative depth error by 7×7 local contrast):

| sources / poses | < 2 grey levels (44 % of pixels) | 2–4 | 4–6 | 6–40 | > 40 |
|---|---|---|---|---|---|
| ±1–2, ground-truth poses | +43 % | +21 % | +8 % | +8..12 % | +2 % |
| ±1–2, COLMAP poses | +30 % | +11 % | +0.3 % | −0.6..+2.6 % | −5 % |
| 1–5, ground-truth poses | +6 % | +2 % | −1 % | ±2 % | −8 % |
| 3–5 only, ground-truth poses | **+1.5 %** | −0.2 % | −1.4 % | ±2 % | −9 % |

The attractor lives in the textureless pixels and is cured by baseline alone (+43 → +1.5 %); the residual bias in
*textured* pixels under exact poses (+8–12 %) is the calibration-residual channel, cured by poses that are
bundle-adjusted to the images (+0.3 %). Figure `docs/figures/c3vd_range_bias.png`.

**Same code path as the patient clips.** The airway station analysis run on the phantom's SfM-pose maps gives a
Theil–Sen slope of −0.017 R per R (8 stations, 38 % positive), because the ±2 bias is nearly flat with depth inside
the station range there and cancels against the canonical; the airway's slope is the same mechanism in a regime with
a smaller lumen and steeper light falloff. The station slope on the phantom, by variant:

| phantom, station analysis (Theil–Sen) | slope R per R | positive at | far − near |
|---|---|---|---|
| ground-truth poses, ±2, raw 800 px | +0.048 | 70 % | +0.068 R |
| ground-truth poses, ±2, raw 1200 px | +0.089 | 91 % | +0.098 R |
| ground-truth poses, ±2, gain-normalised | +0.004 | 56 % | +0.034 R |
| COLMAP poses, ±2, raw | −0.017 | 38 % | −0.010 R |
| **COLMAP poses, sources 3–5, raw (the recipe)** | **−0.005** | 50 % | −0.012 R |

The phantom reproduces the patient clips' signature (a radius that grows with camera distance) with exact poses and
the airway setting, at the size of the gain-normalised 20-V1 slope, and the same two levers move it: photometric
normalisation and poses that are consistent with the images.

**Replication on a second sequence (trans_t2_a, 194 frames, a dwelling scope: median camera step 0.032 mm per
frame, walls at 9–15 mm).** Ground truth verified as before (0.1 % self-consistency, 0.18 mm to the mesh).

| trans_t2_a, ground-truth poses | rel. depth error median / MAD | coverage | station slope (Theil–Sen) |
|---|---|---|---|
| ±2, raw | +89 % / 98 % | 81 % | +0.125 R per R (50 % positive) |
| ±2, gain-normalised | +94 % / 102 % | 81 % | +0.370 R per R |
| sources by baseline ≥ 0.9 mm within 30 frames | **+12 % / 31 %** | 78 % | **−0.014 R per R (46 % positive)** |

With a scope that barely moves, the ±2 window has a baseline of a few hundredths of a millimetre and returns the
attractor almost everywhere (the bias *grows* with texture there: confident matches at zero disparity). Choosing
sources by baseline recovers most of it and takes the station slope to zero; the residual (+56 % below 15 mm,
+60 % in pixels above 180 grey levels, thousands of percent in the most textured bins) sits in the close-range
specular highlights of the silicone, which a saturation mask must remove on top of the baseline rule. This is the
regime of the 26-V2 collapse frames, where the operator stops the scope: without baseline there is no per-frame
stereo, and the rule makes that explicit by finding no sources rather than returning a depth.

**How this joins the velocity bias.** The two nights measured two biases of the same per-frame stereo and they pull
in opposite directions on one knob. The velocity bias of M1 is κ·v·Z/c: it grows with the time gap between the
reference frame and its sources, because the wall moves further in that gap. The range bias measured here shrinks
with the *baseline* between them, i.e. with the same gap for a moving scope. Sources chosen by camera-centre distance
(3–5 frames at 0.3 mm per frame) remove the attractor but hand the moving membrane a longer gap, so on a deforming
airway the motion compensation of M2 is the partner of baseline-adaptive source selection, not an alternative to it:
select the sources for baseline, then compensate their wall motion. The synthetic tube, with strong texture at every
depth, never had the attractor, which is why nights 1–2 saw only the velocity term; the patient clips have both.

**Applied to 20-V1** (`m1/windowed_depth_real.py --min-baseline` chooses sources by camera-centre distance,
`--texture-gate` drops textureless pixels), gain-normalised frames, 337 frames, 800 px:

| 20-V1 | slope R per R (Theil–Sen) | positive at | far − near | area ratio p5–p95 | corr(area, dark lumen) at the best stations | stations passing all sector checks |
|---|---|---|---|---|---|---|
| as captured, ±2 | +0.196 | 100 % | +0.248 R | 0.62–1.42 | −0.57..−0.84 | 0 / 27 |
| gain, ±2 | +0.054 | 100 % | +0.122 R | 0.79–1.31 | −0.47..−0.80 | 4 / 24 |
| gain, sources ≥ 3 median steps | **+0.025** | 90 % | +0.045 R | 0.76–1.22 | −0.32..−0.43 (−0.77 at one) | 4 / 24 |
| gain, sources ≥ 3 steps + texture gate 4 | **+0.016** | 86 % | +0.032 R | 0.75–1.19 | −0.32..−0.41 | 2 / 24 |

The fix acts on the patient clip in the same direction as on the phantom and removes most of what the gain left:
the slope falls another two-fold to three-fold and the area swing narrows, but it does not reach the phantom's zero,
and the image check still fails at the best-observed stations (weaker, −0.4 instead of −0.8). What remains is
either scale drift of the SfM poses along the path, a residual attractor at this lumen's steeper light falloff, or
real wall motion correlated with where the scope happens to be (20-V1 is a malacia clip): separating these needs
the CT-paired calibre of 20-V1 as truth, which is the next measurement.

**Applied to the 26-V2 collapse (canonical-as-map run of the afternoon, now with sources ≥ 3 median steps):**
every one of the 166 frames, including the 14 collapse frames, gets a depth map; during the dwell the baseline rule
reaches across the event for its sources (f1903–1904 before, f1918–1927 after), which is the trade-off of the
previous paragraph made concrete. The result answers three open questions of the afternoon at once:

| 26-V2, model 1, post-event canonical | ±1–2 sources (afternoon) | sources ≥ 3 median steps |
|---|---|---|
| range slope, Theil–Sen | +0.041 R per R | **+0.007 R per R** (65 % positive, far − near +0.009 R) |
| carried frames vs registered neighbours, far stations (1.7–2.1 R) | 0.4–0.7 R off | **within 0.05 R at every station** |
| area ratio right after the event (lost-wall leakage) | 2.2–2.6 | 1.03–1.16 |
| nearest station to the event (3.0 R, camera 1.1 R away) | area min 0.77, inward arc −0.18 R, opposite −0.12 R | area 0.72 → **0.78** → 1.03; **inward arc −0.38 R at +5°, opposite arc −0.01 R, 15 of 27 sectors still** |
| dark-lumen fraction (pose-free) | 0.059 → 0.047 → 0.060 | same |

The carried poses were right all along (the far-station offsets were the range bias), and the collapse now reads as
what the identifiability claim predicts a membranous collapse to look like: one arc folding inward by a third of the
radius while the opposite arc stays still. One caveat keeps it qualitative: the collapse frames' sources come from
before and after the event, so the moving membrane is matched against a wall in a different state; making the
−0.38 R quantitative is exactly the job of the M2 velocity compensation, applied on top of baseline selection.

### Where things stand after night 3

- The range-dependent radius bias that blocked every real-data result has a measured mechanism (the zero-disparity
  attractor of small-baseline stereo, plus a smaller calibration-residual channel) and a validated cure: choose
  stereo sources by baseline, not by frame index, and take poses from bundle adjustment. On the phantom with truth
  this turns +25 % / 42 % into −1.3 % / 9.4 % at full coverage; on 20-V1 it removes most of the remaining slope;
  on 26-V2 it validates the carried poses and produces the first sectoral reading of a real collapse.
- The two nights' biases are one knob pulled two ways: baseline cures the attractor and lengthens the wall's travel
  between reference and sources, so M2's motion compensation is the necessary partner of baseline selection on a
  deforming wall. The synthetic results of nights 1–2 stand (strong texture, no attractor) and now have their
  real-data counterpart.
- Open: the last +0.02 R per R on 20-V1 (SfM scale drift, residual attractor, or real malacia correlated with scope
  position; the CT-paired calibre is the truth to use), a saturation mask for specular highlights (trans_t2_a),
  and M2 on the baseline-selected real frames.

Tools added tonight: `c3vd/prepare.py`, `c3vd/stereo_vs_truth.py`, `c3vd/figure.py`; `m1/windowed_depth_real.py
--min-baseline / --max-gap / --texture-gate`.

Tools added: `m2/real_extract.py` (photometric modes), `m2/real_range_slope.py`, `m2/mc_sweep_vsearch_real.py`,
`m1/bridge_models.py` (Sim(3) bridge through common features, rejected here on evidence), `m1/extrapolate_poses.py`,
`m1/interpolate_poses.py` (now any model, dangling tracks filtered), `m2/real_event_report.py`,
`m2/register_to_canonical.py` (kept with its measured failure).

Code: `m2/mc_sweep.py`, `m2/mc_sweep_vsearch.py`, `m2/mc_sweep_vsearch2.py`, `m2/iterate.sh`, `m2/eval_velocity.py`,
`m2/real_periodicity.py`, `m2/real_checks.py`, `m2/real_sector_estimator.py`, `m2/summary_figure.py`,
`synthetic/distort_frames.py`.
