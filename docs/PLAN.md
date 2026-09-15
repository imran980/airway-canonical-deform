# Plan

## M0 — ground truth we control (synthetic)
`synthetic/deforming_trachea.py` produces, for a chosen scenario, per frame: the deformed surface,
camera pose, the deformation field d(θ, z, t), and the cross-sectional area CSA(z, t).
Scenarios: `static`, `breathing` (periodic membrane motion), `collapse` (a 0.5 s transient posterior
collapse, modelled on 26-V2 f1908–1921), `malacia` (large amplitude breathing, modelled on 20-V1's 45 %).
Done when the tests pass and a rendered sequence exists for each scenario.

## M1 — identifiability experiment
Fit three models to the synthetic sequences from image measurements only:
(a) rigid SfM (the bronchotrust baseline), (b) canonical + unconstrained deformation field,
(c) canonical + posterior-membrane-constrained deformation. Report pose error and CSA(z, t) error.
Prediction: (a) fails on breathing/collapse, (b) absorbs camera motion into tissue motion on the
uniform-contraction cases, (c) recovers both. If (c) does not beat (b) the prior is not doing the work
and the paper needs a different lever. Done when the table exists, whichever way it comes out.

## M2 — streaming front end on real static controls
Per-frame pose + depth (monocular depth network with the pinned intrinsics, poses initialised from the
previous frame), incremental canonical map, no deformation yet. Test: 2-V2 and 19-V1 must reproduce the
CT-validated calibre (0.99 mm pooled, 0.72 mm) with the deformation field held at zero.

## M3 — the collapse case
26-V2 f1600–2100. Success = one canonical map across f1908–1921 with the collapse expressed in the
deformation field, and the dark-lumen trace of the video (bronchotrust `fig_wallcollapse_26-V2`)
reproduced by the model's CSA(t) at the affected station.

## M4 — the failures a rigid pipeline loses
20-V2, 14-V1, 31-V1, 16-V2, 13-V1 (dynamic wall motion class). Report how many become measurable.

## M5 — 4D-CT (porcine)
Respiratory-gated thin-slice CT, ≤ 1 mm, 8–10 phases, same animal bronchoscoped. Compare CSA(z, phase)
from CT against the model's canonical + deformation output at one fixed metric scale. Never re-fit
the scale per phase (the 20-V1 lesson: a re-fitted scale can "match" the collapsed phase spuriously).

## Non-negotiables carried over
Sequential decoding, pinned intrinsics, no per-video tuning, every number with its validity check,
and a span guard on any CT registration.
