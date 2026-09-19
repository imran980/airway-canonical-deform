# airway-canonical-deform

**Streaming canonical airway reconstruction with deformation disentanglement.**
As bronchoscopy frames arrive, estimate camera pose and depth, but do not fuse everything into one rigid
map: maintain a stable **canonical airway** plus a temporary **deformation field** for breathing,
malacia and wall collapse. Rigid, reliable regions update the permanent 3D map; moving or uncertain
tissue is modelled separately instead of corrupting it. Output both the live airway geometry and the
dynamic calibre change.

This repository is the successor to [`airway-recon-colmap`](https://github.com/imran980/airway-recon-colmap) (the pipeline formerly developed under the working name *bronchotrust*), whose
rigid SfM/MVS pipeline and CT-validated measurement layer are the baseline and the source of the real
test data. The motivation comes straight from that cohort: **8 of the 11 airways it could not measure
failed on dynamic wall motion**, one examination (26-V2) breaks into two rigid models at a 14-frame
posterior-wall collapse, and one patient (20-V1) has paired inspiratory/expiratory CT documenting a 45 %
expiratory reduction in tracheal area.

## The scientific claim, stated precisely

The architecture — canonical map plus a time-varying deformation field — is standard (D-NeRF, Nerfies,
EndoNeRF, Endo-4DGS, deformable-Gaussian SLAM). The contribution is not the architecture. It is the
**identifiability problem specific to a forward-looking endoscope in a tube**: with near-zero parallax,
a uniform radial contraction of the wall and a camera translation along the axis produce nearly the
same image change, and a monocular system cannot separate them from photometry alone.

We break the ambiguity with anatomy. In the trachea the **cartilage rings are rigid** and the
**posterior membranous wall is the only structure that deforms**. Rings define the canonical map and
anchor the camera; the deformation field is confined to the posterior sector. The prior is physically
justified, it makes the disentanglement well-posed, and it fails loudly when violated (a lesion on the
anterior wall does not fit) rather than silently absorbing camera motion into fake tissue motion.

## Evaluation plan (in the order it can be done)

1. **Synthetic deforming trachea** (`synthetic/`, needs no new data): a tube with rigid rings and a
   posterior membrane driven by a known time function (breathing plus an optional transient collapse),
   rendered through the calibrated bronchoscope model along a known camera path. Exact ground truth for
   pose and deformation per frame. This is where identifiability is proven or disproven.
2. **Existing videos as a testbed** (`data/manifest.json`, from `bronchotrust`):
   - 26-V2: the transient collapse at f1908–1921 that splits rigid SfM — one canonical map should
     survive it;
   - 20-V1: inspiratory/expiratory CT gives a two-phase check on the deformation magnitude;
   - the 8 wall-motion failures a rigid pipeline loses and this method should recover;
   - 2-V2 and 19-V1: CT-validated static tubes where the deformation field must stay near zero.
3. **Porcine 4D-CT** (pending): respiratory-gated thin-slice CT of the same animal that is
   bronchoscoped, 8–10 phases, for true lumen shape at each phase.

"Real time" is a result to report, not a promise; the first target is *streaming* (frames processed in
order, map updated incrementally).

## Layout

```
synthetic/deforming_trachea.py   ground-truth generator: canonical tube, rigid rings, posterior-membrane
                                 deformation, camera path, per-frame CSA, meshes, renders (--render)
synthetic/verify_m0.py           checks the stored ground truth from outside the generator; verify.png + .mp4
synthetic/score_rigid.py         scores a rigid airway-recon-colmap workspace against the ground truth
                                 (poses via Sim(3), dense cloud in mm, pipeline's own %obstruction)
synthetic/pose_diag.py           alignment-free pose diagnostics (when/where a rigid model went wrong)
synthetic/summarize_rigid.py     runs score_rigid + pose_diag over the scenarios and prints one results table
m1/windowed_depth.py             M1 v0 on a synthetic run: per-frame ±w stereo, canonical cartilage, membrane displacement,
                                 CSA(z,t) and rigid-map baseline against truth; interpolate_poses.py, combine_sides.py,
                                 summarize_m1.py; real_sfm_relaxed.py + windowed_depth_real.py for real clips
docs/M1_RESULTS.md               M1 results, ablations, the velocity-bias analysis, 26-V2 status
data/manifest.json               the real testbed cases and where their videos / calibrations / clouds live
docs/PLAN.md                     milestones and what "done" means for each
docs/M0_VERIFICATION.md          how M0 is verified, and the results
tests/                           checks on the generator (rigid stations constant, membrane tracks A(t))
```

Calibration, frame decoding and the measurement gates are reused from `bronchotrust/pipeline`
(set `BRONCHOTRUST=/path/to/bronchotrust`).

## Status

- [x] problem statement and prior written down
- [x] synthetic generator v0 (geometry, deformation, poses, CSA ground truth; renders when an offscreen
      GL context is available)
- [x] M0 verified three ways (unit tests, external ground-truth checks, rigid-pipeline cross-check), after
      three catches the checks made: an anterior-wall clipping bug, a helical texture symmetry that let rigid SfM
      register a rolled copy of the tube, and a pinhole-vs-distorted intrinsics mismatch in the cross-check.
      On the final data the unchanged rigid pipeline recovers the static tube's cameras to 5 µm and its calibre to
      0.08 mm, keeps exact poses through a 76 % posterior collapse, and reports that collapsed airway as a normal
      tube (14 % obstruction): the rigid baseline and its failure are both quantified. See `docs/M0_VERIFICATION.md`
- [x] M1 v0, identifiability on the synthetic tube (`m1/`, results in `docs/M1_RESULTS.md`). Poses from rigid
      SfM are exact whenever a rigid sector is in view (0.005–0.06 mm; 0.82 mm and 0.6° once the whole wall moves,
      confirming the identifiability claim). Per-frame short-window depth with the posterior-translation prior
      recovers the canonical cartilage to 0.04 mm in every scenario, the breathing amplitude, and the collapse at
      the right instant (the rigid map reads a normal tube), and flags the prior violation (0.51 mm cartilage
      deviation on uniform contraction). Its limit is now measured, not guessed: per-frame stereo of a moving wall
      is biased by κ·v·Z/c (κ ≈ 0.3), even in stereo source side, so exact poses are not enough and the wall's
      velocity must enter the depth estimate. That is the M2 design decision.
- [x] M2 depth on the synthetic tube (`m2/`, results in `docs/M2_RESULTS.md`): motion-compensated plane-sweep
      stereo removes the velocity bias completely when the wall motion is known (0.06–0.12 mm in every scenario,
      cartilage untouched), and a velocity search inside the stereo recovers most of it from the images alone
      (breathing 0.68 → 0.14 mm, collapse 0.48 → 0.17, malacia 2.09 → 0.35), gated by 3-of-4 source agreement;
      under-declaring the moving sector is safe, over-declaring breaks it; the rigidity check still flags a violated
      prior. On real clips (20-V1, 26-V2) the per-frame radius follows the camera's distance to the station
      (+0.12..+0.24 R per R; synthetic control ≈ 0), which by itself produces the apparent respiratory swing; four
      ground-truth-free checks (sector, image, magnitude, camera distance) now gate real per-frame estimates and no
      station passes them yet. Brightness normalisation removes half of the bias.
- [x] The real-data range bias, measured against truth on a real endoscope (C3VD): per-frame stereo with sources
      1–2 frames away reads depth +25 % too far (the zero-disparity attractor of small-baseline stereo, worst in dark
      and textureless pixels, growing with distance); sources 3–5 frames away and bundle-adjusted poses make it
      −1.3 % / 9.4 % at full coverage. Applied to 26-V2, carried poses validate within 0.05 R and the collapse
      reads as a sectoral fold (−0.38 R, opposite arc still). `docs/M2_RESULTS.md`, night 3.
- [x] First quantitative wall motion on a real patient clip (26-V2): with per-ring common-mode shifts removed and a
      permutation test against sham windows, a single arc (narrower than 160°, centres agreeing within 12°) folds
      inward at five neighbouring stations over 2.2–3.0 R of wall, deepening to −0.23 R at the event (p ≤ 0.05,
      control arc within ±0.05 R, 14–20 % of the lumen area). `docs/M2_RESULTS.md`
- [ ] The remaining +0.02 R per R on 20-V1 against its CT-paired calibre; a saturation mask for specular
      highlights; a clip where the wall motion is large enough for the compensation to pay for its noise floor
- [ ] streaming front end (pose + depth per frame) on 2-V2 static control
- [ ] canonical/deformation split on 26-V2 across the collapse
- [ ] porcine 4D-CT comparison
