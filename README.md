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
- [x] M0 verified three ways (unit tests, external ground-truth checks, rigid-pipeline cross-check):
      see `docs/M0_VERIFICATION.md`
- [ ] identifiability experiment on the synthetic tube (rigid-only vs canonical+deformation)
- [ ] streaming front end (pose + depth per frame) on 2-V2 static control
- [ ] canonical/deformation split on 26-V2 across the collapse
- [ ] porcine 4D-CT comparison
