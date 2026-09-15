# Verifying M0 (the synthetic ground truth)

M0 is trusted only if three independent layers agree. The first two are internal to this repository; the third
uses a pipeline that knows nothing about the generator.

## 1. Unit tests on the generator (seconds)

```bash
python -m pytest tests/ -q          # 6 tests
```

They check, on a small instance of every scenario: the static tube has zero deformation; the cartilage sector
never moves while the membrane does; the collapse minimum sits at the event time and at the event station and is
near-occlusive there but absent four sigma away; the uniform scenario moves the whole circumference; the camera
advances monotonically with proper rotations; the saved gt.npz round-trips and the malacia scenario loses the
requested area.

## 2. verify_m0.py: the stored ground truth checked from outside the generator (a minute)

```bash
python synthetic/deforming_trachea.py --scenario static   --out runs/synth_static_30   --fps 30 --duration 6 --calib <intrinsics.json>
python synthetic/deforming_trachea.py --scenario collapse --out runs/synth_collapse_30 --fps 30 --duration 6 --calib <intrinsics.json>
python synthetic/verify_m0.py runs/synth_static_30   --video runs/synth_static_30.mp4
python synthetic/verify_m0.py runs/synth_collapse_30 --video runs/synth_collapse_30.mp4
```

Each run prints a PASS/FAIL line per check and ends with `=> M0 VERIFIED` or `=> M0 HAS FAILURES`:

| check | what it proves |
|---|---|
| cartilage sector rigid | the anatomical prior is actually built into the data (max displacement on rings = 0) |
| membrane peak displacement | the stored field reaches exactly the solved amplitude, at the time breathing and collapse add |
| collapse minimum at t0, depth = target, far station only breathes, lumen recovers | the event is where, when and as deep as requested, and is local in space and time |
| camera speed / rotations valid / jitter bounded | the pose track is what the parameters say |
| area recomputed from displacement field matches stored CSA | CSA(z,t) is a function of the geometry, not a separately typed number |
| canonical mesh cross-section = canonical CSA | the mesh that gets rendered is the same geometry the CSA was computed from |
| texture has no (roll, z-shift) symmetry | rigid SfM cannot lock onto a rolled or ring-shifted copy of the tube (see below). The colour explained by the ring geometry is regressed out first, so only the mucosal texture is tested; the origin's own correlation blob is flood-filled and any peak outside it above 0.35 fails |
| texture correlation length is short | the texture actually carries information at the scale SfM needs: the origin blob is shorter than one ring period in z and narrower than 90° in roll |

Two real bugs were caught by these checks, one by the mesh check and one by the rigid cross-check that then became
the texture check:

1. The anterior-wall cap in `deform_xy` clipped every anterior point by 0.3 mm even with zero displacement, so the
   rendered mesh and the stored canonical CSA disagreed by 0.9 %. The cap now applies only to displaced points and
   follows the wall's circle.
2. The v0 texture was helical: streaks followed `cos(3θ + 0.15 z)` on top of rings periodic in z. Rolling the tube
   by 0.2 rad (11.5°) while shifting it by one ring period (4 mm) reproduced the scene exactly. On the static video
   the rigid pipeline registered all 180 frames at 1.2 px and then, from frame 35 on, rolled the model in steps of
   11.5° and 22.9° (one and two ring periods) to a total of 45°, with the wrong scale, while reporting nothing wrong.
   Whether a given run hit the symmetry was luck: the run before it, on frames differing only by the 0.3 mm cap,
   was exact. Real mucosa has no such symmetry, so the texture is now aperiodic random Fourier fields on the
   cylinder surface, and the rings are jittered per ring in spacing (±15 %) and depth (±25 %), as real tracheal
   rings are, so the geometry has no exact period either. `verify_m0.py` measures the high-passed luminance
   autocorrelation over (z-shift, roll) and fails if any off-origin peak exceeds 0.35. The old texture scores 0.95
   at one ring period and a 12° roll; the new one passes.

The general lesson for the deformation work: on a tube, "N of N frames registered in one model" is not evidence
that the poses are right. Every claim needs the alignment-free relative-rotation check in `pose_diag.py`.

`verify.png` in each run directory shows CSA(t) at three stations, membrane-versus-cartilage displacement, the
camera path, and rendered frames before, at and after the event. Look at it: the frames must go round, slit,
round for the collapse and stay round for the static tube.

## 3. Rigid-pipeline cross-check (tens of minutes, GPUs)

The unchanged `airway-recon-colmap` pipeline is run on the rendered videos, then scored against the exact poses
and cross-sections:

```bash
export BRONCHO_COLMAP=<colmap binary> ; BT=<path to airway-recon-colmap checkout>
python $BT/pipeline/recover_clip.py --video runs/synth_static_30.mp4   --calib <intrinsics.json> --gpus 0,1 --out runs/rigid_static   > runs/rigid_static/run.log
python $BT/pipeline/recover_clip.py --video runs/synth_collapse_30.mp4 --calib <intrinsics.json> --gpus 2,3 --out runs/rigid_collapse > runs/rigid_collapse/run.log
BRONCHOTRUST=$BT python synthetic/score_rigid.py runs/rigid_static   runs/synth_static_30/gt.npz
BRONCHOTRUST=$BT python synthetic/score_rigid.py runs/rigid_collapse runs/synth_collapse_30/gt.npz
```

`score_rigid.py` aligns the recovered camera centres to the ground truth with a similarity transform, reports the
registered fraction, recovered scale, centre RMSE and optical-axis error per model, maps the dense cloud into
millimetres with that transform and measures CSA(z) with an estimator independent of the pipeline, and finally
runs the pipeline's own gated `measure_csa` to see what %obstruction it would report.

What the outcome means:

- **Static tube.** The rigid pipeline must register nearly all frames in one model, the aligned camera path must
  match the truth to well under a millimetre, and the calibre from the dense cloud must match the canonical 9.8 mm
  tube within the paper's CT-validated error (about 1 mm). If it does, the renders, poses, intrinsics and geometry
  are mutually consistent and the rigid baseline is confirmed to work on this data. If it does not, either the
  synthetic data is wrong or the rendering is too poor for real features, and nothing downstream can be trusted.
- **Collapse.** A rigid model cannot represent the 75 % transient: either the model fragments around t0, or the
  frames near the event register with a larger pose residual and the moving membrane is smeared into the dense
  cloud. Whichever happens is recorded, because it is the failure the canonical + deformation method (M1) has to
  fix and the baseline number M1 will be compared against.

## Results

_(filled in from the run below)_
