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

3. A test-setup error rather than a generator bug, caught by the dense comparison: the renders are pure pinhole
   (no lens distortion is applied), but the pipeline was first given the real scope's calibration and so
   undistorted images that were never distorted. Poses were unaffected (they are governed by the image centre),
   but the reconstructed wall shell sat 0.4 to 0.6 mm inside the true wall everywhere. A two-view simulation
   reproduces the sign and size (a wall point at 415 px image radius triangulates 0.28 mm too close to the axis;
   the far periphery is worse). Every run directory now contains `intrinsics_pinhole.json`, which is what the
   pipeline must be given. The same mechanism means that on real video an error in the distortion coefficients
   biases calibre, not just sharpness.

The general lessons for the deformation work: on a tube, "N of N frames registered in one model" is not evidence
that the poses are right, so every claim needs the alignment-free relative-rotation check in `pose_diag.py`; and
the dense cloud must be compared to the truth in millimetres with the scale taken from the cameras, because a scale
fitted to the calibre profile (as CT registration does) would silently absorb a uniform radius bias.

`verify.png` in each run directory shows CSA(t) at three stations, membrane-versus-cartilage displacement, the
camera path, and rendered frames before, at and after the event. Look at it: the frames must go round, slit,
round for the collapse and stay round for the static tube.

## 3. Rigid-pipeline cross-check (tens of minutes, GPUs)

The unchanged `airway-recon-colmap` pipeline is run on the rendered videos, then scored against the exact poses
and cross-sections:

```bash
export BRONCHO_COLMAP=<colmap binary> ; BT=<path to airway-recon-colmap checkout>
python $BT/pipeline/recover_clip.py --video runs/synth_static_30.mp4   --calib runs/synth_static_30/intrinsics_pinhole.json   --session synth_static   --lo 0 --hi 179 --gpus 0,1 --out runs/rigid_static   > runs/rigid_static/run.log
python $BT/pipeline/recover_clip.py --video runs/synth_collapse_30.mp4 --calib runs/synth_collapse_30/intrinsics_pinhole.json --session synth_collapse --lo 0 --hi 179 --gpus 2,3 --out runs/rigid_collapse > runs/rigid_collapse/run.log
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
- **Collapse.** A rigid model cannot represent the 75 % transient. What actually happens (see Results): the
  cameras stay exact, because the rigid cartilage half of the wall anchors them even while the other half moves
  8.8 mm; the moving membrane is fused into the map as a smeared shell pulled into the lumen in the event zone;
  and the pipeline's own gated measurement reads a normal tube, so the event is simply invisible. That is the
  baseline the canonical + deformation method (M1) has to beat: same poses, but a membrane that is where it was
  at each instant and a CSA(t) that shows the event.
- **Breathing and malacia.** Periodic wall motion over the whole clip (12 % and 45 % area change on the
  posterior sector), no transient. They bracket what a rigid map does to a continuously moving wall.

## Results (2026-09-15, final generator, pinhole intrinsics, unchanged `airway-recon-colmap` pipeline)

All four scenarios: 180 frames at 30 fps, 6 s, 60 mm tube of 9.8 mm equivalent diameter, camera advancing 6 mm/s
with 0.35 mm lateral jitter and 2 deg tilt, 26-V2 focal length and principal point, no distortion.

| scenario | wall motion | registered | reproj | camera centre RMSE | viewing axis err | relative-rotation err (median / max) | dense wall vs truth | calibre D_CE error | pipeline's own %obstruction | truth |
|---|---|---|---|---|---|---|---|---|---|---|
| static | none | 180/180, 1 model | 1.29 px | 0.005 mm (max 0.016) | 0.01 deg | 0.01 / 0.02 deg | posterior −4.95, anterior +4.78 (truth ±5.00); 0 % of points inside the lumen | 0.08 mm (bias −0.05) | 5 %, CV 0.02: normal tube | ring corrugation 9 % |
| collapse | 12 % breathing + 76 % transient at 3 s, posterior half | 180/180, 1 model | 1.39 px | 0.063 mm (max 0.157); 0.021 mm during the event | 0.12 deg | 0.04 / 0.11 deg | far from event: median offset −0.006 mm (IQR −0.06 … +0.11); in the event zone: cartilage +0.01 mm, membrane −0.32 mm with 0.89 mm IQR and 20 % of membrane points > 1 mm inside the lumen | 0.47 mm (bias −0.46, all from the event zone) | 14 %, CV 0.05: "measurable tube, within noise floor" | 76 % event |
| breathing | 13 % periodic, posterior 120 deg sector, membrane amplitude 1.8 mm | 180/180, 1 model | 1.35 px | 0.018 mm (max 0.044) | 0.02 deg | 0.02 / 0.06 deg | cartilage −0.003 mm; membrane median −0.07 mm but 0.91 mm IQR, 21 % of membrane points > 1 mm inside the lumen, posterior midline at x = −3.3 (truth −5.0) | 0.47 mm (bias −0.42) | 13.8 %, CV 0.05: normal tube | 13 % twice per breath; time-mean area 94 % of canonical |
| malacia | 47 % periodic, posterior 120 deg sector, membrane amplitude 6.9 mm | 129/180 in the main model: the two breathing peaks (f36–58, f122–149; 35–45 % reduction) are dropped, one of them re-appears as a 22-frame fragment at a nonsense scale | 1.29 px | 0.031 mm (max 0.069) on the 129 | 0.05 deg | 0.01 / 0.07 deg | cartilage +0.004 mm; membrane sector holds 8 % of points for 33 % of the circumference (mostly missing) and its midline is smeared to x = −3.3 (truth −5.0) | 0.52 mm (bias −0.21) | 8.6 %, CV 0.03: normal tube | 47 % twice per breath; time-mean area 78 % of canonical |

What this establishes:

1. **M0 is consistent end to end.** A pipeline that knows nothing about the generator recovers the static tube's
   camera path to 5 µm and its calibre to 0.08 mm from the rendered video alone. Renders, poses, intrinsics and
   geometry agree; the ground truth can be trusted for M1.
2. **The rigid baseline is far stronger than the patient data suggested.** The CT-validated calibre error on
   patient videos is about 1 mm; on a perfect synthetic capture it is 0.08 mm. The patient error is capture and
   real-world effects, not the reconstruction.
3. **Rigid SfM poses survive a 76 % wall collapse.** The cartilage half of the wall anchors the cameras; the frames
   during the event are the best-registered of the clip. Any M1 method must at least match these poses.
4. **What the rigid map gets wrong is the deforming wall, and it gets it wrong silently.** The membrane is fused as a
   smeared shell pulled into the lumen (a fifth of its points more than 1 mm inside), the cartilage next to it is
   exact, and the pipeline's own gated measurement reports a normal tube. A clinician reading the rigid output
   would never know the airway closed by three quarters. This is the failure M1 exists to fix, and the numbers
   above are the baseline it is compared against.

Machine-generated summary (`python synthetic/summarize_rigid.py static collapse breathing malacia`, also written to
`runs/rigid_summary.json`; D_CE numbers are whole-tube means and so include the smeared membrane where there is one):

| scenario | wall motion | models | registered | reproj px | centre RMSE mm | axis err deg | rel-rot err median / max deg | straightness est vs GT | D_CE err mm | D_CE bias mm | 1-mm slabs | pipeline %obstruction | GT %obstruction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static | none | 1 | 180/180 | 1.29 | 0.01 | 0.0 | 0.01 / 0.02 | 0.0132 vs 0.0132 | 0.08 | −0.05 | 47/58 | 5 % (48 st., CV 0.02) | 9 % |
| collapse | 75 % collapse @ 3 s | 1 | 180/180 | 1.39 | 0.06 | 0.1 | 0.04 / 0.11 | 0.0133 vs 0.0132 | 0.47 | −0.46 | 47/58 | 14 % (48 st., CV 0.05) | 7 % static / 76 % at event |
| breathing | 12 % periodic | 1 | 180/180 | 1.35 | 0.02 | 0.0 | 0.02 / 0.06 | 0.0133 vs 0.0132 | 0.47 | −0.42 | 47/58 | 14 % (48 st., CV 0.05) | 9 % |
| malacia | 45 % periodic | 2 | 129/180 | 1.29 | 0.03 | 0.1 | 0.01 / 0.07 | 0.0135 vs 0.0136 | 0.52 | −0.21 | 47/58 | 9 % (48 st., CV 0.03) | 9 % |

Earlier runs on the same scenarios, kept for the record in `runs/old_rigid_*`: with the v0 periodic texture the
static model rolled 45 deg (see bug 2); with the real scope's distortion coefficients applied to the pinhole
renders the dense wall sat 0.4–0.6 mm inside the truth (bug 3). Neither survives the fixes.

Figures: `docs/figures/m0_rigid_static.png`, `docs/figures/m0_rigid_collapse.png` (registered frames, pose
residual against time with the event window marked, dense CSA(z) against the canonical and event-minimum truth),
`docs/figures/m0_verify_collapse.png` (the ground-truth verification figure).
