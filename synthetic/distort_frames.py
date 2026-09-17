"""Barrel-distort the pinhole renders of a synthetic run with a real scope's OPENCV coefficients, so the rigid + per-frame
pipeline can be run on them with (a) the exact coefficients and (b) a deliberately wrong calibration. Tests whether a
distortion residual produces the range-dependent radius bias seen on real clips (wall radius growing with the camera's
distance to the station).
Usage: python synthetic/distort_frames.py runs/synth_static_30 --calib <26-V2_intrinsics.json> --out runs/synth_static_dist --k-scale 0.75"""
import os, json, glob, argparse, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--calib", required=True); ap.add_argument("--out", required=True); ap.add_argument("--k-scale", type=float, default=0.75); ap.add_argument("--fps", type=float, default=30.0)
a = ap.parse_args(); os.makedirs(f"{a.out}/frames", exist_ok=True)
c = json.load(open(a.calib)); K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]]); dist = np.array([c["k1"], c["k2"], c["p1"], c["p2"]])
w, h = c["width"], c["height"]
# for each DISTORTED pixel, the ideal (pinhole) pixel it sees: undistortPoints gives normalised ideal coordinates
uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)); pts = np.stack([uu.ravel(), vv.ravel()], 1)[:, None, :]
und = cv2.undistortPoints(pts, K, dist).reshape(h, w, 2); map_x = (und[..., 0] * c["fx"] + c["cx"]).astype(np.float32); map_y = (und[..., 1] * c["fy"] + c["cy"]).astype(np.float32)
frames = sorted(glob.glob(f"{a.run}/frames/f*.png")); vw = None
for p in frames:
    img = cv2.imread(p); d = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    cv2.imwrite(f"{a.out}/frames/{os.path.basename(p)}", d)
    if vw is None: vw = cv2.VideoWriter(f"{a.out}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    vw.write(d)
vw.release()
for name, sc in (("exact", 1.0), (f"k{int(a.k_scale * 100)}", a.k_scale)):
    cc = dict(c); cc["k1"], cc["k2"] = c["k1"] * sc, c["k2"] * sc; cc["params_colmap"] = [cc["fx"], cc["fy"], cc["cx"], cc["cy"], cc["k1"], cc["k2"], cc["p1"], cc["p2"]]
    cc["note"] = f"synthetic renders distorted with the 26-V2 coefficients; this file has k1,k2 scaled by {sc}"; json.dump(cc, open(f"{a.out}_intrinsics_{name}.json", "w"), indent=1)
# copy the ground truth so the M1 scorer finds it next to the frames
for f in ("gt.npz", "intrinsics_pinhole.json", "canonical_mesh.ply"):
    if os.path.exists(f"{a.run}/{f}") and not os.path.exists(f"{a.out}/{f}"): os.symlink(os.path.abspath(f"{a.run}/{f}"), f"{a.out}/{f}")
black = float((cv2.cvtColor(d, cv2.COLOR_BGR2GRAY) < 8).mean())
print(f"{len(frames)} frames distorted -> {a.out}.mp4; k1 {c['k1']:.4f} k2 {c['k2']:.4f}; black border fraction {black:.3f}; calibrations: exact and k1,k2 x {a.k_scale}")
