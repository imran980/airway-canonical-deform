"""Prepare a real clip for per-frame stereo without throwing away field of view or resolution.

An endoscope frame is a bright disc inside a black rectangle: undistorting the whole rectangle to a pinhole wastes most
of the canvas on black and leaves the airway at a fraction of the available pixels, and COLMAP's undistorter crops the
periphery, which is exactly where the far wall appears. This crops to the disc, undistorts with the clip's own OPENCV
coefficients to a pinhole that covers the WHOLE original field, and writes images plus a sparse model carrying the
existing poses with the new PINHOLE camera (undistortion does not change a camera's pose).

Usage: python m2/real_prepare_hires.py runs/real_20V1_n --calib <intrinsics.json> --out runs/real_20V1_hires --size 900 [--lo 2540 --hi 2700]
"""
import argparse, os, glob, json, subprocess, tempfile, shutil, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("workspace"); ap.add_argument("--calib", required=True); ap.add_argument("--out", required=True); ap.add_argument("--model", default="sparse/0")
ap.add_argument("--size", type=int, default=900); ap.add_argument("--fov-margin", type=float, default=1.02, help="how much beyond the disc's own angular radius the pinhole should cover")
ap.add_argument("--lo", type=int, default=None); ap.add_argument("--hi", type=int, default=None); ap.add_argument("--clahe", type=float, default=2.0)
a = ap.parse_args(); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap"); os.makedirs(f"{a.out}/images", exist_ok=True)
J = json.load(open(a.calib)); K = np.array([[J["fx"], 0, J["cx"]], [0, J["fy"], J["cy"]], [0, 0, 1]]); D = np.array([J["k1"], J["k2"], J["p1"], J["p2"]])
srcs = sorted(glob.glob(f"{a.workspace}/images/f*.png"))
if a.lo is not None: srcs = [p for p in srcs if a.lo <= int(os.path.basename(p)[1:6]) <= a.hi]
if not srcs: raise SystemExit("no source frames")
# the disc: union of non-black pixels over a sample of frames
acc = None
for p in srcs[::max(1, len(srcs) // 40)]:
    g = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY); m = (g > 8).astype(np.uint8)
    acc = m if acc is None else np.maximum(acc, m)
ys, xs = np.where(acc > 0); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
print(f"disc bounding box in the source: x {x0}-{x1}, y {y0}-{y1} ({x1-x0+1} x {y1-y0+1} px of {acc.shape[1]} x {acc.shape[0]})")
# the disc's angular radius under the true (distorted) model: undistort its rim and take the largest ray angle
rim = np.array([[[x0, (y0 + y1) / 2]], [[x1, (y0 + y1) / 2]], [[(x0 + x1) / 2, y0]], [[(x0 + x1) / 2, y1]]], dtype=np.float64)
und = cv2.undistortPoints(rim, K, D).reshape(-1, 2); half = float(np.max(np.hypot(und[:, 0], und[:, 1]))) * a.fov_margin
S = a.size; f_new = (S / 2) / half; cx_new = cy_new = S / 2
print(f"pinhole output {S}x{S}, focal {f_new:.1f} px, half-field {np.degrees(np.arctan(half)):.1f} deg (full field {2*np.degrees(np.arctan(half)):.1f} deg)")
u, v = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
xn = (u - cx_new) / f_new; yn = (v - cy_new) / f_new
pts = np.stack([xn.ravel(), yn.ravel(), np.ones(xn.size, np.float32)], 0)
proj, _ = cv2.projectPoints(pts.T.reshape(-1, 1, 3).astype(np.float64), np.zeros(3), np.zeros(3), K, D)
map_x = proj[:, 0, 0].reshape(S, S).astype(np.float32); map_y = proj[:, 0, 1].reshape(S, S).astype(np.float32)
clahe = cv2.createCLAHE(clipLimit=a.clahe, tileGridSize=(8, 8)); n = 0
for p in srcs:
    img = cv2.imread(p); w = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    lab = cv2.cvtColor(w, cv2.COLOR_BGR2LAB); L = lab[:, :, 0]
    mask = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY) > 8
    lab[:, :, 0] = clahe.apply(L); w2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR); w2[~mask] = 0
    cv2.imwrite(f"{a.out}/images/{os.path.basename(p)}", w2, [cv2.IMWRITE_PNG_COMPRESSION, 1]); n += 1
# the model: same poses, new PINHOLE camera
td = tempfile.mkdtemp(); subprocess.run([COLMAP, "model_converter", "--input_path", f"{a.workspace}/{a.model}", "--output_path", td, "--output_type", "TXT"], capture_output=True, text=True, errors="replace")
L = [l for l in open(f"{td}/images.txt") if l.strip() and not l.startswith("#")]
keep = set(os.path.basename(p) for p in srcs)
open(f"{td}/cameras.txt", "w").write(f"1 PINHOLE {S} {S} {f_new} {f_new} {cx_new} {cy_new}\n")
with open(f"{td}/images.txt", "w") as fo:
    for hdr, pts_ in zip(L[0::2], L[1::2]):
        v_ = hdr.split()
        if v_[9] not in keep: continue
        fo.write(" ".join(v_[:8]) + f" 1 {v_[9]}\n" + "0 0 -1\n")
open(f"{td}/points3D.txt", "w").write("")
for x in ("frames.txt", "rigs.txt"):
    if os.path.exists(f"{td}/{x}"): os.remove(f"{td}/{x}")
os.makedirs(f"{a.out}/sparse/0", exist_ok=True)
r = subprocess.run([COLMAP, "model_converter", "--input_path", td, "--output_path", f"{a.out}/sparse/0", "--output_type", "BIN"], capture_output=True, text=True, errors="replace")
shutil.rmtree(td)
if r.returncode: print(r.stderr[-500:])
json.dump(dict(source=a.workspace, size=S, focal=f_new, full_field_deg=2 * np.degrees(np.arctan(half)), frames=n), open(f"{a.out}/prepare.json", "w"), indent=1)
print(f"{n} frames -> {a.out} (disc fills the frame; poses carried over)")
