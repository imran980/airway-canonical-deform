"""Prepare a C3VD sequence for the range-bias measurement: undistort the 195-degree fisheye colour frames to the pinhole used
by the validated harness (100 deg, 1800 px, f = 755.19), in two photometric variants (raw; per-frame gain to a fixed bright
level), remap the 16-bit ground-truth depth through the same map into float32 mm, and write a COLMAP model with the
ground-truth poses and the pinhole camera.
C3VD conventions: depth = value / 65535 * 100 mm; pose.txt rows = 4x4 [[R,0],[t,1]] row-major, camera-to-world, mm.
Usage: python c3vd/prepare.py /home/mi3dr/dataset/C3VD_official/cecum_t1_a runs/c3vd_cecum [--fov 100 --size 1800]"""
import os, re, glob, json, argparse, subprocess, tempfile, shutil, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("out"); ap.add_argument("--fov", type=float, default=100.0); ap.add_argument("--size", type=int, default=1800); ap.add_argument("--stride", type=int, default=1)
a = ap.parse_args(); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap")
I = dict(cx=678.544839263292, cy=542.975887548343, a0=769.243600037458, a2=-0.000812770624150226, a3=6.25674244578925e-07, a4=-1.19662182144280e-09, c=0.999986882249990, d=0.00288273829525059, e=-0.00296316513429569)
St = np.array([[I["c"], I["d"]], [I["e"], 1.0]]); rho = np.linspace(0, 950, 4000); z = I["a0"] + I["a2"] * rho ** 2 + I["a3"] * rho ** 3 + I["a4"] * rho ** 4; theta = np.arctan2(rho, z)
good = np.concatenate([[True], np.diff(theta) > 0]); rho, theta = rho[good], theta[good]
S = a.size; f = (S / 2) / np.tan(np.radians(a.fov / 2)); cx = cy = S / 2
u, v = np.meshgrid(np.arange(S), np.arange(S)); x = (u - cx) / f; y = (v - cy) / f; rr = np.hypot(x, y); th = np.arctan(rr); phi = np.arctan2(y, x)
rr_fish = np.interp(th, theta, rho, right=np.nan); uvpp = np.stack([rr_fish * np.cos(phi), rr_fish * np.sin(phi)], -1); uvp = uvpp @ St.T
map_x = (uvp[..., 0] + I["cx"]).astype(np.float32); map_y = (uvp[..., 1] + I["cy"]).astype(np.float32); valid = np.isfinite(rr_fish)
np.save(f"{a.out}_valid_mask.npy", valid) if False else None
for d in ("images_raw", "images_n", "gt_depth"): os.makedirs(f"{a.out}/{d}", exist_ok=True)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)); n = 0; frames = []
for fp in sorted(glob.glob(f"{a.seq}/*_color.png"), key=lambda p: int(re.match(r"(\d+)_color", os.path.basename(p)).group(1))):
    k = int(re.match(r"(\d+)_color", os.path.basename(fp)).group(1))
    if k % a.stride: continue
    img = cv2.imread(fp); warp = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=0); warp[~valid] = 0
    cv2.imwrite(f"{a.out}/images_raw/f{k:05d}.png", warp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    lab = cv2.cvtColor(warp, cv2.COLOR_BGR2LAB); L = lab[:, :, 0].astype(np.float32); p95 = np.percentile(L[valid], 95); L = np.clip(L * (150.0 / max(p95, 1.0)), 0, 255)
    lab[:, :, 0] = clahe.apply(L.astype(np.uint8)); im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR); im[~valid] = 0; cv2.imwrite(f"{a.out}/images_n/f{k:05d}.png", im, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    dep = cv2.imread(f"{a.seq}/{k:04d}_depth.tiff", cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0 * 100.0
    dw = cv2.remap(dep, map_x, map_y, cv2.INTER_NEAREST, borderValue=0); dw[~valid] = 0; np.savez_compressed(f"{a.out}/gt_depth/f{k:05d}.npz", d=np.clip(np.round(dw * 100), 0, 65535).astype(np.uint16)); frames.append(k); n += 1
poses = np.loadtxt(f"{a.seq}/pose.txt", delimiter=","); os.makedirs(f"{a.out}/gt_model", exist_ok=True)
def R2q(R):
    t = np.trace(R)
    if t > 0: s = np.sqrt(t + 1) * 2; return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R))); j, kk = (i + 1) % 3, (i + 2) % 3; s = np.sqrt(1 + R[i, i] - R[j, j] - R[kk, kk]) * 2; q = np.zeros(4); q[0] = (R[kk, j] - R[j, kk]) / s; q[1 + i] = 0.25 * s; q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + kk] = (R[kk, i] + R[i, kk]) / s; return q
td = tempfile.mkdtemp()
open(f"{td}/cameras.txt", "w").write(f"1 PINHOLE {S} {S} {f} {f} {cx} {cy}\n"); open(f"{td}/points3D.txt", "w").write("")
with open(f"{td}/images.txt", "w") as fo:
    for i, k in enumerate(frames):
        M = poses[k]; Rc2w = M[:12].reshape(3, 4)[:, :3].T; C = M[12:15]; Rw2c = Rc2w.T; t = -Rw2c @ C; q = R2q(Rw2c)
        fo.write(f"{i + 1} {' '.join(map(repr, q))} {' '.join(map(repr, t))} 1 f{k:05d}.png\n0 0 -1\n")
r = subprocess.run([COLMAP, "model_converter", "--input_path", td, "--output_path", f"{a.out}/gt_model", "--output_type", "BIN"], capture_output=True, text=True, errors="replace"); shutil.rmtree(td)
if r.returncode != 0: print(r.stderr[-800:])
json.dump(dict(camera_model="PINHOLE", width=S, height=S, params_colmap=[f, f, cx, cy], fov_deg=a.fov, seq=a.seq, n_frames=n, depth_mm_per_unit=100 / 65535), open(f"{a.out}/pinhole.json", "w"), indent=1)
print(f"{n} frames -> {a.out}: images_raw, images_n, gt_depth (uint16 hundredths of mm, compressed, pinhole grid), gt_model (GT poses, PINHOLE f={f:.2f} {S}x{S}); valid {valid.mean():.0%}")
