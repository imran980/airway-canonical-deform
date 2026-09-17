"""Streaming front end, pose step: register a frame's depth on the RIGID arc to the canonical tube (the map).

For each requested frame the depth map (from the per-frame stereo run) is back-projected; points whose angle around the
path lies outside the declared moving arc (+ margin) are the rigid wall. The pose is refined as a similarity
(rotation, centre, depth scale s: a carried pose has an uncertain stereo baseline, and a wrong baseline scales every
depth) so that the rigid points' radius matches the canonical wall r_can(station, sector) from the map run, with a
Huber loss and a quadratic prior pulling the pose toward its initial (interpolated) value. Reports the radial
residual before and after, the pose change, and writes a new sparse model with the refined poses for these frames
(other frames unchanged), so the per-frame stereo can be re-run on it.

Usage: python m2/register_to_canonical.py <m1_run> --frames 1908 1920 --arc <centre_deg> --half 60 --out <workspace_out>
       (<m1_run> must still hold dense/stereo/depth_maps and dense/sparse; --workspace is the run's input workspace, whose sparse/0
       supplies the original camera model and whose images are reused by symlink)"""
import os, json, argparse, subprocess, tempfile, shutil, numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
ap = argparse.ArgumentParser(); ap.add_argument("m1_run"); ap.add_argument("--workspace", required=True, help="the workspace whose images/ the new model will use"); ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, nargs=2, required=True); ap.add_argument("--arc", type=float, required=True, help="centre of the MOVING arc, degrees (grid convention)"); ap.add_argument("--half", type=float, default=60.0, help="half-width excluded around the moving arc")
ap.add_argument("--prior", type=float, default=1.0, help="prior weight relative to the data term: a pose change of 1 R (centre), 1 rad or 1 in log depth-scale costs as much as all rigid points at a residual of 1 R"); ap.add_argument("--fix-scale", action="store_true"); ap.add_argument("--stride", type=int, default=3)
ap.add_argument("--joint", action="store_true", help="one correction for the whole frame range, linear in time between a start and an end SE(3) delta (12 parameters), scale fixed; averages the per-frame depth noise")
ap.add_argument("--max-shift", type=float, default=0.3, help="bound on the centre change, R"); ap.add_argument("--max-rot", type=float, default=6.0, help="bound on the rotation change, degrees"); ap.add_argument("--max-logscale", type=float, default=0.35, help="bound on |log depth scale| (0.35: scale within 0.70-1.42)")
a = ap.parse_args(); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap")
def sh(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0: raise RuntimeError(cmd[0] + "\n" + r.stderr[-1500:])
def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
def R2q(R):
    q = Rot.from_matrix(R).as_quat(); return np.array([q[3], q[0], q[1], q[2]])
def read_depth(path):
    with open(path, "rb") as f:
        hdr = b""
        while hdr.count(b"&") < 3: hdr += f.read(1)
        w, h, c = map(int, hdr.decode().split("&")[:3]); return np.fromfile(f, np.float32).reshape(h, w, c)[:, :, 0]
# intrinsics of the UNDISTORTED images (the depth maps live in that geometry) from the dense model; poses and the written model from the workspace model (original camera)
tdd = tempfile.mkdtemp(); sh(["model_converter", "--input_path", f"{a.m1_run}/dense/sparse", "--output_path", tdd, "--output_type", "TXT"])
cam = [l.split() for l in open(f"{tdd}/cameras.txt") if l.strip() and not l.startswith("#")][0]; W, H = int(cam[2]), int(cam[3]); fx, fy, cx, cy = map(float, cam[4:8]); shutil.rmtree(tdd)
td = tempfile.mkdtemp(); sh(["model_converter", "--input_path", f"{a.workspace}/sparse/0", "--output_path", td, "--output_type", "TXT"])
L = [l for l in open(f"{td}/images.txt") if l.strip() and not l.startswith("#")]; imgs = {}
for hdr_, pts in zip(L[0::2], L[1::2]):
    f = hdr_.split(); imgs[f[9]] = dict(hdr=hdr_, pts=pts, q=np.array(list(map(float, f[1:5]))), t=np.array(list(map(float, f[5:8]))))
def fidx(n): return int(os.path.splitext(os.path.basename(n))[0].lstrip("f"))
G = np.load(f"{a.m1_run}/m1_real_grid.npz"); S = G["S"]; s_st = G["s_st"]; R = float(G["R"]); r_can = G["r_can"]; nS, nb = r_can.shape
T = np.gradient(S, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True); N1 = np.zeros_like(S); N2 = np.zeros_like(S); n1 = np.cross(T[0], [0, 0, 1.0]); n1 = n1 if np.linalg.norm(n1) > 1e-3 else np.cross(T[0], [0, 1.0, 0]); n1 /= np.linalg.norm(n1)
for i in range(nS):
    n1 = n1 - (n1 @ T[i]) * T[i]; n1 /= np.linalg.norm(n1); N1[i] = n1; N2[i] = np.cross(T[i], n1)
from scipy.spatial import cKDTree; tree = cKDTree(S); c_rad = np.radians(a.arc); h_rad = np.radians(a.half)
def residuals(Xw):
    """radial residual (units of R) of world points against the canonical wall; NaN where the map has no canonical"""
    _, i = tree.query(Xw); rel = Xw - S[i]; x = (rel * N1[i]).sum(1); y = (rel * N2[i]).sum(1); th = np.arctan2(y, x); rr = np.hypot(x, y)
    b = ((th + np.pi) / (2 * np.pi) * nb).astype(int) % nb; rc = r_can[i, b]; rigid = np.abs(np.angle(np.exp(1j * (th - c_rad)))) > h_rad
    return (rr - rc) / R, rigid & np.isfinite(rc)
report = []
if a.joint:
    sel = [n for n in sorted(imgs, key=fidx) if a.frames[0] <= fidx(n) <= a.frames[1] and os.path.exists(f"{a.m1_run}/dense/stereo/depth_maps/{n}.geometric.bin")]
    data = []
    for n in sel:
        dep = read_depth(f"{a.m1_run}/dense/stereo/depth_maps/{n}.geometric.bin"); h_, w_ = dep.shape; sx, sy = w_ / W, h_ / H; v, u = np.mgrid[0:h_:a.stride, 0:w_:a.stride]; d_ = dep[::a.stride, ::a.stride]; ok = d_ > 0
        Xc = np.stack([(u[ok] - cx * sx) / (fx * sx) * d_[ok], (v[ok] - cy * sy) / (fy * sy) * d_[ok], d_[ok]], 1); Rc2w0 = q2R(imgs[n]["q"]).T; C0 = -Rc2w0 @ imgs[n]["t"]
        r0, rig = residuals((Rc2w0 @ Xc.T).T + C0)
        if rig.sum() < 200: continue
        idx = np.where(rig)[0]; idx = idx[np.linspace(0, len(idx) - 1, min(len(idx), 1500)).astype(int)]
        data.append((n, fidx(n), Xc[idx], Rc2w0, C0, np.median(r0[rig]), np.median(np.abs(r0[rig] - np.median(r0[rig])))))
    k_lo, k_hi = data[0][1], data[-1][1]; npt = sum(len(d[2]) for d in data); wprior = a.prior * np.sqrt(npt) * 0.1
    def pose_at(p, k):
        u = 0.0 if k_hi == k_lo else (k - k_lo) / (k_hi - k_lo); return p[:6] * (1 - u) + p[6:] * u
    def fun(p):
        out = []
        for n, k, Xc, Rc2w0, C0, _, _ in data:
            q = pose_at(p, k); Rc2w = Rot.from_rotvec(q[:3]).as_matrix() @ Rc2w0; C = C0 + q[3:6] * R; r, m = residuals((Rc2w @ Xc.T).T + C); out.append(np.where(m, r, 0.0))
        return np.concatenate(out + [wprior * p])
    rb = np.radians(a.max_rot); lb = np.array(([-rb] * 3 + [-a.max_shift] * 3) * 2); sol = least_squares(fun, np.zeros(12), loss="soft_l1", f_scale=0.15, bounds=(lb, -lb), max_nfev=300); p = sol.x
    print(f"joint fit over f{k_lo}-f{k_hi} ({len(data)} frames, {npt} rigid points): start delta centre {np.linalg.norm(p[3:6]):.3f} R rot {np.degrees(np.linalg.norm(p[:3])):.2f} deg; end delta centre {np.linalg.norm(p[9:12]):.3f} R rot {np.degrees(np.linalg.norm(p[6:9])):.2f} deg")
    for n, k, Xc, Rc2w0, C0, med0, mad0 in data:
        q = pose_at(p, k); Rc2w = Rot.from_rotvec(q[:3]).as_matrix() @ Rc2w0; C = C0 + q[3:6] * R; r, m = residuals((Rc2w @ Xc.T).T + C); r = r[m]
        rec = dict(frame=k, resid_before_median_R=float(med0), resid_before_mad_R=float(mad0), resid_after_median_R=float(np.median(r)), resid_after_mad_R=float(np.median(np.abs(r - np.median(r)))), centre_shift_R=float(np.linalg.norm(q[3:6])), rotation_deg=float(np.degrees(np.linalg.norm(q[:3]))), depth_scale=1.0)
        report.append(rec); print(f"  f{k}: residual median {med0:+.3f} (MAD {mad0:.3f}) -> {rec['resid_after_median_R']:+.3f} (MAD {rec['resid_after_mad_R']:.3f}) R; correction centre {rec['centre_shift_R']:.3f} R, rot {rec['rotation_deg']:.2f} deg")
        Rw2c_new = Rc2w.T; imgs[n]["q"] = R2q(Rw2c_new); imgs[n]["t"] = -Rw2c_new @ C
for n in ([] if a.joint else sorted(imgs, key=fidx)):
    k = fidx(n)
    if k < a.frames[0] or k > a.frames[1]: continue
    fp = f"{a.m1_run}/dense/stereo/depth_maps/{n}.geometric.bin"
    if not os.path.exists(fp): print(f"f{k}: no depth map"); continue
    dep = read_depth(fp); h_, w_ = dep.shape; sx, sy = w_ / W, h_ / H; v, u = np.mgrid[0:h_:a.stride, 0:w_:a.stride]; d_ = dep[::a.stride, ::a.stride]; ok = d_ > 0
    Xc = np.stack([(u[ok] - cx * sx) / (fx * sx) * d_[ok], (v[ok] - cy * sy) / (fy * sy) * d_[ok], d_[ok]], 1)
    Rw2c = q2R(imgs[n]["q"]); Rc2w0 = Rw2c.T; C0 = -Rc2w0 @ imgs[n]["t"]
    res0, rigid0 = residuals((Rc2w0 @ Xc.T).T + C0); n_rig = rigid0.sum()
    if n_rig < 200: print(f"f{k}: only {n_rig} rigid points with a canonical; skipped"); continue
    Xr = Xc[rigid0]; npt = len(Xr); wprior = a.prior * np.sqrt(npt)
    def fun(p):
        dR = Rot.from_rotvec(p[:3]).as_matrix(); s = 1.0 if a.fix_scale else np.exp(p[6]); Rc2w = dR @ Rc2w0; C = C0 + p[3:6] * R
        Xw = (Rc2w @ (s * Xr).T).T + C; r, m = residuals(Xw); r = np.where(m, r, 0.0)
        prior = wprior * np.concatenate([p[:3], p[3:6], [0.0 if a.fix_scale else p[6]]])
        return np.concatenate([r, prior])
    p0 = np.zeros(7); rb = np.radians(a.max_rot); sb = 1e-9 if a.fix_scale else a.max_logscale
    lb = np.array([-rb] * 3 + [-a.max_shift] * 3 + [-sb]); ub = -lb; sol = least_squares(fun, p0, loss="soft_l1", f_scale=0.15, bounds=(lb, ub), max_nfev=300)
    p = sol.x; dR = Rot.from_rotvec(p[:3]).as_matrix(); s = 1.0 if a.fix_scale else float(np.exp(p[6])); Rc2w = dR @ Rc2w0; C = C0 + p[3:6] * R
    res1, m1 = residuals((Rc2w @ (s * Xr).T).T + C); r0 = res0[rigid0]
    rec = dict(frame=k, n_rigid=int(npt), resid_before_median_R=float(np.median(r0)), resid_before_mad_R=float(np.median(np.abs(r0 - np.median(r0)))), resid_after_median_R=float(np.median(res1[m1])), resid_after_mad_R=float(np.median(np.abs(res1[m1] - np.median(res1[m1])))),
               centre_shift_R=float(np.linalg.norm(p[3:6])), rotation_deg=float(np.degrees(np.linalg.norm(p[:3]))), depth_scale=s)
    report.append(rec); print(f"f{k}: rigid pts {npt}; radial residual median {rec['resid_before_median_R']:+.3f} (MAD {rec['resid_before_mad_R']:.3f}) -> {rec['resid_after_median_R']:+.3f} (MAD {rec['resid_after_mad_R']:.3f}) R; pose change: centre {rec['centre_shift_R']:.3f} R, rotation {rec['rotation_deg']:.2f} deg, depth scale {s:.3f}")
    Rw2c_new = Rc2w.T; imgs[n]["q"] = R2q(Rw2c_new); imgs[n]["t"] = -Rw2c_new @ C; imgs[n]["scale"] = s
# write the refined model (poses only; the stereo must be re-run on it)
os.makedirs(f"{a.out}/sparse/0", exist_ok=True)
with open(f"{td}/images.txt", "w") as f:
    for n, v in imgs.items():
        fld = v["hdr"].split(); f.write(f"{fld[0]} {' '.join(map(repr, v['q']))} {' '.join(map(repr, v['t']))} {fld[8]} {n}\n{v['pts']}")
for extra_f in ("frames.txt", "rigs.txt"):
    if os.path.exists(f"{td}/{extra_f}"): os.remove(f"{td}/{extra_f}")
sh(["model_converter", "--input_path", td, "--output_path", f"{a.out}/sparse/0", "--output_type", "BIN"]); shutil.rmtree(td)
if not os.path.exists(f"{a.out}/images"): os.symlink(os.path.abspath(f"{a.workspace}/images"), f"{a.out}/images")
json.dump(report, open(f"{a.out}/register_report.json", "w"), indent=1); print(f"refined model -> {a.out}/sparse/0 ({len(report)} frames refined, original camera model kept); re-run the per-frame stereo on it")
