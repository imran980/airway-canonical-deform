"""Eyeball the reconstruction of a real per-frame run as a point cloud: an oblique 3D view with the camera path, a
longitudinal section through the moving arc (arclength vs signed radius), and a cross-section at the event station.
Points are back-projected from the per-frame geometric depth maps and coloured by the undistorted image.
Usage: python m2/eyeball_cloud.py runs/m1_real_26V2d_b3 --event 1908 1921 --pre 1860 1907 --arc 5 --out docs/figures/x.png"""
import argparse, os, glob, numpy as np, cv2, json, subprocess, tempfile, shutil
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--event", type=int, nargs=2, required=True); ap.add_argument("--pre", type=int, nargs=2, required=True); ap.add_argument("--post", type=int, nargs=2, default=None, help="reference frames after the event (the re-opened wall)")
ap.add_argument("--arc", type=float, default=0.0, help="angle (deg, grid convention) of the section plane"); ap.add_argument("--out", required=True); ap.add_argument("--stride", type=int, default=5); ap.add_argument("--frame-step", type=int, default=1); ap.add_argument("--max-points", type=int, default=250000)
a = ap.parse_args(); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap")
def sh(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0: raise RuntimeError(cmd[0] + "\n" + r.stderr[-1000:])
def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
def read_depth(p):
    with open(p, "rb") as f:
        hdr = b""
        while hdr.count(b"&") < 3: hdr += f.read(1)
        w, h, c = map(int, hdr.decode().split("&")[:3]); return np.fromfile(f, np.float32).reshape(h, w, c)[:, :, 0]
td = tempfile.mkdtemp(); sh(["model_converter", "--input_path", f"{a.run}/dense/sparse", "--output_path", td, "--output_type", "TXT"])
cam = [l.split() for l in open(f"{td}/cameras.txt") if l.strip() and not l.startswith("#")][0]; W, H = int(cam[2]), int(cam[3]); fx, fy, cx, cy = map(float, cam[4:8])
imgs = {}
for l in [l for l in open(f"{td}/images.txt") if l.strip() and not l.startswith("#")][0::2]:
    v = l.split(); imgs[v[9]] = (np.array(list(map(float, v[1:5]))), np.array(list(map(float, v[5:8]))))
shutil.rmtree(td)
G = np.load(f"{a.run}/m1_real_grid.npz"); S = G["S"]; R = float(G["R"]); s_st = G["s_st"]; C_all = G["C"]; fr_all = G["frames"]
T = np.gradient(S, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True); N1 = np.zeros_like(S); N2 = np.zeros_like(S); n1 = np.cross(T[0], [0, 0, 1.0]); n1 = n1 if np.linalg.norm(n1) > 1e-3 else np.cross(T[0], [0, 1.0, 0]); n1 /= np.linalg.norm(n1)
for i in range(len(S)):
    n1 = n1 - (n1 @ T[i]) * T[i]; n1 /= np.linalg.norm(n1); N1[i] = n1; N2[i] = np.cross(T[i], n1)
from scipy.spatial import cKDTree; tree = cKDTree(S)
P, COL, FR = [], [], []
names = sorted(imgs, key=lambda n: int(n[1:6]))
for n in names[::a.frame_step]:
    k = int(n[1:6]); fp = f"{a.run}/dense/stereo/depth_maps/{n}.geometric.bin"
    if not os.path.exists(fp): continue
    dep = read_depth(fp); h_, w_ = dep.shape; sx, sy = w_ / W, h_ / H; img = cv2.imread(f"{a.run}/dense/images/{n}")
    v, u = np.mgrid[0:h_:a.stride, 0:w_:a.stride]; d = dep[::a.stride, ::a.stride]; ok = d > 0
    if ok.sum() < 50: continue
    Xc = np.stack([(u[ok] - cx * sx) / (fx * sx) * d[ok], (v[ok] - cy * sy) / (fy * sy) * d[ok], d[ok]], 1)
    Rw2c = q2R(imgs[n][0]); Rc2w = Rw2c.T; Cw = -Rc2w @ imgs[n][1]; Xw = (Rc2w @ Xc.T).T + Cw
    P.append(Xw); FR.append(np.full(len(Xw), k))
    if img is not None:
        col = cv2.resize(img, (w_, h_))[::a.stride, ::a.stride][ok][:, ::-1] / 255.0; COL.append(col)
    else: COL.append(np.full((len(Xw), 3), 0.6))
P = np.vstack(P); COL = np.vstack(COL); FR = np.concatenate(FR)
# station coordinates: nearest station, arclength, radius, angle
_, idx = tree.query(P); rel = P - S[idx]; s_of = s_st[idx] / R; rr = np.hypot((rel * N1[idx]).sum(1), (rel * N2[idx]).sum(1)) / R; th = np.degrees(np.arctan2((rel * N2[idx]).sum(1), (rel * N1[idx]).sum(1)))
keep = (rr < 3.0) & (np.abs((rel * T[idx]).sum(1)) / R < 0.3)
ev = (FR >= a.event[0]) & (FR <= a.event[1]); pre = (FR >= a.pre[0]) & (FR <= a.pre[1]); post = (FR >= a.post[0]) & (FR <= a.post[1]) if a.post else np.zeros(len(FR), bool)
print(f"{len(P)} points from {len(set(FR))} frames; kept {keep.sum()} within 3 R of the path; event points {(keep&ev).sum()}, pre-event {(keep&pre).sum()}")
fig = plt.figure(figsize=(17, 6.2)); gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.35, 1])
# A: oblique 3D, image colour
ax = fig.add_subplot(gs[0, 0], projection="3d"); m = keep
sub = np.random.default_rng(0).choice(np.where(m)[0], min(m.sum(), a.max_points // 3), replace=False)
ax.scatter(P[sub, 0] / R, P[sub, 1] / R, P[sub, 2] / R, c=COL[sub], s=0.6, lw=0, depthshade=False)
ax.plot(C_all[:, 0] / R, C_all[:, 1] / R, C_all[:, 2] / R, "k-", lw=2, label="camera path")
ax.set_title(f"the whole reconstructed segment, image colour\n({len(set(FR))} frames, camera path in black)", fontsize=9.5); ax.legend(fontsize=7)
Pn = np.vstack([P[sub], C_all]) / R; ctr = Pn.mean(0); half = max((np.percentile(Pn, 97, axis=0) - np.percentile(Pn, 3, axis=0)).max() / 2, 1.5)
for setter, c in ((ax.set_xlim, ctr[0]), (ax.set_ylim, ctr[1]), (ax.set_zlim, ctr[2])): setter(c - half, c + half)
Tm = T[max(0, len(S) // 2)]; ax.view_init(elev=12, azim=np.degrees(np.arctan2(Tm[1], Tm[0])) + 90); ax.set_box_aspect((1, 1, 1)); ax.tick_params(labelsize=6)
# B: longitudinal section through the arc plane: arclength vs signed radius
ax = fig.add_subplot(gs[0, 1]); dth = np.abs((th - a.arc + 180) % 360 - 180); side = np.where(dth < 90, 1.0, -1.0); insec = keep & (np.minimum(dth, 180 - dth) < 25)
trio = [(insec & pre, "0.55", f"before f{a.pre[0]}–{a.pre[1]}"), (insec & post, "tab:blue", f"after f{a.post[0]}–{a.post[1]}" if a.post else ""), (insec & ev, "tab:red", f"collapse f{a.event[0]}–{a.event[1]}")]
for m_, c_, lab in trio:
    if m_.sum() and lab: ax.plot(s_of[m_], side[m_] * rr[m_], ".", ms=1.2, color=c_, alpha=0.45, label=f"{lab} ({m_.sum()} pts)")
ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("arclength along the path (R)"); ax.set_ylabel(f"signed radius (R), section through {a.arc:+.0f}° / {a.arc+180:+.0f}°")
ax.set_title("longitudinal section of the wall: does the lumen pinch during the collapse?", fontsize=10); ax.legend(fontsize=8, markerscale=8); ax.grid(alpha=.3); ax.set_ylim(-2.5, 2.5)
for i in range(len(S)):
    pass
# C: cross-section at the station nearest the event camera
cnt = np.bincount(idx[keep & ev], minlength=len(S)); i_ev = int(np.argmax(cnt)); print(f"cross-section station {i_ev} with {cnt[i_ev]} collapse points")
ax = fig.add_subplot(gs[0, 2]); band = keep & (np.abs(s_of - s_st[i_ev] / R) < 0.35)
for m_, c_, lab in ((band & pre, "0.55", "before"), (band & post, "tab:blue", "after"), (band & ev, "tab:red", "collapse")):
    if m_.sum() > 5: x = rr[m_] * np.cos(np.radians(th[m_])); y = rr[m_] * np.sin(np.radians(th[m_])); ax.plot(x, y, ".", ms=1.6, color=c_, alpha=0.5, label=f"{lab} ({m_.sum()} pts)")
ax.plot(0, 0, "k+", ms=10); ax.set_aspect("equal"); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_xlabel("R"); ax.set_ylabel("R"); ax.grid(alpha=.3); ax.legend(fontsize=8, markerscale=6)
ax.set_title(f"cross-section at station {i_ev} ({s_st[i_ev]/R:.1f} R),\nslab ±0.35 R", fontsize=10)
fig.suptitle(f"{os.path.basename(a.run)}: the reconstruction itself (points back-projected from the per-frame depth maps; radii in tube radii R)", fontweight="bold")
fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=115); print("figure:", a.out)
