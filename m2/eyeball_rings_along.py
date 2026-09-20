"""Eyeball a real reconstruction the way it is used: the wall cross-section at several stations along the path, built from
the depth-map points themselves, with the open-phase and folded-phase frames drawn separately.
Usage: python m2/eyeball_rings_along.py <run_with_maps> --grid <run_with_grid> --folded f1 f2 --open f1 f2 --arc -140 --out x.png"""
import argparse, os, subprocess, tempfile, shutil, numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--grid", default=None); ap.add_argument("--folded", type=int, nargs="+", required=True); ap.add_argument("--open", type=int, nargs="+", required=True)
ap.add_argument("--arc", type=float, default=None); ap.add_argument("--stations", type=int, nargs="*", default=None); ap.add_argument("--slab", type=float, default=0.25); ap.add_argument("--stride", type=int, default=3); ap.add_argument("--out", required=True); ap.add_argument("--label", default=None)
a = ap.parse_args(); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap"); grid_run = a.grid or a.run
def sh(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0: raise RuntimeError(cmd[0] + "\n" + r.stderr[-800:])
def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
def read_depth(p):
    with open(p, "rb") as f:
        h = b""
        while h.count(b"&") < 3: h += f.read(1)
        w, hh, c = map(int, h.decode().split("&")[:3]); return np.fromfile(f, np.float32).reshape(hh, w, c)[:, :, 0]
td = tempfile.mkdtemp(); sh(["model_converter", "--input_path", f"{a.run}/dense/sparse", "--output_path", td, "--output_type", "TXT"])
cam = [l.split() for l in open(f"{td}/cameras.txt") if l.strip() and not l.startswith("#")][0]; W, H = int(cam[2]), int(cam[3]); fx, fy, cx, cy = map(float, cam[4:8])
poses = {l.split()[9]: (np.array(list(map(float, l.split()[1:5]))), np.array(list(map(float, l.split()[5:8])))) for l in [l for l in open(f"{td}/images.txt") if l.strip() and not l.startswith("#")][0::2]}
shutil.rmtree(td)
G = np.load(f"{grid_run}/m1_real_grid.npz"); S = G["S"]; R = float(G["R"]); s_st = G["s_st"]; r_grid = G["r_grid"]; fr = G["frames"]
T = np.gradient(S, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True); N1 = np.zeros_like(S); N2 = np.zeros_like(S)
n1 = np.cross(T[0], [0, 0, 1.0]); n1 = n1 if np.linalg.norm(n1) > 1e-3 else np.cross(T[0], [0, 1.0, 0]); n1 /= np.linalg.norm(n1)
for i in range(len(S)):
    n1 = n1 - (n1 @ T[i]) * T[i]; n1 /= np.linalg.norm(n1); N1[i] = n1; N2[i] = np.cross(T[i], n1)
can = np.nanpercentile(r_grid, 75, axis=0); can[np.isfinite(r_grid).sum(0) < 5] = np.nan
nb = r_grid.shape[2]; tb = (np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi
def cloud(frames):
    P = []
    for k in frames:
        n = f"f{int(k):05d}.png"; fp = f"{a.run}/dense/stereo/depth_maps/{n}.geometric.bin"
        if n not in poses or not os.path.exists(fp): continue
        dep = read_depth(fp); h_, w_ = dep.shape; sx = w_ / W; v, u = np.mgrid[0:h_:a.stride, 0:w_:a.stride]; d = dep[::a.stride, ::a.stride]; ok = d > 0
        if ok.sum() < 50: continue
        Xc = np.stack([(u[ok] - cx * sx) / (fx * sx) * d[ok], (v[ok] - cy * sx) / (fy * sx) * d[ok], d[ok]], 1)
        Rw2c = q2R(poses[n][0]); Rc2w = Rw2c.T; C = -Rc2w @ poses[n][1]; P.append((Rc2w @ Xc.T).T + C)
    return np.vstack(P) if P else np.zeros((0, 3))
fold_fr = list(range(a.folded[0], a.folded[1] + 1)) if len(a.folded) == 2 else a.folded
open_fr = list(range(a.open[0], a.open[1] + 1)) if len(a.open) == 2 else a.open
Pf = cloud(fold_fr); Po = cloud(open_fr); print(f"folded {len(Pf)} points from {len(fold_fr)} frames; open {len(Po)} points from {len(open_fr)} frames")
obs = np.isfinite(r_grid).sum(2); sts = a.stations or [int(i) for i in np.argsort(-(obs >= 12).sum(0))[:6]]
sts = sorted(sts); n = len(sts); fig, axs = plt.subplots(1, n, figsize=(3.4 * n, 4.0))
axs = np.atleast_1d(axs)
def sector_profile(P, i):
    """what the pipeline measures: the median wall radius in each of 36 sectors, and the point count per sector"""
    rel = P - S[i]; al = rel @ T[i]; m = np.abs(al) < a.slab * R
    if m.sum() < 50: return None, None
    x = (rel[m] @ N1[i]) / R; y = (rel[m] @ N2[i]) / R; rr = np.hypot(x, y); th = np.arctan2(y, x)
    k = rr < 2.5; rr, th = rr[k], th[k]
    b = ((th + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    med = np.array([np.median(rr[b == q]) if (b == q).sum() >= 5 else np.nan for q in range(nb)])
    cnt = np.array([(b == q).sum() for q in range(nb)]); return med, cnt
for ax, i in zip(axs, sts):
    for P, col, lab in ((Po, "tab:blue", "open phase"), (Pf, "tab:red", "folded phase")):
        if not len(P): continue
        rel = P - S[i]; al = rel @ T[i]; m = np.abs(al) < a.slab * R
        if m.sum() > 50:
            x = (rel[m] @ N1[i]) / R; y = (rel[m] @ N2[i]) / R; k = np.hypot(x, y) < 2.2
            ax.plot(x[k], y[k], ".", ms=0.7, color=col, alpha=0.05)
        med, cnt = sector_profile(P, i)
        if med is None: continue
        ok = np.isfinite(med); th = np.concatenate([tb[ok], tb[ok][:1]]); rr = np.concatenate([med[ok], med[ok][:1]])
        ax.plot(rr * np.cos(th), rr * np.sin(th), "-", lw=2.2, color=col, label=f"{lab} (median wall)")
    okc = np.isfinite(can[i])
    if okc.sum() > 6:
        th = np.concatenate([tb[okc], tb[okc][:1]]); rr = np.concatenate([can[i][okc], can[i][okc][:1]]) / R
        ax.plot(rr * np.cos(th), rr * np.sin(th), "k--", lw=1.2, alpha=0.7, label="canonical (75th pct)")
    if a.arc is not None:
        for sgn in (-1, 1):
            th = np.radians(a.arc + sgn * 60); ax.plot([0, 2.1 * np.cos(th)], [0, 2.1 * np.sin(th)], color="0.4", lw=0.8, ls=":")
        th = np.radians(a.arc); ax.annotate("fold arc", xy=(1.6 * np.cos(th), 1.6 * np.sin(th)), fontsize=8, color="0.3", ha="center")
    ax.plot(0, 0, "k+", ms=8); ax.set_aspect("equal"); ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.grid(alpha=.3)
    ax.set_title(f"station {i}: {s_st[i]/R:.1f} R along the path", fontsize=9.5); ax.tick_params(labelsize=7)
axs[0].set_ylabel("radii (R)"); axs[0].legend(fontsize=7, loc="lower left", markerscale=6)
fig.suptitle(f"{a.label or os.path.basename(a.run)}: the measured wall at consecutive stations — open phase (blue) against folded phase (red); thick line = median radius per sector, dots = the depth-map points behind it", fontweight="bold", fontsize=11)
fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=120); print("figure:", a.out)
