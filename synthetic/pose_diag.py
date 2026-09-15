"""Alignment-independent pose diagnostics of a COLMAP sparse model against the synthetic ground truth.

score_rigid.py aligns the whole camera path with one similarity transform; when part of the path is wrong the
alignment splits the difference and every frame looks bad. These quantities do not depend on any alignment:

  1. rotation of camera k relative to camera 0 (estimated vs truth)      -> when did the orientation go wrong?
  2. angle between each camera's viewing axis and its own trajectory tangent (estimated vs truth)
  3. path straightness (RMS off-line distance / path length)              -> did the model bend?
  4. radial distance of sparse points from the estimated path line, relative to path length (truth: 5/36 = 0.139)
plus a side view of the estimated model with viewing-axis arrows.

Usage: python synthetic/pose_diag.py <workspace> <gt.npz> [--model sparse/0] [--out fig.png]
Env:   BRONCHO_COLMAP (default colmap)"""
import sys, os, json, argparse, subprocess, tempfile
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("workspace"); ap.add_argument("gt"); ap.add_argument("--model", default="sparse/0"); ap.add_argument("--out", default=None)
a = ap.parse_args(); ws = a.workspace.rstrip("/"); tag = os.path.basename(ws); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap")
g = np.load(a.gt); t = g["t"]; c2w = g["poses_c2w"]; C_gt = c2w[:, :3, 3]; R_gt = c2w[:, :3, :3]
with tempfile.TemporaryDirectory() as td:
    subprocess.run([COLMAP, "model_converter", "--input_path", os.path.join(ws, a.model), "--output_path", td, "--output_type", "TXT"], check=True, capture_output=True)
    L = [l for l in open(f"{td}/images.txt") if l.strip() and not l.startswith("#")]
    pts = np.array([[float(x) for x in l.split()[1:4]] for l in open(f"{td}/points3D.txt") if l.strip() and not l.startswith("#")])


def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


idx, Rw2c, C = [], [], []
for l in L[0::2]:
    f = l.split(); b = os.path.splitext(os.path.basename(f[9]))[0].lstrip("f")
    if not b.isdigit(): continue
    R = q2R(list(map(float, f[1:5]))); tv = np.array(list(map(float, f[5:8]))); idx.append(int(b)); Rw2c.append(R); C.append(-R.T @ tv)
o = np.argsort(idx); idx = np.array(idx)[o]; Rw2c = np.array(Rw2c)[o]; C = np.array(C)[o]; Rc2w = np.transpose(Rw2c, (0, 2, 1))


def relang(Rk, R0):
    Rr = R0.T @ Rk; return np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1)))


def fwd_vs_tan(Rc, Cc, h=5):
    out = np.full(len(Cc), np.nan)
    for j in range(h, len(Cc) - h):
        tan = Cc[j + h] - Cc[j - h]; tan /= np.linalg.norm(tan); f = Rc[j] @ np.array([0, 0, 1.0]); out[j] = np.degrees(np.arccos(np.clip(f @ tan, -1, 1)))
    return out


def straight(Cc):
    m = Cc.mean(0); u = np.linalg.svd(Cc - m)[2][0]; d = (Cc - m) - np.outer((Cc - m) @ u, u); return np.sqrt((d ** 2).sum(1).mean()), np.ptp((Cc - m) @ u)


rel_e = np.array([relang(Rc2w[j], Rc2w[0]) for j in range(len(idx))]); rel_g = np.array([relang(R_gt[k], R_gt[idx[0]]) for k in idx])
fv_e, fv_g = fwd_vs_tan(Rc2w, C), fwd_vs_tan(R_gt[idx], C_gt[idx]); se, Le = straight(C); sg, Lg = straight(C_gt[idx])
m = C.mean(0); u = np.linalg.svd(C - m)[2][0]; u = u if u @ (C[-1] - C[0]) > 0 else -u
ax_pts = (pts - m) @ u; rad = np.linalg.norm((pts - m) - np.outer(ax_pts, u), axis=1)
print(f"== {tag}: {len(idx)} cams, {len(pts)} sparse pts")
print(f"  rotation relative to frame 0 (deg), every 20th frame:\n    est {np.round(rel_e[::20], 1).tolist()}\n    GT  {np.round(rel_g[::20], 1).tolist()}")
print(f"  |est - GT| relative rotation: median {np.median(np.abs(rel_e - rel_g)):.2f} deg, max {np.abs(rel_e - rel_g).max():.2f} deg; first frame exceeding 5 deg: "
      + (f"f{idx[np.argmax(np.abs(rel_e - rel_g) > 5)]} (t={t[idx[np.argmax(np.abs(rel_e - rel_g) > 5)]]:.2f}s)" if (np.abs(rel_e - rel_g) > 5).any() else "none"))
print(f"  viewing axis vs own trajectory tangent (deg): est median {np.nanmedian(fv_e):.1f} max {np.nanmax(fv_e):.1f} | GT median {np.nanmedian(fv_g):.1f} max {np.nanmax(fv_g):.1f}")
print(f"  path straightness (RMS off-line / length): est {se / Le:.4f} | GT {sg / Lg:.4f}")
print(f"  sparse-point radial distance / path length: median {np.median(rad) / Le:.3f} (p10 {np.percentile(rad, 10) / Le:.3f}, p90 {np.percentile(rad, 90) / Le:.3f}) | GT tube 5/36 = 0.139")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
axs[0].plot(t[idx], rel_e, label="estimated"); axs[0].plot(t[idx], rel_g, "k--", label="truth"); axs[0].set_title("camera rotation relative to frame 0 (deg)"); axs[0].legend(); axs[0].grid(alpha=.3); axs[0].set_xlabel("t (s)")
axs[1].plot(t[idx], fv_e, label="estimated"); axs[1].plot(t[idx], fv_g, "k--", label="truth"); axs[1].set_title("viewing axis vs own trajectory tangent (deg)"); axs[1].legend(); axs[1].grid(alpha=.3); axs[1].set_xlabel("t (s)")
v = np.linalg.svd(pts - m)[2]; perp = v[1] - (v[1] @ u) * u; perp /= np.linalg.norm(perp)
axs[2].scatter(ax_pts, (pts - m) @ perp, s=1, c="0.6"); axs[2].plot((C - m) @ u, (C - m) @ perp, "r.-", ms=3, label="cameras")
for j in range(0, len(C), 15): f = Rc2w[j] @ np.array([0, 0, 1.0]); axs[2].arrow((C[j] - m) @ u, (C[j] - m) @ perp, 1.5 * (f @ u), 1.5 * (f @ perp), color="b", head_width=.15)
axs[2].set_aspect("equal"); axs[2].set_title("estimated model, side view (scene units); arrows = viewing axes"); axs[2].legend(); axs[2].grid(alpha=.3)
fig.suptitle(f"pose diagnostics — {tag}", fontweight="bold"); fig.tight_layout(); out = a.out or os.path.join(ws, f"pose_diag_{tag}.png"); fig.savefig(out, dpi=110); print("  fig:", out)
