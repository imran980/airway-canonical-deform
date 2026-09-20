"""Where do the unobserved wall sectors go? For one station and a few frames, project the canonical ring into the image and
label every sector: outside the image, inside but with no depth, depth but failing the support gate, or trusted.
Usage: python m2/coverage_diagnosis.py <run> --station 25 --frames 2560 2575 2590 --out docs/figures/x.png"""
import argparse, os, subprocess, tempfile, shutil, numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--station", type=int, required=True); ap.add_argument("--frames", type=int, nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--label", default=None)
a = ap.parse_args(); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap")
def sh(c):
    r = subprocess.run([COLMAP] + c, capture_output=True, text=True, errors="replace")
    if r.returncode != 0: raise RuntimeError(r.stderr[-600:])
def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def rd(p):
    f = open(p, "rb"); h = b""
    while h.count(b"&") < 3: h += f.read(1)
    w, hh, c = map(int, h.decode().split("&")[:3]); return np.fromfile(f, np.float32).reshape(hh, w, c)[:, :, 0]
td = tempfile.mkdtemp(); sh(["model_converter", "--input_path", f"{a.run}/dense/sparse", "--output_path", td, "--output_type", "TXT"])
cam = [l.split() for l in open(f"{td}/cameras.txt") if l.strip() and not l.startswith("#")][0]; W, H = int(cam[2]), int(cam[3]); fx, fy, cx, cy = map(float, cam[4:8])
poses = {l.split()[9]: (np.array(list(map(float, l.split()[1:5]))), np.array(list(map(float, l.split()[5:8])))) for l in [l for l in open(f"{td}/images.txt") if l.strip() and not l.startswith("#")][0::2]}
shutil.rmtree(td)
G = np.load(f"{a.run}/m1_real_grid.npz"); S = G["S"]; R = float(G["R"]); fr = G["frames"]; r_grid = G["r_grid"]; nb = r_grid.shape[2]
sup = G["support"] if "support" in G.files else np.zeros_like(r_grid, bool)
can = np.nanpercentile(r_grid, 75, axis=0)
T = np.gradient(S, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True); N1 = np.zeros_like(S); N2 = np.zeros_like(S)
n1 = np.cross(T[0], [0, 0, 1.0]); n1 /= np.linalg.norm(n1)
for i in range(len(S)):
    n1 = n1 - (n1 @ T[i]) * T[i]; n1 /= np.linalg.norm(n1); N1[i] = n1; N2[i] = np.cross(T[i], n1)
i = a.station; tb = (np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi
rr = np.where(np.isfinite(can[i]), can[i], np.nanmedian(can[i]))
P = S[i][None] + rr[:, None] * (np.cos(tb)[:, None] * N1[i][None] + np.sin(tb)[:, None] * N2[i][None])
fig, axs = plt.subplots(2, len(a.frames), figsize=(4.2 * len(a.frames), 8.0), squeeze=False)
stats = []
for c, k in enumerate(a.frames):
    n = f"f{int(k):05d}.png"; j = int(np.where(fr == k)[0][0]) if (fr == k).any() else None
    img = cv2.imread(f"{a.run}/dense/images/{n}"); dep = rd(f"{a.run}/dense/stereo/depth_maps/{n}.geometric.bin") if os.path.exists(f"{a.run}/dense/stereo/depth_maps/{n}.geometric.bin") else None
    Rw2c = q2R(poses[n][0]); C = -Rw2c.T @ poses[n][1]; Xc = (Rw2c @ (P - C).T).T
    u = fx * Xc[:, 0] / np.maximum(Xc[:, 2], 1e-6) + cx; v = fy * Xc[:, 1] / np.maximum(Xc[:, 2], 1e-6) + cy
    inimg = (Xc[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    lab = np.zeros(nb, int)                                   # 0 outside image, 1 in image no depth, 2 depth but not trusted, 3 trusted
    hd, wd = (dep.shape if dep is not None else (H, W)); sxy = wd / W
    for b in range(nb):
        if not inimg[b]: continue
        lab[b] = 1
        if dep is not None:
            uu, vv = int(u[b] * sxy), int(v[b] * sxy)
            if 0 <= uu < wd and 0 <= vv < hd and dep[vv, uu] > 0: lab[b] = 2
        if j is not None and sup[j, i, b]: lab[b] = 3
    stats.append(lab)
    ax = axs[0][c]; ax.imshow(img[:, :, ::-1] if img is not None else np.zeros((H, W)), zorder=0)
    cols = {0: "0.4", 1: "tab:orange", 2: "tab:blue", 3: "tab:green"}
    for b in range(nb):
        if inimg[b]: ax.plot(u[b], v[b], "o", ms=5, color=cols[lab[b]], zorder=3)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_title(f"f{k}: ring of station {i} projected\ngreen trusted {np.sum(lab==3)}, blue depth {np.sum(lab==2)}, orange no depth {np.sum(lab==1)}, off-image {np.sum(lab==0)}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax = axs[1][c]
    if dep is not None:
        d = dep.copy(); d[d <= 0] = np.nan; im = ax.imshow(d, cmap="viridis"); plt.colorbar(im, ax=ax, fraction=0.03)
        for b in range(nb):
            if inimg[b]: ax.plot(u[b] * sxy, v[b] * sxy, "o", ms=4, color=cols[lab[b]])
    ax.set_title("depth map (blank = no depth)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
L = np.array(stats)
print(f"[{a.label or os.path.basename(a.run)}] station {i}, {len(a.frames)} frames, {nb} sectors each:")
for nm, v in (("off-image", 0), ("in image, no depth", 1), ("depth, fails support", 2), ("trusted", 3)): print(f"   {nm:>22s}: {100*np.mean(L==v):5.0f} %")
fig.suptitle(f"{a.label or os.path.basename(a.run)}: why sectors are missing at station {i}", fontweight="bold")
fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=110); print("figure:", a.out)
