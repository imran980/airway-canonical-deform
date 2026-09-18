"""Eyeball a C3VD reconstruction against ground truth: the ground-truth cloud and one or two reconstructions in the same
box (image colour), plus the per-pixel error distributions. Every cloud is built from the SAME pixels, so the shapes are
comparable; an over-reading stereo shows as a cloud that spills outside the true wall.
Usage: python c3vd/eyeball_truth.py runs/c3vd_cecum --run st_eye --run2 st_eye_old --gt-poses <pose.txt> --out x.png"""
import argparse, os, json, subprocess, tempfile, shutil, numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("prep"); ap.add_argument("--run", required=True); ap.add_argument("--run2", default=None)
ap.add_argument("--label", default="sources 3–5 frames + COLMAP poses (the recipe)"); ap.add_argument("--label2", default="sources ±1–2 frames (the old setting)")
ap.add_argument("--out", required=True); ap.add_argument("--stride", type=int, default=6); ap.add_argument("--frame-step", type=int, default=2); ap.add_argument("--gt-poses", required=True)
a = ap.parse_args(); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap")
def sh(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0: raise RuntimeError(cmd[0] + "\n" + r.stderr[-1000:])
def read_depth(p):
    with open(p, "rb") as f:
        hdr = b""
        while hdr.count(b"&") < 3: hdr += f.read(1)
        w, h, c = map(int, hdr.decode().split("&")[:3]); return np.fromfile(f, np.float32).reshape(h, w, c)[:, :, 0]
def load_gt(base):
    if os.path.exists(base + ".npz"): return np.load(base + ".npz")["d"].astype(np.float32) / 100.0
    return np.load(base + ".npy")
P = json.load(open(f"{a.prep}/pinhole.json")); f0, _, cx0, cy0 = P["params_colmap"][:4]; S0 = P["width"]; gtp = np.loadtxt(a.gt_poses, delimiter=",")
def collect(run_name):
    run = f"{a.prep}/{run_name}"; scale = json.load(open(f"{run}/summary.json")).get("scale_mm_per_unit", 1.0)
    td = tempfile.mkdtemp(); sh(["model_converter", "--input_path", f"{run}/dense/sparse", "--output_path", td, "--output_type", "TXT"])
    names = sorted([l.split()[9] for l in [l for l in open(f"{td}/images.txt") if l.strip() and not l.startswith("#")][0::2]], key=lambda n: int(n[1:6])); shutil.rmtree(td)
    E, G, COL, ERR = [], [], [], []
    for n in names[::a.frame_step]:
        k = int(n[1:6]); fp = f"{run}/dense/stereo/depth_maps/{n}.geometric.bin"
        if not os.path.exists(fp): continue
        est = read_depth(fp) * scale; h_, w_ = est.shape; s_ = w_ / S0; img = cv2.imread(f"{run}/dense/images/{n}")
        gt = cv2.resize(load_gt(f"{a.prep}/gt_depth/{n[:-4]}"), (w_, h_), interpolation=cv2.INTER_NEAREST)
        v, u = np.mgrid[0:h_:a.stride, 0:w_:a.stride]; e = est[::a.stride, ::a.stride]; g = gt[::a.stride, ::a.stride]; ok = (e > 0) & (g > 0.5)
        if ok.sum() < 50: continue
        fx_, cx_, cy_ = f0 * s_, cx0 * s_, cy0 * s_; x = (u[ok] - cx_) / fx_; y = (v[ok] - cy_) / fx_
        M = gtp[k]; Rc2w = M[:12].reshape(3, 4)[:, :3].T; Cw = M[12:15]
        E.append((Rc2w @ np.stack([x * e[ok], y * e[ok], e[ok]], 1).T).T + Cw); G.append((Rc2w @ np.stack([x * g[ok], y * g[ok], g[ok]], 1).T).T + Cw)
        ERR.append((e[ok] - g[ok]) / g[ok]); COL.append(cv2.resize(img, (w_, h_))[::a.stride, ::a.stride][ok][:, ::-1] / 255.0 if img is not None else np.full((ok.sum(), 3), 0.6))
    return np.vstack(E), np.vstack(G), np.vstack(COL), np.concatenate(ERR)
E1, G1, C1, R1 = collect(a.run); print(f"{a.run}: {len(E1)} points, rel error median {100*np.median(R1):+.1f}%")
pair2 = collect(a.run2) if a.run2 else None
if pair2: print(f"{a.run2}: {len(pair2[0])} points, rel error median {100*np.median(pair2[3]):+.1f}%")
panels = [(G1, C1, "ground truth (C3VD depth, C3VD poses)", None)] + ([(pair2[0], pair2[2], a.label2, pair2[3])] if pair2 else []) + [(E1, C1, a.label, R1)]
fig = plt.figure(figsize=(5.2 * len(panels), 10.4)); gs = fig.add_gridspec(2, len(panels))
rng = np.random.default_rng(0); lim = np.percentile(G1, [1, 99], axis=0); ctr = lim.mean(0); half = (lim[1] - lim[0]).max() / 2 * 1.15
def inbox(X): return np.all(np.abs(X - ctr) < half, axis=1)          # the SAME box for every cloud: truth's 1-99 percentile extent
for j, (X, C, ttl, err) in enumerate(panels):
    keepb = inbox(X); sub = rng.choice(np.where(keepb)[0], min(int(keepb.sum()), 90000), replace=False); ax = fig.add_subplot(gs[0, j], projection="3d")
    ax.scatter(X[sub, 0], X[sub, 1], X[sub, 2], c=C[sub], s=0.5, lw=0, depthshade=False)
    sub_t = f"\nmedian {100*np.median(err):+.1f} %, |error| {np.median(np.abs(err))*100:.0f} %\n{100*(1-keepb.mean()):.0f} % of points outside the true extent" if err is not None else "\n(the wall as it really is)"
    ax.set_title(ttl + sub_t, fontsize=9.5); ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=18, azim=-60); ax.tick_params(labelsize=6)
    ax.set_xlim(ctr[0] - half, ctr[0] + half); ax.set_ylim(ctr[1] - half, ctr[1] + half); ax.set_zlim(ctr[2] - half, ctr[2] + half); ax.set_xlabel("mm", fontsize=7)
# slab through the phantom: the three walls on top of each other
zc = np.median(G1[:, 2]); ax = fig.add_subplot(gs[1, 0:max(1, len(panels) - 1)])
for X, c_, lab, lw_ in ((G1, "k", "ground truth", 0), ) + (((pair2[0], "tab:orange", a.label2, 0),) if pair2 else ()) + ((E1, "tab:blue", a.label, 0),):
    m_ = np.abs(X[:, 2] - zc) < 1.5
    ax.plot(X[m_, 0], X[m_, 1], ".", ms=1.4, color=c_, alpha=0.45 if c_ != "k" else 0.8, label=f"{lab} ({m_.sum()} pts)")
ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.grid(alpha=.3); ax.legend(fontsize=9, markerscale=8)
ax.set_title(f"a 3 mm slab through the phantom at z = {zc:.0f} mm: where each method puts the wall", fontsize=11)
ax.set_xlim(ctr[0] - half, ctr[0] + half); ax.set_ylim(ctr[1] - half, ctr[1] + half)
ax = fig.add_subplot(gs[1, max(1, len(panels) - 1):]); bins = np.linspace(-60, 90, 151)
if pair2: ax.hist(100 * pair2[3], bins=bins, color="tab:orange", alpha=0.65, label=f"{a.label2}: {100*np.median(pair2[3]):+.1f} %")
ax.hist(100 * R1, bins=bins, color="tab:blue", alpha=0.7, label=f"{a.label}: {100*np.median(R1):+.1f} %")
ax.axvline(0, color="k", lw=1.2); ax.set_xlabel("relative depth error per pixel (%)"); ax.set_ylabel("pixels"); ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_title("error against truth, same pixels", fontsize=10)
fig.suptitle(f"C3VD {os.path.basename(a.prep)} (real colonoscope, CT-registered phantom): the airway pipeline's per-frame stereo, seen against truth", fontweight="bold")
fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=115); print("figure:", a.out)
