"""Per-frame stereo on a C3VD sequence (COLMAP patch-match, the airway pipeline's settings: +-w frame window, geometric
consistency, min 2 consistent) with a given pose model and image set, then every geometric depth map against the
remapped ground truth. Errors are binned by ground-truth depth (camera distance), pixel brightness, normalised image
radius and the wall's incidence angle (from the ground-truth depth gradient), and the slope of relative depth error
against distance is reported: the range-bias measurement with truth.
Usage: python c3vd/stereo_vs_truth.py runs/c3vd_cecum --model gt_model --images images_raw --out runs/c3vd_cecum/st_raw800 --max-size 800 --gpu 0"""
import os, json, argparse, subprocess, tempfile, glob, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("prep"); ap.add_argument("--model", default="gt_model"); ap.add_argument("--images", default="images_raw"); ap.add_argument("--out", required=True)
ap.add_argument("--max-size", type=int, default=800); ap.add_argument("--window", type=int, default=2); ap.add_argument("--gpu", default="0"); ap.add_argument("--skip-stereo", action="store_true"); ap.add_argument("--frames", type=int, nargs=2, default=None)
ap.add_argument("--min-baseline", type=float, default=0.0, help="mm: choose sources by camera-centre distance >= this (within --max-gap frames) instead of the fixed window")
ap.add_argument("--max-gap", type=int, default=12); ap.add_argument("--texture-gate", type=float, default=0.0, help="drop pixels whose 7x7 local std (grey levels) is below this before scoring")
ap.add_argument("--src-min", type=int, default=1, help="exclude source frames closer than this many frames (tests the small-baseline attractor)")
ap.add_argument("--gt-poses", default=None, help="C3VD pose.txt: when the model is an SfM model, its scale is recovered by Umeyama on the camera centres and applied to the depths")
a = ap.parse_args(); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap"); dense = f"{a.out}/dense"; os.makedirs(a.out, exist_ok=True)
def sh(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0: raise RuntimeError(cmd[0] + "\n" + r.stderr[-2000:])
def load_gt(base):
    """ground-truth depth in mm: compressed uint16 hundredths of a mm (.npz) or the older float32 .npy"""
    if os.path.exists(base + ".npz"): return np.load(base + ".npz")["d"].astype(np.float32) / 100.0
    return np.load(base + ".npy")
def read_depth(path):
    with open(path, "rb") as f:
        hdr = b""
        while hdr.count(b"&") < 3: hdr += f.read(1)
        w, h, c = map(int, hdr.decode().split("&")[:3]); return np.fromfile(f, np.float32).reshape(h, w, c)[:, :, 0]
P = json.load(open(f"{a.prep}/pinhole.json")); f0, _, cx0, cy0 = P["params_colmap"][:4]; S0 = P["width"]
scale = 1.0
def q2R_(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
tdc = tempfile.mkdtemp(); sh(["model_converter", "--input_path", f"{a.prep}/{a.model}", "--output_path", tdc, "--output_type", "TXT"])
centres = {}
for l in [l for l in open(f"{tdc}/images.txt") if l.strip() and not l.startswith("#")][0::2]:
    v = l.split(); q = np.array(list(map(float, v[1:5]))); t = np.array(list(map(float, v[5:8]))); centres[int(v[9][1:6])] = -q2R_(q).T @ t
import shutil as _sh; _sh.rmtree(tdc)
if a.gt_poses:
    def q2R(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    td = tempfile.mkdtemp(); sh(["model_converter", "--input_path", f"{a.prep}/{a.model}", "--output_path", td, "--output_type", "TXT"])
    L = [l for l in open(f"{td}/images.txt") if l.strip() and not l.startswith("#")][0::2]; cs = {}
    for l in L:
        v = l.split(); q = np.array(list(map(float, v[1:5]))); t = np.array(list(map(float, v[5:8]))); cs[int(v[9][1:6])] = -q2R(q).T @ t
    poses = np.loadtxt(a.gt_poses, delimiter=","); ks_ = sorted(cs); A_ = np.array([cs[k] for k in ks_]); B_ = poses[ks_][:, 12:15]
    ma, mb = A_.mean(0), B_.mean(0); Aa, Bb = A_ - ma, B_ - mb; U, sv, Vt = np.linalg.svd(Bb.T @ Aa / len(A_)); d = np.ones(3); d[-1] = np.sign(np.linalg.det(U @ Vt)); Rs = U @ np.diag(d) @ Vt
    scale = float((sv * d).sum() / (Aa ** 2).sum(1).mean()); resid = np.linalg.norm((scale * (Rs @ Aa.T)).T + mb - B_, axis=1)
    print(f"SfM model: {len(ks_)} cameras; Sim(3) to GT: scale {scale:.4f} mm/unit, centre residual median {np.median(resid):.2f} mm (p90 {np.percentile(resid, 90):.2f})"); import shutil; shutil.rmtree(td)
names = sorted(os.path.basename(p) for p in glob.glob(f"{a.prep}/{a.images}/f*.png")); ks = [int(n[1:6]) for n in names]
if a.frames: names = [n for n, k in zip(names, ks) if a.frames[0] <= k <= a.frames[1]]; ks = [int(n[1:6]) for n in names]
if not a.skip_stereo:
    subprocess.run(["rm", "-rf", dense])
    sh(["image_undistorter", "--image_path", f"{a.prep}/{a.images}", "--input_path", f"{a.prep}/{a.model}", "--output_path", dense, "--output_type", "COLMAP", "--max_image_size", str(a.max_size)])
    with open(f"{dense}/stereo/patch-match.cfg", "w") as fo:
        byk = {k: n for k, n in zip(ks, names)}
        for k, n in zip(ks, names):
            if a.min_baseline > 0:
                cand = [j for j in range(k - a.max_gap, k + a.max_gap + 1) if j != k and j in byk and np.linalg.norm(centres[j] - centres[k]) * (scale if a.gt_poses else 1.0) >= a.min_baseline]
                src = [byk[j] for j in sorted(cand, key=lambda j: abs(j - k))[:6]]
            else: src = [byk[j] for j in range(k - a.window, k + a.window + 1) if abs(j - k) >= a.src_min and j in byk]
            if not src: src = [byk[sorted(byk, key=lambda j: abs(j - k))[1]]]
            fo.write(n + "\n" + ", ".join(src) + "\n")
    dmin, dmax = (5.0, 120.0) if not a.gt_poses else (5.0 / scale, 120.0 / scale)
    sh(["patch_match_stereo", "--workspace_path", dense, "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true", "--PatchMatchStereo.max_image_size", str(a.max_size), "--PatchMatchStereo.gpu_index", a.gpu,
        "--PatchMatchStereo.depth_min", f"{dmin:.4f}", "--PatchMatchStereo.depth_max", f"{dmax:.4f}", "--PatchMatchStereo.filter_min_triangulation_angle", "0.25", "--PatchMatchStereo.filter_min_num_consistent", "2", "--PatchMatchStereo.window_radius", "7"])
    print("stereo done", flush=True)
# comparison
rows = []; acc = dict(depth=[], rel=[], bright=[], radius=[], inc=[], frame=[], texture=[])
for k, n in zip(ks, names):
    fp = f"{dense}/stereo/depth_maps/{n}.geometric.bin"
    if not os.path.exists(fp): continue
    est = read_depth(fp) * scale; h, w = est.shape; s = w / S0; gt = cv2.resize(load_gt(f"{a.prep}/gt_depth/{n[:-4]}"), (w, h), interpolation=cv2.INTER_NEAREST)
    img = cv2.imread(f"{dense}/images/{n}", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    mu = cv2.blur(img, (7, 7)); tex = np.sqrt(np.maximum(cv2.blur(img * img, (7, 7)) - mu * mu, 0))            # local contrast: std in the NCC window
    # incidence angle from the GT depth: normal of the surface vs the viewing ray
    f_ = f0 * s; cx_, cy_ = cx0 * s, cy0 * s; v, u = np.mgrid[0:h, 0:w]; x = (u - cx_) / f_; y = (v - cy_) / f_
    X = np.stack([x * gt, y * gt, gt], -1); gx = np.gradient(X, axis=1); gy = np.gradient(X, axis=0); nrm = np.cross(gx, gy); nn = np.linalg.norm(nrm, axis=-1) + 1e-9; nrm /= nn[..., None]
    ray = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9); inc = np.degrees(np.arccos(np.clip(np.abs((nrm * ray).sum(-1)), 0, 1)))
    ok = (est > 0) & (gt > 0.5)
    if a.texture_gate > 0: ok &= tex >= a.texture_gate
    if ok.sum() < 100: continue
    rel = (est[ok] - gt[ok]) / gt[ok]; rad = np.hypot(u[ok] - cx_, v[ok] - cy_) / (S0 * s / 2)
    rows.append(dict(frame=k, n=int(ok.sum()), coverage=float(ok.mean()), rel_median=float(np.median(rel)), rel_mad=float(np.median(np.abs(rel - np.median(rel)))), abs_median_mm=float(np.median(np.abs(est[ok] - gt[ok]))), gt_depth_median=float(np.median(gt[ok]))))
    sub = np.random.default_rng(k).choice(ok.sum(), min(ok.sum(), 4000), replace=False)
    acc["depth"].append(gt[ok][sub]); acc["rel"].append(rel[sub]); acc["bright"].append(img[ok][sub]); acc["radius"].append(rad[sub]); acc["inc"].append(inc[ok][sub]); acc["frame"].append(np.full(len(sub), k)); acc["texture"].append(tex[ok][sub])
A = {k: np.concatenate(v) for k, v in acc.items()}; np.savez_compressed(f"{a.out}/samples.npz", **A); json.dump(rows, open(f"{a.out}/per_frame.json", "w"), indent=0)
def binned(x, y, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= 200: out.append((lo, hi, float(np.median(y[m])), float(np.median(np.abs(y[m] - np.median(y[m])))), int(m.sum())))
    return out
def ts(x, y):
    rng = np.random.default_rng(0); idx = rng.choice(len(x), min(len(x), 3000), replace=False); xx, yy = x[idx], y[idx]; i, j = np.triu_indices(len(xx), 1); dx = xx[i] - xx[j]; m = np.abs(dx) > 0.1 * (np.percentile(x, 90) - np.percentile(x, 10))
    return float(np.median((yy[i] - yy[j])[m] / dx[m]))
rel = A["rel"]; d = A["depth"]
summary = dict(out=a.out, images=a.images, model=a.model, max_size=a.max_size, scale_mm_per_unit=scale, window=a.window, src_min=a.src_min, min_baseline_mm=a.min_baseline, texture_gate=a.texture_gate, frames=len(rows), points=int(len(rel)), rel_median=float(np.median(rel)), rel_mad=float(np.median(np.abs(rel - np.median(rel)))), abs_median_mm=float(np.median([r["abs_median_mm"] for r in rows])),
               coverage_median=float(np.median([r["coverage"] for r in rows])), slope_rel_per_mm=ts(d, rel), slope_rel_per_relative_depth=ts(d / np.median(d), rel),
               by_depth=binned(d, rel, [5, 15, 20, 25, 30, 35, 40, 50, 65, 90]), by_brightness=binned(A["bright"], rel, [0, 30, 50, 70, 90, 110, 140, 180, 256]), by_radius=binned(A["radius"], rel, [0, 0.2, 0.4, 0.6, 0.8, 1.0]), by_incidence=binned(A["inc"], rel, [0, 20, 40, 55, 65, 75, 90]), by_texture=binned(A["texture"], rel, [0, 2, 4, 6, 9, 13, 20, 40, 300]))
json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)
print(f"[{os.path.basename(a.out)}] {len(rows)} frames, {len(rel)} samples, coverage {summary['coverage_median']:.2f}; rel depth error median {100*summary['rel_median']:+.2f}% (MAD {100*summary['rel_mad']:.2f}%), abs {summary['abs_median_mm']:.2f} mm; slope of rel error vs GT depth {100*summary['slope_rel_per_mm']:+.3f} %/mm = {100*summary['slope_rel_per_relative_depth']:+.2f} % per median-depth")
print("  by depth (mm): " + "; ".join(f"{lo:.0f}-{hi:.0f}: {100*m:+.1f}%" for lo, hi, m, _, _ in summary["by_depth"]))
print("  by brightness: " + "; ".join(f"{lo:.0f}-{hi:.0f}: {100*m:+.1f}%" for lo, hi, m, _, _ in summary["by_brightness"]))
print("  by texture (local std, grey levels): " + "; ".join(f"{lo:.0f}-{hi:.0f}: {100*m:+.1f}% (n {n_})" for lo, hi, m, _, n_ in summary["by_texture"]))
print("  by radius: " + "; ".join(f"{lo:.1f}-{hi:.1f}: {100*m:+.1f}%" for lo, hi, m, _, _ in summary["by_radius"]) + " | by incidence: " + "; ".join(f"{lo:.0f}-{hi:.0f}: {100*m:+.1f}%" for lo, hi, m, _, _ in summary["by_incidence"]))
