"""M2: motion-compensated plane-sweep stereo for a deforming airway wall.

M1 showed that short-window stereo of a laterally moving wall is biased by kappa * v * Z / c: over the stereo window
the wall's image motion is indistinguishable from parallax. The fix is to stop assuming a static scene. For frame k
and source k+D, every hypothesised 3D point is first displaced by the wall motion between the two instants,
    X'(k+D) = X(k) + [d(k+D, z) - d(k, z)] * w(theta) * x_anterior,
and only then projected into the source. Cartilage (w = 0) is untouched; the membrane's texture is looked for where
it actually is at time k+D. The displacement field d(z, t) is the quantity under estimation, so it is iterated:
    d^0 from M1 (biased) -> compensated depth -> M1 estimator -> d^1 -> ...  (--deform <m1_grid.npz>)
--deform oracle uses the true field (ceiling); --deform none is plain stereo (must reproduce the M1 bias).

Stereo: fronto-parallel plane sweep in inverse depth, 5x5 NCC against each source, per pixel the mean of the two
best source costs, parabolic sub-plane refinement, masks for weak NCC, textureless windows, range ends and
cross-source disagreement. Depth maps are written in COLMAP's .geometric.bin layout under <out>/dense so that
m1/windowed_depth.py --dense-from <out>/dense scores them unchanged.

Usage:
  python m2/mc_sweep.py runs/rigid_collapse runs/synth_collapse_30 --out runs/m2_collapse_oracle --deform oracle
  python m2/mc_sweep.py runs/rigid_collapse runs/synth_collapse_30 --out runs/m2_collapse_it1 --deform runs/m1_collapse/m1_grid.npz
Env: BRONCHO_COLMAP (default colmap)"""
import sys, os, json, argparse, subprocess, tempfile, shutil, time
import numpy as np, cv2, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("workspace"); ap.add_argument("synth_run"); ap.add_argument("--out", required=True)
ap.add_argument("--deform", default="none", help="none | oracle | path to an m1_grid.npz (uses est_d_smooth)")
ap.add_argument("--window", type=int, default=2); ap.add_argument("--scale", type=float, default=0.5); ap.add_argument("--planes", type=int, default=128)
ap.add_argument("--zmin", type=float, default=1.5, help="mm"); ap.add_argument("--zmax", type=float, default=70.0, help="mm")
ap.add_argument("--ncc-win", type=int, default=5); ap.add_argument("--ncc-min", type=float, default=0.5); ap.add_argument("--min-consistent", type=int, default=2)
ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--frames", default=None, help="lo,hi subset")
a = ap.parse_args(); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap"); dev = torch.device(f"cuda:{a.gpu}")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "synthetic")); import deforming_trachea as dt
os.makedirs(f"{a.out}/dense/stereo/depth_maps", exist_ok=True)
if not os.path.isdir(f"{a.out}/dense/sparse"): shutil.copytree(f"{a.workspace}/sparse/0", f"{a.out}/dense/sparse")


def read_model(model_dir):
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([COLMAP, "model_converter", "--input_path", model_dir, "--output_path", td, "--output_type", "TXT"], check=True, capture_output=True)
        cam = [l.split() for l in open(f"{td}/cameras.txt") if l.strip() and not l.startswith("#")][0]
        L = [l for l in open(f"{td}/images.txt") if l.strip() and not l.startswith("#")]
        imgs = {l.split()[9]: (np.array(list(map(float, l.split()[1:5]))), np.array(list(map(float, l.split()[5:8])))) for l in L[0::2]}
    return (int(cam[2]), int(cam[3]), np.array(list(map(float, cam[4:8])))), imgs


def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


(W0, H0, prm), imgs = read_model(f"{a.workspace}/sparse/0"); fx, fy, cx, cy = prm[:4] * a.scale; W, H = int(round(W0 * a.scale)), int(round(H0 * a.scale))
names = sorted(imgs, key=lambda n: int(n[1:6])); frames = np.array([int(n[1:6]) for n in names]); byidx = {k: j for j, k in enumerate(frames)}
Rc2w = np.array([q2R(imgs[n][0]).T for n in names]); C = np.array([-q2R(imgs[n][0]).T @ imgs[n][1] for n in names])
g = np.load(f"{a.synth_run}/gt.npz"); P = json.loads(str(g["params"])); p = dt.Params(**P); t, z, th, c2w, D = g["t"], g["z"], g["theta"], g["poses_c2w"], g["deformation"].astype(np.float32)
w_memb, _ = dt.sector_weights(th, p)
# similarity scene -> tube frame (mm): scale from centres, rotation from orientations (as in m1/windowed_depth.py)
Cg = c2w[frames, :3, 3]; mu_s, mu_d = C.mean(0), Cg.mean(0); S_, D_ = C - mu_s, Cg - mu_d; U, sv, Vt = np.linalg.svd(D_.T @ S_ / len(C)); dd = np.ones(3); dd[-1] = np.sign(np.linalg.det(U @ Vt))
s = (sv * dd).sum() / (S_ ** 2).sum(1).mean(); Mo = sum(c2w[k, :3, :3] @ Rc2w[j].T for j, k in enumerate(frames)); Uo, _, Vto = np.linalg.svd(Mo); Ro = Uo @ np.diag([1, 1, np.sign(np.linalg.det(Uo @ Vto))]) @ Vto; tro = mu_d - s * Ro @ mu_s
x_ant_world = Ro.T @ np.array([1.0, 0, 0])                                  # anterior direction of the tube in scene coordinates
print(f"{len(names)} frames, {W}x{H}, scale {s:.4f} mm/unit, window ±{a.window}, planes {a.planes}, deform = {a.deform}")

# deformation model d(k, z) in mm
if a.deform == "none": dfun = None
elif a.deform == "oracle":
    Dt = torch.tensor(D, device=dev)                                        # (N, nz, nt) exact field incl. taper
    def dfun(k, zmm, thmm):                                                 # displacement (mm) of the wall at (z, theta) at frame k
        iz = torch.clamp(((zmm - z[0]) / (z[1] - z[0])).round().long(), 0, len(z) - 1); it = torch.clamp((torch.remainder(thmm, 2 * np.pi) / (2 * np.pi) * len(th)).round().long(), 0, len(th) - 1)
        return Dt[k][iz, it]
else:
    G = np.load(a.deform); zs, dsm = G["zs"], G["est_d_smooth"]                # (N, nz) on 1-mm stations; NaN = unknown
    dsm = np.where(np.isfinite(dsm), dsm, np.nan)
    for i in range(dsm.shape[1]):                                           # fill unknowns along time by nearest valid, else 0
        col = dsm[:, i]; fin = np.isfinite(col)
        dsm[:, i] = np.interp(np.arange(len(col)), np.where(fin)[0], col[fin]) if fin.sum() >= 2 else 0.0
    dsm = np.clip(np.nan_to_num(dsm), 0, None); Dz = torch.tensor(dsm, device=dev, dtype=torch.float32); zs_t = torch.tensor(zs, device=dev, dtype=torch.float32)
    wm = torch.tensor(w_memb, device=dev, dtype=torch.float32)
    def dfun(k, zmm, thmm):
        iz = torch.clamp(((zmm - zs_t[0]) / (zs_t[1] - zs_t[0])).round().long(), 0, len(zs) - 1); it = torch.clamp((torch.remainder(thmm, 2 * np.pi) / (2 * np.pi) * len(th)).round().long(), 0, len(th) - 1)
        return Dz[k][iz] * wm[it]

# geometry tensors
Kinv = torch.tensor(np.linalg.inv(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])), device=dev, dtype=torch.float32)
vv, uu = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
rays = (Kinv @ torch.stack([uu.flatten(), vv.flatten(), torch.ones_like(uu.flatten())], 0))                          # (3, HW) camera rays at unit depth
inv_planes = torch.linspace(1.0 / (a.zmax / s), 1.0 / (a.zmin / s), a.planes, device=dev); Zs = 1.0 / inv_planes         # scene units, near->far reversed below
Zs, _ = torch.sort(Zs)                                                                                                  # ascending depth
Ro_t = torch.tensor(Ro, device=dev, dtype=torch.float32); tro_t = torch.tensor(tro, device=dev, dtype=torch.float32); xant = torch.tensor(x_ant_world, device=dev, dtype=torch.float32)
pad = a.ncc_win // 2


def load_gray(k):
    im = cv2.imread(f"{a.synth_run}/frames/f{k:05d}.png", cv2.IMREAD_GRAYSCALE); im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    return torch.tensor(im, device=dev, dtype=torch.float32) / 255.0


def ncc(ref, src):
    """windowed NCC over ncc_win x ncc_win; ref, src: (1,1,H,W) -> (H,W)"""
    kw = a.ncc_win; mr = F.avg_pool2d(ref, kw, 1, pad); ms = F.avg_pool2d(src, kw, 1, pad)
    vr = F.avg_pool2d(ref * ref, kw, 1, pad) - mr * mr; vs = F.avg_pool2d(src * src, kw, 1, pad) - ms * ms; cov = F.avg_pool2d(ref * src, kw, 1, pad) - mr * ms
    return (cov / torch.sqrt(torch.clamp(vr * vs, min=1e-8)))[0, 0], vr[0, 0]


def write_depth(path, depth):
    with open(path, "wb") as f: f.write(f"{depth.shape[1]}&{depth.shape[0]}&1&".encode()); depth.astype(np.float32).tofile(f)


lo_k, hi_k = (map(int, a.frames.split(",")) if a.frames else (frames.min(), frames.max())); t0 = time.time(); n_done = 0
for j, k in enumerate(frames):
    if k < lo_k or k > hi_k: continue
    src_ks = [kk for kk in range(k - a.window, k + a.window + 1) if kk != k and kk in byidx]
    if not src_ks: continue
    ref = load_gray(k)[None, None]; Rk = torch.tensor(Rc2w[j], device=dev, dtype=torch.float32); Ck = torch.tensor(C[j], device=dev, dtype=torch.float32)
    srcs = [(load_gray(kk)[None, None], torch.tensor(Rc2w[byidx[kk]], device=dev, dtype=torch.float32), torch.tensor(C[byidx[kk]], device=dev, dtype=torch.float32), kk) for kk in src_ks]
    cost_all = torch.empty((len(srcs), a.planes, H, W), device=dev)
    _, vref = ncc(ref, ref)
    for pi_, Z in enumerate(Zs):
        Xw = (Rk @ (rays * Z)) + Ck[:, None]                                                                              # (3, HW) world points
        if dfun is not None:
            Xmm = (s * (Ro_t @ Xw)) + tro_t[:, None]; zmm = Xmm[2]; thmm = torch.atan2(Xmm[1], Xmm[0]); d_k = dfun(k, zmm, thmm)
        for si, (img_s, Rs, Cs, kk) in enumerate(srcs):
            Xw_s = Xw
            if dfun is not None:
                delta = (dfun(kk, zmm, thmm) - d_k) / s                                                                    # scene units, along the anterior axis
                Xw_s = Xw + xant[:, None] * delta[None, :]
            Xc = Rs.T @ (Xw_s - Cs[:, None]); zc = Xc[2].clamp(min=1e-6); u = fx * Xc[0] / zc + cx; v = fy * Xc[1] / zc + cy
            grid = torch.stack([(u / (W - 1)) * 2 - 1, (v / (H - 1)) * 2 - 1], -1).view(1, H, W, 2)
            warped = F.grid_sample(img_s, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            n_, _ = ncc(ref, warped); behind = (Xc[2] <= 0).view(H, W) | (u < 0).view(H, W) | (u > W - 1).view(H, W) | (v < 0).view(H, W) | (v > H - 1).view(H, W)
            cost_all[si, pi_] = torch.where(behind, torch.full_like(n_, 2.0), 1.0 - n_)
    # aggregate: mean of the two best sources per pixel/plane
    k_best = min(2, len(srcs)); agg = torch.topk(cost_all, k_best, dim=0, largest=False).values.mean(0)                     # (planes, H, W)
    best = agg.argmin(0); cmin = agg.gather(0, best[None])[0]
    # sub-plane parabolic refinement in inverse depth
    bm = (best - 1).clamp(0, a.planes - 1); bp = (best + 1).clamp(0, a.planes - 1); c0, cm, cp = cmin, agg.gather(0, bm[None])[0], agg.gather(0, bp[None])[0]
    denom = (cm - 2 * c0 + cp); off = torch.where(denom.abs() > 1e-6, 0.5 * (cm - cp) / denom, torch.zeros_like(c0)).clamp(-1, 1)
    inv_sorted = 1.0 / Zs; inv_best = inv_sorted[best] + off * (inv_sorted[bp] - inv_sorted[bm]) / 2; depth = 1.0 / inv_best
    # masks
    per_src_best = cost_all.argmin(1)                                                                                     # (S, H, W)
    consistent = ((per_src_best - best[None]).abs() <= 2).sum(0) >= min(a.min_consistent, len(srcs))
    ok = (cmin < 1.0 - a.ncc_min) & consistent & (best > 0) & (best < a.planes - 1) & (vref > 1e-4)
    depth = torch.where(ok, depth, torch.zeros_like(depth)); write_depth(f"{a.out}/dense/stereo/depth_maps/f{k:05d}.png.geometric.bin", depth.cpu().numpy()); n_done += 1
    if n_done % 20 == 0: print(f"  {n_done} frames, {time.time() - t0:.0f} s, valid {float(ok.float().mean()):.2f}", flush=True)
print(f"done: {n_done} depth maps in {time.time() - t0:.0f} s -> {a.out}/dense/stereo/depth_maps")
json.dump(dict(vars(a), scale_mm_per_unit=float(s)), open(f"{a.out}/sweep_params.json", "w"), indent=1)
