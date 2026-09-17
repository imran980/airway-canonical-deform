"""M2, joint estimation v2: velocity search inside the stereo with a finer grid, consistent-pixel scoring, scores
accumulated over neighbouring frames, and temporal smoothing of the chosen velocity field.

What changed from mc_sweep_vsearch.py (which found slow motion but not fast):
  * grid: 0, ±1, ±2, ±3, ±4, ±6, ±8, ±11, ±15, ±20, ±27 mm/s (21 hypotheses; --vel)
  * score: the SUM of best-NCC over membrane pixels that pass a loose gate under the hypothesis, i.e. it rewards a
    hypothesis both for making more pixels consistent and for matching them better (the mean NCC alone was flat)
  * decision per (frame, station) on scores summed over ±2 neighbouring frames: wall velocity is smooth in time,
    independent per-frame picks were what failed
  * accepted only if it beats "no motion" by a relative margin; then a temporal median over ±2 frames per station
    and linear filling of short gaps; unknown -> 0 (no compensation)
  * pass 1 (search) is saved to <out>/scores.npz so passes 2-3 can be re-run cheaply (--scores-from)
Pass 3 sweeps the final full-resolution depth with the smoothed v(z) per frame. Outputs as before: COLMAP-layout
depth maps under <out>/dense, <out>/velocity.npz (vstar raw, vsmooth, scores).

Usage: python m2/mc_sweep_vsearch2.py runs/rigid_collapse runs/synth_collapse_30 --out runs/m2_collapse_vs2 --gpu 0
Env:   BRONCHO_COLMAP (default colmap)"""
import sys, os, json, argparse, subprocess, tempfile, shutil, time
import numpy as np, cv2, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("workspace"); ap.add_argument("synth_run"); ap.add_argument("--out", required=True)
ap.add_argument("--window", type=int, default=2); ap.add_argument("--planes", type=int, default=128); ap.add_argument("--search-planes", type=int, default=64); ap.add_argument("--search-scale", type=float, default=0.5)
ap.add_argument("--zmin", type=float, default=1.5); ap.add_argument("--zmax", type=float, default=70.0)
ap.add_argument("--ncc-win", type=int, default=7); ap.add_argument("--ncc-min", type=float, default=0.6); ap.add_argument("--min-consistent", type=int, default=3); ap.add_argument("--best-k", type=int, default=3); ap.add_argument("--median", type=int, default=5)
ap.add_argument("--vel", default="1,2,3,4,6,8,11,15,20,27"); ap.add_argument("--margin", type=float, default=0.05, help="relative gain over v=0 needed to accept a velocity")
ap.add_argument("--tagg", type=int, default=2, help="frames on each side whose scores are summed before the decision"); ap.add_argument("--tsmooth", type=int, default=2, help="half-width of the temporal median on v*"); ap.add_argument("--maxgap", type=int, default=6)
ap.add_argument("--search-ncc-min", type=float, default=0.4); ap.add_argument("--search-min-consistent", type=int, default=2); ap.add_argument("--min-pixels", type=int, default=20)
ap.add_argument("--scores-from", default=None, help="reuse pass-1 scores from another run directory"); ap.add_argument("--no-final", action="store_true", help="stop after the velocity field (no depth maps)")
ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--frames", default=None)
ap.add_argument("--taper", default="true", help="true = the generator's sector weights; box:<half_angle_deg> = a plain posterior box (mis-specified prior ablation)")
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


(W0, H0, prm0), imgs = read_model(f"{a.workspace}/sparse/0")
names = sorted(imgs, key=lambda n: int(n[1:6])); frames = np.array([int(n[1:6]) for n in names]); byidx = {k: j for j, k in enumerate(frames)}
Rc2w = np.array([q2R(imgs[n][0]).T for n in names]); C = np.array([-q2R(imgs[n][0]).T @ imgs[n][1] for n in names])
g = np.load(f"{a.synth_run}/gt.npz"); P = json.loads(str(g["params"])); p = dt.Params(**P); t, z, th, c2w = g["t"], g["z"], g["theta"], g["poses_c2w"]; N = len(t)
w_memb, _ = dt.sector_weights(th, p); fps = p.fps
if a.taper.startswith("box:"):
    half = np.radians(float(a.taper.split(":")[1])); w_memb = (np.abs(np.angle(np.exp(1j * (th - np.pi)))) <= half).astype(np.float64); print(f"taper: posterior box ±{np.degrees(half):.0f} deg (mis-specified prior)")
Cg = c2w[frames, :3, 3]; mu_s, mu_d = C.mean(0), Cg.mean(0); S_, D_ = C - mu_s, Cg - mu_d; U, sv, Vt = np.linalg.svd(D_.T @ S_ / len(C)); dd = np.ones(3); dd[-1] = np.sign(np.linalg.det(U @ Vt))
s = (sv * dd).sum() / (S_ ** 2).sum(1).mean(); Mo = sum(c2w[k, :3, :3] @ Rc2w[j].T for j, k in enumerate(frames)); Uo, _, Vto = np.linalg.svd(Mo); Ro = Uo @ np.diag([1, 1, np.sign(np.linalg.det(Uo @ Vto))]) @ Vto; tro = mu_d - s * Ro @ mu_s
Ro_t = torch.tensor(Ro, device=dev, dtype=torch.float32); tro_t = torch.tensor(tro, device=dev, dtype=torch.float32); xant = torch.tensor(Ro.T @ np.array([1.0, 0, 0]), device=dev, dtype=torch.float32)
wm = torch.tensor(w_memb, device=dev, dtype=torch.float32); zs = np.arange(1.0, P["length_mm"], 1.0); nz = len(zs)
vels = sorted(set([0.0] + [float(v) for v in a.vel.split(",")] + [-float(v) for v in a.vel.split(",")])); n_v = len(vels); i0 = vels.index(0.0)
print(f"{len(names)} frames; scale {s:.4f} mm/unit; {n_v} velocity hypotheses; tagg ±{a.tagg}, tsmooth ±{a.tsmooth}, margin {a.margin}")
cache = {}


def gray(k, scale):
    key = (k, scale)
    if key not in cache:
        im = cv2.imread(f"{a.synth_run}/frames/f{k:05d}.png", cv2.IMREAD_GRAYSCALE); W, H = int(round(W0 * scale)), int(round(H0 * scale))
        cache[key] = torch.tensor(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA), device=dev, dtype=torch.float32)[None, None] / 255.0
        if len(cache) > 40: cache.pop(next(iter(cache)))
    return cache[key]


def ncc(ref, src, kw):
    pad = kw // 2; mr = F.avg_pool2d(ref, kw, 1, pad); ms = F.avg_pool2d(src, kw, 1, pad)
    vr = F.avg_pool2d(ref * ref, kw, 1, pad) - mr * mr; vs = F.avg_pool2d(src * src, kw, 1, pad) - ms * ms; cov = F.avg_pool2d(ref * src, kw, 1, pad) - mr * ms
    return (cov / torch.sqrt(torch.clamp(vr * vs, min=1e-8)))[0, 0], vr[0, 0]


def sweep(j, k, src_ks, scale, planes, vel_of_z, ncc_min, min_cons):
    W, H = int(round(W0 * scale)), int(round(H0 * scale)); fx, fy, cx, cy = prm0[:4] * scale
    Kinv = torch.tensor(np.linalg.inv(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])), device=dev, dtype=torch.float32)
    vv, uu = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    rays = Kinv @ torch.stack([uu.flatten(), vv.flatten(), torch.ones_like(uu.flatten())], 0)
    Zs, _ = torch.sort(1.0 / torch.linspace(1.0 / (a.zmax / s), 1.0 / (a.zmin / s), planes, device=dev))
    ref = gray(k, scale); Rk = torch.tensor(Rc2w[j], device=dev, dtype=torch.float32); Ck = torch.tensor(C[j], device=dev, dtype=torch.float32)
    srcs = [(gray(kk, scale), torch.tensor(Rc2w[byidx[kk]], device=dev, dtype=torch.float32), torch.tensor(C[byidx[kk]], device=dev, dtype=torch.float32), kk) for kk in src_ks]
    cost_all = torch.empty((len(srcs), planes, H, W), device=dev); _, vref = ncc(ref, ref, a.ncc_win)
    zmm_pl = torch.empty((planes, H * W), device=dev); th_pl = torch.empty((planes, H * W), device=dev)
    for pi_, Z in enumerate(Zs):
        Xw = (Rk @ (rays * Z)) + Ck[:, None]; Xmm = (s * (Ro_t @ Xw)) + tro_t[:, None]; zmm = Xmm[2]; thmm = torch.atan2(Xmm[1], Xmm[0]); zmm_pl[pi_] = zmm; th_pl[pi_] = thmm
        it = torch.clamp((torch.remainder(thmm, 2 * np.pi) / (2 * np.pi) * len(th)).round().long(), 0, len(th) - 1); wpt = wm[it]
        vel = vel_of_z(zmm) if vel_of_z is not None else None
        for si, (img_s, Rs, Cs, kk) in enumerate(srcs):
            Xw_s = Xw if vel is None else Xw + xant[:, None] * ((vel * ((kk - k) / fps) * wpt) / s)[None, :]
            Xc = Rs.T @ (Xw_s - Cs[:, None]); zc = Xc[2].clamp(min=1e-6); u = fx * Xc[0] / zc + cx; v = fy * Xc[1] / zc + cy
            grid = torch.stack([(u / (W - 1)) * 2 - 1, (v / (H - 1)) * 2 - 1], -1).view(1, H, W, 2)
            n_, _ = ncc(ref, F.grid_sample(img_s, grid, mode="bilinear", padding_mode="zeros", align_corners=True), a.ncc_win)
            behind = ((Xc[2] <= 0) | (u < 0) | (u > W - 1) | (v < 0) | (v > H - 1)).view(H, W); cost_all[si, pi_] = torch.where(behind, torch.full_like(n_, 2.0), 1.0 - n_)
    agg = torch.topk(cost_all, min(a.best_k, len(srcs)), dim=0, largest=False).values.mean(0); best = agg.argmin(0); cmin = agg.gather(0, best[None])[0]
    bm = (best - 1).clamp(0, planes - 1); bp = (best + 1).clamp(0, planes - 1); cm, cp = agg.gather(0, bm[None])[0], agg.gather(0, bp[None])[0]
    denom = cm - 2 * cmin + cp; off = torch.where(denom.abs() > 1e-6, 0.5 * (cm - cp) / denom, torch.zeros_like(cmin)).clamp(-1, 1)
    inv = 1.0 / Zs; depth = 1.0 / (inv[best] + off * (inv[bp] - inv[bm]) / 2)
    consistent = ((cost_all.argmin(1) - best[None]).abs() <= 2).sum(0) >= min(min_cons, len(srcs))
    ok = (cmin < 1.0 - ncc_min) & consistent & (best > 0) & (best < planes - 1) & (vref > 1e-4)
    zb = zmm_pl.gather(0, best.flatten()[None])[0].view(H, W); thb = th_pl.gather(0, best.flatten()[None])[0].view(H, W)
    wb = wm[torch.clamp((torch.remainder(thb, 2 * np.pi) / (2 * np.pi) * len(th)).round().long(), 0, len(th) - 1)]
    return cmin, depth, ok, zb, wb


def write_depth(path, depth):
    with open(path, "wb") as f: f.write(f"{depth.shape[1]}&{depth.shape[0]}&1&".encode()); depth.astype(np.float32).tofile(f)


lo_k, hi_k = (map(int, a.frames.split(",")) if a.frames else (frames.min(), frames.max())); t0 = time.time()
# ---------------------------------------------------------------- pass 1: search scores
if a.scores_from:
    Sz = np.load(f"{a.scores_from}/scores.npz"); S_sum, S_cnt = Sz["S_sum"], Sz["S_cnt"]; assert list(Sz["vels"]) == vels; print("scores reused from", a.scores_from)
else:
    S_sum = np.zeros((N, nz, n_v)); S_cnt = np.zeros((N, nz, n_v)); n_done = 0
    for j, k in enumerate(frames):
        if k < lo_k or k > hi_k: continue
        src_ks = [kk for kk in range(k - a.window, k + a.window + 1) if kk != k and kk in byidx]
        if not src_ks: continue
        for hi_, vh in enumerate(vels):
            cmin, depth, ok, zb, wb = sweep(j, k, src_ks, a.search_scale, a.search_planes, (lambda zmm, vh=vh: torch.full_like(zmm, vh)), a.search_ncc_min, a.search_min_consistent)
            sel = ok & (wb > 0.6); iz = torch.clamp(((zb - zs[0]) / 1.0).round().long(), 0, nz - 1)[sel]; val = (1.0 - cmin)[sel]
            S_sum[k, :, hi_] = torch.zeros(nz, device=dev).index_add_(0, iz, val).cpu().numpy(); S_cnt[k, :, hi_] = torch.zeros(nz, device=dev).index_add_(0, iz, torch.ones_like(val)).cpu().numpy()
        n_done += 1
        if n_done % 20 == 0: print(f"  pass 1: {n_done} frames, {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(f"{a.out}/scores.npz", S_sum=S_sum, S_cnt=S_cnt, vels=np.array(vels), zs=zs)
# ---------------------------------------------------------------- pass 2: decision with temporal aggregation, then smoothing
A = np.zeros_like(S_sum); Cn = np.zeros_like(S_cnt)
for dk in range(-a.tagg, a.tagg + 1):
    A += np.roll(S_sum, dk, axis=0); Cn += np.roll(S_cnt, dk, axis=0)
vstar = np.full((N, nz), np.nan)
for k in range(N):
    for i in range(nz):
        row = A[k, i]
        if Cn[k, i].max() < a.min_pixels: continue
        h = int(np.argmax(row)); vstar[k, i] = vels[h] if row[h] > row[i0] * (1 + a.margin) and row[i0] >= 0 else 0.0
vsmooth = np.full_like(vstar, np.nan)
for i in range(nz):
    col = vstar[:, i]
    for k in range(N):
        seg = col[max(0, k - a.tsmooth):k + a.tsmooth + 1]; seg = seg[np.isfinite(seg)]
        if len(seg) >= 3: vsmooth[k, i] = np.median(seg)
    fin = np.where(np.isfinite(vsmooth[:, i]))[0]
    if len(fin) >= 2:
        filled = np.interp(np.arange(N), fin, vsmooth[fin, i]); gap_ok = np.zeros(N, bool)
        for k in range(N):
            near = fin[np.argmin(np.abs(fin - k))]; gap_ok[k] = abs(near - k) <= a.maxgap
        vsmooth[:, i] = np.where(gap_ok, filled, np.nan)
np.savez_compressed(f"{a.out}/velocity.npz", vstar=vstar, vsmooth=vsmooth, vels=np.array(vels), zs=zs, t=t, S_sum=S_sum, S_cnt=S_cnt)
print(f"pass 2: decided {np.isfinite(vstar).mean():.2f} of cells, smoothed field defined on {np.isfinite(vsmooth).mean():.2f}; {time.time() - t0:.0f} s")
if a.no_final: sys.exit(0)
# ---------------------------------------------------------------- pass 3: final compensated sweep
n_done = 0
for j, k in enumerate(frames):
    if k < lo_k or k > hi_k: continue
    src_ks = [kk for kk in range(k - a.window, k + a.window + 1) if kk != k and kk in byidx]
    if not src_ks: continue
    vz = torch.tensor(np.nan_to_num(vsmooth[k], nan=0.0), device=dev, dtype=torch.float32)
    cmin, depth, ok, zb, wb = sweep(j, k, src_ks, 1.0, a.planes, (lambda zmm: vz[torch.clamp(((zmm - zs[0]) / 1.0).round().long(), 0, nz - 1)]), a.ncc_min, a.min_consistent)
    depth = torch.where(ok, depth, torch.zeros_like(depth)); H, W = depth.shape
    if a.median > 1:
        m_ = a.median; pd_ = m_ // 2; dpad = F.pad(depth[None, None], (pd_, pd_, pd_, pd_), mode="replicate")[0, 0]
        patches = dpad.unfold(0, m_, 1).unfold(1, m_, 1).reshape(H, W, -1); valid = patches > 0; nval = valid.sum(-1)
        srt, _ = torch.where(valid, patches, torch.full_like(patches, float("inf"))).sort(-1); med = srt.gather(-1, ((nval - 1) // 2).clamp(min=0)[..., None])[..., 0]
        keep = ok & (nval >= (m_ * m_) // 3) & ((depth - med).abs() < 0.05 * med); depth = torch.where(keep, depth, torch.zeros_like(depth))
    write_depth(f"{a.out}/dense/stereo/depth_maps/f{k:05d}.png.geometric.bin", depth.cpu().numpy()); n_done += 1
    if n_done % 30 == 0: print(f"  pass 3: {n_done} frames, {time.time() - t0:.0f} s", flush=True)
print(f"done: {n_done} depth maps in {time.time() - t0:.0f} s -> {a.out}")
