"""M2 velocity search on a REAL clip (port of mc_sweep_vsearch2.py from the synthetic tube to the station frames of
m1/windowed_depth_real.py).

Geometry: the stations S_i along the smoothed camera path with parallel-transported normals (N1, N2) are rebuilt from the
per-frame run's m1_real_grid.npz exactly as windowed_depth_real.py built them. A candidate 3D point is assigned to its
nearest station, with angle theta = atan2(X.N2, X.N1) around the path. The wall hypothesis: points inside the declared
arc (centre c, half-width h; --arc auto derives c from the grid, --arc <deg> declares it; under-declare!) move RADIALLY
INWARD at v (in units of R per second), so a source frame dt away sees them at X - e_r * v * R * dt. Velocity hypotheses
are scored per (frame, station) by the sum of best-NCC over consistent arc pixels, summed over +-tagg frames, accepted
against v = 0 by a relative margin, temporally median-filtered, and the final full-resolution sweep uses the smoothed
field. Output: COLMAP-layout depth maps under <out>/dense (with the undistorted sparse model), so
  python m1/windowed_depth_real.py <workspace> --out <out> --skip-stereo --max-size <same>
scores them like any other per-frame run.

Usage: python m2/mc_sweep_vsearch_real.py <workspace> <m1_run> --out runs/vs_real_20V1 --arc auto --half 30 --gpu 0 --max-size 800"""
import sys, os, json, argparse, subprocess, tempfile, shutil, time
import numpy as np, cv2, torch, torch.nn.functional as F
ap = argparse.ArgumentParser()
ap.add_argument("workspace"); ap.add_argument("m1_run"); ap.add_argument("--out", required=True); ap.add_argument("--model", default="sparse/0"); ap.add_argument("--max-size", type=int, default=800)
ap.add_argument("--window", type=int, default=2); ap.add_argument("--planes", type=int, default=128); ap.add_argument("--search-planes", type=int, default=64); ap.add_argument("--search-scale", type=float, default=0.5)
ap.add_argument("--zmin", type=float, default=0.3, help="min depth in R"); ap.add_argument("--zmax", type=float, default=8.0, help="max depth in R")
ap.add_argument("--ncc-win", type=int, default=7); ap.add_argument("--ncc-min", type=float, default=0.6); ap.add_argument("--min-consistent", type=int, default=3); ap.add_argument("--best-k", type=int, default=3); ap.add_argument("--median", type=int, default=5)
ap.add_argument("--vel", default="0.15,0.3,0.5,0.75,1.0,1.4,2.0,2.8,4.0", help="hypotheses in R per second (both signs added, 0 included)"); ap.add_argument("--margin", type=float, default=0.05)
ap.add_argument("--tagg", type=int, default=2); ap.add_argument("--tsmooth", type=int, default=2); ap.add_argument("--maxgap", type=int, default=3)
ap.add_argument("--search-ncc-min", type=float, default=0.4); ap.add_argument("--search-min-consistent", type=int, default=2); ap.add_argument("--min-pixels", type=int, default=20)
ap.add_argument("--arc", default="auto", help="'auto' (from the grid: circular median over stations of each station's most inward-moving 120-deg arc centre) or a centre in degrees in the grid's sector convention"); ap.add_argument("--half", type=float, default=30.0, help="half-width of the declared arc, degrees")
ap.add_argument("--fps", type=float, default=29.97); ap.add_argument("--scores-from", default=None); ap.add_argument("--no-final", action="store_true"); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--frames", default=None)
ap.add_argument("--plain", action="store_true", help="no search: final sweep with zero velocity everywhere (the uncompensated baseline with identical gates)")
ap.add_argument("--mode", default="velocity", choices=["velocity", "displacement"], help="velocity: the wall moves at v (R/s), so a source dt away sees it v*dt further in (right for a moving scope with near sources). displacement: the reference frame's wall is folded inward by d (R) relative to the OPEN wall its sources see (right when the sources straddle a brief event, as baseline-selected sources do on a dwell)")
ap.add_argument("--disp", default="0.05,0.1,0.15,0.2,0.3,0.4,0.55,0.7", help="displacement hypotheses in R (both signs added, 0 included)")
ap.add_argument("--event", type=int, nargs=2, default=None, help="displacement mode: frames inside the event are NOT treated as open, so they get no warp")
ap.add_argument("--min-baseline", type=float, default=0.0, help="choose sources by camera-centre distance >= this many median inter-frame steps (within --max-gap frames) instead of the fixed +-window")
ap.add_argument("--max-gap", type=int, default=12)
a = ap.parse_args(); os.makedirs(f"{a.out}/dense/stereo/depth_maps", exist_ok=True); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap"); dev = torch.device(f"cuda:{a.gpu}")
def sh(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0: raise RuntimeError(cmd[0] + "\n" + r.stderr[-1500:])
# undistorted pinhole workspace (images + sparse) at the same size as the per-frame run
if not os.path.isdir(f"{a.out}/dense/sparse"):
    sh(["image_undistorter", "--image_path", f"{a.workspace}/images", "--input_path", f"{a.workspace}/{a.model}", "--output_path", f"{a.out}/dense", "--output_type", "COLMAP", "--max_image_size", str(a.max_size)])
def read_model(model_dir):
    with tempfile.TemporaryDirectory() as td:
        sh(["model_converter", "--input_path", model_dir, "--output_path", td, "--output_type", "TXT"])
        cam = [l.split() for l in open(f"{td}/cameras.txt") if l.strip() and not l.startswith("#")][0]
        L = [l for l in open(f"{td}/images.txt") if l.strip() and not l.startswith("#")]
        imgs = {l.split()[9]: (np.array(list(map(float, l.split()[1:5]))), np.array(list(map(float, l.split()[5:8])))) for l in L[0::2]}
    return cam, imgs
def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
def fidx(n): return int(os.path.splitext(os.path.basename(n))[0].lstrip("f"))
cam, imgs = read_model(f"{a.out}/dense/sparse"); W0, H0 = int(cam[2]), int(cam[3]); prm0 = np.array(list(map(float, cam[4:8])))
names = sorted(imgs, key=fidx); frames = np.array([fidx(n) for n in names]); byidx = {k: j for j, k in enumerate(frames)}; N = len(names)
Rc2w = np.array([q2R(imgs[n][0]).T for n in names]); C = np.array([-q2R(imgs[n][0]).T @ imgs[n][1] for n in names])
# stations from the per-frame run's grid (S, s_st, R); the parallel-transported basis is rebuilt as windowed_depth_real.py does
G = np.load(f"{a.m1_run}/m1_real_grid.npz"); S = G["S"]; s_st = G["s_st"]; R = float(G["R"]); nS = len(S); dev_g = G["dev"]
T = np.gradient(S, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
N1 = np.zeros_like(S); N2 = np.zeros_like(S); n1 = np.cross(T[0], [0, 0, 1.0]); n1 = n1 if np.linalg.norm(n1) > 1e-3 else np.cross(T[0], [0, 1.0, 0]); n1 /= np.linalg.norm(n1)
for i in range(nS):
    n1 = n1 - (n1 @ T[i]) * T[i]; n1 /= np.linalg.norm(n1); N1[i] = n1; N2[i] = np.cross(T[i], n1)
# the arc: grid convention is theta = atan2(Q.N2, Q.N1) in (-180, 180], sector b = floor((theta+180)/10)
nb = dev_g.shape[2]; tb = (np.arange(nb) + 0.5) / nb * 360 - 180
if a.arc == "auto":
    d = dev_g.copy(); d[(d > 0.3) | (np.abs(d) > 1.5)] = np.nan; centres = []
    for i in range(nS):
        D = d[:, i, :]
        if (np.isfinite(D).sum(1) >= 6).sum() < 60: continue
        best = None
        for b in range(nb):
            arc = [(b + q) % nb for q in range(12)]; sub = D[:, arc]; m = np.isfinite(sub).sum(1) >= 6
            if m.sum() < 60: continue
            with np.errstate(all="ignore"): med = np.nanmedian(sub[m], axis=1)
            p5 = np.percentile(med, 5)
            if best is None or p5 < best[0]: best = (p5, (tb[b] + 60 + 180) % 360 - 180)
        if best: centres.append(best[1])
    ang = np.radians(centres); c_deg = float(np.degrees(np.arctan2(np.median(np.sin(ang)), np.median(np.cos(ang))))); spread = float(np.degrees(np.median(np.abs(np.angle(np.exp(1j * (ang - np.radians(c_deg))))))))
    print(f"arc auto: {len(centres)} stations, centre {c_deg:+.0f} deg (median absolute spread {spread:.0f} deg), half-width {a.half:.0f} deg")
else: c_deg = float(a.arc); print(f"arc declared: centre {c_deg:+.0f} deg, half-width {a.half:.0f} deg")
c_rad = np.radians(c_deg); h_rad = np.radians(a.half)
St = torch.tensor(S, device=dev, dtype=torch.float32); Tt = torch.tensor(T, device=dev, dtype=torch.float32); N1t = torch.tensor(N1, device=dev, dtype=torch.float32); N2t = torch.tensor(N2, device=dev, dtype=torch.float32)
_hyp = a.disp if a.mode == "displacement" else a.vel
vels = sorted(set([0.0] + [float(v) for v in _hyp.split(",")] + [-float(v) for v in _hyp.split(",")])); n_v = len(vels); i0 = vels.index(0.0); fps = a.fps
UNIT = "R (inward displacement of the reference wall)" if a.mode == "displacement" else "R/s (inward wall velocity)"
ev_lo, ev_hi = (a.event if a.event else (10**9, -10**9))
print(f"{N} frames f{frames.min()}-{frames.max()}; {nS} stations, R = {R:.4f} scene units; {n_v} hypotheses in {UNIT}; mode {a.mode}; image {W0}x{H0}")
cache = {}
def gray(j, scale):
    key = (j, scale)
    if key not in cache:
        im = cv2.imread(f"{a.out}/dense/images/{names[j]}", cv2.IMREAD_GRAYSCALE); W, H = int(round(W0 * scale)), int(round(H0 * scale))
        cache[key] = torch.tensor(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA), device=dev, dtype=torch.float32)[None, None] / 255.0
        if len(cache) > 40: cache.pop(next(iter(cache)))
    return cache[key]
def ncc(ref, src, kw):
    pad = kw // 2; mr = F.avg_pool2d(ref, kw, 1, pad); ms = F.avg_pool2d(src, kw, 1, pad)
    vr = F.avg_pool2d(ref * ref, kw, 1, pad) - mr * mr; vs = F.avg_pool2d(src * src, kw, 1, pad) - ms * ms; cov = F.avg_pool2d(ref * src, kw, 1, pad) - mr * ms
    return (cov / torch.sqrt(torch.clamp(vr * vs, min=1e-8)))[0, 0], vr[0, 0]
def station_geom(Xw):
    """Xw (3,P) -> station index, arc weight (1 inside the declared arc, 0 outside), inward radial unit vector (3,P)"""
    d2 = torch.cdist(Xw.T[None], St[None])[0]; i = d2.argmin(1); rel = Xw.T - St[i]; x = (rel * N1t[i]).sum(1); y = (rel * N2t[i]).sum(1); th = torch.atan2(y, x)
    w = (torch.remainder(th - c_rad + np.pi, 2 * np.pi) - np.pi).abs() <= h_rad; er = (N1t[i] * torch.cos(th)[:, None] + N2t[i] * torch.sin(th)[:, None]).T
    return i, w.float(), er
def sweep(j, src_js, scale, planes, vel_of_station, ncc_min, min_cons):
    W, H = int(round(W0 * scale)), int(round(H0 * scale)); fx, fy, cx, cy = prm0 * scale
    Kinv = torch.tensor(np.linalg.inv(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])), device=dev, dtype=torch.float32)
    vv, uu = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    rays = Kinv @ torch.stack([uu.flatten(), vv.flatten(), torch.ones_like(uu.flatten())], 0)
    Zs, _ = torch.sort(1.0 / torch.linspace(1.0 / (a.zmax * R), 1.0 / (a.zmin * R), planes, device=dev))
    ref = gray(j, scale); Rk = torch.tensor(Rc2w[j], device=dev, dtype=torch.float32); Ck = torch.tensor(C[j], device=dev, dtype=torch.float32)
    def fac_of(jj):
        if a.mode == "velocity": return (frames[jj] - frames[j]) / fps                     # warp = v * R * dt
        return 0.0 if ev_lo <= frames[jj] <= ev_hi else -1.0                                # warp = -d * R  (the source sees the OPEN wall, d outward of the reference)
    srcs = [(gray(jj, scale), torch.tensor(Rc2w[jj], device=dev, dtype=torch.float32), torch.tensor(C[jj], device=dev, dtype=torch.float32), fac_of(jj)) for jj in src_js]
    cost_all = torch.empty((len(srcs), planes, H, W), device=dev); _, vref = ncc(ref, ref, a.ncc_win)
    st_pl = torch.empty((planes, H * W), device=dev, dtype=torch.long); w_pl = torch.empty((planes, H * W), device=dev)
    for pi_, Z in enumerate(Zs):
        Xw = (Rk @ (rays * Z)) + Ck[:, None]; i_st, wpt, er = station_geom(Xw); st_pl[pi_] = i_st; w_pl[pi_] = wpt
        vel = vel_of_station(i_st) if vel_of_station is not None else None
        for si, (img_s, Rs, Cs, dt) in enumerate(srcs):
            Xw_s = Xw if vel is None or dt == 0.0 else Xw - er * ((vel * R * dt) * wpt)[None, :]
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
    stb = st_pl.gather(0, best.flatten()[None])[0].view(H, W); wb = w_pl.gather(0, best.flatten()[None])[0].view(H, W)
    return cmin, depth, ok, stb, wb
def write_depth(path, depth):
    with open(path, "wb") as f: f.write(f"{depth.shape[1]}&{depth.shape[0]}&1&".encode()); depth.astype(np.float32).tofile(f)
lo_k, hi_k = (map(int, a.frames.split(",")) if a.frames else (frames.min(), frames.max())); t0 = time.time()
_step = float(np.median(np.linalg.norm(np.diff(C, axis=0), axis=1)))
def sources(j):
    k = frames[j]
    if a.min_baseline > 0:
        cand = [kk for kk in range(k - a.max_gap, k + a.max_gap + 1) if kk != k and kk in byidx and np.linalg.norm(C[byidx[kk]] - C[j]) >= a.min_baseline * _step]
        return [byidx[kk] for kk in sorted(cand, key=lambda q: abs(q - k))[:6]]
    return [byidx[kk] for kk in range(k - a.window, k + a.window + 1) if kk != k and kk in byidx]
# pass 1: search scores per (frame, station, hypothesis)
if a.plain:
    S_sum = np.zeros((N, nS, n_v)); S_cnt = np.zeros((N, nS, n_v)); print("plain sweep: no velocity search")
elif a.scores_from:
    Sz = np.load(f"{a.scores_from}/scores.npz"); S_sum, S_cnt = Sz["S_sum"], Sz["S_cnt"]; assert list(Sz["vels"]) == vels; print("scores reused from", a.scores_from)
else:
    S_sum = np.zeros((N, nS, n_v)); S_cnt = np.zeros((N, nS, n_v)); n_done = 0
    for j in range(N):
        if frames[j] < lo_k or frames[j] > hi_k: continue
        src = sources(j)
        if not src: continue
        for hi_, vh in enumerate(vels):
            cmin, depth, ok, stb, wb = sweep(j, src, a.search_scale, a.search_planes, (lambda i_st, vh=vh: torch.full((i_st.shape[0],), vh, device=dev)), a.search_ncc_min, a.search_min_consistent)
            sel = ok & (wb > 0.5); ist = stb[sel]; val = (1.0 - cmin)[sel]
            S_sum[j, :, hi_] = torch.zeros(nS, device=dev).index_add_(0, ist, val).cpu().numpy(); S_cnt[j, :, hi_] = torch.zeros(nS, device=dev).index_add_(0, ist, torch.ones_like(val)).cpu().numpy()
        n_done += 1
        if n_done % 20 == 0: print(f"  pass 1: {n_done} frames, {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(f"{a.out}/scores.npz", S_sum=S_sum, S_cnt=S_cnt, vels=np.array(vels), frames=frames, s_st=s_st, R=R, arc_centre_deg=c_deg, half_deg=a.half)
# pass 2: decision with temporal aggregation, temporal median, gap fill
A = np.zeros_like(S_sum); Cn = np.zeros_like(S_cnt)
for dk in range(-a.tagg, a.tagg + 1): A += np.roll(S_sum, dk, axis=0); Cn += np.roll(S_cnt, dk, axis=0)
vstar = np.full((N, nS), np.nan)
for j in range(N):
    for i in range(nS):
        row = A[j, i]
        if Cn[j, i].max() < a.min_pixels: continue
        h = int(np.argmax(row)); vstar[j, i] = vels[h] if row[h] > row[i0] * (1 + a.margin) and row[i0] >= 0 else 0.0
vsmooth = np.full_like(vstar, np.nan)
if a.plain: vstar[:] = 0.0; vsmooth[:] = 0.0
for i in range(0 if a.plain else nS):
    col = vstar[:, i]
    for j in range(N):
        seg = col[max(0, j - a.tsmooth):j + a.tsmooth + 1]; seg = seg[np.isfinite(seg)]
        if len(seg) >= 3: vsmooth[j, i] = np.median(seg)
    fin = np.where(np.isfinite(vsmooth[:, i]))[0]
    if len(fin) >= 2:
        filled = np.interp(np.arange(N), fin, vsmooth[fin, i]); gap_ok = np.array([abs(fin[np.argmin(np.abs(fin - j))] - j) <= a.maxgap for j in range(N)]); vsmooth[:, i] = np.where(gap_ok, filled, np.nan)
dec = np.isfinite(vstar); nz_ = dec & (vstar != 0)
print(f"pass 2: decided {dec.mean():.2f} of cells, non-zero in {nz_.sum() / max(dec.sum(), 1):.2f} of decided; magnitude median of non-zero {np.median(np.abs(vstar[nz_])) if nz_.any() else float('nan'):.2f} [{UNIT}]; smoothed field defined on {np.isfinite(vsmooth).mean():.2f}; {time.time() - t0:.0f} s")
np.savez_compressed(f"{a.out}/velocity.npz", vstar=vstar, vsmooth=vsmooth, vels=np.array(vels), frames=frames, s_st=s_st, R=R, S_sum=S_sum, S_cnt=S_cnt, arc_centre_deg=c_deg, half_deg=a.half, mode=a.mode, event=np.array(a.event if a.event else [0, 0]))
if a.no_final: sys.exit(0)
# pass 3: final compensated sweep at full resolution -> COLMAP-layout depth maps
n_done = 0
for j in range(N):
    if frames[j] < lo_k or frames[j] > hi_k: continue
    src = sources(j)
    if not src: continue
    vz = torch.tensor(np.nan_to_num(vsmooth[j], nan=0.0), device=dev, dtype=torch.float32)
    cmin, depth, ok, stb, wb = sweep(j, src, 1.0, a.planes, (lambda i_st: vz[i_st]), a.ncc_min, a.min_consistent)
    depth = torch.where(ok, depth, torch.zeros_like(depth)); H, W = depth.shape
    if a.median > 1:
        m_ = a.median; pd_ = m_ // 2; dpad = F.pad(depth[None, None], (pd_, pd_, pd_, pd_), mode="replicate")[0, 0]
        patches = dpad.unfold(0, m_, 1).unfold(1, m_, 1).reshape(H, W, -1); valid = patches > 0; nval = valid.sum(-1)
        srt, _ = torch.where(valid, patches, torch.full_like(patches, float("inf"))).sort(-1); med = srt.gather(-1, ((nval - 1) // 2).clamp(min=0)[..., None])[..., 0]
        keep = ok & (nval >= (m_ * m_) // 3) & ((depth - med).abs() < 0.05 * med); depth = torch.where(keep, depth, torch.zeros_like(depth))
    write_depth(f"{a.out}/dense/stereo/depth_maps/{names[j]}.geometric.bin", depth.cpu().numpy()); n_done += 1
    if n_done % 30 == 0: print(f"  pass 3: {n_done} frames, {time.time() - t0:.0f} s", flush=True)
print(f"done: {n_done} depth maps in {time.time() - t0:.0f} s -> {a.out}")
