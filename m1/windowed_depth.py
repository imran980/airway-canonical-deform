"""M1 v0: canonical map + per-frame deformation from short-window stereo, scored against exact ground truth.

Idea. The rigid cross-check showed that COLMAP's camera poses stay exact through breathing and collapse because the
rigid cartilage anchors them; what a rigid map gets wrong is the deforming membrane, which time-fusion smears. So:
  * poses      : from the rigid model (cartilage-anchored; verified exact in docs/M0_VERIFICATION.md)
  * canonical  : the wall where it is rigid (cartilage sector), the same in every frame
  * deformation: per frame k, depth is estimated with patch-match stereo against ONLY frames k-w..k+w and never
                 fused across time. The membrane's position in frame k relative to the canonical wall is the
                 deformation at time t_k.
Per frame and per station ahead of the camera this gives: a model-free lumen area (free space inside the canonical
disc bounded by that frame's wall points), a model-based area using the declared anatomical prior (the posterior
sector translates anteriorly by d(z,t), so d is the one unknown per station), the cartilage radius (must equal the
canonical radius: the rigidity check), and the membrane displacement d(z,t) itself. Everything is compared with
gt.npz on the (t, z) grid where the station was observed, and with the rigid pipeline's time-constant CSA(z).

Usage:
  python m1/windowed_depth.py runs/rigid_collapse runs/synth_collapse_30 --out runs/m1_collapse [--window 2] [--gpus 0,1,2,3]
Env: BRONCHO_COLMAP (default colmap)"""
import sys, os, json, argparse, subprocess, tempfile, glob
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("workspace"); ap.add_argument("synth_run"); ap.add_argument("--out", required=True)
ap.add_argument("--window", type=int, default=2, help="source frames k-w..k+w for the depth of frame k")
ap.add_argument("--gpus", default="0,1,2,3"); ap.add_argument("--max-size", type=int, default=1600)
ap.add_argument("--ahead", default="4,18", help="stations measured this many mm ahead of the camera")
ap.add_argument("--skip-stereo", action="store_true", help="reuse existing depth maps in <out>/dense")
ap.add_argument("--min-pts", type=int, default=25, help="minimum posterior points for a displacement estimate")
ap.add_argument("--dense-from", default=None, help="reuse the depth maps of another run's dense dir (implies --skip-stereo)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap"); MIN_PTS = a.min_pts
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "synthetic")); import deforming_trachea as dt

g = np.load(f"{a.synth_run}/gt.npz"); P = json.loads(str(g["params"])); p = dt.Params(**P)
t, z, th, r0, CSA_gt, c2w, D = g["t"], g["z"], g["theta"], g["r_canonical"], g["csa_mm2"], g["poses_c2w"], g["deformation"].astype(np.float32)
N, dz = len(t), z[1] - z[0]; w_memb, cart_w = dt.sector_weights(th, p); Rr = P["radius_mm"]
from scipy.ndimage import binary_dilation, label
dense = a.dense_from if a.dense_from else os.path.join(a.out, "dense")
if a.dense_from: a.skip_stereo = True


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError(" ".join(cmd[:3]) + "\n" + r.stderr[-1500:])
    return r.stdout


def read_model(model_dir):
    with tempfile.TemporaryDirectory() as td:
        sh([COLMAP, "model_converter", "--input_path", model_dir, "--output_path", td, "--output_type", "TXT"])
        cams = {}
        for l in open(f"{td}/cameras.txt"):
            if l.startswith("#") or not l.strip(): continue
            f = l.split(); cams[int(f[0])] = (f[1], int(f[2]), int(f[3]), np.array(list(map(float, f[4:]))))
        imgs = {}
        L = [l for l in open(f"{td}/images.txt") if l.strip() and not l.startswith("#")]
        for l in L[0::2]:
            f = l.split(); imgs[f[9]] = (np.array(list(map(float, f[1:5]))), np.array(list(map(float, f[5:8]))), int(f[8]))
    return cams, imgs


def q2R(q):
    w, x, y, z_ = q
    return np.array([[1 - 2 * (y * y + z_ * z_), 2 * (x * y - z_ * w), 2 * (x * z_ + y * w)], [2 * (x * y + z_ * w), 1 - 2 * (x * x + z_ * z_), 2 * (y * z_ - x * w)], [2 * (x * z_ - y * w), 2 * (y * z_ + x * w), 1 - 2 * (x * x + y * y)]])


def fidx(name): return int(os.path.splitext(os.path.basename(name))[0].lstrip("f"))


def read_depth(path):
    with open(path, "rb") as f:
        hdr = b""
        while hdr.count(b"&") < 3: hdr += f.read(1)
        w, h, c = map(int, hdr.decode().split("&")[:3]); return np.fromfile(f, np.float32).reshape(h, w, c)[:, :, 0]


# ---------------------------------------------------------------- 1. dense workspace + per-frame source windows
if not a.skip_stereo:
    if os.path.isdir(dense): subprocess.run(["rm", "-rf", dense])
    sh([COLMAP, "image_undistorter", "--image_path", f"{a.synth_run}/frames", "--input_path", f"{a.workspace}/sparse/0", "--output_path", dense,
        "--output_type", "COLMAP", "--max_image_size", str(a.max_size)])
cams, imgs = read_model(f"{dense}/sparse")
names = sorted(imgs, key=fidx); frames = np.array([fidx(n) for n in names]); byidx = {k: n for k, n in zip(frames, names)}
print(f"{len(names)} registered frames of {N}; window ±{a.window}; undistorted camera: {cams[imgs[names[0]][2]][0]} {cams[imgs[names[0]][2]][3][:4]}")

# similarity (scene -> GT mm): scale from centres (Umeyama), rotation from camera orientations
Rc2w = np.array([q2R(imgs[n][0]).T for n in names]); C = np.array([-q2R(imgs[n][0]).T @ imgs[n][1] for n in names]); Cg = c2w[frames, :3, 3]
mu_s, mu_d = C.mean(0), Cg.mean(0); S_, D_ = C - mu_s, Cg - mu_d; U, sv, Vt = np.linalg.svd(D_.T @ S_ / len(C)); dd = np.ones(3); dd[-1] = np.sign(np.linalg.det(U @ Vt))
s = (sv * dd).sum() / (S_ ** 2).sum(1).mean(); Mo = sum(c2w[k, :3, :3] @ Rc2w[j].T for j, k in enumerate(frames)); Uo, _, Vto = np.linalg.svd(Mo)
Ro = Uo @ np.diag([1, 1, np.sign(np.linalg.det(Uo @ Vto))]) @ Vto; tro = mu_d - s * Ro @ mu_s
print(f"scene->mm: scale {s:.4f} mm/unit, camera-centre RMSE after alignment {np.sqrt((np.linalg.norm((s * (Ro @ C.T)).T + tro - Cg, axis=1) ** 2).mean()):.3f} mm")

if not a.skip_stereo:
    with open(f"{dense}/stereo/patch-match.cfg", "w") as f:
        for k, n in zip(frames, names):
            src = [byidx[j] for j in range(k - a.window, k + a.window + 1) if j != k and j in byidx]
            f.write(n + "\n" + ", ".join(src) + "\n")
    sh([COLMAP, "patch_match_stereo", "--workspace_path", dense, "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.max_image_size", str(a.max_size), "--PatchMatchStereo.gpu_index", a.gpus,
        "--PatchMatchStereo.depth_min", f"{1.0 / s:.4f}", "--PatchMatchStereo.depth_max", f"{(P['length_mm'] + 15) / s:.4f}",
        "--PatchMatchStereo.filter_min_triangulation_angle", "0.25", "--PatchMatchStereo.filter_min_num_consistent", "2",
        "--PatchMatchStereo.window_radius", "7", "--PatchMatchStereo.num_samples", "15"])
    print("patch-match stereo done")

# ---------------------------------------------------------------- 2. per-frame points -> per-station measurements
lo_ahead, hi_ahead = map(float, a.ahead.split(",")); zs = np.arange(1.0, P["length_mm"], 1.0); nz = len(zs)
est = {k: np.full((N, nz), np.nan) for k in ("csa_free", "csa_model", "d_est", "r_cart_dev", "cov")}
gt = {k: np.full((N, nz), np.nan) for k in ("csa", "d")}
iz_gt = np.array([int(np.argmin(np.abs(z - zz))) for zz in zs]); memb_mid = np.abs(np.angle(np.exp(1j * (th - np.pi)))) < np.radians(20)
grid = np.arange(-Rr - 1, Rr + 1.0001, 0.1); GX, GY = np.meshgrid(grid, grid, indexing="ij"); disc = np.hypot(GX, GY) <= Rr + 0.3
anchor = (int(np.argmin(np.abs(grid - (Rr - 1.0)))), int(np.argmin(np.abs(grid))))   # 1 mm inside the (rigid) anterior wall


def free_area(x, y, delta=0.2):
    """model-free lumen area: free space inside the canonical disc, farther than delta from any wall point, connected
    to a point just inside the anterior wall (which never moves)."""
    occ = np.zeros(GX.shape, bool); ix = np.clip(((x - grid[0]) / 0.1).round().astype(int), 0, len(grid) - 1); iy = np.clip(((y - grid[0]) / 0.1).round().astype(int), 0, len(grid) - 1)
    occ[ix, iy] = True; r = int(round(delta / 0.1)); yy, xx = np.ogrid[-r:r + 1, -r:r + 1]; occ = binary_dilation(occ, structure=(xx ** 2 + yy ** 2) <= r * r)
    free = disc & ~occ; lab, _ = label(free); L = lab[anchor]
    return float((lab == L).sum() * 0.01) if L > 0 else np.nan


def membrane_d(x, y, r_can):
    """declared prior: the posterior sector translates along +x with the known taper w(theta0). A point keeps its y, so
    its rest position on the canonical wall is (-xa, y) with xa = sqrt(r^2 - y^2) and rest angle theta0; it predicts
    d = (x + xa) / w(theta0). Robust median over all posterior-side points with w > 0.3 that are not on the anterior
    wall (x < xa - 0.8: there a membrane hugging the wall is indistinguishable from the wall itself). At rest the
    median is ~0; too few points -> unknown (NaN), never 0."""
    inside = np.abs(y) < 0.85 * r_can; xa = np.sqrt(np.maximum(r_can ** 2 - y[inside] ** 2, 0)); xi, yi = x[inside], y[inside]
    th0 = np.mod(np.arctan2(yi, -xa), 2 * np.pi); w0 = np.interp(th0, th, w_memb)
    use = (w0 > 0.6) & (xi < xa - 0.8) & (xi + xa > -1.0)
    return float(np.median((xi[use] + xa[use]) / w0[use])) if use.sum() >= MIN_PTS else np.nan


depth_dir = f"{dense}/stereo/depth_maps"; n_used = 0
for j, (k, n) in enumerate(zip(frames, names)):
    fp = f"{depth_dir}/{n}.geometric.bin"
    if not os.path.exists(fp): continue
    dep = read_depth(fp); model, W, H, prm = cams[imgs[n][2]]; fx, fy, cx, cy = prm[:4]
    h_, w_ = dep.shape; sx, sy = w_ / W, h_ / H; fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy      # depth maps may be downscaled relative to the camera record
    v, u = np.mgrid[0:h_:2, 0:w_:2]; d_ = dep[::2, ::2]; ok = d_ > 0
    Xc = np.stack([(u[ok] - cx) / fx * d_[ok], (v[ok] - cy) / fy * d_[ok], d_[ok]], 1)
    Xw = (Rc2w[j] @ Xc.T).T + C[j]; Pmm = (s * (Ro @ Xw.T)).T + tro; zc = float(((s * (Ro @ C[j])) + tro)[2])
    n_used += 1
    for i, zz in enumerate(zs):
        if not (zc + lo_ahead <= zz <= zc + hi_ahead): continue
        Q = Pmm[np.abs(Pmm[:, 2] - zz) < 0.5]
        if len(Q) < 60: continue
        x, y = Q[:, 0], Q[:, 1]
        from scipy.spatial import cKDTree
        nn = cKDTree(np.stack([x, y], 1)).query_ball_point(np.stack([x, y], 1), 0.3, return_length=True); keep = nn >= 4   # drop isolated floaters inside the lumen
        if keep.sum() < 60: continue
        x, y = x[keep], y[keep]; ang = np.arctan2(y, x); b = ((ang + np.pi) / (2 * np.pi) * 36).astype(int) % 36
        est["cov"][k, i] = len(np.unique(b)) / 36.0
        rc = r0[iz_gt[i]]; r_can = float(rc[np.abs(th) < np.radians(40)].mean())         # canonical anterior (cartilage) radius at this station
        r_memb0 = float(rc[w_memb > 0.9].mean())                                          # canonical rest radius of the membrane sector (no rings there)
        ant = np.abs(ang) < np.radians(40); est["r_cart_dev"][k, i] = float(np.median(np.hypot(x[ant], y[ant])) - r_can) if ant.sum() >= 20 else np.nan
        est["csa_free"][k, i] = free_area(x, y)
        dm = membrane_d(x, y, r_memb0); est["d_est"][k, i] = dm
        if np.isfinite(dm):
            Xm, Ym = dt.deform_xy(r0[iz_gt[i]:iz_gt[i] + 1], th, (max(dm, 0.0) * w_memb)[None, :], p); est["csa_model"][k, i] = float(dt.csa_from_xy(Xm, Ym)[0])
        gt["csa"][k, i] = CSA_gt[k, iz_gt[i]]; gt["d"][k, i] = float(D[k, iz_gt[i], memb_mid].max())
print(f"depth maps used: {n_used}/{len(names)}; (t,z) cells measured: {np.isfinite(est['csa_free']).sum()}")
# temporal regularisation of the deformation field: 3-frame median per station (one frame of latency in a streaming system)
est["d_smooth"] = np.full((N, nz), np.nan); est["csa_model_s"] = np.full((N, nz), np.nan)
for i in range(nz):
    col = est["d_est"][:, i]
    for k in range(N):
        win = col[max(0, k - 1):k + 2]; win = win[np.isfinite(win)]
        if len(win) >= 2 and np.isfinite(col[k]):
            est["d_smooth"][k, i] = float(np.median(win))
            Xm, Ym = dt.deform_xy(r0[iz_gt[i]:iz_gt[i] + 1], th, (max(est["d_smooth"][k, i], 0.0) * w_memb)[None, :], p); est["csa_model_s"][k, i] = float(dt.csa_from_xy(Xm, Ym)[0])

# ---------------------------------------------------------------- 3. metrics vs truth and vs the rigid map
ok = np.isfinite(est["csa_model"]) & np.isfinite(gt["csa"]) & (est["cov"] >= 0.6); okf = ok & np.isfinite(est["csa_free"])
rel_free = est["csa_free"][okf] / gt["csa"][okf] - 1; rel_model = est["csa_model"][ok] / gt["csa"][ok] - 1
seen = np.isfinite(est["cov"]) & (est["cov"] >= 0.6)
res = dict(workspace=a.workspace, synth_run=a.synth_run, window=a.window, frames_registered=int(len(names)), frames_total=int(N), cells=int(ok.sum()), cells_seen=int(seen.sum()), d_estimated_fraction=float(ok.sum() / max(seen.sum(), 1)),
           csa_free_rel_err_median=float(np.median(rel_free)), csa_free_rel_err_iqr=[float(np.percentile(rel_free, 25)), float(np.percentile(rel_free, 75))], csa_free_rel_err_p90_abs=float(np.percentile(np.abs(rel_free), 90)),
           csa_model_rel_err_median=float(np.median(rel_model)), csa_model_rel_err_iqr=[float(np.percentile(rel_model, 25)), float(np.percentile(rel_model, 75))], csa_model_rel_err_p90_abs=float(np.percentile(np.abs(rel_model), 90)),
           cartilage_radius_dev_mm_median=float(np.nanmedian(np.abs(est["r_cart_dev"][ok]))), cartilage_radius_dev_mm_p90=float(np.nanpercentile(np.abs(est["r_cart_dev"][ok]), 90)),
           membrane_d_err_mm_median=float(np.median(np.abs(est["d_est"][ok] - gt["d"][ok]))), membrane_d_err_mm_p90=float(np.percentile(np.abs(est["d_est"][ok] - gt["d"][ok]), 90)))
oks = ok & np.isfinite(est["csa_model_s"]); rel_s = est["csa_model_s"][oks] / gt["csa"][oks] - 1
res.update(csa_model_smoothed_rel_err_median=float(np.median(rel_s)), csa_model_smoothed_rel_err_iqr=[float(np.percentile(rel_s, 25)), float(np.percentile(rel_s, 75))], csa_model_smoothed_rel_err_p90_abs=float(np.percentile(np.abs(rel_s), 90)),
           membrane_d_smoothed_err_mm_median=float(np.median(np.abs(est["d_smooth"][oks] - gt["d"][oks]))), membrane_d_smoothed_err_mm_p90=float(np.percentile(np.abs(est["d_smooth"][oks] - gt["d"][oks]), 90)))
# rigid baseline: time-constant CSA(z) from the rigid dense cloud, evaluated on the same cells
rig = None; sr = glob.glob(f"{a.workspace}/score_rigid_*.json")
if sr:
    S = json.load(open(sr[0])); dn = S.get("dense", {})
    if dn.get("zc"): rig = np.interp(zs, np.array(dn["zc"]), np.array(dn["csa_z_mm2"], float), left=np.nan, right=np.nan); rig_grid = np.broadcast_to(rig[None, :], (N, nz))
    okr = ok & np.isfinite(rig_grid); rel_rig = rig_grid[okr] / gt["csa"][okr] - 1
    res.update(rigid_rel_err_median=float(np.median(rel_rig)), rigid_rel_err_p90_abs=float(np.percentile(np.abs(rel_rig), 90)))
# the event station
if P.get("collapse_target", 0) > 0 or P.get("breath_target", 0) > 0:
    i0 = int(np.argmin(np.abs(zs - P["collapse_z0_mm"]))) if P.get("collapse_target", 0) > 0 else nz // 2
    kk = np.where(ok[:, i0])[0]
    if len(kk):
        e, gg = np.nan_to_num(est["csa_free"][kk, i0], nan=np.nanmedian(est["csa_free"][kk, i0])), gt["csa"][kk, i0]; em = est["csa_model"][kk, i0]
        res["event_station"] = dict(z_mm=float(zs[i0]), frames=[int(kk.min()), int(kk.max())], t=[float(t[kk.min()]), float(t[kk.max()])],
                                    gt_min_csa=float(gg.min()), gt_min_t=float(t[kk[np.argmin(gg)]]), est_free_min_csa=float(e.min()), est_free_min_t=float(t[kk[np.argmin(e)]]),
                                    est_model_min_csa=float(em.min()), gt_reduction_seen=float(1 - gg.min() / CSA_gt[0, iz_gt[i0]]), est_free_reduction=float(1 - e.min() / CSA_gt[0, iz_gt[i0]]),
                                    est_model_reduction=float(1 - em.min() / CSA_gt[0, iz_gt[i0]]), rmse_free=float(np.sqrt(((e - gg) ** 2).mean())), rmse_model=float(np.sqrt(((em - gg) ** 2).mean())),
                                    rigid_csa=float(rig[i0]) if rig is not None and np.isfinite(rig[i0]) else None,
                                    rmse_rigid=float(np.sqrt(((rig[i0] - gg) ** 2).mean())) if rig is not None and np.isfinite(rig[i0]) else None)
        seen_k = np.where(seen[:, i0])[0]; ev = np.abs(t[seen_k] - (P["collapse_t0_s"] if P.get("collapse_target", 0) > 0 else t[seen_k].mean())) < 3 * P.get("collapse_sigma_s", 0.5)
        res["event_station"].update(frames_seen=int(len(seen_k)), frames_estimated=int(len(kk)), coverage=float(len(kk) / max(len(seen_k), 1)),
                                    event_window_frames_seen=int(ev.sum()), event_window_frames_estimated=int(np.isin(seen_k[ev], kk).sum()))
        es = est["csa_model_s"][kk, i0]; oks_ = np.isfinite(es)
        if oks_.any(): res["event_station"].update(est_smoothed_min_csa=float(np.nanmin(es)), est_smoothed_reduction=float(1 - np.nanmin(es) / CSA_gt[0, iz_gt[i0]]), rmse_smoothed=float(np.sqrt(((es[oks_] - gg[oks_]) ** 2).mean())))
json.dump(res, open(f"{a.out}/m1_result.json", "w"), indent=1); np.savez_compressed(f"{a.out}/m1_grid.npz", zs=zs, t=t, **{f"est_{k}": v for k, v in est.items()}, **{f"gt_{k}": v for k, v in gt.items()})
for k_, v_ in res.items():
    if k_ != "event_station": print(f"  {k_}: {v_}")
if "event_station" in res: print("  event station:", json.dumps(res["event_station"]))

# ---------------------------------------------------------------- 4. figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(15, 9)); i0 = int(np.argmin(np.abs(zs - P.get("collapse_z0_mm", 30)))) if P.get("collapse_target", 0) > 0 else nz // 2
ax = axs[0, 0]; ax.plot(t, CSA_gt[:, iz_gt[i0]], "k-", lw=2, label="truth")
if rig is not None: ax.axhline(rig[i0], color="tab:red", ls="--", label="rigid map (time-constant)")
ax.plot(t, est["csa_free"][:, i0], "o", ms=3, color="tab:blue", label=f"M1 model-free (window ±{a.window})"); ax.plot(t, est["csa_model"][:, i0], "s", ms=3, color="tab:green", alpha=.5, label="M1 with anatomical prior (per frame)"); ax.plot(t, est["csa_model_s"][:, i0], "-", lw=2, color="tab:green", label="M1 prior + 3-frame median")
ax.set_xlabel("t (s)"); ax.set_ylabel("CSA (mm²)"); ax.set_title(f"cross-sectional area at z = {zs[i0]:.0f} mm"); ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_ylim(bottom=0)
ax = axs[0, 1]; ax.plot(t, gt["d"][:, i0], "k-", lw=2, label="truth"); ax.plot(t, est["d_est"][:, i0], "o", ms=3, color="tab:green", alpha=.5, label="M1 per frame"); ax.plot(t, est["d_smooth"][:, i0], "-", lw=2, color="tab:green", label="M1 3-frame median")
ax.set_xlabel("t (s)"); ax.set_ylabel("membrane displacement d (mm)"); ax.set_title(f"deformation field at z = {zs[i0]:.0f} mm"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax = axs[1, 0]; m = ax.imshow(np.where(ok, est["csa_model_s"], np.nan).T, origin="lower", aspect="auto", extent=[t[0], t[-1], zs[0], zs[-1]], cmap="viridis", vmin=0, vmax=np.nanmax(CSA_gt))
ax.set_xlabel("t (s)"); ax.set_ylabel("z (mm)"); ax.set_title("M1 (prior, smoothed) CSA(z, t) where observed"); plt.colorbar(m, ax=ax, label="mm²")
ax = axs[1, 1]; m = ax.imshow(np.where(ok, gt["csa"], np.nan).T, origin="lower", aspect="auto", extent=[t[0], t[-1], zs[0], zs[-1]], cmap="viridis", vmin=0, vmax=np.nanmax(CSA_gt))
ax.set_xlabel("t (s)"); ax.set_ylabel("z (mm)"); ax.set_title("truth on the same cells"); plt.colorbar(m, ax=ax, label="mm²")
fig.suptitle(f"M1 v0 — {os.path.basename(a.synth_run)}: canonical cartilage + per-frame membrane (poses from the rigid model)", fontweight="bold"); fig.tight_layout()
fig.savefig(f"{a.out}/m1_{os.path.basename(a.synth_run)}.png", dpi=110); print("figure:", f"{a.out}/m1_{os.path.basename(a.synth_run)}.png")
