"""M1 v0 on a real bronchoscopy clip: per-frame short-window depth, canonical wall = time median, deformation = per-frame
deviation. No ground truth: the test is whether a documented wall event appears as ONE sector moving while the others
hold (the anatomical prior checked, not imposed), and whether the per-frame CSA(t) shows the event that the rigid map
cannot.

Geometry is in scene units and expressed relative to the tube radius R (median polar radius over the clip), because a
single real clip has no scale. Stations are points on the smoothed camera path; for frame k the stations between
0.8 R and 3.5 R in front of the camera along its viewing axis are measured from that frame's depth map alone.

Usage: python m1/windowed_depth_real.py runs/real_26V2 --out runs/m1_real_26V2 --event 1908 1921 [--window 2] [--gpus 0,1,2,3] [--skip-stereo]
Env:   BRONCHO_COLMAP (default colmap)"""
import sys, os, json, argparse, subprocess, tempfile
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("workspace"); ap.add_argument("--out", required=True); ap.add_argument("--model", default="sparse/0")
ap.add_argument("--event", type=int, nargs=2, default=None, help="frame range of the documented wall event"); ap.add_argument("--window", type=int, default=2)
ap.add_argument("--gpus", default="0,1,2,3"); ap.add_argument("--max-size", type=int, default=1600); ap.add_argument("--skip-stereo", action="store_true"); ap.add_argument("--ahead", default="0.8,3.5")
ap.add_argument("--canonical-frames", type=int, nargs=2, default=None, help="build the canonical wall from this frame range only (e.g. the post-event pullback), falling back to all frames where it has < 5 observations")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap"); dense = f"{a.out}/dense"
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree


def sh(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0: raise RuntimeError(cmd[0] + "\n" + r.stderr[-1500:])


def read_model(model_dir):
    with tempfile.TemporaryDirectory() as td:
        sh(["model_converter", "--input_path", model_dir, "--output_path", td, "--output_type", "TXT"])
        cams = {}
        for l in open(f"{td}/cameras.txt"):
            if l.startswith("#") or not l.strip(): continue
            f = l.split(); cams[int(f[0])] = (f[1], int(f[2]), int(f[3]), np.array(list(map(float, f[4:]))))
        imgs = {}; L = [l for l in open(f"{td}/images.txt") if l.strip() and not l.startswith("#")]
        for l in L[0::2]:
            f = l.split(); imgs[f[9]] = (np.array(list(map(float, f[1:5]))), np.array(list(map(float, f[5:8]))), int(f[8]))
        pts = np.array([[float(x) for x in l.split()[1:4]] for l in open(f"{td}/points3D.txt") if l.strip() and not l.startswith("#")])
    return cams, imgs, pts


def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def fidx(n): return int(os.path.splitext(os.path.basename(n))[0].lstrip("f"))


def read_depth(path):
    with open(path, "rb") as f:
        hdr = b""
        while hdr.count(b"&") < 3: hdr += f.read(1)
        w, h, c = map(int, hdr.decode().split("&")[:3]); return np.fromfile(f, np.float32).reshape(h, w, c)[:, :, 0]


# ------------------------------------------------------------------ 1. stereo with per-frame windows
_, imgs0, pts0 = read_model(f"{a.workspace}/{a.model}")
if not a.skip_stereo:
    subprocess.run(["rm", "-rf", dense])
    sh(["image_undistorter", "--image_path", f"{a.workspace}/images", "--input_path", f"{a.workspace}/{a.model}", "--output_path", dense, "--output_type", "COLMAP", "--max_image_size", str(a.max_size)])
cams, imgs, _ = read_model(f"{dense}/sparse"); names = sorted(imgs, key=fidx); frames = np.array([fidx(n) for n in names]); byidx = {k: n for k, n in zip(frames, names)}
Rc2w = np.array([q2R(imgs[n][0]).T for n in names]); C = np.array([-q2R(imgs[n][0]).T @ imgs[n][1] for n in names]); N = len(names)
# depth range from the sparse points as seen from the cameras
dep_sparse = np.concatenate([((pts0 - C[j]) @ Rc2w[j][:, 2]) for j in range(0, N, max(1, N // 20))]); dep_sparse = dep_sparse[dep_sparse > 0]
dmin, dmax = np.percentile(dep_sparse, 1) * 0.5, np.percentile(dep_sparse, 99) * 1.5
print(f"{N} frames f{frames.min()}-{frames.max()}; window ±{a.window}; depth range {dmin:.3f}-{dmax:.3f} scene units")
if not a.skip_stereo:
    with open(f"{dense}/stereo/patch-match.cfg", "w") as f:
        for k, n in zip(frames, names):
            src = [byidx[j] for j in range(k - a.window, k + a.window + 1) if j != k and j in byidx]
            if not src: src = [byidx[frames[np.argsort(np.abs(frames - k))[1]]]]
            f.write(n + "\n" + ", ".join(src) + "\n")
    sh(["patch_match_stereo", "--workspace_path", dense, "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true", "--PatchMatchStereo.max_image_size", str(a.max_size),
        "--PatchMatchStereo.gpu_index", a.gpus, "--PatchMatchStereo.depth_min", f"{dmin:.5f}", "--PatchMatchStereo.depth_max", f"{dmax:.5f}",
        "--PatchMatchStereo.filter_min_triangulation_angle", "0.25", "--PatchMatchStereo.filter_min_num_consistent", "2", "--PatchMatchStereo.window_radius", "7"])
    print("patch-match stereo done", flush=True)

# ------------------------------------------------------------------ 2. centreline (smoothed camera path) and stations
Cs = uniform_filter1d(C, size=min(15, N // 4 * 2 + 1), axis=0, mode="nearest")
seg = np.linalg.norm(np.diff(Cs, axis=0), axis=1); arc = np.concatenate([[0], np.cumsum(seg)]); L_path = arc[-1]
# first pass: tube radius R from all frames' points (distance from the path)
depth_dir = f"{dense}/stereo/depth_maps"; sample_r = []
tree_path = cKDTree(Cs)
for j in range(0, N, max(1, N // 12)):
    fp = f"{depth_dir}/{names[j]}.geometric.bin"
    if not os.path.exists(fp): continue
    dep = read_depth(fp); model, W, H, prm = cams[imgs[names[j]][2]]; fx, fy, cx, cy = prm[:4]; h_, w_ = dep.shape; sx, sy = w_ / W, h_ / H
    v, u = np.mgrid[0:h_:4, 0:w_:4]; d_ = dep[::4, ::4]; ok = d_ > 0
    Xc = np.stack([(u[ok] - cx * sx) / (fx * sx) * d_[ok], (v[ok] - cy * sy) / (fy * sy) * d_[ok], d_[ok]], 1); Xw = (Rc2w[j] @ Xc.T).T + C[j]
    dist, idx = tree_path.query(Xw); inside = (idx > 2) & (idx < N - 3); sample_r.append(dist[inside])
R = float(np.median(np.concatenate(sample_r))); print(f"tube radius estimate R = {R:.4f} scene units; path length {L_path / R:.1f} R")
# extend the path straight along its end tangents so that stations exist ahead of the first (and last) cameras: on a
# withdrawal the camera looks at wall the path never reaches
ext = 4.0 * R; nb_ = min(10, N - 1)
ax0 = Rc2w[:nb_, :, 2].mean(0); ax0 /= np.linalg.norm(ax0); ax1 = Rc2w[-nb_:, :, 2].mean(0); ax1 /= np.linalg.norm(ax1)       # where the first / last cameras look
body0 = Cs[min(N - 1, 30)] - Cs[0]; body1 = Cs[max(0, N - 31)] - Cs[-1]                                                         # direction into the path body from each end
pre = [Cs[0] + ext * ax0] if (ax0 @ body0) < 0.3 * np.linalg.norm(body0) else []                                                # extend only where the view leaves the path
post = [Cs[-1] + ext * ax1] if (ax1 @ body1) < 0.3 * np.linalg.norm(body1) else []
Cs = np.vstack(pre + [Cs] + post); seg = np.linalg.norm(np.diff(Cs, axis=0), axis=1); arc = np.concatenate([[0], np.cumsum(seg)]); L_path = arc[-1]
print(f"path extended at start: {bool(pre)}, at end: {bool(post)} (along the mean viewing axis of the end frames)")
ds = 0.2 * R; s_st = np.arange(0, L_path, ds); S = np.stack([np.interp(s_st, arc, Cs[:, i]) for i in range(3)], 1)
T = np.gradient(S, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
# parallel-transported normal basis
N1 = np.zeros_like(S); N2 = np.zeros_like(S); n1 = np.cross(T[0], [0, 0, 1.0]); n1 = n1 if np.linalg.norm(n1) > 1e-3 else np.cross(T[0], [0, 1.0, 0]); n1 /= np.linalg.norm(n1)
for i in range(len(S)):
    n1 = n1 - (n1 @ T[i]) * T[i]; n1 /= np.linalg.norm(n1); N1[i] = n1; N2[i] = np.cross(T[i], n1)
lo_a, hi_a = [float(x) * R for x in a.ahead.split(",")]; nS, nb = len(S), 36
r_grid = np.full((N, nS, nb), np.nan); csa_free = np.full((N, nS), np.nan)
for j in range(N):
    fp = f"{depth_dir}/{names[j]}.geometric.bin"
    if not os.path.exists(fp): continue
    dep = read_depth(fp); model, W, H, prm = cams[imgs[names[j]][2]]; fx, fy, cx, cy = prm[:4]; h_, w_ = dep.shape; sx, sy = w_ / W, h_ / H
    v, u = np.mgrid[0:h_:2, 0:w_:2]; d_ = dep[::2, ::2]; ok = d_ > 0
    Xc = np.stack([(u[ok] - cx * sx) / (fx * sx) * d_[ok], (v[ok] - cy * sy) / (fy * sy) * d_[ok], d_[ok]], 1); Xw = (Rc2w[j] @ Xc.T).T + C[j]
    axis = Rc2w[j][:, 2]; i_c = int(np.argmin(np.linalg.norm(S - C[j], axis=1)))                      # closest path station to the camera
    sgn = 1.0 if (T[i_c] @ axis) >= 0 else -1.0                                                          # which way along the path the camera looks
    along = sgn * (s_st - s_st[i_c]); rel_S = S - C[j]; cosang = (rel_S @ axis) / (np.linalg.norm(rel_S, axis=1) + 1e-9)
    for i in np.where((along >= lo_a) & (along <= hi_a) & (cosang > 0.5))[0]:                            # ahead along the path, within ~60 deg of the view axis
        rel = Xw - S[i]; along = rel @ T[i]; Q = rel[np.abs(along) < 0.1 * R]
        if len(Q) < 60: continue
        x, y = Q @ N1[i], Q @ N2[i]
        nn = cKDTree(np.stack([x, y], 1)).query_ball_point(np.stack([x, y], 1), 0.06 * R, return_length=True); keep = nn >= 4
        if keep.sum() < 60: continue
        x, y = x[keep], y[keep]; ang = np.arctan2(y, x); b = ((ang + np.pi) / (2 * np.pi) * nb).astype(int) % nb; rr = np.hypot(x, y)
        rm = np.array([np.median(rr[b == q]) if (b == q).sum() >= 3 else np.nan for q in range(nb)]); r_grid[j, i] = rm
        okb = np.isfinite(rm)
        if okb.mean() >= 0.75:
            tb = (np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi; xx, yy = rm[okb] * np.cos(tb[okb]), rm[okb] * np.sin(tb[okb]); csa_free[j, i] = 0.5 * abs(np.dot(xx, np.roll(yy, -1)) - np.dot(yy, np.roll(xx, -1)))
seen = np.isfinite(r_grid).any(2); print(f"(frame, station) cells measured: {seen.sum()}; stations observed by >= 5 frames: {(seen.sum(0) >= 5).sum()}/{nS}")

# ------------------------------------------------------------------ 3. canonical wall (time median) and deformation
r_can = np.nanmedian(r_grid, axis=0); n_obs = np.isfinite(r_grid).sum(0); r_can[n_obs < 5] = np.nan
if a.canonical_frames:
    cf = (frames >= a.canonical_frames[0]) & (frames <= a.canonical_frames[1]); r_sub = np.nanmedian(r_grid[cf], axis=0); n_sub = np.isfinite(r_grid[cf]).sum(0)
    use = n_sub >= 5; r_can = np.where(use, r_sub, r_can); print(f"canonical wall from f{a.canonical_frames[0]}-f{a.canonical_frames[1]} on {use.sum()} (station, sector) cells, all-frame fallback on {(~use & np.isfinite(r_can)).sum()}")
dev = (r_grid - r_can[None]) / R                                   # deformation in units of R (negative = wall moved into the lumen)
csa_can = np.full(nS, np.nan); tb = (np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi
for i in range(nS):
    okb = np.isfinite(r_can[i])
    if okb.mean() >= 0.75: xx, yy = r_can[i][okb] * np.cos(tb[okb]), r_can[i][okb] * np.sin(tb[okb]); csa_can[i] = 0.5 * abs(np.dot(xx, np.roll(yy, -1)) - np.dot(yy, np.roll(xx, -1)))
# per-frame CSA with missing sectors filled from the canonical wall (like-for-like with csa_can)
csa_fill = np.full((N, nS), np.nan)
for j in range(N):
    for i in np.where(seen[j] & np.isfinite(csa_can))[0]:
        rr = np.where(np.isfinite(r_grid[j, i]), r_grid[j, i], r_can[i]); okb = np.isfinite(rr)
        if okb.mean() >= 0.75 and np.isfinite(r_grid[j, i]).mean() >= 0.5: xx, yy = rr[okb] * np.cos(tb[okb]), rr[okb] * np.sin(tb[okb]); csa_fill[j, i] = 0.5 * abs(np.dot(xx, np.roll(yy, -1)) - np.dot(yy, np.roll(xx, -1)))
ratio = csa_fill / csa_can[None]
res = dict(workspace=a.workspace, frames=[int(frames.min()), int(frames.max())], n_frames=int(N), window=a.window, R_scene=R, n_stations=int(nS), cells=int(seen.sum()),
           csa_ratio_median_all=float(np.nanmedian(ratio)), csa_ratio_iqr_all=[float(np.nanpercentile(ratio, 25)), float(np.nanpercentile(ratio, 75))],
           dev_R_iqr_all=[float(np.nanpercentile(dev, 25)), float(np.nanpercentile(dev, 75))])
if a.event:
    ev = (frames >= a.event[0]) & (frames <= a.event[1]); pre = (frames < a.event[0]) & (frames >= a.event[0] - 60); post = (frames > a.event[1]) & (frames <= a.event[1] + 60)
    # which stations see the event, and in which sector does the wall move?
    dev_ev = np.nanmean(dev[ev], axis=0); dev_out = np.nanmean(dev[pre | post], axis=0)      # (nS, nb)
    diff = dev_ev - dev_out; score = np.nanmin(diff, axis=1)                                  # most inward-moving sector per station
    good = np.where(np.isfinite(score) & (np.isfinite(r_grid[ev]).any(2).sum(0) >= 3))[0]
    if len(good):
        i_best = int(good[np.argmin(score[good])]); b_best = int(np.nanargmin(diff[i_best])); d_row = diff[i_best]
        moving = np.isfinite(d_row) & (d_row < -0.1); still = np.isfinite(d_row) & (np.abs(d_row) < 0.05)
        res["event"] = dict(frames=a.event, n_event_frames_registered=int(ev.sum()), station_index=i_best, station_arclength_R=float(s_st[i_best] / R), moving_sector_deg=float(tb[b_best] * 180 / np.pi),
                            moving_sector_inward_R=float(d_row[b_best]), n_sectors_moving_gt_0p1R=int(moving.sum()), n_sectors_still_lt_0p05R=int(still.sum()), n_sectors_valid=int(np.isfinite(d_row).sum()),
                            csa_ratio_event_min=float(np.nanmin(ratio[ev][:, i_best])) if np.isfinite(ratio[ev][:, i_best]).any() else None,
                            csa_ratio_pre_median=float(np.nanmedian(ratio[pre][:, i_best])) if np.isfinite(ratio[pre][:, i_best]).any() else None,
                            csa_ratio_post_median=float(np.nanmedian(ratio[post][:, i_best])) if np.isfinite(ratio[post][:, i_best]).any() else None,
                            stations_with_event_signal=[int(i) for i in good if score[i] < -0.15])
        print("event:", json.dumps(res["event"]))
json.dump(res, open(f"{a.out}/m1_real_result.json", "w"), indent=1)
np.savez_compressed(f"{a.out}/m1_real_grid.npz", frames=frames, s_st=s_st, R=R, r_grid=r_grid, r_can=r_can, dev=dev, csa_fill=csa_fill, csa_can=csa_can, csa_free=csa_free, C=C, S=S)
print({k: v for k, v in res.items() if k != "event"})

# ------------------------------------------------------------------ 4. figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, cv2
fig, axs = plt.subplots(2, 2, figsize=(15, 9))
ax = axs[0, 0]
if a.event and "event" in res:
    ib = res["event"]["station_index"]; picks = [i for i in (ib - 4, ib - 2, ib, ib + 2, ib + 4) if 0 <= i < nS]
    for i in picks: ax.plot(frames, ratio[:, i], ".-", ms=3, lw=.8, label=f"station {s_st[i] / R:.1f} R")
    ax.axvspan(a.event[0], a.event[1], color="tab:red", alpha=.15, label="documented event")
else:
    for i in range(0, nS, max(1, nS // 5)): ax.plot(frames, ratio[:, i], ".-", ms=3, lw=.8, label=f"station {s_st[i] / R:.1f} R")
ax.axhline(1, color="k", lw=1); ax.set_xlabel("frame"); ax.set_ylabel("CSA(t) / canonical CSA"); ax.set_title("per-frame lumen area relative to the canonical wall"); ax.legend(fontsize=7); ax.grid(alpha=.3)
ax = axs[0, 1]
if a.event and "event" in res:
    m = ax.imshow(dev[:, ib, :].T, aspect="auto", origin="lower", extent=[frames.min(), frames.max(), -180, 180], cmap="RdBu", vmin=-0.6, vmax=0.6); plt.colorbar(m, ax=ax, label="wall deviation from canonical (R)")
    ax.axvline(a.event[0], color="k", ls="--"); ax.axvline(a.event[1], color="k", ls="--"); ax.set_xlabel("frame"); ax.set_ylabel("sector (deg)"); ax.set_title(f"deformation field at station {s_st[ib] / R:.1f} R")
ax = axs[1, 0]
if a.event and "event" in res:
    d_row = np.nan_to_num(np.nanmean(dev[ev], 0)[ib] - np.nanmean(dev[pre | post], 0)[ib]); ax.bar(tb * 180 / np.pi, d_row, width=9, color=np.where(d_row < -0.1, "tab:red", "0.6"))
    ax.set_xlabel("sector (deg)"); ax.set_ylabel("event minus baseline deviation (R)"); ax.set_title("which part of the wall moved during the event"); ax.grid(alpha=.3)
ax = axs[1, 1]; ax.set_axis_off()
if a.event:
    k_ev = frames[ev][len(frames[ev]) // 2] if ev.any() else None
    fp = f"{a.workspace}/images/{byidx.get(k_ev, names[N // 2])}" if k_ev is not None else f"{a.workspace}/images/{names[N // 2]}"
    if os.path.exists(fp): ax.imshow(cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)); ax.set_title(f"frame {k_ev if k_ev is not None else fidx(names[N // 2])} (event)" if k_ev is not None else "mid frame", fontsize=9)
fig.suptitle(f"M1 v0 on real video — {os.path.basename(a.workspace)} (window ±{a.window}, poses from relaxed rigid SfM)", fontweight="bold"); fig.tight_layout(); fig.savefig(f"{a.out}/m1_real.png", dpi=110); print("figure:", f"{a.out}/m1_real.png")
