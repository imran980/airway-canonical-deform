"""Calibrate the range-dependent radius bias out of a real per-frame grid and re-save it as a new run.

For each station the per-frame whole-circumference median radius is regressed on the camera's distance to the station;
each frame's radii are divided by (1 + k * (distance - reference distance)), the canonical wall, deviation and filled
area are recomputed with the same rules as m1/windowed_depth_real.py. The fit uses the same data it corrects, so the
result is a preview of what a calibration on a rigid segment would give, not an independent measurement; the image
checks (dark-lumen / brightness) remain independent of it.

Usage: python m2/real_range_correct.py runs/m1_real_20V1 --out runs/m1_real_20V1_rc"""
import os, json, argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--out", required=True); ap.add_argument("--min-frames", type=int, default=60); a = ap.parse_args()
G = np.load(f"{a.run}/m1_real_grid.npz"); r_grid = G["r_grid"].copy(); C = G["C"]; S = G["S"]; R = float(G["R"]); N, nS, nb = r_grid.shape
r0 = r_grid.copy(); r0[(r0 - np.nanmedian(r0, 0)[None]) / R > 0.3] = np.nan          # lost-wall sectors excluded from the fit
slopes = {}
for i in range(nS):
    dist = np.linalg.norm(C - S[i], axis=1) / R; y = np.nanmedian(r0[:, i, :], axis=1) / R; ok = np.isfinite(y) & np.isfinite(dist)
    if ok.sum() < a.min_frames or np.std(dist[ok]) < 0.1: continue
    A = np.vstack([dist[ok], np.ones(ok.sum())]).T; k, b = np.linalg.lstsq(A, y[ok], rcond=None)[0]; d_ref = np.median(dist[ok])
    r_grid[:, i, :] = r_grid[:, i, :] / (1 + k / (k * d_ref + b) * (dist - d_ref))[:, None]        # relative slope about the reference distance
    slopes[i] = dict(slope_R_per_R=float(k), rel_slope_per_R=float(k / (k * d_ref + b)), d_ref_R=float(d_ref), n=int(ok.sum()))
# recompute canonical, deviation, filled area (same rules as windowed_depth_real.py)
r_can = np.nanmedian(r_grid, axis=0); n_obs = np.isfinite(r_grid).sum(0); r_can[n_obs < 5] = np.nan
dev = (r_grid - r_can[None]) / R; tb = (np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi
def area(rr, okb): xx, yy = rr[okb] * np.cos(tb[okb]), rr[okb] * np.sin(tb[okb]); return 0.5 * abs(np.dot(xx, np.roll(yy, -1)) - np.dot(yy, np.roll(xx, -1)))
csa_can = np.full(nS, np.nan); csa_fill = np.full((N, nS), np.nan)
for i in range(nS):
    okb = np.isfinite(r_can[i])
    if okb.mean() >= 0.75: csa_can[i] = area(r_can[i], okb)
for j in range(N):
    for i in np.where(np.isfinite(csa_can))[0]:
        rr = np.where(np.isfinite(r_grid[j, i]), r_grid[j, i], r_can[i]); okb = np.isfinite(rr)
        if okb.mean() >= 0.75 and np.isfinite(r_grid[j, i]).mean() >= 0.5: csa_fill[j, i] = area(rr, okb)
os.makedirs(a.out, exist_ok=True)
np.savez_compressed(f"{a.out}/m1_real_grid.npz", frames=G["frames"], s_st=G["s_st"], R=R, r_grid=r_grid, r_can=r_can, dev=dev, csa_fill=csa_fill, csa_can=csa_can, csa_free=G["csa_free"], C=C, S=S)
json.dump(dict(source=a.run, stations_corrected=len(slopes), slopes=slopes), open(f"{a.out}/range_correction.json", "w"), indent=1)
ratio = csa_fill / csa_can[None]; ratio0 = G["csa_fill"] / G["csa_can"][None]
print(f"{len(slopes)} stations corrected; median relative slope {np.median([v['rel_slope_per_R'] for v in slopes.values()]):+.3f} per R; area ratio IQR before {np.nanpercentile(ratio0, 25):.3f}-{np.nanpercentile(ratio0, 75):.3f}, after {np.nanpercentile(ratio, 25):.3f}-{np.nanpercentile(ratio, 75):.3f}; p5-p95 before {np.nanpercentile(ratio0, 5):.2f}-{np.nanpercentile(ratio0, 95):.2f}, after {np.nanpercentile(ratio, 5):.2f}-{np.nanpercentile(ratio, 95):.2f}")
