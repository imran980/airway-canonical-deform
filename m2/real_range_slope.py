"""Robust range dependence of a real per-frame grid: per-station Theil-Sen slope of the whole-circumference median wall
radius (in R) against the camera's distance to the station (in R), after rejecting lost-wall sectors.
Usage: python m2/real_range_slope.py runs/m1_real_20V1 [runs/m1_real_20V1n ...]"""
import sys, json, numpy as np
def ts(a, v):
    rng = np.random.default_rng(0); idx = rng.choice(len(a), min(len(a), 600), replace=False); aa = a[idx]; vv = v[idx]
    i, j = np.triu_indices(len(aa), 1); da = aa[i] - aa[j]; m = np.abs(da) > 0.3
    return float(np.median((vv[i] - vv[j])[m] / da[m])) if m.sum() > 50 else np.nan
for run in sys.argv[1:]:
    G = np.load(f"{run}/m1_real_grid.npz"); dev = G["dev"].copy(); C = G["C"]; S = G["S"]; R = float(G["R"]); dev[(dev > 0.3) | (np.abs(dev) > 1.5)] = np.nan
    with np.errstate(all="ignore"): med = np.nanmedian(dev, axis=2)
    sl = []; fn = []
    for i in range(dev.shape[1]):
        dist = np.linalg.norm(C - S[i], axis=1) / R; y = med[:, i]; ok = np.isfinite(y)
        if ok.sum() < 60 or np.std(dist[ok]) < 0.3: continue
        s = ts(dist[ok], y[ok]); sl.append(s); a = dist[ok]; v = y[ok]; lo, hi = np.percentile(a, [25, 75]); fn.append(float(np.median(v[a >= hi]) - np.median(v[a <= lo])))
    sl = np.array(sl); fn = np.array(fn); ratio = G["csa_fill"] / G["csa_can"][None]
    res = dict(run=run, stations=int(len(sl)), slope_median_R_per_R=float(np.median(sl)), slope_iqr=[float(np.percentile(sl, 25)), float(np.percentile(sl, 75))], frac_positive=float(np.mean(sl > 0)), far_minus_near_R=float(np.median(fn)),
               area_ratio_p5_p95=[float(np.nanpercentile(ratio, 5)), float(np.nanpercentile(ratio, 95))], cells=int(np.isfinite(ratio).sum()))
    json.dump(res, open(f"{run}/range_slope.json", "w"), indent=1)
    print(f"[slope] {run}: {res['stations']} stations; Theil-Sen slope median {res['slope_median_R_per_R']:+.3f} R/R (IQR {res['slope_iqr'][0]:+.3f}..{res['slope_iqr'][1]:+.3f}), positive at {100*res['frac_positive']:.0f}%; far-near {res['far_minus_near_R']:+.3f} R; area ratio p5-p95 {res['area_ratio_p5_p95'][0]:.2f}-{res['area_ratio_p5_p95'][1]:.2f}; cells {res['cells']}")
