"""Motion-compensated estimate from two one-sided stereo runs.

Short-window stereo on a laterally moving wall converts the wall's image motion into parallax; the resulting bias in
the membrane displacement is proportional to the wall velocity and has opposite sign for sources taken before and
after the reference frame. So with d_past (sources k-w..k-1) and d_future (sources k+1..k+w):
    d = (d_past + d_future) / 2          bias cancels to first order (no calibration constant)
    d_past - d_future  ~  wall velocity  (sign and scale reported against the truth)
Usage: python m1/combine_sides.py runs/m1_collapse_past runs/m1_collapse_future runs/synth_collapse_30 --out runs/m1_collapse_sides
"""
import sys, os, json, argparse
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("past"); ap.add_argument("future"); ap.add_argument("synth_run"); ap.add_argument("--out", required=True); ap.add_argument("--rigid", default=None, help="score_rigid json of the rigid workspace (for the baseline line)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "synthetic")); import deforming_trachea as dt
g = np.load(f"{a.synth_run}/gt.npz"); P = json.loads(str(g["params"])); p = dt.Params(**P); t, z, th, r0, CSA_gt, D = g["t"], g["z"], g["theta"], g["r_canonical"], g["csa_mm2"], g["deformation"].astype(np.float32)
w_memb, _ = dt.sector_weights(th, p)
A, B = np.load(f"{a.past}/m1_grid.npz"), np.load(f"{a.future}/m1_grid.npz"); zs = A["zs"]; iz = np.array([int(np.argmin(np.abs(z - zz))) for zz in zs]); N, nz = len(t), len(zs)
dp, df = A["est_d_smooth"], B["est_d_smooth"]; both = np.isfinite(dp) & np.isfinite(df)
d_comb = np.where(both, 0.5 * (dp + df), np.where(np.isfinite(dp), dp, df))          # fall back to whichever side exists (biased) and flag it
one_sided = np.isfinite(d_comb) & ~both
vel_proxy = np.where(both, dp - df, np.nan)
gt_d = A["gt_d"]; cov = np.fmax(np.nan_to_num(A["est_cov"], nan=0), np.nan_to_num(B["est_cov"], nan=0))
csa = np.full((N, nz), np.nan)
for k in range(N):
    for i in np.where(np.isfinite(d_comb[k]))[0]:
        X, Y = dt.deform_xy(r0[iz[i]:iz[i] + 1], th, (max(d_comb[k, i], 0.0) * w_memb)[None, :], p); csa[k, i] = float(dt.csa_from_xy(X, Y)[0])
gt_csa = np.array([[CSA_gt[k, iz[i]] for i in range(nz)] for k in range(N)])
ok = np.isfinite(csa) & (cov >= 0.6); okb = ok & both
rel = csa[okb] / gt_csa[okb] - 1; derr = np.abs(d_comb[okb] - gt_d[okb])
res = dict(cells_both=int(okb.sum()), cells_one_sided=int((ok & one_sided).sum()), csa_rel_err_median=float(np.median(rel)), csa_rel_err_iqr=[float(np.percentile(rel, 25)), float(np.percentile(rel, 75))], csa_rel_err_p90_abs=float(np.percentile(np.abs(rel), 90)),
           d_err_mm_median=float(np.median(derr)), d_err_mm_p90=float(np.percentile(derr, 90)))
for lab, arr in (("past", dp), ("future", df)):
    m = okb & np.isfinite(arr); e = np.abs(arr[m] - gt_d[m]); res[f"d_err_{lab}_median"] = float(np.median(e)); res[f"d_err_{lab}_p90"] = float(np.percentile(e, 90))
# velocity proxy vs truth: d_past - d_future against -d(d_gt)/dt
ddt = np.gradient(gt_d, t, axis=0); m = okb & np.isfinite(vel_proxy) & (np.abs(ddt) > 0.5)
if m.sum() > 20:
    slope = np.polyfit(ddt[m], vel_proxy[m], 1); res.update(velocity_proxy_slope_mm_per_mm_s=float(slope[0]), velocity_proxy_corr=float(np.corrcoef(ddt[m], vel_proxy[m])[0, 1]))
# event station
if P.get("collapse_target", 0) > 0 or P.get("breath_target", 0) > 0:
    i0 = int(np.argmin(np.abs(zs - P["collapse_z0_mm"]))) if P.get("collapse_target", 0) > 0 else nz // 2; kk = np.where(okb[:, i0])[0]
    if len(kk):
        e, gg = csa[kk, i0], gt_csa[kk, i0]; res["event_station"] = dict(z_mm=float(zs[i0]), n_frames=int(len(kk)), gt_reduction=float(1 - gg.min() / CSA_gt[0, iz[i0]]), est_reduction=float(1 - e.min() / CSA_gt[0, iz[i0]]), est_min_t=float(t[kk[np.argmin(e)]]), gt_min_t=float(t[kk[np.argmin(gg)]]), rmse=float(np.sqrt(((e - gg) ** 2).mean())))
        if a.rigid and os.path.exists(a.rigid):
            S = json.load(open(a.rigid)); dn = S.get("dense", {})
            if dn.get("zc"): rig = float(np.interp(zs[i0], dn["zc"], dn["csa_z_mm2"])); res["event_station"].update(rigid_csa=rig, rmse_rigid=float(np.sqrt(((rig - gg) ** 2).mean())))
json.dump(res, open(f"{a.out}/sides_result.json", "w"), indent=1); np.savez_compressed(f"{a.out}/sides_grid.npz", zs=zs, t=t, d_comb=d_comb, d_past=dp, d_future=df, vel_proxy=vel_proxy, csa=csa, gt_csa=gt_csa, gt_d=gt_d)
print(json.dumps(res, indent=1))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8)); i0 = int(np.argmin(np.abs(zs - P.get("collapse_z0_mm", 30)))) if P.get("collapse_target", 0) > 0 else nz // 2
ax = axs[0]; ax.plot(t, gt_d[:, i0], "k-", lw=2, label="truth"); ax.plot(t, dp[:, i0], ".", ms=4, color="tab:orange", label="sources in the past"); ax.plot(t, df[:, i0], ".", ms=4, color="tab:cyan", label="sources in the future"); ax.plot(t, d_comb[:, i0], "-", lw=2, color="tab:purple", label="mean of both")
ax.set_xlabel("t (s)"); ax.set_ylabel("membrane displacement (mm)"); ax.set_title(f"one-sided stereo biases cancel: z = {zs[i0]:.0f} mm"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax = axs[1]; ax.plot(t, gt_csa[:, i0], "k-", lw=2, label="truth"); ax.plot(t, csa[:, i0], "-", lw=2, color="tab:purple", label="M1 (mean of one-sided)")
if "event_station" in res and "rigid_csa" in res["event_station"]: ax.axhline(res["event_station"]["rigid_csa"], color="tab:red", ls="--", label="rigid map")
ax.set_xlabel("t (s)"); ax.set_ylabel("CSA (mm²)"); ax.set_title("cross-sectional area at the event station"); ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_ylim(bottom=0)
ax = axs[2]; mm = okb & np.isfinite(vel_proxy); ax.plot(ddt[mm], vel_proxy[mm], ".", ms=2, alpha=.4); ax.set_xlabel("true d(d)/dt (mm/s)"); ax.set_ylabel("d_past − d_future (mm)"); ax.set_title("the difference of the two sides measures wall velocity"); ax.grid(alpha=.3)
fig.suptitle(f"motion-compensated M1 — {os.path.basename(a.synth_run)}", fontweight="bold"); fig.tight_layout(); fig.savefig(f"{a.out}/sides.png", dpi=110); print("figure:", f"{a.out}/sides.png")
