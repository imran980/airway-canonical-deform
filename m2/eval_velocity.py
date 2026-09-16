"""Compare the wall velocity chosen inside the stereo (velocity.npz from mc_sweep_vsearch.py) with the true d(d)/dt.
Usage: python m2/eval_velocity.py runs/m2_collapse_vs runs/synth_collapse_30"""
import sys, json, numpy as np
out, run = sys.argv[1], sys.argv[2]
V = np.load(f"{out}/velocity.npz"); vstar, zs, t, vels = V["vstar"], V["zs"], V["t"], V["vels"]
g = np.load(f"{run}/gt.npz"); D = g["deformation"].astype(np.float32); th = g["theta"]; z = g["z"]; zc = g["poses_c2w"][:, 2, 3]
mid = np.abs(np.angle(np.exp(1j * (th - np.pi)))) < np.radians(10); dmid = D[:, :, mid].max(2); ddt = np.gradient(dmid, t, axis=0)
iz = np.array([int(np.argmin(np.abs(z - zz))) for zz in zs]); truth = ddt[:, iz]
ok = np.isfinite(vstar); K, I = np.where(ok); ahead = zs[I] - zc[K]; v, tr = vstar[ok], truth[ok]
res = dict(decided_fraction=float(ok.mean()), n=int(ok.sum()), corr=float(np.corrcoef(v, tr)[0, 1]) if ok.sum() > 10 else None,
           abs_err_median=float(np.median(np.abs(v - tr))), abs_err_p90=float(np.percentile(np.abs(v - tr), 90)),
           sign_agreement_when_moving=float(np.mean(np.sign(v[np.abs(tr) > 1]) == np.sign(tr[np.abs(tr) > 1]))) if (np.abs(tr) > 1).sum() > 10 else None,
           chose_zero_when_still=float(np.mean(v[np.abs(tr) < 0.5] == 0)) if (np.abs(tr) < 0.5).sum() > 10 else None)
for lo, hi in ((4, 8), (8, 12), (12, 18)):
    m = (ahead >= lo) & (ahead < hi)
    if m.sum() > 10: res[f"ahead_{lo}_{hi}mm"] = dict(n=int(m.sum()), corr=float(np.corrcoef(v[m], tr[m])[0, 1]), abs_err_median=float(np.median(np.abs(v[m] - tr[m]))))
for lo, hi in ((0, 1), (1, 3), (3, 6), (6, 12), (12, 30)):
    m = (np.abs(tr) >= lo) & (np.abs(tr) < hi)
    if m.sum() > 10: res[f"true_speed_{lo}_{hi}"] = dict(n=int(m.sum()), median_ratio=float(np.median(np.abs(v[m]) / np.maximum(np.abs(tr[m]), 1e-3))) if lo > 0 else None, abs_err_median=float(np.median(np.abs(v[m] - tr[m]))))
print(json.dumps(res, indent=1)); json.dump(res, open(f"{out}/velocity_eval.json", "w"), indent=1)
