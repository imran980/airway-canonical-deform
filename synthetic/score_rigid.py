"""Score a RIGID reconstruction (bronchotrust recover_clip workspace) against the synthetic ground truth.

This is the external cross-check for M0: the unchanged rigid pipeline is run on the rendered video and its output
is compared with the exact poses and cross-sections the generator wrote. The synthetic data is only trusted if a
pipeline that knows nothing about the generator recovers the static tube.

For every sparse model in <workspace>/sparse/*:
  * camera centres are matched to ground-truth frames by file name (f%05d.png) and aligned with a similarity
    transform (Umeyama). Reported: registered fraction, recovered scale (mm per scene unit), centre RMSE (mm),
    optical-axis angular error, per-frame residual against time (the collapse instant is marked).
For the dense cloud <workspace>/dense*/fused.ply (largest model):
  * the cloud is mapped into ground-truth millimetres with the camera-derived similarity; CSA(z) is measured by
    an estimator independent of the pipeline (polar median radius in 36 sectors, 1-mm slabs) and compared with
    the canonical CSA(z);
  * the pipeline's own gated measurement (pipeline/measure_csa.py) gives the %obstruction it would report.

Usage: python synthetic/score_rigid.py <workspace> <gt.npz> [--out-dir DIR] [--label NAME]
Env:   BRONCHOTRUST (default ../bronchotrust), BRONCHO_COLMAP (default colmap)"""
import sys, os, json, glob, argparse, subprocess, tempfile
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("workspace"); ap.add_argument("gt"); ap.add_argument("--out-dir", default=None); ap.add_argument("--label", default=None)
a = ap.parse_args(); ws = a.workspace.rstrip("/"); lab = a.label or os.path.basename(ws); out_dir = a.out_dir or ws; os.makedirs(out_dir, exist_ok=True)
BT = os.environ.get("BRONCHOTRUST", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bronchotrust"))
COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap")

g = np.load(a.gt); P = json.loads(str(g["params"])); t, z, CSA, c2w = g["t"], g["z"], g["csa_mm2"], g["poses_c2w"]
C_gt = c2w[:, :3, 3]; F_gt = c2w[:, :3, :3] @ np.array([0, 0, 1.0])
t0, sig_t = P.get("collapse_t0_s", None), P.get("collapse_sigma_s", 0.0); has_event = P.get("collapse_target", 0) > 0
report = dict(label=lab, workspace=ws, gt=a.gt, models=[])


def read_model_txt(model_dir):
    """Convert a COLMAP binary model to text (robust across COLMAP versions) and parse images.txt."""
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([COLMAP, "model_converter", "--input_path", model_dir, "--output_path", td, "--output_type", "TXT"], capture_output=True, text=True)
        if r.returncode != 0: raise RuntimeError(r.stderr[-500:])
        lines = [l for l in open(os.path.join(td, "images.txt")) if l.strip() and not l.startswith("#")]
    imgs = {}
    for l in lines[0::2]:
        f = l.split(); q = np.array(list(map(float, f[1:5]))); tv = np.array(list(map(float, f[5:8]))); imgs[f[9]] = (q, tv)
    return imgs


def quat_to_R(q):
    w, x, y, z_ = q
    return np.array([[1 - 2 * (y * y + z_ * z_), 2 * (x * y - z_ * w), 2 * (x * z_ + y * w)],
                     [2 * (x * y + z_ * w), 1 - 2 * (x * x + z_ * z_), 2 * (y * z_ - x * w)],
                     [2 * (x * z_ - y * w), 2 * (y * z_ + x * w), 1 - 2 * (x * x + y * y)]])


def umeyama(src, dst):
    """Similarity dst ~ s R src + tr (Umeyama 1991). Returns s, R, tr."""
    mu_s, mu_d = src.mean(0), dst.mean(0); S, D = src - mu_s, dst - mu_d
    U, sv, Vt = np.linalg.svd(D.T @ S / len(src)); d = np.ones(3); d[-1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag(d) @ Vt; s = (sv * d).sum() / (S ** 2).sum(axis=1).mean(); tr = mu_d - s * R @ mu_s
    return s, R, tr


def frame_index(name):
    b = os.path.splitext(os.path.basename(name))[0]; return int(b.lstrip("f")) if b.lstrip("f").isdigit() else None


models = sorted(glob.glob(os.path.join(ws, "sparse", "*")), key=lambda d: -len(read_model_txt(d)) if os.path.isdir(d) else 0)
models = [m for m in models if os.path.isdir(m)]
print(f"{lab}: {len(models)} sparse model(s); GT {len(t)} frames, {'collapse at t0=%.2fs' % t0 if has_event else 'no event'}")
best = None
for m in models:
    imgs = read_model_txt(m); idx, C_e, F_e, Rc_e = [], [], [], []
    for name, (q, tv) in imgs.items():
        k = frame_index(name)
        if k is None or k >= len(t): continue
        R = quat_to_R(q); idx.append(k); C_e.append(-R.T @ tv); F_e.append(R.T @ np.array([0, 0, 1.0])); Rc_e.append(R.T)
    idx = np.array(idx); o = np.argsort(idx); idx, C_e, F_e, Rc_e = idx[o], np.array(C_e)[o], np.array(F_e)[o], np.array(Rc_e)[o]
    if len(idx) < 3: print(f"  model {os.path.basename(m)}: {len(idx)} frames, too few to align"); continue
    s, R, tr = umeyama(C_e, C_gt[idx]); C_al = (s * (R @ C_e.T)).T + tr; res = np.linalg.norm(C_al - C_gt[idx], axis=1)
    # rotation from the camera ORIENTATIONS: centres alone fix the roll about a near-straight path only through the 0.35 mm
    # jitter, which is too weak for sector-resolved comparison of the dense cloud
    Mo = sum(c2w[k, :3, :3] @ Rc_e[j].T for j, k in enumerate(idx)); Uo, _, Vto = np.linalg.svd(Mo)
    Ro = Uo @ np.diag([1, 1, np.sign(np.linalg.det(Uo @ Vto))]) @ Vto; tro = C_gt[idx].mean(0) - s * Ro @ C_e.mean(0)
    res_o = np.linalg.norm((s * (Ro @ C_e.T)).T + tro - C_gt[idx], axis=1); rot_gap = np.degrees(np.arccos(np.clip((np.trace(Ro @ R.T) - 1) / 2, -1, 1)))
    F_al = (R @ F_e.T).T; ang = np.degrees(np.arccos(np.clip((F_al * F_gt[idx]).sum(1), -1, 1)))
    gaps = np.diff(idx); span = (int(idx.min()), int(idx.max()))
    rec = dict(model=os.path.basename(m), n_registered=int(len(idx)), frac_registered=float(len(idx) / len(t)), span=span, max_gap=int(gaps.max()) if len(gaps) else 0,
               scale_mm_per_unit=float(s), centre_rmse_mm=float(np.sqrt((res ** 2).mean())), centre_max_mm=float(res.max()),
               centre_rmse_orientR_mm=float(np.sqrt((res_o ** 2).mean())), centre_vs_orientation_rotation_gap_deg=float(rot_gap),
               axis_err_deg_median=float(np.median(ang)), axis_err_deg_max=float(ang.max()), frames=idx.tolist(), residual_mm=res.tolist(), t=t[idx].tolist())
    rec["axis_err_deg"] = ang.tolist()
    if has_event:
        near = np.abs(t[idx] - t0) < 3 * sig_t
        rec["residual_near_event_mm"] = float(np.sqrt((res[near] ** 2).mean())) if near.any() else None
        rec["residual_far_from_event_mm"] = float(np.sqrt((res[~near] ** 2).mean())) if (~near).any() else None
        rec["n_registered_near_event"] = int(near.sum()); rec["n_gt_near_event"] = int((np.abs(t - t0) < 3 * sig_t).sum())
        if (~near).sum() >= 3:   # anchor the similarity on the unaffected frames only: is the error local to the event or everywhere?
            s2, R2, tr2 = umeyama(C_e[~near], C_gt[idx][~near]); res2 = np.linalg.norm((s2 * (R2 @ C_e.T)).T + tr2 - C_gt[idx], axis=1)
            ang2 = np.degrees(np.arccos(np.clip(((R2 @ F_e.T).T * F_gt[idx]).sum(1), -1, 1)))
            rec["anchored_far"] = dict(scale_mm_per_unit=float(s2), rmse_far_mm=float(np.sqrt((res2[~near] ** 2).mean())), rmse_near_mm=float(np.sqrt((res2[near] ** 2).mean())) if near.any() else None,
                                       max_near_mm=float(res2[near].max()) if near.any() else None, axis_far_deg_median=float(np.median(ang2[~near])), axis_near_deg_median=float(np.median(ang2[near])) if near.any() else None,
                                       residual_mm=res2.tolist(), axis_err_deg=ang2.tolist())
    report["models"].append(rec)
    print(f"  model {rec['model']}: {rec['n_registered']}/{len(t)} frames (f{span[0]}-{span[1]}, max gap {rec['max_gap']}), scale {s:.4f} mm/unit, "
          f"centre RMSE {rec['centre_rmse_mm']:.3f} mm (max {rec['centre_max_mm']:.3f}), axis err median {rec['axis_err_deg_median']:.2f} deg (max {rec['axis_err_deg_max']:.2f}); "
          f"orientation-based R: centre RMSE {rec['centre_rmse_orientR_mm']:.3f} mm, gap to centre-based R {rot_gap:.2f} deg")
    if has_event:
        print(f"    near event (|t-t0|<3 sigma): {rec['n_registered_near_event']}/{rec['n_gt_near_event']} frames registered, RMSE {rec['residual_near_event_mm']:.3f} mm vs {rec['residual_far_from_event_mm']:.3f} mm elsewhere")
        if "anchored_far" in rec:
            af = rec["anchored_far"]; print(f"    Sim(3) anchored on far frames only: scale {af['scale_mm_per_unit']:.4f}, RMSE far {af['rmse_far_mm']:.3f} mm / near {af['rmse_near_mm']:.3f} mm (max near {af['max_near_mm']:.3f}), axis err far {af['axis_far_deg_median']:.2f} / near {af['axis_near_deg_median']:.2f} deg")
            print("    per-frame (every 10th): t  GT z  |  centre err (mm)  axis err (deg)  [far-anchored]")
            for j in range(0, len(idx), 10): print(f"      {t[idx[j]]:5.2f}  {C_gt[idx[j],2]:6.2f}  |  {af['residual_mm'][j]:6.2f}  {af['axis_err_deg'][j]:6.2f}")
    if best is None or rec["n_registered"] > best[0]["n_registered"]: best = (rec, s, Ro, tro)

# ---------------- dense cloud -----------------
plys = sorted(glob.glob(os.path.join(ws, "dense*", "fused.ply")))
csa_z = None; pipe = None
if plys and best is not None:
    rec, s, R, tr = best
    sys.path.insert(0, os.path.join(BT, "pipeline"))
    import open3d as o3d
    from measure_csa import load_clean, measure
    Pc = load_clean(plys[0]); Pmm = (s * (R @ Pc.T)).T + tr
    # where did the posterior (membrane) wall land? Points in a sagittal band |y| < 1.5 mm: x ~ -R is the canonical posterior
    # wall, x ~ +R the anterior wall; anything in between is wall smeared into the lumen. Compared inside the event zone
    # (|z - z0| < 2 sigma_z) and far from it; the anterior wall is the control.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import deforming_trachea as dt
    p_ = dt.Params(**P); band = np.abs(Pmm[:, 1]) < 1.5; Rr = P["radius_mm"]

    def wall_stats(mask):
        xs = Pmm[mask, 0]
        if len(xs) < 20: return dict(n=int(len(xs)))
        return dict(n=int(len(xs)), posterior_median_x=float(np.median(xs[xs < -0.5 * Rr])) if (xs < -0.5 * Rr).sum() >= 10 else None,
                    anterior_median_x=float(np.median(xs[xs > 0.5 * Rr])) if (xs > 0.5 * Rr).sum() >= 10 else None,
                    interior_fraction=float(np.mean(np.abs(xs) < 0.7 * Rr)))

    if has_event:
        zone = np.abs(Pmm[:, 2] - P["collapse_z0_mm"]) < 2 * P["collapse_sigma_z_mm"]
        far = (np.abs(Pmm[:, 2] - P["collapse_z0_mm"]) > 3 * P["collapse_sigma_z_mm"]) & (Pmm[:, 2] > z.min() + 3) & (Pmm[:, 2] < z.max() - 3)
        iz0 = int(np.argmin(np.abs(z - P["collapse_z0_mm"]))); Dk = g["deformation"].astype(np.float32); kpk = int(np.argmax(Dk[:, iz0, :].max(1)))
        Xp, _ = dt.deform_xy(g["r_canonical"][iz0:iz0 + 1], g["theta"], Dk[kpk, iz0:iz0 + 1, :], p_); x_peak = float(Xp[0, int(np.argmin(np.abs(g["theta"] - np.pi)))])
        walls = dict(event_zone=wall_stats(band & zone), far=wall_stats(band & far), gt_posterior_x_canonical=-Rr, gt_posterior_x_at_peak=x_peak, gt_anterior_x=Rr)
        print(f"  posterior wall (sagittal band) in event zone: {walls['event_zone']} | far from event: {walls['far']} | GT posterior x canonical {-Rr:.2f}, at peak {x_peak:.2f}; anterior {Rr:.2f}")
    else:
        walls = dict(all=wall_stats(band & (Pmm[:, 2] > z.min() + 3) & (Pmm[:, 2] < z.max() - 3)), gt_posterior_x=-Rr, gt_anterior_x=Rr)
        print(f"  walls (sagittal band): {walls['all']} | GT posterior x {-Rr:.2f}, anterior {Rr:.2f}")
    # independent estimator in GT coordinates: 1-mm slabs perpendicular to the GT axis (z), polar median radius in 36 sectors
    zc = np.arange(z.min() + 1, z.max() - 1, 1.0); csa_z, cov_z = np.full(len(zc), np.nan), np.zeros(len(zc))
    for i, zb in enumerate(zc):
        Q = Pmm[np.abs(Pmm[:, 2] - zb) < 0.5]
        if len(Q) < 30: continue
        th = np.arctan2(Q[:, 1], Q[:, 0]); r = np.hypot(Q[:, 0], Q[:, 1]); b = ((th + np.pi) / (2 * np.pi) * 36).astype(int) % 36
        rm = np.array([np.median(r[b == k]) if (b == k).sum() >= 3 else np.nan for k in range(36)]); cov_z[i] = np.isfinite(rm).mean()
        if cov_z[i] < 0.75: continue
        thb = (np.arange(36) + 0.5) / 36 * 2 * np.pi - np.pi; ok = np.isfinite(rm); x, y = rm[ok] * np.cos(thb[ok]), rm[ok] * np.sin(thb[ok])
        csa_z[i] = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    gt_z = np.interp(zc, z, CSA[0]); ok = np.isfinite(csa_z)
    d_ce = 2 * np.sqrt(csa_z[ok] / np.pi); d_gt = 2 * np.sqrt(gt_z[ok] / np.pi)
    report["dense"] = dict(ply=plys[0], walls=walls, n_points_clean=int(len(Pc)), n_slabs=int(len(zc)), n_slabs_measured=int(ok.sum()), z_measured=[float(zc[ok].min()), float(zc[ok].max())] if ok.any() else None,
                           calibre_err_mm_mean=float(np.mean(np.abs(d_ce - d_gt))) if ok.any() else None, calibre_bias_mm=float(np.mean(d_ce - d_gt)) if ok.any() else None,
                           csa_ratio_est_over_gt_median=float(np.median(csa_z[ok] / gt_z[ok])) if ok.any() else None, csa_z_mm2=csa_z.tolist(), gt_csa_z_mm2=gt_z.tolist(), zc=zc.tolist(), cov=cov_z.tolist())
    print(f"  dense: {len(Pc)} clean points; {ok.sum()}/{len(zc)} 1-mm slabs measured (z {report['dense']['z_measured']}); "
          f"D_CE error mean {report['dense']['calibre_err_mm_mean']} mm, bias {report['dense']['calibre_bias_mm']} mm (GT D_CE {2*np.sqrt(CSA[0].mean()/np.pi):.2f} mm)")
    try:
        pipe, _ = measure(Pc); report["pipeline_measure_csa"] = pipe
        keys = [k for k in pipe if "obstr" in k or k in ("tier", "n_accepted", "cv")]
        print("  pipeline measure_csa:", {k: pipe[k] for k in keys})
    except Exception as e: print("  pipeline measure_csa failed:", e)
    gt_pct = 100 * (1 - CSA[0].min() / np.percentile(CSA[0], 90)); report["gt_static_pct_obstruction"] = float(gt_pct)
    if has_event: report["gt_event_pct_obstruction"] = float(100 * (1 - CSA.min() / np.percentile(CSA[0], 90)))
    print(f"  GT canonical %obstruction (1 - min/p90, ring corrugation only) = {gt_pct:.1f}%" + (f"; GT event depth = {report['gt_event_pct_obstruction']:.0f}%" if has_event else ""))
elif not plys: print("  no dense cloud in workspace (MVS not finished or not run)")

json.dump(report, open(os.path.join(out_dir, f"score_rigid_{lab}.json"), "w"), indent=1)

# ---------------- figure -----------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6)); ax = axs[0]
ax.plot(t, C_gt[:, 2], "k-", lw=2, label="GT camera z")
for rec in report["models"]:
    ax.plot(rec["t"], np.array(rec["residual_mm"]) * 0 + np.interp(rec["t"], t, C_gt[:, 2]), ".", ms=4, label=f"registered ({rec['model']}: {rec['n_registered']}/{len(t)})")
if has_event: ax.axvspan(t0 - 3 * sig_t, t0 + 3 * sig_t, color="tab:red", alpha=.15, label="collapse ±3σ")
ax.set_xlabel("t (s)"); ax.set_ylabel("z (mm)"); ax.set_title("which frames the rigid pipeline registered"); ax.legend(fontsize=7); ax.grid(alpha=.3)
ax = axs[1]
for rec in report["models"]: ax.plot(rec["t"], rec["residual_mm"], ".-", ms=4, lw=.8, label=f"{rec['model']}: RMSE {rec['centre_rmse_mm']:.2f} mm, scale {rec['scale_mm_per_unit']:.3f}")
if has_event: ax.axvspan(t0 - 3 * sig_t, t0 + 3 * sig_t, color="tab:red", alpha=.15)
ax.set_xlabel("t (s)"); ax.set_ylabel("camera-centre residual after Sim(3) (mm)"); ax.set_title("pose error vs time"); ax.legend(fontsize=7); ax.grid(alpha=.3)
ax = axs[2]
ax.plot(z, CSA[0], "k-", lw=2, label="GT canonical CSA(z)")
if has_event: ax.plot(z, CSA.min(0), "r--", lw=1, label="GT min over time (event)")
if csa_z is not None: ax.plot(zc, csa_z, "o-", ms=3, lw=.8, color="tab:blue", label="rigid dense cloud → mm (independent estimator)")
ax.set_xlabel("z (mm)"); ax.set_ylabel("CSA (mm²)"); ax.set_title("cross-section: rigid recon vs truth"); ax.legend(fontsize=7); ax.grid(alpha=.3); ax.set_ylim(bottom=0)
fig.suptitle(f"rigid cross-check — {lab}", fontweight="bold"); fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"score_rigid_{lab}.png"), dpi=110)
print("wrote", os.path.join(out_dir, f"score_rigid_{lab}.json"), "and .png")
