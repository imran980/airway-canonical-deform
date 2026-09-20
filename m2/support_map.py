"""Per-sector support: which parts of the airway wall are actually measured, and which are unknown.

Every (frame, station, sector) cell of a real per-frame run carries three support measurements, written by
m1/windowed_depth_real.py: how many depth points fell in the sector, how tightly they agree in radius (MAD, in R), and
how square-on the camera saw it (|cos| between the view ray and the sector's radial direction; a sector seen edge-on is
not measured however many points it has). A cell is TRUSTED when all three pass. Everything else is unknown and must
stay unknown - never filled from the canonical, never averaged into a deformation.

Prints where the support is, by angle and by station, and draws a support map. Also reports, for a declared arc, how
much support that arc has compared with the rest of the ring, which is the check that decides whether a measured
"fold" there can be believed at all.

Usage: python m2/support_map.py runs/m1_real_20V1_eye [--arc -140 --width 120] [--out docs/figures/x.png]
"""
import argparse, os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--arc", type=float, default=None); ap.add_argument("--width", type=float, default=120.0)
ap.add_argument("--points", type=int, default=None); ap.add_argument("--mad", type=float, default=None); ap.add_argument("--cos", type=float, default=None)
ap.add_argument("--se", type=float, default=None, help="principled gate: keep a sector when the standard error of its median radius, 1.25*MAD/sqrt(n), is below this (in R). Use with --points as a floor")
ap.add_argument("--sensitivity", action="store_true", help="show how the verdict moves with the threshold")
ap.add_argument("--save-mask", action="store_true", help="write the mask back into the run's grid so every downstream tool uses it")
ap.add_argument("--stations", type=int, nargs="*", default=None); ap.add_argument("--out", default=None); ap.add_argument("--label", default=None)
a = ap.parse_args(); G = np.load(f"{a.run}/m1_real_grid.npz")
if "n_grid" not in G.files: raise SystemExit(f"{a.run} has no support arrays: re-run m1/windowed_depth_real.py (it records them now)")
n = G["n_grid"]; mad = G["mad_grid"]; cosg = G["cos_grid"]; r = G["r_grid"]; R = float(G["R"]); fr = G["frames"]; s_st = G["s_st"]
P0, M0, C0 = (G["support_params"] if "support_params" in G.files else [15, 0.12, 0.25])
pts = a.points if a.points is not None else int(P0); mmax = a.mad if a.mad is not None else float(M0); cmin = a.cos if a.cos is not None else float(C0)
with np.errstate(all="ignore"): se = 1.25 * mad / np.sqrt(np.maximum(n, 1))
if a.se is not None: sup = (n >= pts) & (se <= a.se) & (cosg >= cmin) & np.isfinite(r)
else: sup = (n >= pts) & (mad <= mmax) & (cosg >= cmin) & np.isfinite(r)
N, nS, nb = r.shape; tb = (np.arange(nb) + 0.5) / nb * 360 - 180
lab = a.label or os.path.basename(a.run)
rule = f"at least {pts} points, standard error of the median radius <= {a.se:.3f} R, |cos| >= {cmin:.2f}" if a.se is not None else f"at least {pts} points, radial spread <= {mmax:.2f} R, |cos| >= {cmin:.2f}"
print(f"[{lab}] support = {rule}")
print(f"[{lab}] {100*sup.mean():.1f} % of all cells trusted; {100*np.isfinite(r).mean():.1f} % have any radius; so {100*(np.isfinite(r).sum()-sup.sum())/max(np.isfinite(r).sum(),1):.0f} % of measured sectors fail the gate")
fail = np.isfinite(r) & ~sup
if fail.any():
    why = dict(too_few_points=float(np.mean((n < pts)[fail])), too_spread=float(np.mean((mad > mmax)[fail])), too_oblique=float(np.mean((cosg < cmin)[fail])))
    print(f"[{lab}] why they fail: too few points {100*why['too_few_points']:.0f} %, radii too spread {100*why['too_spread']:.0f} %, seen edge-on {100*why['too_oblique']:.0f} % (reasons overlap)")
obs_st = sup.sum(2).sum(0); sts = a.stations or [int(i) for i in np.argsort(-obs_st)[:8]]
print(f"{'st':>3s} {'arcR':>5s} {'frames w/ >=50% ring':>21s} {'median trusted sectors':>23s} {'best-covered angle':>19s} {'worst':>7s}")
rows = []
for i in sorted(sts):
    fr_ok = sup[:, i, :].sum(1); have = fr_ok > 0
    if have.sum() < 5: continue
    per_ang = sup[:, i, :].sum(0); best = int(np.argmax(per_ang)); worst = int(np.argmin(per_ang))
    rows.append(dict(station=i, arclength_R=float(s_st[i] / R), frames_half_ring=int((fr_ok >= nb // 2).sum()), median_sectors=float(np.median(fr_ok[have])), best_angle=float(tb[best]), best_count=int(per_ang[best]), worst_count=int(per_ang[worst])))
    print(f"{i:3d} {s_st[i]/R:5.1f} {int((fr_ok>=nb//2).sum()):21d} {np.median(fr_ok[have]):18.0f}/{nb} {tb[best]:+15.0f} deg {per_ang[worst]:7d}")
if a.arc is not None:
    inarc = np.abs(np.angle(np.exp(1j * (np.radians(tb) - np.radians(a.arc))))) <= np.radians(a.width / 2)
    print(f"\n[{lab}] the declared arc {a.arc:+.0f} deg +-{a.width/2:.0f}:")
    print(f"{'st':>3s} {'arcR':>5s} {'trusted in arc':>15s} {'trusted opposite':>17s} {'support ratio':>14s} {'verdict':>28s}")
    for i in sorted(sts):
        ia = sup[:, i, inarc].mean(); io = sup[:, i, ~inarc].mean()
        ratio = ia / max(io, 1e-9)
        verdict = "usable" if ia >= 0.25 and ratio >= 0.5 else ("too little support" if ia < 0.25 else "one-sided")
        print(f"{i:3d} {s_st[i]/R:5.1f} {100*ia:14.0f}% {100*io:16.0f}% {ratio:14.2f} {verdict:>28s}")
        rows = [dict(q, arc_support=float(ia), opposite_support=float(io), support_ratio=float(ratio), verdict=verdict) if q["station"] == i else q for q in rows]
if a.sensitivity:
    print(f"\n[{lab}] sensitivity: how much of the ring survives, and the arc's share")
    print(f"{'rule':>34s} {'cells kept':>11s} {'median sectors/ring':>20s}" + (f" {'arc support':>12s}" if a.arc is not None else ""))
    for nm, msk in (("points>=5, se<=0.08 R", (n >= 5) & (se <= 0.08) & (cosg >= cmin) & np.isfinite(r)), ("points>=8, se<=0.05 R", (n >= 8) & (se <= 0.05) & (cosg >= cmin) & np.isfinite(r)),
                    ("points>=15, se<=0.05 R", (n >= 15) & (se <= 0.05) & (cosg >= cmin) & np.isfinite(r)), ("points>=15, MAD<=0.12 R (default)", (n >= 15) & (mad <= 0.12) & (cosg >= cmin) & np.isfinite(r)),
                    ("points>=3 only (old behaviour)", (n >= 3) & np.isfinite(r))):
        per = msk.sum(2); med = np.median(per[per > 0]) if (per > 0).any() else 0
        extra = ""
        if a.arc is not None:
            ia = np.abs(np.angle(np.exp(1j * (np.radians(tb) - np.radians(a.arc))))) <= np.radians(a.width / 2)
            extra = f" {100*msk[:, :, ia].mean():11.0f}%"
        print(f"{nm:>34s} {100*msk.mean():10.1f}% {med:16.0f}/{nb}{extra}")
if a.save_mask:
    d = dict(np.load(f"{a.run}/m1_real_grid.npz")); d["support"] = sup; np.savez_compressed(f"{a.run}/m1_real_grid.npz", **d); print(f"[{lab}] mask written back into {a.run}/m1_real_grid.npz")
json.dump(dict(run=a.run, points=pts, mad=mmax, cos=cmin, trusted_fraction=float(sup.mean()), stations=rows), open(f"{a.run}/support_map.json", "w"), indent=1)
if a.out:
    show = [q["station"] for q in rows][:4] or sts[:4]
    fig, axs = plt.subplots(2, len(show), figsize=(4.0 * len(show), 7.2), squeeze=False)
    for c, i in enumerate(show):
        ax = axs[0][c]; m = ax.imshow(sup[:, i, :].T, aspect="auto", origin="lower", extent=[fr.min() - .5, fr.max() + .5, -180, 180], cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"station {i} ({s_st[i]/R:.1f} R): trusted sectors", fontsize=10); ax.set_xlabel("frame")
        if c == 0: ax.set_ylabel("sector angle (deg)")
        if a.arc is not None:
            for e in (a.arc - a.width / 2, a.arc + a.width / 2): ax.axhline((e + 180) % 360 - 180, color="tab:red", lw=1.2, ls="--")
        ax = axs[1][c]; per_ang = sup[:, i, :].mean(0); raw = np.isfinite(r[:, i, :]).mean(0)
        ax.plot(tb, 100 * raw, color="0.6", lw=1.2, label="has a radius")
        ax.plot(tb, 100 * per_ang, color="tab:green", lw=2, label="trusted")
        if a.arc is not None: ax.axvspan(a.arc - a.width / 2, a.arc + a.width / 2, color="tab:red", alpha=0.12, label="declared arc")
        ax.set_xlabel("sector angle (deg)"); ax.set_ylim(0, 100); ax.grid(alpha=.3)
        if c == 0: ax.set_ylabel("% of frames"); ax.legend(fontsize=8)
    fig.suptitle(f"{lab}: where the wall is actually measured (green = trusted; grey = a radius exists but fails the support gate)", fontweight="bold")
    fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=115); print("figure:", a.out)
