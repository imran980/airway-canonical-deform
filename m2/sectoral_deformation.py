"""Sectoral deformation after removing per-ring common-mode shifts, with a permutation test.

Per (frame, station) the measured wall deviation is split into a common-mode part (the whole ring moving together:
residual range bias, scale drift, pose error) and a sectoral part (a real fold). For a candidate arc the common mode
is the median deviation over every sector OUTSIDE that arc (plus a guard band), so the arc opposite the candidate is
only part of the reference and its residual stays free as a control. The arc is chosen per station as the most
inward-moving one over the window, which is a free parameter fitted to the same frames, so the same procedure is run
on sham windows drawn from outside the event: the sham distribution is the noise floor the real excursion has to beat.

Reports per station: the chosen arc centre, the common mode removed, the residual excursion at the arc, the residual
at the opposite arc (the control), the sham distribution and a permutation p-value; and across stations, whether the
arc centres agree (the same wall segment) and how far along the path the excursion extends.

Usage: python m2/sectoral_deformation.py runs/m2_26V2_disp --event 1908 1921 --arc-width 120 --out docs/figures/x.png
"""
import argparse, os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--event", type=int, nargs=2, required=True)
ap.add_argument("--arc-width", type=float, default=120.0, help="width of the candidate arc, degrees"); ap.add_argument("--guard", type=int, default=2, help="sectors excluded on each side of the arc when estimating the common mode")
ap.add_argument("--min-frames", type=int, default=4, help="frames with a usable ring needed at a station"); ap.add_argument("--min-sectors", type=int, default=12)
ap.add_argument("--sham", type=int, default=200, help="number of sham windows for the permutation test"); ap.add_argument("--exclude", type=int, default=15, help="frames around the event excluded from the sham pool")
ap.add_argument("--out", default=None); ap.add_argument("--label", default=None)
a = ap.parse_args()
G = np.load(f"{a.run}/m1_real_grid.npz"); fr = G["frames"]; R = float(G["R"]); s_st = G["s_st"]; S = G["S"]; C = G["C"]
if a.canonical_pct is not None:
    _r = G["r_grid"]; _R = float(G["R"])
    with np.errstate(all="ignore"): _can = np.nanpercentile(_r, a.canonical_pct, axis=0)
    _n = np.isfinite(_r).sum(0); _can[_n < 5] = np.nan
    dev = (_r - _can[None]) / _R; print(f"canonical rebuilt from the {a.canonical_pct:.0f}th percentile (the open phase) on {int(np.isfinite(_can).sum())} (station, sector) cells")
else: dev = G["dev"].copy()
dev[(dev > 0.3) | (np.abs(dev) > 1.5)] = np.nan            # lost-wall sectors
N, nS, nb = dev.shape; A = max(1, int(round(a.arc_width / 360 * nb))); tb = (np.arange(nb) + 0.5) / nb * 360 - 180
def arc_idx(b, w): return [(b + q) % nb for q in range(w)]
def window_stat(W, i, b):
    """common-mode-removed arc excursion at station i, arc starting at b, over the frames W.
    returns (arc residual, opposite-arc residual, control scatter, n frames used)"""
    arc = arc_idx(b, A); ctrl = [q for q in range(nb) if min((q - b) % nb, (b - q) % nb, (q - (b + A - 1)) % nb, ((b + A - 1) - q) % nb) > a.guard and q not in arc]
    opp = arc_idx((b + nb // 2) % nb, A)
    ra, ro, sc, n = [], [], [], 0
    for j in W:
        d = dev[j, i]
        if np.isfinite(d).sum() < a.min_sectors: continue
        c = d[ctrl]; c = c[np.isfinite(c)]
        if len(c) < 6: continue
        off = np.median(c); res = d - off
        va = res[arc]; va = va[np.isfinite(va)]; vo = res[opp]; vo = vo[np.isfinite(vo)]
        if len(va) < max(3, A // 3): continue
        ra.append(np.median(va)); ro.append(np.median(vo) if len(vo) >= 3 else np.nan); sc.append(np.median(np.abs(res[ctrl][np.isfinite(res[ctrl])]))); n += 1
    if n < a.min_frames: return None
    return float(np.median(ra)), float(np.nanmedian(ro)) if np.isfinite(ro).any() else np.nan, float(np.median(sc)), n
def best_arc(W, i):
    out = []
    for b in range(nb):
        r = window_stat(W, i, b)
        if r: out.append((r[0], b, r))
    if not out: return None
    r0, b0, rr = min(out, key=lambda t: t[0]); return dict(arc_start=b0, arc_centre_deg=float((tb[b0] + (A - 1) / 2 * 360 / nb + 180) % 360 - 180), excursion=rr[0], opposite=rr[1], control_scatter=rr[2], n_frames=rr[3])
ev_idx = np.where((fr >= a.event[0]) & (fr <= a.event[1]))[0]; L = len(ev_idx)
pool = np.where((fr < a.event[0] - a.exclude) | (fr > a.event[1] + a.exclude))[0]
rng = np.random.default_rng(0)
sham_windows = []
for _ in range(a.sham):
    if len(pool) < L: break
    st = rng.integers(0, len(pool) - L + 1); w = pool[st:st + L]
    if fr[w].max() - fr[w].min() < 3 * L: sham_windows.append(w)          # contiguous in frame number, like the event
rows = []
for i in range(nS):
    real = best_arc(ev_idx, i)
    if real is None: continue
    sham = [best_arc(w, i) for w in sham_windows]; sham = [s["excursion"] for s in sham if s]
    if len(sham) < 20: p = None; sh_med = sh_p5 = np.nan
    else: p = float(np.mean(np.array(sham) <= real["excursion"])); sh_med = float(np.median(sham)); sh_p5 = float(np.percentile(sham, 5))
    rows.append(dict(station=i, arclength_R=float(s_st[i] / R), cam_dist_R=float(np.linalg.norm(C[ev_idx] - S[i], axis=1).mean() / R), **real, sham_n=len(sham), sham_median=sh_med, sham_p5=sh_p5, p_value=p))
sig = [r for r in rows if r["p_value"] is not None and r["p_value"] <= 0.05 and r["excursion"] < 0]
ang = np.radians([r["arc_centre_deg"] for r in sig]); coh = float(np.degrees(np.abs(np.angle(np.mean(np.exp(1j * ang)))))) if len(sig) else np.nan
spread = float(np.degrees(np.median(np.abs(np.angle(np.exp(1j * (ang - np.angle(np.mean(np.exp(1j * ang)))))))))) if len(sig) else np.nan
lab = a.label or os.path.basename(a.run)
print(f"[{lab}] arc {a.arc_width:.0f} deg, event f{a.event[0]}-{a.event[1]} ({L} frames), {len(sham_windows)} sham windows; {len(rows)} stations measurable, {len(sig)} with an inward excursion at p<=0.05")
print(f"{'st':>3s} {'arc R':>6s} {'cam R':>6s} {'arc centre':>11s} {'excursion':>10s} {'opposite':>9s} {'ctrl scat':>10s} {'sham med':>9s} {'sham p5':>8s} {'p':>6s}")
for r in sorted(rows, key=lambda q: q["arclength_R"]):
    f = lambda x: "   nan" if x is None or not np.isfinite(x) else f"{x:+.2f}"
    pv = "  none" if r["p_value"] is None else f"{r['p_value']:6.3f}"
    print(f"{r['station']:3d} {r['arclength_R']:6.1f} {r['cam_dist_R']:6.1f} {r['arc_centre_deg']:+10.0f}  {r['excursion']:+10.2f} {f(r['opposite']):>9s} {r['control_scatter']:10.2f} {f(r['sham_median']):>9s} {f(r['sham_p5']):>8s} {pv}")
if len(sig): print(f"[{lab}] significant stations span {min(r['arclength_R'] for r in sig):.1f}-{max(r['arclength_R'] for r in sig):.1f} R; arc centres agree at {coh:+.0f} deg with a median spread of {spread:.0f} deg")
json.dump(dict(run=a.run, event=a.event, arc_width=a.arc_width, n_sham=len(sham_windows), stations=rows, significant=[r["station"] for r in sig], arc_centre_mean_deg=coh, arc_centre_spread_deg=spread), open(f"{a.run}/sectoral_deformation.json", "w"), indent=1)
if a.out and rows:
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    # A: angular profile of the common-mode-removed deviation during the event, one line per station near the event
    near = sorted(rows, key=lambda q: q["cam_dist_R"])[:5]
    for r in near:
        i = r["station"]; prof = []
        for q in range(nb):
            vals = []
            for j in ev_idx:
                d = dev[j, i]
                if np.isfinite(d).sum() < a.min_sectors: continue
                ctrl = [c for c in range(nb) if min((c - r["arc_start"]) % nb, (r["arc_start"] - c) % nb) > a.guard and c not in arc_idx(r["arc_start"], A)]
                cc = d[ctrl]; cc = cc[np.isfinite(cc)]
                if len(cc) < 6 or not np.isfinite(d[q]): continue
                vals.append(d[q] - np.median(cc))
            prof.append(np.median(vals) if len(vals) >= 2 else np.nan)
        axs[0, 0].plot(tb, prof, ".-", ms=3, lw=1, label=f"station {r['arclength_R']:.1f} R (p {r['p_value'] if r['p_value'] is not None else float('nan'):.3f})")
    axs[0, 0].axhline(0, color="k", lw=1); axs[0, 0].set_xlabel("sector angle (deg)"); axs[0, 0].set_ylabel("deviation after common-mode removal (R)")
    axs[0, 0].set_title("during the event: the same sectors fold inward at neighbouring stations", fontsize=10.5); axs[0, 0].legend(fontsize=8); axs[0, 0].grid(alpha=.3)
    # B: excursion along the path with the sham band
    st_ = [r["arclength_R"] for r in rows]; ex = [r["excursion"] for r in rows]; shm = [r["sham_median"] for r in rows]; sh5 = [r["sham_p5"] for r in rows]
    axs[0, 1].plot(st_, ex, "o-", color="tab:red", label="event"); axs[0, 1].plot(st_, shm, "-", color="0.5", label="sham windows, median")
    axs[0, 1].fill_between(st_, sh5, np.zeros(len(st_)), color="0.75", alpha=0.6, label="sham windows, down to the 5th percentile")
    axs[0, 1].axhline(0, color="k", lw=1); axs[0, 1].set_xlabel("station arclength (R)"); axs[0, 1].set_ylabel("arc excursion (R)"); axs[0, 1].legend(fontsize=8); axs[0, 1].grid(alpha=.3)
    axs[0, 1].set_title("how far along the wall the fold extends (below the grey band = beyond the noise floor)", fontsize=10.5)
    # C: time course at the significant stations
    for r in (sig or near)[:5]:
        i = r["station"]; arc = arc_idx(r["arc_start"], A); ctrl = [c for c in range(nb) if c not in arc]
        ts = []
        for j in range(N):
            d = dev[j, i]
            if np.isfinite(d).sum() < a.min_sectors: ts.append(np.nan); continue
            cc = d[ctrl]; cc = cc[np.isfinite(cc)]; va = d[arc]; va = va[np.isfinite(va)]
            ts.append(np.median(va) - np.median(cc) if len(cc) >= 6 and len(va) >= 3 else np.nan)
        axs[1, 0].plot(fr, ts, ".-", ms=3, lw=0.9, label=f"station {r['arclength_R']:.1f} R")
    axs[1, 0].axvspan(a.event[0] - .5, a.event[1] + .5, color="orange", alpha=0.2); axs[1, 0].axhline(0, color="k", lw=1)
    axs[1, 0].set_xlabel("frame"); axs[1, 0].set_ylabel("arc excursion (R)"); axs[1, 0].legend(fontsize=8); axs[1, 0].grid(alpha=.3); axs[1, 0].set_title("time course at the folding arc (event shaded)", fontsize=10.5)
    # D: permutation test at the most inward station
    r = min(rows, key=lambda q: q["excursion"]); i = r["station"]
    sham_v = [best_arc(w, i) for w in sham_windows]; sham_v = [s["excursion"] for s in sham_v if s]
    axs[1, 1].hist(sham_v, bins=25, color="0.7", label=f"sham windows ({len(sham_v)})")
    axs[1, 1].axvline(r["excursion"], color="tab:red", lw=2, label=f"event: {r['excursion']:+.2f} R (p {r['p_value'] if r['p_value'] is not None else float('nan'):.3f})")
    axs[1, 1].set_xlabel("best arc excursion (R)"); axs[1, 1].set_ylabel("windows"); axs[1, 1].legend(fontsize=8); axs[1, 1].grid(alpha=.3)
    axs[1, 1].set_title(f"permutation test at station {r['arclength_R']:.1f} R: is the fold beyond the noise floor?", fontsize=10.5)
    fig.suptitle(f"{lab}: sectoral deformation after removing per-ring common-mode shifts", fontweight="bold"); fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=115); print("figure:", a.out)
