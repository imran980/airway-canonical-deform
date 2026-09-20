"""Sectoral wall motion on a clip with no discrete event: do neighbouring stations move on the SAME arc, together in time?

Per (frame, station) the per-ring common mode is removed exactly as in sectoral_deformation.py (median over the sectors
outside the candidate arc plus a guard band), leaving a sectoral excursion time series x(t) for every candidate arc.
For each pair of neighbouring stations the arc that maximises the correlation between their series is chosen, and the
null is built by circularly shifting one series in time, which preserves each series' own autocorrelation and amplitude
while destroying the alignment between them, and letting the shifted pair choose its own best arc too. A pair that beats
its own shifted copies is moving on the same arc at the same times.

Also reports, for the winning arc, the amplitude (5th-95th percentile of the excursion), the equivalent lumen-area
change, and where the excursion's power sits in the respiratory band, with the same circular-shift null.

Usage: python m2/sectoral_coherence.py runs/m1_real_20V1_eye --arc-width 120 --out docs/figures/x.png
"""
import argparse, os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--arc-width", type=float, default=120.0); ap.add_argument("--guard", type=int, default=2)
ap.add_argument("--min-overlap", type=int, default=40, help="frames both stations must share"); ap.add_argument("--min-sectors", type=int, default=12)
ap.add_argument("--shifts", type=int, default=300); ap.add_argument("--min-shift", type=int, default=10); ap.add_argument("--fps", type=float, default=29.97)
ap.add_argument("--out", default=None); ap.add_argument("--label", default=None); ap.add_argument("--max-stations", type=int, default=40)
a = ap.parse_args()
G = np.load(f"{a.run}/m1_real_grid.npz"); fr = G["frames"]; R = float(G["R"]); s_st = G["s_st"]; S = G["S"]; C = G["C"]
dev = G["dev"].copy(); dev[(dev > 0.3) | (np.abs(dev) > 1.5)] = np.nan
N, nS, nb = dev.shape; A = max(1, int(round(a.arc_width / 360 * nb))); tb = (np.arange(nb) + 0.5) / nb * 360 - 180
def arc_idx(b): return [(b + q) % nb for q in range(A)]
def series(i):
    """common-mode-removed arc excursion x[frame, arc_start] at station i (NaN where the ring is unusable)"""
    X = np.full((N, nb), np.nan)
    for j in range(N):
        d = dev[j, i]
        if np.isfinite(d).sum() < a.min_sectors: continue
        for b in range(nb):
            arc = arc_idx(b); ctrl = [q for q in range(nb) if q not in arc and min((q - b) % nb, (b - q) % nb, (q - (b + A - 1)) % nb, ((b + A - 1) - q) % nb) > a.guard]
            c = d[ctrl]; c = c[np.isfinite(c)]; v = d[arc]; v = v[np.isfinite(v)]
            if len(c) >= 6 and len(v) >= max(3, A // 3): X[j, b] = np.median(v) - np.median(c)
    return X
obs = np.isfinite(dev).sum(2); cand = [i for i in range(nS) if (obs[:, i] >= a.min_sectors).sum() >= a.min_overlap][:a.max_stations]
print(f"[{a.label or os.path.basename(a.run)}] arc {a.arc_width:.0f} deg; {len(cand)} stations with >= {a.min_overlap} usable frames")
Xs = {i: series(i) for i in cand}
rng = np.random.default_rng(0); rows = []
def best_corr(Xa, Xb, ok):
    """max over arcs of the correlation between the two stations' excursion series on the shared frames"""
    best = (-2, None)
    for b in range(nb):
        u, v = Xa[:, b], Xb[:, b]; m = ok & np.isfinite(u) & np.isfinite(v)
        if m.sum() < a.min_overlap: continue
        if np.std(u[m]) < 1e-6 or np.std(v[m]) < 1e-6: continue
        r = float(np.corrcoef(u[m], v[m])[0, 1])
        if r > best[0]: best = (r, b)
    return best
for k in range(len(cand) - 1):
    i, i2 = cand[k], cand[k + 1]
    if abs(s_st[i2] - s_st[i]) / R > 0.6: continue
    Xa, Xb = Xs[i], Xs[i2]; ok = np.ones(N, bool)
    r, b = best_corr(Xa, Xb, ok)
    if b is None: continue
    # arc-rotation null: the same pair, but the second station's arc rotated away from the first. A genuine localized
    # fold correlates best at zero offset; a whole-ring or per-frame effect correlates just as well at any offset.
    rot = []
    for dlt in range(3, nb - 2):
        bestr = -2
        for b in range(nb):
            u, v = Xa[:, b], Xb[:, (b + dlt) % nb]; m = np.isfinite(u) & np.isfinite(v)
            if m.sum() < a.min_overlap or np.std(u[m]) < 1e-6 or np.std(v[m]) < 1e-6: continue
            bestr = max(bestr, float(np.corrcoef(u[m], v[m])[0, 1]))
        if bestr > -2: rot.append(bestr)
    p_arc = float(np.mean(np.array(rot) >= r)) if len(rot) >= 10 else None
    null = []
    for _ in range(a.shifts):
        sft = int(rng.integers(a.min_shift, N - a.min_shift)); rn, _bn = best_corr(Xa, np.roll(Xb, sft, axis=0), ok)
        if _bn is not None: null.append(rn)
    p = float(np.mean(np.array(null) >= r)) if len(null) >= 30 else None
    u, v = Xa[:, b], Xb[:, b]; m = np.isfinite(u) & np.isfinite(v); amp = float(np.percentile(u[m], 95) - np.percentile(u[m], 5)) if m.sum() else np.nan
    inward = float(-np.percentile(u[m], 5)) if m.sum() else np.nan
    area = float((a.arc_width / 360) * (1 - (1 - inward) ** 2)) if np.isfinite(inward) else np.nan
    rows.append(dict(station=i, station2=i2, arclength_R=float(s_st[i] / R), arc_centre_deg=float((tb[b] + (A - 1) / 2 * 360 / nb + 180) % 360 - 180), corr=r, p_value=p, p_arc=p_arc, rot_median=float(np.median(rot)) if rot else None, rot_max=float(np.max(rot)) if rot else None,
                     null_median=float(np.median(null)) if null else None, null_p95=float(np.percentile(null, 95)) if null else None, n_frames=int(m.sum()),
                     amplitude_R=amp, inward_p5_R=inward, area_change_fraction=area))
sig = [r for r in rows if r["p_value"] is not None and r["p_value"] <= 0.05 and r["p_arc"] is not None and r["p_arc"] <= 0.05]
print(f"{'st':>3s}-{'st2':<3s} {'arc R':>6s} {'arc centre':>11s} {'corr':>6s} {'shift p95':>9s} {'p_time':>7s} {'rot med':>8s} {'rot max':>8s} {'p_arc':>6s} {'inward':>7s} {'area':>6s} {'n':>4s}")
for r in rows:
    pv = "  none" if r["p_value"] is None else f"{r['p_value']:6.3f}"
    g = lambda x, w=9, d=2: " " * (w - 3) + "nan" if x is None or not np.isfinite(x) else f"{x:{w}.{d}f}"
    pa = "  none" if r["p_arc"] is None else f"{r['p_arc']:6.3f}"
    print(f"{r['station']:3d}-{r['station2']:<3d} {r['arclength_R']:6.1f} {r['arc_centre_deg']:+11.0f} {g(r['corr'], 6)} {g(r['null_p95'])} {pv:>7s} {g(r['rot_median'], 8)} {g(r['rot_max'], 8)} {pa} {g(r['inward_p5_R'], 7)} {g(100*r['area_change_fraction'] if r['area_change_fraction'] is not None else None, 5, 0)}% {r['n_frames']:4d}")
ang = np.radians([r["arc_centre_deg"] for r in sig]); coh = float(np.degrees(np.angle(np.mean(np.exp(1j * ang))))) if len(sig) else np.nan
spread = float(np.degrees(np.median(np.abs(np.angle(np.exp(1j * (ang - np.radians(coh)))))))) if len(sig) else np.nan
print(f"[{a.label or os.path.basename(a.run)}] {len(sig)}/{len(rows)} neighbouring pairs coherent in TIME and ARC at p<=0.05" + (f"; their arcs agree at {coh:+.0f} deg with a median spread of {spread:.0f} deg; inward excursion (5th pct) median {np.median([r['inward_p5_R'] for r in sig]):.2f} R = {100*np.median([r['area_change_fraction'] for r in sig]):.0f} % of the lumen area" if len(sig) else ""))
json.dump(dict(run=a.run, arc_width=a.arc_width, pairs=rows, n_significant=len(sig), arc_centre_deg=coh, arc_spread_deg=spread), open(f"{a.run}/sectoral_coherence.json", "w"), indent=1)
if a.out and rows:
    fig, axs = plt.subplots(2, 2, figsize=(15, 9)); lab = a.label or os.path.basename(a.run)
    show = (sig or rows)[:4]
    for r in show:
        b = int(round((r["arc_centre_deg"] - (A - 1) / 2 * 360 / nb + 180) % 360 / 360 * nb)) % nb
        axs[0, 0].plot(fr, Xs[r["station"]][:, b], ".-", ms=2.5, lw=0.8, label=f"station {r['arclength_R']:.1f} R (r {r['corr']:.2f}, p {r['p_value'] if r['p_value'] is not None else float('nan'):.3f})")
    axs[0, 0].axhline(0, color="k", lw=1); axs[0, 0].set_xlabel("frame"); axs[0, 0].set_ylabel("arc excursion after common-mode removal (R)")
    axs[0, 0].set_title("sectoral excursion at the winning arc, neighbouring stations", fontsize=10.5); axs[0, 0].legend(fontsize=8); axs[0, 0].grid(alpha=.3)
    if show:
        r0 = show[0]; i, i2 = r0["station"], r0["station2"]; rs = []; dl = list(range(0, nb))
        for dlt in dl:
            bestr = np.nan
            for b in range(nb):
                u, v = Xs[i][:, b], Xs[i2][:, (b + dlt) % nb]; m = np.isfinite(u) & np.isfinite(v)
                if m.sum() >= a.min_overlap and np.std(u[m]) > 1e-6 and np.std(v[m]) > 1e-6:
                    rr_ = np.corrcoef(u[m], v[m])[0, 1]; bestr = rr_ if not np.isfinite(bestr) else max(bestr, rr_)
            rs.append(bestr)
        axs[0, 1].plot([d * 360 / nb for d in dl], rs, "o-", ms=3, color="tab:red", label="best correlation at this relative rotation")
        if r0["rot_median"] is not None: axs[0, 1].axhline(r0["rot_median"], color="0.5", ls="--", label="median over rotations (the arc-permutation null)")
        axs[0, 1].set_xlabel("rotation of the second station's arc relative to the first (deg)"); axs[0, 1].set_ylabel("correlation"); axs[0, 1].legend(fontsize=8); axs[0, 1].grid(alpha=.3)
        axs[0, 1].set_title(f"is it the SAME arc? (stations {r0['arclength_R']:.1f} R and next)", fontsize=10.5)
    axs[1, 0].plot([r["arclength_R"] for r in rows], [r["corr"] for r in rows], "o-", color="tab:red", label="best-arc correlation")
    axs[1, 0].plot([r["arclength_R"] for r in rows], [r["null_p95"] for r in rows], "-", color="0.5", label="shifted null, 95th percentile")
    axs[1, 0].set_xlabel("station arclength (R)"); axs[1, 0].set_ylabel("correlation"); axs[1, 0].legend(fontsize=8); axs[1, 0].grid(alpha=.3); axs[1, 0].set_title("coherence along the wall", fontsize=10.5)
    ax = axs[1, 1]
    for r in show:
        b = int(round((r["arc_centre_deg"] - (A - 1) / 2 * 360 / nb + 180) % 360 / 360 * nb)) % nb; x = Xs[r["station"]][:, b]; m = np.isfinite(x)
        if m.sum() < 30: continue
        t = (fr - fr.min()) / a.fps; xi = np.interp(t, t[m], x[m]); P = np.abs(np.fft.rfft(xi - xi.mean())) ** 2; f_ = np.fft.rfftfreq(len(xi), 1 / a.fps); band = (f_ > 1 / 8) & (f_ < 1.5)
        ax.plot(60 * f_[band], P[band] / max(P[band].max(), 1e-9), lw=1, label=f"station {r['arclength_R']:.1f} R")
    ax.set_xlabel("cycles per minute"); ax.set_ylabel("normalised power"); ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_title("where the sectoral motion's power sits (respiratory band)", fontsize=10.5)
    fig.suptitle(f"{lab}: is the same arc moving at neighbouring stations, at the same times?", fontweight="bold"); fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=115); print("figure:", a.out)
