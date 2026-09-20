"""Slide the sectoral-fold analysis across a whole clip: where, when and on which arc does the wall fold?

For every station the per-ring common mode is removed for each candidate arc (median over the sectors outside it plus a
guard), giving X[frame, arc] = the arc's excursion. Then a window of L frames is slid along the clip; in each window
every station picks its most inward arc, and the window is summarised by how many stations fold beyond a threshold and
by how much their chosen arcs agree. A real, localized fold shows as a window where several neighbouring stations fold
at the SAME angle; measurement noise shows as deep excursions with scattered angles.

The null is the clip itself: the distribution of the same statistics over all windows, so a window is "unusual" only
relative to the rest of the recording. Outputs a per-window table, a JSON, and a figure.

Usage: python m2/sectoral_scan.py runs/m1_real_20V1_eye --length 13 --step 3 --out docs/figures/x.png
"""
import argparse, os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--arc-width", type=float, default=120.0); ap.add_argument("--guard", type=int, default=2); ap.add_argument("--canonical-pct", type=float, default=None, help="rebuild the canonical wall per (station, sector) as this percentile of the observed radii (e.g. 75 = the OPEN phase) instead of using the run's time-median canonical; a moving wall makes the median canonical a mixture of open and folded states")
ap.add_argument("--length", type=int, default=13, help="window length in frames"); ap.add_argument("--step", type=int, default=3); ap.add_argument("--min-sectors", type=int, default=12)
ap.add_argument("--min-frames", type=int, default=4, help="frames a station needs inside a window"); ap.add_argument("--min-obs", type=int, default=30, help="usable frames a station needs overall")
ap.add_argument("--fold", type=float, default=0.12, help="excursion (R) counted as a fold"); ap.add_argument("--agree", type=float, default=30.0, help="degrees within which arcs count as the same")
ap.add_argument("--max-roughness", type=float, default=3.0, help="reject a fold whose arc sits on a canonical ring this many times rougher than the rest of the ring")
ap.add_argument("--event", type=int, nargs=2, default=None, help="mark a known event on the figure"); ap.add_argument("--fps", type=float, default=29.97)
ap.add_argument("--out", default=None); ap.add_argument("--label", default=None); ap.add_argument("--max-stations", type=int, default=30); ap.add_argument("--recompute", action="store_true")
a = ap.parse_args()
G = np.load(f"{a.run}/m1_real_grid.npz"); fr = G["frames"]; R = float(G["R"]); s_st = G["s_st"]
if a.canonical_pct is not None:
    _r = G["r_grid"]; _R = float(G["R"])
    with np.errstate(all="ignore"): _can = np.nanpercentile(_r, a.canonical_pct, axis=0)
    _n = np.isfinite(_r).sum(0); _can[_n < 5] = np.nan
    dev = (_r - _can[None]) / _R; print(f"canonical rebuilt from the {a.canonical_pct:.0f}th percentile (the open phase) on {int(np.isfinite(_can).sum())} (station, sector) cells")
else: dev = G["dev"].copy()
dev[(dev > 0.3) | (np.abs(dev) > 1.5)] = np.nan
N, nS, nb = dev.shape; A = max(1, int(round(a.arc_width / 360 * nb))); tb = (np.arange(nb) + 0.5) / nb * 360 - 180
ARC = np.array([[(b + q) % nb for q in range(A)] for b in range(nb)])
CTRL = [np.array([q for q in range(nb) if q not in set(ARC[b]) and min((q - b) % nb, (b - q) % nb, (q - (b + A - 1)) % nb, ((b + A - 1) - q) % nb) > a.guard]) for b in range(nb)]
obs = np.isfinite(dev).sum(2); cand = [i for i in range(nS) if (obs[:, i] >= a.min_sectors).sum() >= a.min_obs][:a.max_stations]
# how trustworthy is the canonical ring, sector by sector: a wall that is jagged in one arc gives folds there for free
rc = (_can if a.canonical_pct is not None else G["r_can"]); ROUGH = np.full((nS, nb), np.nan)
for i in range(nS):
    rr = rc[i] / R; ROUGH[i] = np.abs(rr - 0.5 * (np.roll(rr, 1) + np.roll(rr, -1)))
cache = f"{a.run}/sectoral_series_{int(a.arc_width)}.npz"
if os.path.exists(cache) and not a.recompute:
    Z = np.load(cache); X = Z["X"]; cand = list(Z["stations"]); print(f"series reused from {cache}")
else:
    X = np.full((len(cand), N, nb), np.nan)
    for si, i in enumerate(cand):
        for j in range(N):
            d = dev[j, i]
            if np.isfinite(d).sum() < a.min_sectors: continue
            for b in range(nb):
                c = d[CTRL[b]]; c = c[np.isfinite(c)]; v = d[ARC[b]]; v = v[np.isfinite(v)]
                if len(c) >= 6 and len(v) >= max(3, A // 3): X[si, j, b] = np.median(v) - np.median(c)
    np.savez_compressed(cache, X=X, stations=np.array(cand)); print(f"series computed for {len(cand)} stations -> {cache}")
starts = list(range(0, N - a.length + 1, a.step)); rows = []
for st in starts:
    W = slice(st, st + a.length); exc, arcs, sts, rgh = [], [], [], []
    for si, i in enumerate(cand):
        w = X[si, W]; n_ok = np.isfinite(w).any(1).sum()
        if n_ok < a.min_frames: continue
        with np.errstate(all="ignore"): m = np.nanmedian(w, axis=0)
        if not np.isfinite(m).any(): continue
        b = int(np.nanargmin(m)); exc.append(float(m[b])); arcs.append(float((tb[b] + (A - 1) / 2 * 360 / nb + 180) % 360 - 180)); sts.append(i)
        with np.errstate(all="ignore"):
            r_in = np.nanmedian(ROUGH[i, ARC[b]]); r_out = np.nanmedian(np.delete(ROUGH[i], ARC[b]))
        rgh.append(float(r_in / r_out) if np.isfinite(r_in) and np.isfinite(r_out) and r_out > 1e-6 else np.nan)
    if len(exc) < 3: continue
    exc = np.array(exc); arcs = np.array(arcs); rgh = np.array(rgh); folding = exc <= -a.fold
    if folding.sum() >= 2:
        ang = np.radians(arcs[folding]); mu = np.angle(np.mean(np.exp(1j * ang))); spread = np.degrees(np.median(np.abs(np.angle(np.exp(1j * (ang - mu))))))
        agree = int(np.sum(np.abs(np.degrees(np.angle(np.exp(1j * (ang - mu))))) <= a.agree)); mu_deg = float(np.degrees(mu))
    else: spread = np.nan; agree = int(folding.sum()); mu_deg = float(arcs[np.argmin(exc)]) if len(exc) else np.nan
    with np.errstate(all="ignore"): rq = float(np.nanmedian(rgh[folding])) if folding.sum() else float(np.nanmedian(rgh))
    rows.append(dict(frame=int(fr[st + a.length // 2]), n_stations=len(exc), n_folding=int(folding.sum()), n_agreeing=agree, canonical_roughness_ratio=rq if np.isfinite(rq) else None, trusted=bool(np.isfinite(rq) and rq <= a.max_roughness), arc_centre_deg=mu_deg, arc_spread_deg=float(spread) if np.isfinite(spread) else None,
                     excursion_median=float(np.median(exc)), excursion_min=float(exc.min()), stations=[int(s) for s in np.array(sts)[folding]]))
lab = a.label or os.path.basename(a.run)
agr = np.array([r["n_agreeing"] for r in rows]); dep = np.array([r["excursion_min"] for r in rows]); frs = np.array([r["frame"] for r in rows])
print(f"[{lab}] {len(rows)} windows of {a.length} frames (step {a.step}), {len(cand)} stations; fold threshold {a.fold} R, agreement {a.agree:.0f} deg")
tr = np.array([bool(r["trusted"]) for r in rows]); good = (agr >= 3) & tr
print(f"[{lab}] windows with >=3 stations folding on the same arc: {int((agr >= 3).sum())} of {len(rows)} ({100*(agr>=3).mean():.0f} %); of those, {int(good.sum())} sit on a trustworthy canonical (roughness ratio <= {a.max_roughness:.0f})")
print(f"[{lab}] best window overall f{frs[np.argmax(agr)]} with {agr.max()} agreeing stations; best TRUSTED window " + (f"f{frs[np.where(good)[0][np.argmax(agr[good])]]} with {agr[good].max()} agreeing stations" if good.any() else "none"))
top = sorted(rows, key=lambda r: (-r["n_agreeing"], r["excursion_min"]))[:8]
print(f"{'frame':>6s} {'stations':>8s} {'folding':>7s} {'agreeing':>8s} {'arc':>6s} {'spread':>7s} {'deepest':>8s} {'rough':>6s} {'trusted':>8s}")
for r in top:
    sp = "    nan" if r["arc_spread_deg"] is None else f"{r['arc_spread_deg']:7.0f}"
    rq = "   nan" if r["canonical_roughness_ratio"] is None else f"{r['canonical_roughness_ratio']:6.1f}"
    print(f"{r['frame']:6d} {r['n_stations']:8d} {r['n_folding']:7d} {r['n_agreeing']:8d} {r['arc_centre_deg']:+6.0f} {sp} {r['excursion_min']:8.2f} {rq} {str(r['trusted']):>8s}")
json.dump(dict(run=a.run, arc_width=a.arc_width, length=a.length, step=a.step, fold=a.fold, windows=rows), open(f"{a.run}/sectoral_scan.json", "w"), indent=1)
if a.out:
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axs[0].plot(frs, agr, "k.-", ms=3, lw=0.9); axs[0].set_ylabel("stations folding\non the same arc"); axs[0].grid(alpha=.3)
    axs[0].set_title(f"{lab}: sliding {a.length}-frame windows — how many neighbouring stations fold on one arc", fontsize=11)
    axs[1].plot(frs, dep, "-", color="tab:red", lw=1, label="deepest station in the window")
    axs[1].plot(frs, [r["excursion_median"] for r in rows], "-", color="0.5", lw=1, label="median station")
    axs[1].axhline(-a.fold, color="0.3", ls=":", label=f"fold threshold {-a.fold:.2f} R"); axs[1].set_ylabel("excursion (R)"); axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)
    cols = ["tab:red" if r["n_agreeing"] >= 3 else "0.7" for r in rows]
    axs[2].scatter(frs, [r["arc_centre_deg"] for r in rows], c=cols, s=14); axs[2].set_ylabel("arc centre (deg)"); axs[2].set_xlabel("frame"); axs[2].grid(alpha=.3)
    axs[2].set_title("chosen arc per window (red: >=3 stations agreeing — a localized fold; grey: no agreement)", fontsize=10)
    if a.event:
        for ax in axs: ax.axvspan(a.event[0] - .5, a.event[1] + .5, color="orange", alpha=0.2)
    fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=115); print("figure:", a.out)
