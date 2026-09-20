"""Is the reconstructed airway a full ring at each station, and how complete is it along the path?

No single endoscope frame sees a whole cross-section: the ring is completed by pooling the frames that image each sector
well. For every station and 10-degree sector this counts the supported observations (the per-sector support gate of
m1/windowed_depth_real.py), forms the pooled wall radius, and reports completeness: how many sectors reach the required
number of observations, the largest angular gap, and the eccentricity of the camera path inside the ring. Sectors that
never get enough observations stay empty - they are unknown, not interpolated.

Usage: python m2/ring_completeness.py runs/m1_20V1_hires_std [--min-obs 3] [--out docs/figures/x.png] [--label A]
"""
import argparse, os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--min-obs", type=int, default=3); ap.add_argument("--pct", type=float, default=75.0)
ap.add_argument("--max-se", type=float, default=None, help="a sector is known only if its CANONICAL radius is well determined: 1.25*MAD_across_frames/sqrt(observations) <= this (in R). Unlike an agreement gate this tolerates a wall that genuinely moves, provided enough frames measured it")
ap.add_argument("--max-disagree", type=float, default=None, help="a sector is known only if its observations AGREE: median absolute deviation across frames, in R (e.g. 0.15). Frames seeing a sector from far away or edge-on disagree, and a ring stitched from them is spiky rather than anatomical")
ap.add_argument("--stations", type=int, nargs="*", default=None); ap.add_argument("--out", default=None); ap.add_argument("--labels", nargs="*", default=None); ap.add_argument("--ignore-support", action="store_true")
a = ap.parse_args()
def load(run):
    G = np.load(f"{run}/m1_real_grid.npz"); r = G["r_grid"].copy(); R = float(G["R"]); s_st = G["s_st"]
    sup = G["support"] if ("support" in G.files and not a.ignore_support) else np.isfinite(r)
    r = np.where(sup, r, np.nan); return G, r, R, s_st
res = []
for ri, run in enumerate(a.runs):
    G, r, R, s_st = load(run); N, nS, nb = r.shape; lab = (a.labels[ri] if a.labels and ri < len(a.labels) else os.path.basename(run))
    obs = np.isfinite(r).sum(0)                                  # supported observations per (station, sector)
    with np.errstate(all="ignore"):
        can = np.nanpercentile(np.where(np.isfinite(r), r, np.nan), a.pct, axis=0)
        med = np.nanmedian(r, axis=0); dis = np.nanmedian(np.abs(r - med[None]), axis=0) / R      # do the frames agree about this sector?
    can[obs < a.min_obs] = np.nan
    if a.max_disagree is not None:
        dropped = int((np.isfinite(can) & (dis > a.max_disagree)).sum()); can[dis > a.max_disagree] = np.nan
        print(f"[{lab}] agreement gate {a.max_disagree:.2f} R: dropped {dropped} sectors whose frames disagree (median disagreement {np.nanmedian(dis):.3f} R)")
    if a.max_se is not None:
        with np.errstate(all="ignore"): se_can = 1.25 * dis / np.sqrt(np.maximum(obs, 1))
        dropped = int((np.isfinite(can) & (se_can > a.max_se)).sum()); can[se_can > a.max_se] = np.nan
        print(f"[{lab}] canonical standard-error gate {a.max_se:.3f} R: dropped {dropped} sectors (median standard error {np.nanmedian(se_can):.3f} R, median spread {np.nanmedian(dis):.3f} R)")
    full = np.isfinite(can)
    def gap(row):
        if row.all(): return 0.0
        if not row.any(): return 360.0
        idx = np.where(~row)[0]; runs_ = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        if row[0] and row[-1]: pass
        else:
            if len(runs_) > 1 and runs_[0][0] == 0 and runs_[-1][-1] == nb - 1: runs_ = [np.concatenate([runs_[-1], runs_[0]])] + runs_[1:-1]
        return max(len(x) for x in runs_) * 360.0 / nb
    stat = []
    for i in range(nS):
        if obs[i].max() < a.min_obs: continue
        comp = float(full[i].mean()); g = gap(full[i])
        rr = can[i]; ok = np.isfinite(rr)
        ecc = np.nan
        if ok.sum() >= nb // 2:
            th = (np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi
            cx = np.mean(rr[ok] * np.cos(th[ok])); cy = np.mean(rr[ok] * np.sin(th[ok])); ecc = float(np.hypot(cx, cy) / max(np.nanmedian(rr), 1e-6))
        stat.append(dict(station=i, arclength_R=float(s_st[i] / R), completeness=comp, largest_gap_deg=g, median_obs=float(np.median(obs[i])), eccentricity=ecc))
    comp = np.array([q["completeness"] for q in stat]); gaps = np.array([q["largest_gap_deg"] for q in stat])
    print(f"[{lab}] {len(stat)} stations with any wall; completeness (sectors with >= {a.min_obs} supported observations):")
    print(f"        median {100*np.median(comp):.0f} %, stations >= 90 % complete: {int((comp>=0.9).sum())}, >= 75 %: {int((comp>=0.75).sum())}, >= 50 %: {int((comp>=0.5).sum())}")
    print(f"        largest angular gap: median {np.median(gaps):.0f} deg, best {gaps.min():.0f} deg; median observations per sector {np.median([q['median_obs'] for q in stat]):.0f}")
    res.append((lab, run, stat, can, obs, R, s_st))
    json.dump(dict(run=run, min_obs=a.min_obs, stations=stat), open(f"{run}/ring_completeness.json", "w"), indent=1)
if a.out:
    nrow = len(res); best = sorted(res[0][2], key=lambda q: -q["completeness"])[:6]
    sts = a.stations or [q["station"] for q in sorted(res[0][2], key=lambda q: q["arclength_R"])[::max(1, len(res[0][2]) // 6)]][:6]
    fig, axs = plt.subplots(nrow, len(sts) + 1, figsize=(3.1 * (len(sts) + 1), 3.3 * nrow), squeeze=False)
    for ri, (lab, run, stat, can, obs, R, s_st) in enumerate(res):
        nb = can.shape[1]; th = (np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi
        for c, i in enumerate(sts):
            ax = axs[ri][c]; rr = can[i] if i < len(can) else np.full(nb, np.nan); ok = np.isfinite(rr)
            if ok.sum() >= 3:
                for b in range(nb):
                    b2 = (b + 1) % nb
                    if ok[b] and ok[b2]: ax.plot([rr[b] * np.cos(th[b]) / R, rr[b2] * np.cos(th[b2]) / R], [rr[b] * np.sin(th[b]) / R, rr[b2] * np.sin(th[b2]) / R], "-", color="tab:green", lw=2)
                sc = ax.scatter(rr[ok] * np.cos(th[ok]) / R, rr[ok] * np.sin(th[ok]) / R, c=np.minimum(obs[i][ok], 40), cmap="viridis", s=14, vmin=0, vmax=40, zorder=3)
            miss = ~ok
            if miss.any(): ax.scatter(1.9 * np.cos(th[miss]), 1.9 * np.sin(th[miss]), marker="x", color="tab:red", s=18, label="unknown" if c == 0 else None)
            ax.plot(0, 0, "k+", ms=8); ax.set_aspect("equal"); ax.set_xlim(-2.1, 2.1); ax.set_ylim(-2.1, 2.1); ax.grid(alpha=.3); ax.tick_params(labelsize=6)
            q = next((x for x in stat if x["station"] == i), None)
            ax.set_title(f"st {i}" + (f": {100*q['completeness']:.0f} % complete\ngap {q['largest_gap_deg']:.0f} deg" if q else ": no wall"), fontsize=8.5)
            if c == 0: ax.set_ylabel(lab, fontsize=10)
        ax = axs[ri][len(sts)]
        ax.plot([q["arclength_R"] for q in stat], [100 * q["completeness"] for q in stat], "o-", ms=3, color="tab:green")
        ax.set_ylim(0, 100); ax.set_xlabel("arclength (R)"); ax.set_ylabel("% of sectors known"); ax.grid(alpha=.3); ax.set_title("completeness along the path", fontsize=9)
    fig.suptitle("ring completeness: green = wall known (colour = supported observations), red x = unknown sector", fontweight="bold")
    fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=115); print("figure:", a.out)
