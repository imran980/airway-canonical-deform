"""Why is the canonical wall jagged in one arc: a bad reconstruction there, or a wall that moves?

A time-median canonical mixes the open and folded states of a moving wall, and per-frame stereo also loses the wall more
often while it moves, so real motion can make the canonical jagged in exactly the moving arc. That looks identical to a
badly reconstructed sector unless the two are told apart. Four diagnostics, each computed inside the suspect arc and
outside it:
  1. temporal continuity - lag-1 autocorrelation of each sector's radius series. A wall that moves is smooth in time;
     noise is not.
  2. phase mixing - the canonical rebuilt from the OPEN phase (a high percentile of each sector's radius) instead of
     the median. If mixing caused the jaggedness, the open-phase canonical is smooth.
  3. bimodality - the dip statistic of each sector's radius distribution (open vs folded is two-humped, noise is one).
  4. wall loss - how often the sector has no measurement at all, and whether it is lost more while folded.
Usage: python m2/canonical_diagnosis.py runs/m1_real_20V1_b3_g0 --arc -136 --width 120 --stations 21 22 23 24 25
"""
import argparse, os, json, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--arc", type=float, required=True, help="centre of the suspect arc, degrees")
ap.add_argument("--width", type=float, default=120.0); ap.add_argument("--stations", type=int, nargs="*", default=None); ap.add_argument("--open-pct", type=float, default=75.0)
ap.add_argument("--min-obs", type=int, default=20); ap.add_argument("--label", default=None)
a = ap.parse_args()
G = np.load(f"{a.run}/m1_real_grid.npz"); r = G["r_grid"].copy(); R = float(G["R"]); s_st = G["s_st"]; rc = G["r_can"]
N, nS, nb = r.shape; tb = (np.arange(nb) + 0.5) / nb * 360 - 180
d = np.abs(np.angle(np.exp(1j * (np.radians(tb) - np.radians(a.arc))))) <= np.radians(a.width / 2)
sts = a.stations if a.stations else [int(i) for i in np.argsort(-np.isfinite(r).any(2).sum(0))[:5]]
def rough(profile):
    p = profile / R; return np.abs(p - 0.5 * (np.roll(p, 1) + np.roll(p, -1)))
def dip(x):
    """a crude two-humpedness score: the gap between the two halves of the distribution relative to their spread"""
    if len(x) < 12: return np.nan
    q = np.sort(x); lo = q[:len(q) // 2]; hi = q[len(q) // 2:]
    sp = np.std(x)
    return float((np.median(hi) - np.median(lo)) / sp) if sp > 1e-9 else np.nan
print(f"[{a.label or os.path.basename(a.run)}] suspect arc {a.arc:+.0f} deg +-{a.width/2:.0f}; stations {sts}")
print(f"{'st':>3s} {'arcR':>5s} | {'rough med':>9s} {'rough open':>10s} | {'autocorr':>8s} {'autocorr out':>12s} | {'bimodality':>10s} {'out':>5s} | {'lost %':>6s} {'out':>5s} | {'range R':>8s} {'out':>6s}")
rows = []
for i in sts:
    ri = r[:, i, :]
    med = np.nanmedian(ri, axis=0); opn = np.nanpercentile(ri, a.open_pct, axis=0)
    ok = np.isfinite(ri).sum(0) >= a.min_obs
    rmed = rough(np.where(ok, med, np.nan)); ropn = rough(np.where(ok, opn, np.nan))
    ac, bm, lost, rng = [], [], [], []
    for b in range(nb):
        x = ri[:, b]; f = np.isfinite(x)
        lost.append(1 - f.mean())
        if f.sum() < a.min_obs: ac.append(np.nan); bm.append(np.nan); rng.append(np.nan); continue
        idx = np.where(f)[0]; adj = idx[:-1][np.diff(idx) == 1]
        ac.append(float(np.corrcoef(x[adj], x[adj + 1])[0, 1]) if len(adj) >= 10 and np.std(x[adj]) > 1e-9 else np.nan)
        bm.append(dip(x[f] / R)); rng.append(float(np.percentile(x[f], 95) - np.percentile(x[f], 5)) / R)
    ac = np.array(ac); bm = np.array(bm); lost = np.array(lost); rng = np.array(rng)
    g = lambda v, m: float(np.nanmedian(v[m & ok])) if (m & ok).any() else np.nan
    row = dict(station=i, arclength_R=float(s_st[i] / R), rough_median_in=g(rmed, d), rough_open_in=g(ropn, d), rough_median_out=g(rmed, ~d), rough_open_out=g(ropn, ~d),
               autocorr_in=g(ac, d), autocorr_out=g(ac, ~d), bimodality_in=g(bm, d), bimodality_out=g(bm, ~d), lost_in=g(lost, d), lost_out=g(lost, ~d), range_in=g(rng, d), range_out=g(rng, ~d))
    rows.append(row)
    print(f"{i:3d} {row['arclength_R']:5.1f} | {row['rough_median_in']:9.3f} {row['rough_open_in']:10.3f} | {row['autocorr_in']:8.2f} {row['autocorr_out']:12.2f} | {row['bimodality_in']:10.2f} {row['bimodality_out']:5.2f} | {100*row['lost_in']:6.0f} {100*row['lost_out']:5.0f} | {row['range_in']:8.2f} {row['range_out']:6.2f}")
m = lambda k: float(np.nanmedian([r_[k] for r_ in rows]))
print(f"summary: inside the arc the canonical is {m('rough_median_in')/max(m('rough_median_out'),1e-9):.1f}x rougher than outside;")
print(f"         rebuilt from the open phase ({a.open_pct:.0f}th percentile) it is {m('rough_open_in')/max(m('rough_median_in'),1e-9):.2f}x the median-canonical roughness ({m('rough_open_in'):.3f} vs {m('rough_median_in'):.3f} R)")
print(f"         temporal continuity inside {m('autocorr_in'):+.2f} vs outside {m('autocorr_out'):+.2f}; bimodality {m('bimodality_in'):.2f} vs {m('bimodality_out'):.2f}; wall lost {100*m('lost_in'):.0f} % vs {100*m('lost_out'):.0f} %; radius range {m('range_in'):.2f} vs {m('range_out'):.2f} R")
json.dump(dict(run=a.run, arc=a.arc, width=a.width, stations=rows), open(f"{a.run}/canonical_diagnosis.json", "w"), indent=1)
