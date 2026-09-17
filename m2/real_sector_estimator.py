"""Prior-based displacement on a real clip, from the saved per-frame grid (m1_real_grid.npz), instead of the free-space area.

The synthetic estimator that survived M1/M2 was a median displacement over a known posterior sector. On a real clip the
sector is unknown, so: (1) reject lost-wall sectors (outward deviation > 0.3 R, or |dev| > 1.5 R); (2) for each frame
take the median deviation over every 120-degree arc (12 of 36 sectors, >= 6 finite); (3) the moving arc of a station
is the arc with the largest inward excursion of its 3-frame-median-smoothed series (p5); the arc opposite to it is
the reference and should be still if the motion is a membrane and not a pose swing. Checks reported per station:
opposite-arc range, correlation of the moving-arc displacement with the dark-lumen fraction (real inward motion
shrinks the dark hole: positive correlation expected), correlation with the camera's distance to the station (a wall does
not know where the camera is; |corr| > 0.5 is a range-dependent depth bias), and coherence with the neighbouring station.

Usage: python m2/real_sector_estimator.py runs/m1_real_26V2_m0 [--images dir | --stats dark_lumen.npz] [--min-frames 60]"""
import os, json, argparse, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--images", default=None); ap.add_argument("--stats", default=None); ap.add_argument("--min-frames", type=int, default=60); ap.add_argument("--arc", type=int, default=12)
a = ap.parse_args()
G = np.load(f"{a.run}/m1_real_grid.npz"); fr = G["frames"]; dev = G["dev"].copy(); R = float(G["R"]); s_st = G["s_st"]; N, nS, nb = dev.shape; tb = (np.arange(nb) + 0.5) / nb * 360 - 180
dev[(dev > 0.3) | (np.abs(dev) > 1.5)] = np.nan                     # lost-wall sectors
dark = {}
if a.stats:
    S_ = np.load(a.stats); dark = {int(k): float(d) for k, d in zip(S_["frames"], S_["dark"])}
elif a.images:
    for k in fr:
        p = f"{a.images}/f{int(k):05d}.png"
        if os.path.exists(p):
            g = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY).astype(float); m = g > 8
            if m.sum() > 1000: v = g[m]; dark[int(k)] = float((v < 0.25 * np.percentile(v, 95)).mean())
d_series = np.array([dark.get(int(f), np.nan) for f in fr])
def smooth3(y):
    out = np.full_like(y, np.nan)
    for k in range(len(y)):
        w = y[max(0, k - 1):k + 2]; w = w[np.isfinite(w)]
        if len(w): out[k] = np.median(w)
    return out
A = a.arc; arcs = np.array([[(b + q) % nb for q in range(A)] for b in range(nb)])   # (nb, A)
rows = []; d_move = np.full((N, nS), np.nan); arc_of = np.full(nS, -1)
for i in range(nS):
    D = dev[:, i, :]; ok_fr = np.isfinite(D).sum(1) >= 6
    if ok_fr.sum() < a.min_frames: continue
    med = np.full((N, nb), np.nan)                                    # arc medians per frame and arc start
    for b in range(nb):
        sub = D[:, arcs[b]]; n = np.isfinite(sub).sum(1); m = n >= 6
        if m.any(): med[m, b] = np.nanmedian(sub[m], axis=1)
    sm = np.column_stack([smooth3(med[:, b]) for b in range(nb)])
    p5 = np.nanpercentile(sm, 5, axis=0); p5[np.isfinite(sm).sum(0) < a.min_frames] = np.nan
    if not np.isfinite(p5).any(): continue
    b = int(np.nanargmin(p5)); bo = (b + nb // 2) % nb; arc_of[i] = b; d_move[:, i] = sm[:, b]
    y, yo = sm[:, b], sm[:, bo]; okd = np.isfinite(y) & np.isfinite(d_series)
    dist = np.linalg.norm(G["C"] - G["S"][i], axis=1) / R; okc = np.isfinite(y) & np.isfinite(dist)   # camera distance to the station, in R
    rows.append(dict(station=i, arclength_R=float(s_st[i] / R), n_frames=int(np.isfinite(y).sum()), arc_start_deg=float(tb[b]), arc_centre_deg=float((tb[b] + A / 2 * 360 / nb + 180) % 360 - 180),
                     moving_arc_p5_R=float(np.nanpercentile(y, 5)), moving_arc_p95_R=float(np.nanpercentile(y, 95)), moving_arc_range_R=float(np.nanpercentile(y, 95) - np.nanpercentile(y, 5)),
                     opposite_arc_range_R=float(np.nanpercentile(yo, 95) - np.nanpercentile(yo, 5)) if np.isfinite(yo).sum() >= 20 else None,
                     corr_d_darkfraction=float(np.corrcoef(y[okd], d_series[okd])[0, 1]) if okd.sum() >= 30 else None,
                     corr_d_cameradistance=float(np.corrcoef(y[okc], dist[okc])[0, 1]) if okc.sum() >= 30 and np.std(dist[okc]) > 1e-6 else None))
# coherence with the neighbouring station: correlation of the two moving-arc series over common frames
for r in rows:
    i = r["station"]; nb_ = [q for q in (i - 1, i + 1) if 0 <= q < nS and arc_of[q] >= 0]
    cs = []
    for q in nb_:
        ok = np.isfinite(d_move[:, i]) & np.isfinite(d_move[:, q])
        if ok.sum() >= 30: cs.append(float(np.corrcoef(d_move[ok, i], d_move[ok, q])[0, 1]))
    r["neighbour_coherence"] = float(np.mean(cs)) if cs else None
    same = [abs(((tb[arc_of[q]] - tb[arc_of[i]] + 180) % 360) - 180) <= 40 for q in nb_]; r["neighbour_same_arc"] = bool(all(same)) if same else None
    ok_ref = r["opposite_arc_range_R"] is not None and r["opposite_arc_range_R"] < 0.5 * max(r["moving_arc_range_R"], 1e-6)
    img = r["corr_d_darkfraction"]; ok_img = img is None or img > -0.2
    cam = r["corr_d_cameradistance"]; ok_cam = cam is None or abs(cam) < 0.5            # a real wall does not know where the camera is
    r["verdict"] = ("sectoral, reference still, image-consistent, camera-independent" if ok_ref and ok_img and ok_cam and r["moving_arc_p5_R"] < -0.15 else
                    "no motion beyond noise" if r["moving_arc_p5_R"] >= -0.15 else "reference arc moves too (pose/scale swing)" if not ok_ref else
                    "against the image" if not ok_img else "follows the camera distance (range-dependent depth bias)")
out = dict(run=a.run, arc_deg=A * 360 / nb, stations=rows); json.dump(out, open(f"{a.run}/sector_estimator.json", "w"), indent=1)
np.savez_compressed(f"{a.run}/sector_estimator.npz", frames=fr, d_move=d_move, arc_of=arc_of, s_st=s_st, R=R)
rows_s = sorted(rows, key=lambda r: -r["n_frames"])
for r in rows_s[:14]:
    print(f"station {r['station']:3d} ({r['arclength_R']:.1f} R, n={r['n_frames']:3d}): arc centre {r['arc_centre_deg']:+4.0f} deg; moving p5/p95 {r['moving_arc_p5_R']:+.2f}/{r['moving_arc_p95_R']:+.2f} R (range {r['moving_arc_range_R']:.2f}); opposite range {r['opposite_arc_range_R'] if r['opposite_arc_range_R'] is None else round(r['opposite_arc_range_R'], 2)}; corr(d, dark) {r['corr_d_darkfraction'] if r['corr_d_darkfraction'] is None else round(r['corr_d_darkfraction'], 2)}; corr(d, cam dist) {r['corr_d_cameradistance'] if r['corr_d_cameradistance'] is None else round(r['corr_d_cameradistance'], 2)}; neighbour coh {r['neighbour_coherence'] if r['neighbour_coherence'] is None else round(r['neighbour_coherence'], 2)} same arc {r['neighbour_same_arc']} -> {r['verdict']}")
import collections; print("verdicts:", dict(collections.Counter(r["verdict"] for r in rows)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 1, figsize=(14, 7.5)); picks = [r["station"] for r in rows_s[:5]]
for i in picks: axs[0].plot(fr, d_move[:, i], ".-", ms=3, lw=.8, label=f"station {s_st[i] / R:.1f} R, arc {rows[[r['station'] for r in rows].index(i)]['arc_centre_deg']:+.0f} deg")
axs[0].axhline(0, color="k", lw=1); axs[0].set_ylabel("moving-arc displacement (R), inward < 0"); axs[0].set_xlabel("frame"); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3); axs[0].set_title("prior-based sector estimator: median deviation over the most inward-moving 120-degree arc (lost-wall sectors rejected)")
st = [r["arclength_R"] for r in rows]; axs[1].bar(st, [r["moving_arc_range_R"] for r in rows], width=0.15, label="moving arc, p5-p95 range"); axs[1].bar([s + 0.15 for s in st], [r["opposite_arc_range_R"] or 0 for r in rows], width=0.15, label="opposite arc (reference)")
axs[1].set_xlabel("station arclength (R)"); axs[1].set_ylabel("range of displacement (R)"); axs[1].legend(fontsize=8); axs[1].grid(alpha=.3); axs[1].set_title("a membrane moves its own arc and leaves the opposite arc still; a pose swing moves both")
fig.suptitle(f"sector estimator — {os.path.basename(a.run)}", fontweight="bold"); fig.tight_layout(); fig.savefig(f"{a.run}/sector_estimator.png", dpi=110); print("figure:", f"{a.run}/sector_estimator.png")
