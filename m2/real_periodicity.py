"""Respiratory periodicity in per-frame CSA(t) of a real clip (output of m1/windowed_depth_real.py).

For each station with enough frames, the CSA(t)/canonical series is detrended (median over 3 s), and its
autocorrelation and periodogram are computed; the dominant period is reported with its amplitude (half the
peak-to-trough of the smoothed cycle) and how many stations agree. A child's respiratory rate is 15-40 /min
(1.5-4 s period); a clip of 300+ frames at 30 fps holds 3-7 breaths. On 20-V1 (tracheobronchomalacia, CT insp/exp
45 % area change) the cycle should be visible if the per-frame method sees the wall move.

Usage: python m2/real_periodicity.py runs/m1_real_20V1 [--fps 30] [--out figure.png]"""
import sys, os, json, argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--fps", type=float, default=29.97); ap.add_argument("--min-frames", type=int, default=60); ap.add_argument("--out", default=None)
a = ap.parse_args()
G = np.load(f"{a.run}/m1_real_grid.npz"); fr = G["frames"]; ratio = G["csa_fill"] / G["csa_can"][None]; s_st = G["s_st"]; R = float(G["R"]); dev = G["dev"]
t = (fr - fr.min()) / a.fps; rows = []
for i in range(ratio.shape[1]):
    y = ratio[:, i]; ok = np.isfinite(y)
    if ok.sum() < a.min_frames: continue
    # interpolate gaps for the spectral analysis, detrend with a 3-s running median
    yi = np.interp(t, t[ok], y[ok]); w = int(3 * a.fps) | 1; pad = w // 2
    trend = np.array([np.median(yi[max(0, k - pad):k + pad + 1]) for k in range(len(yi))]); d = yi - trend
    d[~ok] = 0.0
    # periodogram over 1-6 s periods
    n = len(d); freqs = np.fft.rfftfreq(n, 1 / a.fps); Pw = np.abs(np.fft.rfft(d - d.mean())) ** 2
    band = (freqs >= 1 / 6.0) & (freqs <= 1 / 1.0)
    if band.sum() < 3: continue
    fpk = freqs[band][np.argmax(Pw[band])]; period = 1 / fpk; power_frac = float(Pw[band].max() / (Pw[1:].sum() + 1e-12))
    # autocorrelation at the peak period as an independent check
    lag = int(round(period * a.fps)); ac = float(np.corrcoef(d[:-lag], d[lag:])[0, 1]) if lag < n - 10 else np.nan
    # cycle amplitude: fold the series at the period and take half the peak-to-trough of the folded median
    ph = ((t * fpk) % 1.0); bins = np.floor(ph * 8).astype(int); fold = np.array([np.median(d[(bins == b) & ok]) if ((bins == b) & ok).sum() >= 3 else np.nan for b in range(8)])
    amp = float(0.5 * (np.nanmax(fold) - np.nanmin(fold))) if np.isfinite(fold).sum() >= 4 else np.nan
    rows.append(dict(station=int(i), arclength_R=float(s_st[i] / R), n_frames=int(ok.sum()), period_s=float(period), rate_per_min=float(60 * fpk), power_fraction=power_frac, autocorr_at_period=ac, amplitude_ratio=amp, csa_ratio_iqr=[float(np.nanpercentile(y, 25)), float(np.nanpercentile(y, 75))]))
rows.sort(key=lambda r: -r["n_frames"])
per = np.array([r["period_s"] for r in rows]); good = [r for r in rows if np.isfinite(r["autocorr_at_period"]) and r["autocorr_at_period"] > 0.3]
summary = dict(run=a.run, fps=a.fps, n_stations=len(rows), period_median_s=float(np.median(per)) if len(per) else None, period_iqr_s=[float(np.percentile(per, 25)), float(np.percentile(per, 75))] if len(per) else None,
               stations_with_autocorr_gt_0p3=len(good), their_periods_s=[round(r["period_s"], 2) for r in good][:12], their_amplitudes=[round(r["amplitude_ratio"], 3) for r in good if np.isfinite(r["amplitude_ratio"])][:12], stations=rows[:20])
print(json.dumps({k: v for k, v in summary.items() if k != "stations"}, indent=1))
for r in rows[:12]: print(f"  station {r['station']:3d} ({r['arclength_R']:.1f} R): n={r['n_frames']:3d} period {r['period_s']:.2f} s ({r['rate_per_min']:.0f}/min) power {r['power_fraction']:.2f} autocorr {r['autocorr_at_period']:+.2f} amp ±{r['amplitude_ratio']:.3f}")
json.dump(summary, open(f"{a.run}/periodicity.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 1, figsize=(14, 7)); picks = [r["station"] for r in (good if len(good) >= 3 else rows)[:4]]
for i in picks: axs[0].plot(fr, ratio[:, i], ".-", ms=3, lw=.8, label=f"station {s_st[i] / R:.1f} R")
axs[0].axhline(1, color="k", lw=1); axs[0].set_xlabel("frame"); axs[0].set_ylabel("CSA(t) / canonical"); axs[0].set_title("per-frame lumen area at the best-observed stations"); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
for i in picks:
    y = ratio[:, i]; ok = np.isfinite(y)
    if ok.sum() < a.min_frames: continue
    yi = np.interp(t, t[ok], y[ok]); n = len(yi); freqs = np.fft.rfftfreq(n, 1 / a.fps); Pw = np.abs(np.fft.rfft(yi - yi.mean())) ** 2; band = (freqs > 1 / 8) & (freqs < 1.5)
    axs[1].plot(60 * freqs[band], Pw[band] / Pw[band].max(), label=f"station {s_st[i] / R:.1f} R")
axs[1].set_xlabel("cycles per minute"); axs[1].set_ylabel("normalised power"); axs[1].set_title("periodogram (respiratory band 7-90 /min)"); axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)
fig.suptitle(f"respiratory periodicity — {os.path.basename(a.run)}", fontweight="bold"); fig.tight_layout(); fig.savefig(a.out or f"{a.run}/periodicity.png", dpi=110); print("figure:", a.out or f"{a.run}/periodicity.png")
