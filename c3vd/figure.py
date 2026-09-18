"""Figure: the range bias of per-frame stereo against C3VD truth, by distance and by local texture, for the airway setting
(+-2 frames), COLMAP poses, and far sources (3-5 frames). Usage: python c3vd/figure.py [--out docs/figures/c3vd_range_bias.png]"""
import argparse, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/c3vd_range_bias.png"); a = ap.parse_args()
RUNS = [("st_raw800", "ground-truth poses, sources ±1–2 (airway setting)", "#b58a00"), ("st_sfm800", "COLMAP poses, sources ±1–2", "#7a7a7a"), ("st_w5", "ground-truth poses, sources 1–5", "#1f6fb2"), ("st_w5min3", "ground-truth poses, sources 3–5 only", "#2a9d5c")]
def load(r):
    p = f"runs/c3vd_cecum/{r}/samples.npz"
    return np.load(p) if os.path.exists(p) else None
def binmed(x, y, edges):
    xs, ys, lo_, hi_ = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= 300: xs.append(0.5 * (lo + hi)); ys.append(np.median(y[m])); q = np.percentile(y[m], [25, 75]); lo_.append(q[0]); hi_.append(q[1])
    return np.array(xs), np.array(ys), np.array(lo_), np.array(hi_)
fig, axs = plt.subplots(1, 2, figsize=(14, 5.4))
for r, lab, col in RUNS:
    A = load(r)
    if A is None: continue
    x, y, lo, hi = binmed(A["depth"], 100 * A["rel"], [5, 12, 16, 20, 24, 28, 32, 36, 40, 46, 54, 65, 90]); axs[0].plot(x, y, "o-", color=col, lw=2, ms=5, label=lab); axs[0].fill_between(x, lo, hi, color=col, alpha=0.10)
    if "texture" in A.files:
        x, y, lo, hi = binmed(A["texture"], 100 * A["rel"], [0, 1, 2, 3, 4, 6, 9, 13, 20, 40, 300]); axs[1].plot(x, y, "o-", color=col, lw=2, ms=5, label=lab)
axs[0].axhline(0, color="k", lw=1); axs[0].set_xlabel("ground-truth depth (mm)"); axs[0].set_ylabel("relative depth error, median (%)  (shaded: interquartile)"); axs[0].set_title("per-frame stereo against truth: bias grows with distance in the airway setting", fontsize=10.5, loc="left"); axs[0].grid(alpha=.3); axs[0].legend(fontsize=8.5, frameon=False); axs[0].set_ylim(-20, 70)
axs[1].axhline(0, color="k", lw=1); axs[1].set_xscale("log"); axs[1].set_xlabel("local texture: 7×7 std of grey level"); axs[1].set_ylabel("relative depth error, median (%)"); axs[1].set_title("the bias lives in textureless pixels (zero-disparity attractor)", fontsize=10.5, loc="left"); axs[1].grid(alpha=.3); axs[1].legend(fontsize=8.5, frameon=False)
fig.suptitle("C3VD cecum_t1_a (real colonoscope, CT-registered phantom), 800 px COLMAP patch-match: what the airway pipeline's stereo does on real optics", fontweight="bold", fontsize=11); fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=120); print("figure:", a.out)
