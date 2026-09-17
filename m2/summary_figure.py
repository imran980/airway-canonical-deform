"""Night-2 summary figure: membrane displacement error per scenario and method, from the saved eval JSONs.
Usage: python m2/summary_figure.py [--out docs/figures/m2_night2_summary.png]"""
import json, os, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/m2_night2_summary.png"); a = ap.parse_args()
def load(run):
    p = f"runs/{run}/m1_result.json"
    if not os.path.exists(p): return None
    r = json.load(open(p)); return dict(med=r["membrane_d_smoothed_err_mm_median"], p90=r["membrane_d_smoothed_err_mm_p90"], cells=r["cells"], cart=r["cartilage_radius_dev_mm_median"], csa90=r["csa_model_smoothed_rel_err_p90_abs"])
METHODS = [("M1: COLMAP per-frame stereo", "m1_{s}", "#7a7a7a"), ("M2 sweep, no compensation", "m2_{s}_none_eval", "#b58a00"),
           ("M2 + velocity search v2", "m2_{s}_vs2_eval", "#1f6fb2"), ("M2 + true motion (ceiling)", "m2_{s}_oracle_eval", "#2a9d5c")]
SCEN = [("static", "static"), ("breathing", "breathing 13 %"), ("collapse", "collapse 76 %"), ("malacia", "malacia 47 %"), ("uniform", "uniform 30 %\n(prior violated)")]
fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw=dict(width_ratios=[3.2, 1.6]))
ax = axs[0]; W = 0.19; x = np.arange(len(SCEN)); YMAX = 3.6
for j, (name, pat, col) in enumerate(METHODS):
    for i, (s, lab) in enumerate(SCEN):
        r = load(pat.format(s=s))
        if r is None: continue
        xx = x[i] + (j - 1.5) * W
        ax.bar(xx, r["med"], W * 0.9, color=col, label=name, zorder=3)
        if r["p90"] <= YMAX: ax.plot([xx], [r["p90"]], marker="_", ms=14, mew=2.2, color=col, zorder=4)
        else: ax.annotate(f'p90 {r["p90"]:.1f}', (xx, YMAX - 0.05), ha="center", va="top", fontsize=7.5, color=col, rotation=90)
        if r["cart"] > 0.3: ax.text(xx, r["med"] + 0.08, "flagged\nby rigidity\ncheck", ha="center", va="bottom", fontsize=7, color="#444")
h, l = ax.get_legend_handles_labels(); seen = {}; [seen.setdefault(ll, hh) for hh, ll in zip(h, l)]
ax.legend(seen.values(), seen.keys(), fontsize=8.5, loc="upper left", frameon=False, title="bar = median, tick = p90", title_fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in SCEN], fontsize=9.5); ax.set_ylabel("membrane displacement error (mm)"); ax.set_ylim(0, YMAX)
ax.axhline(0.25, color="#999", lw=0.8, ls=":", zorder=2); ax.text(-0.45, 0.29, "0.25 mm", fontsize=8, color="#666", ha="left")
ax.set_title("synthetic deforming trachea (R = 5 mm): recovering the posterior-wall motion", fontsize=11, loc="left"); ax.grid(axis="y", alpha=0.25, zorder=0); ax.spines[["top", "right"]].set_visible(False)
# camera-speed panel: 6 mm/s reference vs the clean 3 / 12 mm/s pair (event 16 mm ahead at t0 for every speed); the old 12 mm/s video (event behind the camera) is not shown
ax = axs[1]; speeds = [("slow", 3), ("30", 6), ("fast2", 12)]
for j, (name, pat, col) in enumerate(METHODS):
    xs, ys, p9 = [], [], []
    for tag, v in speeds:
        s = "collapse" if tag == "30" else f"collapse_{tag}"; r = load(pat.format(s=s))
        if r is None: continue
        xs.append(v); ys.append(r["med"]); p9.append(r["p90"])
    if xs:
        ax.plot(xs, ys, "o-", color=col, lw=2, ms=6, label=name, zorder=3); ax.plot(xs, p9, "_", color=col, ms=12, mew=2, zorder=3)
ax.set_xscale("log"); ax.set_xticks([3, 6, 12]); ax.set_xticklabels(["3", "6", "12"]); ax.set_xlabel("camera speed (mm/s)"); ax.set_ylabel("membrane displacement error (mm)")
ax.set_title("collapse: same event, camera speed varied", fontsize=11, loc="left"); ax.grid(alpha=0.25, zorder=0); ax.spines[["top", "right"]].set_visible(False); ax.set_ylim(0, None)
ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2); ax.set_ylim(0, 2.0)
fig.suptitle("M2 (motion-compensated stereo) after night 2: per-frame posterior-wall displacement, all scenarios", fontweight="bold", fontsize=12)
fig.tight_layout(rect=(0, 0.06, 1, 1)); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=120); print("figure:", a.out)
