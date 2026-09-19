"""What the real compensation actually estimated: the deformation field over frames and stations, with the event window
marked. A field that is non-zero only inside the event (and near zero before and after) is evidence the search found the
wall event; a field that is non-zero everywhere is fitting the images.
Usage: python m2/real_deform_field.py runs/m2_26V2_disp --event 1908 1921 --out docs/figures/x.png"""
import argparse, os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--event", type=int, nargs=2, required=True); ap.add_argument("--out", default=None)
a = ap.parse_args(); V = np.load(f"{a.run}/velocity.npz", allow_pickle=True); vs = V["vsmooth"]; vstar = V["vstar"]; fr = V["frames"]; s_st = V["s_st"]; R = float(V["R"]); mode = str(V["mode"]) if "mode" in V.files else "velocity"
unit = "R inward" if mode == "displacement" else "R/s inward"
ev = (fr >= a.event[0]) & (fr <= a.event[1]); out = ~ev
dec = np.isfinite(vstar)
def stat(m):
    v = vs[m]; v = v[np.isfinite(v)]
    return (np.nan, np.nan, 0) if not len(v) else (float(np.median(np.abs(v))), float(np.percentile(np.abs(v), 90)), len(v))
me, pe, ne = stat(ev); mo, po, no = stat(out)
res = dict(run=a.run, mode=mode, decided_fraction=float(dec.mean()), event_median_abs=me, event_p90_abs=pe, event_cells=ne, outside_median_abs=mo, outside_p90_abs=po, outside_cells=no,
           inward_fraction_event=float(np.mean(vs[ev][np.isfinite(vs[ev])] > 0)) if ne else None, inward_fraction_outside=float(np.mean(vs[out][np.isfinite(vs[out])] > 0)) if no else None)
json.dump(res, open(f"{a.run}/deform_field.json", "w"), indent=1)
print(f"[{os.path.basename(a.run)}] mode {mode}; decided {100*dec.mean():.0f} % of (frame, station) cells")
print(f"  inside the event f{a.event[0]}-{a.event[1]}: |value| median {me:.2f} p90 {pe:.2f} {unit} ({ne} cells, {100*(res['inward_fraction_event'] or 0):.0f} % inward)")
print(f"  outside:                     |value| median {mo:.2f} p90 {po:.2f} {unit} ({no} cells, {100*(res['inward_fraction_outside'] or 0):.0f} % inward)")
if a.out:
    fig, axs = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
    lim = np.nanpercentile(np.abs(vs), 98) if np.isfinite(vs).any() else 1.0
    m = axs[0].imshow(vs.T, aspect="auto", origin="lower", extent=[fr.min() - .5, fr.max() + .5, 0, len(s_st)], cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    axs[0].set_ylabel("station index"); axs[0].set_title(f"estimated deformation field ({unit}); red = wall folded inward", fontsize=11); plt.colorbar(m, ax=axs[0], fraction=0.03, label=unit)
    with np.errstate(all="ignore"): prof = np.nanmedian(vs, axis=1); n_dec = np.isfinite(vstar).sum(1)
    axs[1].plot(fr, prof, "k.-", ms=3, lw=1, label="median over stations")
    axs[1].fill_between(fr, np.nanpercentile(vs, 25, axis=1), np.nanpercentile(vs, 75, axis=1), color="0.7", alpha=0.6, label="interquartile over stations")
    axs[1].axhline(0, color="k", lw=0.8); axs[1].set_xlabel("frame"); axs[1].set_ylabel(unit); axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)
    ax2 = axs[1].twinx(); ax2.plot(fr, n_dec, color="tab:blue", lw=0.8, alpha=0.5); ax2.set_ylabel("stations decided", color="tab:blue", fontsize=8); ax2.tick_params(labelsize=7, colors="tab:blue")
    for ax in axs: ax.axvspan(a.event[0] - .5, a.event[1] + .5, color="orange", alpha=0.18)
    fig.suptitle(f"{os.path.basename(a.run)}: what the compensation estimated (event shaded)", fontweight="bold"); fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=115); print("figure:", a.out)
