"""Event report for a real clip whose dropped frames were carried (interpolated) inside one rigid model, with the canonical
wall taken from a reference frame range (the post-event pullback): is the carried pose valid, and did the wall move?

Per station observed before, during and after the event: area ratio (median pre / min during / median post), the most
inward-moving 120-degree arc during the event against the canonical and its opposite arc (a membrane folds on one
side and leaves the other still; a pose error shifts the whole circumference), and the whole-circumference median
deviation in the carried frames against the registered frames next to them (the pose-validity measure). Also the
dark-lumen fraction of the frames (pose-free). Figure: CSA(t) and arc deviations across the event with the carried
frames shaded.
Usage: python m2/real_event_report.py runs/m1_real_26V2d --event 1908 1921 --carried 1908 1920 --images runs/real_26V2d/images"""
import os, json, argparse, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--event", type=int, nargs=2, required=True); ap.add_argument("--carried", type=int, nargs=2, required=True); ap.add_argument("--images", default=None); ap.add_argument("--pre", type=int, default=40); ap.add_argument("--post", type=int, default=40)
a = ap.parse_args()
G = np.load(f"{a.run}/m1_real_grid.npz"); fr = G["frames"]; dev = G["dev"]; r = G["r_grid"]; R = float(G["R"]); C = G["C"]; S = G["S"]; s_st = G["s_st"]; ratio = G["csa_fill"] / G["csa_can"][None]; nb = dev.shape[2]; tb = (np.arange(nb) + 0.5) / nb * 360 - 180
ev = (fr >= a.event[0]) & (fr <= a.event[1]); car = (fr >= a.carried[0]) & (fr <= a.carried[1]); pre = (fr < a.event[0]) & (fr >= a.event[0] - a.pre); post = (fr > a.event[1]) & (fr <= a.event[1] + a.post)
dark = {}
if a.images:
    for k in fr:
        p = f"{a.images}/f{int(k):05d}.png"
        if os.path.exists(p):
            g = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY).astype(float); m = g > 8
            if m.sum() > 1000: v = g[m]; dark[int(k)] = float((v < 0.25 * np.percentile(v, 95)).mean())
d_series = np.array([dark.get(int(f), np.nan) for f in fr])
def arcs(diff):
    return np.array([np.nanmean(np.roll(diff, -b)[:12]) if np.isfinite(np.roll(diff, -b)[:12]).sum() >= 6 else np.nan for b in range(nb)])
def wmed(D): 
    with np.errstate(all="ignore"): return np.nanmedian(D, axis=-1)
rows = []
print(f"frames f{fr.min()}-f{fr.max()}, R = {R:.3f} units; event f{a.event[0]}-{a.event[1]}, carried f{a.carried[0]}-{a.carried[1]}; dark-lumen fraction pre {np.nanmean(d_series[pre]):.3f} / event {np.nanmean(d_series[ev]):.3f} / post {np.nanmean(d_series[post]):.3f}")
for i in range(dev.shape[1]):
    ok = np.isfinite(dev[:, i]).sum(1) >= 6; n_pre, n_ev, n_post = ok[pre].sum(), ok[ev].sum(), ok[post].sum()
    if n_ev < 3 or n_post < 5: continue
    with np.errstate(all="ignore"): d_ev = np.nanmedian(dev[ev, i], 0)
    fin = np.isfinite(d_ev)
    if fin.sum() < 12: continue
    A = arcs(d_ev); b = int(np.nanargmin(A)); bo = (b + nb // 2) % nb
    # pose validity: whole-circumference median deviation of carried frames vs the registered frames within 6 frames of the carried range
    near = ((fr >= a.carried[0] - 6) & (fr < a.carried[0])) | ((fr > a.carried[1]) & (fr <= a.carried[1] + 6)); near &= ~car
    wc_car = wmed(dev[car & ok, i]); wc_near = wmed(dev[near & ok, i])
    rec = dict(station=i, arclength_R=float(s_st[i] / R), n_pre=int(n_pre), n_event=int(n_ev), n_post=int(n_post), cam_dist_event_R=float(np.linalg.norm(C[ev] - S[i], axis=1).mean() / R),
               ratio_pre=float(np.nanmedian(ratio[pre, i])) if np.isfinite(ratio[pre, i]).any() else None, ratio_event_min=float(np.nanmin(ratio[ev, i])) if np.isfinite(ratio[ev, i]).any() else None, ratio_event_median=float(np.nanmedian(ratio[ev, i])) if np.isfinite(ratio[ev, i]).any() else None, ratio_post=float(np.nanmedian(ratio[post, i])) if np.isfinite(ratio[post, i]).any() else None,
               inward_arc_centre_deg=float((tb[b] + 60 + 180) % 360 - 180), inward_arc_R=float(A[b]), opposite_arc_R=float(A[bo]) if np.isfinite(A[bo]) else None, sectors_still=int((fin & (np.abs(d_ev) < 0.05)).sum()), sectors_valid=int(fin.sum()),
               carried_whole_circ_median_R=float(np.nanmedian(wc_car)) if np.isfinite(wc_car).any() else None, neighbours_whole_circ_median_R=float(np.nanmedian(wc_near)) if np.isfinite(wc_near).any() else None)
    rows.append(rec)
    print(f"station {i:2d} ({rec['arclength_R']:.1f} R, cam {rec['cam_dist_event_R']:.1f} R away): frames pre/event/post {n_pre}/{n_ev}/{n_post}; area ratio pre {rec['ratio_pre'] if rec['ratio_pre'] is None else round(rec['ratio_pre'],2)} -> event min {rec['ratio_event_min'] if rec['ratio_event_min'] is None else round(rec['ratio_event_min'],2)} (median {rec['ratio_event_median'] if rec['ratio_event_median'] is None else round(rec['ratio_event_median'],2)}) -> post {rec['ratio_post'] if rec['ratio_post'] is None else round(rec['ratio_post'],2)}; inward arc {rec['inward_arc_centre_deg']:+.0f} deg {rec['inward_arc_R']:+.2f} R, opposite {rec['opposite_arc_R'] if rec['opposite_arc_R'] is None else round(rec['opposite_arc_R'],2)} R, still {rec['sectors_still']}/{rec['sectors_valid']}; carried frames whole-circ {rec['carried_whole_circ_median_R'] if rec['carried_whole_circ_median_R'] is None else round(rec['carried_whole_circ_median_R'],2)} vs registered neighbours {rec['neighbours_whole_circ_median_R'] if rec['neighbours_whole_circ_median_R'] is None else round(rec['neighbours_whole_circ_median_R'],2)} R")
json.dump(dict(run=a.run, event=a.event, carried=a.carried, dark_pre=float(np.nanmean(d_series[pre])), dark_event=float(np.nanmean(d_series[ev])), dark_post=float(np.nanmean(d_series[post])), stations=rows), open(f"{a.run}/event_report.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
picks = sorted(rows, key=lambda q: -q["n_event"])[:4]; fig, axs = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
for q in picks:
    i = q["station"]; axs[0].plot(fr, ratio[:, i], ".-", ms=3, lw=.8, label=f"station {q['arclength_R']:.1f} R")
    with np.errstate(all="ignore"):
        b = int(np.nanargmin(arcs(np.nanmedian(dev[ev, i], 0)))); arcsel = [(b + k) % nb for k in range(12)]; opp = [(b + nb // 2 + k) % nb for k in range(12)]
        axs[1].plot(fr, np.nanmedian(dev[:, i, arcsel], 1), ".-", ms=3, lw=.8, label=f"station {q['arclength_R']:.1f} R, inward arc {q['inward_arc_centre_deg']:+.0f} deg")
        axs[1].plot(fr, np.nanmedian(dev[:, i, opp], 1), ":", lw=1.2, color=axs[1].lines[-1].get_color(), label=f"  opposite arc")
axs[0].axhline(1, color="k", lw=1); axs[0].set_ylabel("area / canonical (post-event)"); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3); axs[0].set_title("per-frame lumen area against the post-event canonical wall")
axs[1].axhline(0, color="k", lw=1); axs[1].set_ylabel("arc deviation (R), inward < 0"); axs[1].legend(fontsize=7, ncol=2); axs[1].grid(alpha=.3); axs[1].set_title("most inward-moving 120-degree arc during the event, and the arc opposite it")
axs[2].plot(fr, d_series, "k.-", ms=3, lw=.8); axs[2].set_ylabel("dark-lumen fraction (pose-free)"); axs[2].set_xlabel("frame"); axs[2].grid(alpha=.3)
for ax in axs: ax.axvspan(a.carried[0] - 0.5, a.carried[1] + 0.5, color="orange", alpha=0.15); ax.axvspan(a.event[0] - 0.5, a.event[1] + 0.5, color="red", alpha=0.08)
fig.suptitle(f"{os.path.basename(a.run)}: collapse f{a.event[0]}-{a.event[1]} (red), carried poses f{a.carried[0]}-{a.carried[1]} (orange)", fontweight="bold"); fig.tight_layout(); fig.savefig(f"{a.run}/event_report.png", dpi=110); print("figure:", f"{a.run}/event_report.png")
