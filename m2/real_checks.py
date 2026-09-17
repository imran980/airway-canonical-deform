"""Three checks that a per-frame lumen change on a real clip is wall motion and not an artefact.

1. Sector check. Between the widest and the narrowest frames at a station, which wall sectors moved? A membranous
   collapse or malacia is confined to a contiguous sector; a change common to the whole circumference is either
   a circumferential malacia or, far more often, a per-frame pose / scale / illumination artefact.
2. Image check. The dark-lumen fraction of a frame (pixels much darker than the frame's own bright wall) does not
   depend on any pose or depth. Real narrowing shrinks the dark hole and brightens the image (the scope tip is
   closer to the walls); an estimate that is anti-correlated with the dark fraction, or positively correlated with
   brightness, is following illumination and camera distance, not the wall.
3. Magnitude check. A wall folds inward by up to a radius but cannot move outward by more than a fraction of it; an
   outward excursion > 0.5 R in the wide frames, or > 5 % of frames dilated by > 50 %, is a lost wall (free-space fill
   leaking through holes in the depth map), not motion.

Usage: python m2/real_checks.py runs/m1_real_20V1 --images runs/real_20V1/images  [--video <mp4> | --stats dark_lumen.npz]  [--stations 29,30]
(frame statistics come from --stats if given, else are decoded sequentially from --video, else read from --images as f%05d.png)"""
import sys, os, json, argparse, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--images", default=None); ap.add_argument("--video", default=None); ap.add_argument("--stations", default=None); ap.add_argument("--min-frames", type=int, default=60); ap.add_argument("--stats", default=None, help="dark_lumen.npz (frames, dark, bright) saved by an earlier run; use for the normalised runs so the image check sees the ORIGINAL frames")
a = ap.parse_args()
G = np.load(f"{a.run}/m1_real_grid.npz"); fr = G["frames"]; dev = G["dev"]; R = float(G["R"]); s_st = G["s_st"]; ratio = G["csa_fill"] / G["csa_can"][None]; nb = dev.shape[2]; tb = (np.arange(nb) + 0.5) / nb * 360 - 180
# frame-level image statistics
dark, bright = {}, {}
def stats(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float); m = g > 8
    if m.sum() < 1000: return None
    v = g[m]; return float((v < 0.25 * np.percentile(v, 95)).mean()), float(np.percentile(v, 50))
if a.stats:
    S_ = np.load(a.stats)
    for k, dk, bk in zip(S_["frames"], S_["dark"], S_["bright"]): dark[int(k)], bright[int(k)] = float(dk), float(bk)
elif a.video:
    cap = cv2.VideoCapture(a.video); fi = 0; lo, hi = int(fr.min()), int(fr.max())
    while True:
        ok, f = cap.read()
        if not ok or fi > hi: break
        if fi >= lo:
            st = stats(f)
            if st: dark[fi], bright[fi] = st
        fi += 1
    cap.release()
elif a.images:
    for k in fr:
        p = f"{a.images}/f{int(k):05d}.png"
        if os.path.exists(p):
            st = stats(cv2.imread(p))
            if st: dark[int(k)], bright[int(k)] = st
d_series = np.array([dark.get(int(f), np.nan) for f in fr]); b_series = np.array([bright.get(int(f), np.nan) for f in fr])
# stations: given, or the best observed
if a.stations: sts = [int(x) for x in a.stations.split(",")]
else:
    n_obs = np.isfinite(ratio).sum(0); sts = [int(i) for i in np.argsort(-n_obs)[:4] if n_obs[i] >= a.min_frames]
out = dict(run=a.run, stations=[])
for i in sts:
    ok = np.isfinite(ratio[:, i]); k = np.where(ok)[0]
    if len(k) < a.min_frames: continue
    r_ = ratio[k, i]; hi_ = k[r_ > np.percentile(r_, 80)]; lo_ = k[r_ < np.percentile(r_, 20)]
    diff = np.nanmean(dev[hi_, i, :], 0) - np.nanmean(dev[lo_, i, :], 0); fin = np.isfinite(diff); moving = fin & (diff > 0.08)
    m2 = np.concatenate([moving, moving]); best = cur = 0
    for v in m2: cur = cur + 1 if v else 0; best = max(best, cur)
    rec = dict(station=i, arclength_R=float(s_st[i] / R), n_frames=int(len(k)), ratio_p5_p95=[float(np.percentile(r_, 5)), float(np.percentile(r_, 95))],
               sector_dev_median_R=float(np.nanmedian(diff)), sector_dev_max_R=float(np.nanmax(diff)), sector_dev_max_at_deg=float(tb[int(np.nanargmax(diff))]),
               sectors_moving=int(moving.sum()), sectors_valid=int(fin.sum()), longest_moving_arc_deg=float(min(best, nb) * 360 / nb), sectors_still=int((fin & (np.abs(diff) < 0.03)).sum()))
    okd = ok & np.isfinite(d_series)
    if okd.sum() >= 30: rec.update(corr_csa_darkfraction=float(np.corrcoef(ratio[okd, i], d_series[okd])[0, 1]), corr_csa_brightness=float(np.corrcoef(ratio[okd, i], b_series[okd])[0, 1]))
    # 3. magnitude check: a wall can fold inward by more than a radius (collapse) but cannot move OUTWARD by more than a
    #    fraction of the radius (the posterior membrane bulges, the cartilage holds); an outward excursion > 0.5 R between the
    #    narrow and the wide frames, or many frames with > 50 % dilation, means the wall was LOST in the wide frames
    #    (free-space fill leaking through holes / depth outliers), not that it moved
    rec["frac_frames_ratio_gt_1p5"] = float((r_ > 1.5).mean()); rec["frac_frames_ratio_lt_0p5"] = float((r_ < 0.5).mean())
    implausible = rec["sector_dev_max_R"] > 0.5 or rec["frac_frames_ratio_gt_1p5"] > 0.05
    common_mode = rec["longest_moving_arc_deg"] >= 300 and rec["sectors_still"] == 0
    image_conflict = rec.get("corr_csa_darkfraction", 0) < -0.3 or rec.get("corr_csa_brightness", 0) > 0.3
    failed = [n for n, f in (("common-mode (whole circumference moves together)", common_mode), ("against the image (dark-lumen / brightness correlation)", image_conflict),
                             ("implausible magnitude (wall lost in the wide frames: outward > 0.5 R or > 5 % of frames dilated > 50 %)", implausible)) if f]
    rec["failed_checks"] = failed; rec["verdict"] = "consistent with sectoral wall motion" if not failed else "not wall motion: " + "; ".join(failed)
    out["stations"].append(rec)
    print(f"station {i} ({rec['arclength_R']:.1f} R, n={rec['n_frames']}): ratio p5-p95 {rec['ratio_p5_p95'][0]:.2f}-{rec['ratio_p5_p95'][1]:.2f}; moving arc {rec['longest_moving_arc_deg']:.0f} deg, still sectors {rec['sectors_still']}, max dev {rec['sector_dev_max_R']:+.2f} R at {rec['sector_dev_max_at_deg']:+.0f} deg; corr(CSA, dark) {rec.get('corr_csa_darkfraction', float('nan')):+.2f}, corr(CSA, bright) {rec.get('corr_csa_brightness', float('nan')):+.2f}; frames >1.5x {100*rec['frac_frames_ratio_gt_1p5']:.0f}% -> {rec['verdict']}")
json.dump(out, open(f"{a.run}/real_checks.json", "w"), indent=1)
