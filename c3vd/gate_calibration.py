"""Calibrate the per-sector support gate against ground truth.

The per-sector support gate decides which parts of a wall ring may be believed. Its incidence term is the one that
matters most on real clips (a sector seen edge-on), but the threshold has so far been a guess. On C3VD the true wall is
known, so the threshold can be measured: run the identical station analysis on the estimated depth maps and on the
ground-truth depth maps, then plot the radial error of each sector against how square-on the camera saw it, how many
points it had, and how tightly they agreed.

Step 1 writes the ground-truth depth maps in COLMAP's format next to a copy of the estimated run's model and images.
Step 2 is left to m1/windowed_depth_real.py --skip-stereo on both. Step 3 (this script, --compare) does the comparison.

Usage: python c3vd/gate_calibration.py runs/c3vd_cecum --run st_gate --scale 3.3215 --write-gt
       python c3vd/gate_calibration.py runs/c3vd_cecum --est runs/c3vd_cecum/gate_est --gt runs/c3vd_cecum/gate_gt --compare --out docs/figures/x.png
"""
import argparse, os, glob, json, shutil, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("prep"); ap.add_argument("--run", default="st_gate"); ap.add_argument("--scale", type=float, default=None)
ap.add_argument("--write-gt", action="store_true"); ap.add_argument("--compare", action="store_true"); ap.add_argument("--est", default=None); ap.add_argument("--gt", default=None); ap.add_argument("--out", default=None)
a = ap.parse_args()
def write_depth(path, d):
    with open(path, "wb") as f: f.write(f"{d.shape[1]}&{d.shape[0]}&1&".encode()); d.astype(np.float32).tofile(f)
def load_gt(base):
    if os.path.exists(base + ".npz"): return np.load(base + ".npz")["d"].astype(np.float32) / 100.0
    return np.load(base + ".npy")
if a.write_gt:
    src = f"{a.prep}/{a.run}"; scale = a.scale or json.load(open(f"{src}/summary.json")).get("scale_mm_per_unit", 1.0)
    for tag in ("gate_gt",):
        out = f"{a.prep}/{tag}"; os.makedirs(f"{out}/dense/stereo/depth_maps", exist_ok=True)
        for sub in ("sparse", "images"):
            if os.path.islink(f"{out}/dense/{sub}") or os.path.exists(f"{out}/dense/{sub}"): continue
            os.symlink(os.path.abspath(f"{src}/dense/{sub}"), f"{out}/dense/{sub}")
        n = 0
        for p in sorted(glob.glob(f"{src}/dense/stereo/depth_maps/*.geometric.bin")):
            nm = os.path.basename(p).replace(".geometric.bin", "")
            est = open(p, "rb").read(64)
            hdr = est[:est.find(b"&", est.find(b"&", est.find(b"&") + 1) + 1) + 1].decode(); w, h, _ = map(int, hdr.split("&")[:3])
            g = load_gt(f"{a.prep}/gt_depth/{nm[:-4]}")
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST) / scale                 # ground truth in the model's own units
            write_depth(f"{out}/dense/stereo/depth_maps/{nm}.geometric.bin", g); n += 1
        print(f"wrote {n} ground-truth depth maps -> {out}/dense/stereo/depth_maps (scale {scale:.4f} mm/unit)")
if a.compare:
    E = np.load(f"{a.est}/m1_real_grid.npz"); T = np.load(f"{a.gt}/m1_real_grid.npz")
    re_, rt = E["r_grid"], T["r_grid"]; R = float(E["R"]); cos = E["cos_grid"]; n = E["n_grid"]; mad = E["mad_grid"]
    m = np.isfinite(re_) & np.isfinite(rt)
    err = np.abs(re_ - rt) / R
    print(f"sectors measured by both: {m.sum()}; radial error median {np.median(err[m]):.3f} R, p90 {np.percentile(err[m],90):.3f} R")
    print(f"\n{'incidence |cos|':>16s} {'sectors':>9s} {'median error':>13s} {'p75':>7s} {'p90':>7s} {'% within 0.05 R':>16s}")
    edges = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.70, 1.01]; rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = m & (cos >= lo) & (cos < hi)
        if k.sum() < 200: continue
        e = err[k]; rows.append((lo, hi, int(k.sum()), float(np.median(e)), float(np.percentile(e, 75)), float(np.percentile(e, 90)), float(np.mean(e <= 0.05))))
        print(f"{lo:6.2f}-{hi:<9.2f} {k.sum():9d} {np.median(e):13.3f} {np.percentile(e,75):7.3f} {np.percentile(e,90):7.3f} {100*np.mean(e<=0.05):15.0f}%")
    print(f"\n{'points in sector':>16s} {'sectors':>9s} {'median error':>13s} {'p90':>7s}")
    for lo, hi in ((3, 5), (5, 10), (10, 20), (20, 50), (50, 200), (200, 10 ** 9)):
        k = m & (n >= lo) & (n < hi)
        if k.sum() < 200: continue
        print(f"{lo:6d}-{hi if hi<10**9 else 0:<9d} {k.sum():9d} {np.median(err[k]):13.3f} {np.percentile(err[k],90):7.3f}")
    json.dump(dict(est=a.est, gt=a.gt, by_incidence=rows), open(f"{a.est}/gate_calibration.json", "w"), indent=1)
    if a.out and rows:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        x = [0.5 * (r[0] + r[1]) for r in rows]
        axs[0].plot(x, [r[3] for r in rows], "o-", label="median"); axs[0].plot(x, [r[4] for r in rows], "s--", label="75th percentile"); axs[0].plot(x, [r[5] for r in rows], "^:", label="90th percentile")
        axs[0].axhline(0.05, color="tab:red", lw=1, ls="--", label="0.05 R"); axs[0].set_xlabel("how square-on the sector was seen, |cos|"); axs[0].set_ylabel("radial error against truth (R)")
        axs[0].set_title("the incidence gate, measured against ground truth", fontsize=11); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3); axs[0].set_yscale("log")
        axs[1].plot(x, [100 * r[6] for r in rows], "o-", color="tab:green"); axs[1].set_xlabel("how square-on the sector was seen, |cos|"); axs[1].set_ylabel("% of sectors within 0.05 R of truth")
        axs[1].grid(alpha=.3); axs[1].set_title("what fraction can be believed", fontsize=11); axs[1].set_ylim(0, 100)
        fig.suptitle("C3VD: calibrating the per-sector support gate", fontweight="bold"); fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=115); print("figure:", a.out)
