"""Direction-resolved, pose-free check of a sectoral wall fold.

A fold at one angle of the airway must show in the image as the lumen boundary moving inward IN THAT DIRECTION, and not
on the opposite side. For every frame and every folding station this projects the canonical wall points of the arc into
the image, takes the image direction they fall in, and measures how far the dark lumen extends along that direction
(the radius at which the image crosses from lumen-dark to wall-bright). The same is measured in the opposite direction
as a control. The estimated sectoral fold is then correlated with both.

Real inward motion: the fold and the lumen boundary in its own direction shrink together (POSITIVE correlation), while
the opposite direction is unaffected. A measurement artefact has no reason to respect direction.

Usage: python m2/image_sector_check.py <grid_run> --images <dense_dir_with_images_and_sparse> --arc -140 --width 120 --stations 21 25 29
"""
import argparse, os, json, subprocess, tempfile, shutil, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--images", required=True, help="a dense workspace (images/ + sparse/) built from the same poses")
ap.add_argument("--arc", type=float, required=True); ap.add_argument("--width", type=float, default=120.0); ap.add_argument("--stations", type=int, nargs="*", default=None)
ap.add_argument("--canonical-pct", type=float, default=75.0); ap.add_argument("--ignore-support", action="store_true", help="use sectors that fail the per-sector support gate (they are excluded by default)"); ap.add_argument("--frames", type=int, nargs=2, default=None, help="restrict to this frame range (use the event window on a clip with a brief event)"); ap.add_argument("--dirs", type=int, default=12, help="directions sampled around the image for the null"); ap.add_argument("--min-frames", type=int, default=40); ap.add_argument("--label", default=None)
a = ap.parse_args(); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap")
def sh(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0: raise RuntimeError(cmd[0] + "\n" + r.stderr[-800:])
def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
td = tempfile.mkdtemp(); sh(["model_converter", "--input_path", f"{a.images}/sparse", "--output_path", td, "--output_type", "TXT"])
cam = [l.split() for l in open(f"{td}/cameras.txt") if l.strip() and not l.startswith("#")][0]; W, H = int(cam[2]), int(cam[3]); fx, fy, cx, cy = map(float, cam[4:8])
poses = {l.split()[9]: (np.array(list(map(float, l.split()[1:5]))), np.array(list(map(float, l.split()[5:8])))) for l in [l for l in open(f"{td}/images.txt") if l.strip() and not l.startswith("#")][0::2]}
shutil.rmtree(td)
G = np.load(f"{a.run}/m1_real_grid.npz"); r = G["r_grid"]; R = float(G["R"]); fr = G["frames"]; S = G["S"]; s_st = G["s_st"]; nb = r.shape[2]
can = np.nanpercentile(r, a.canonical_pct, axis=0); can[np.isfinite(r).sum(0) < 5] = np.nan
dev = (r - can[None]) / R; dev[(dev > 0.3) | (np.abs(dev) > 1.5)] = np.nan
if "support" in G.files and not getattr(a, "ignore_support", False):
    _sup = G["support"]; _before = np.isfinite(dev).sum(); dev = np.where(_sup, dev, np.nan)
    print(f"per-sector support gate: {int(np.isfinite(dev).sum())} of {int(_before)} measured cells kept ({100*np.isfinite(dev).sum()/max(_before,1):.0f} %)")
tb = (np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi; tdeg = np.degrees(tb)
T = np.gradient(S, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True); N1 = np.zeros_like(S); N2 = np.zeros_like(S)
n1 = np.cross(T[0], [0, 0, 1.0]); n1 = n1 if np.linalg.norm(n1) > 1e-3 else np.cross(T[0], [0, 1.0, 0]); n1 /= np.linalg.norm(n1)
for i in range(len(S)):
    n1 = n1 - (n1 @ T[i]) * T[i]; n1 /= np.linalg.norm(n1); N1[i] = n1; N2[i] = np.cross(T[i], n1)
ARC = np.abs(np.angle(np.exp(1j * (tb - np.radians(a.arc))))) <= np.radians(a.width / 2)
sts = a.stations if a.stations else [int(i) for i in np.argsort(-np.isfinite(r).any(2).sum(0))[:6]]
def lumen_radius(img, ang, cxx, cyy):
    """how far the dark lumen extends from the image centre along direction ang (pixels), by the first crossing of the
    midpoint between the frame's dark and bright levels"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32); m = g > 8
    if m.sum() < 1000: return np.nan
    v = g[m]; lo = np.percentile(v, 5); hi = np.percentile(v, 95); thr = 0.5 * (lo + hi)
    rmax = 0.45 * min(img.shape[0], img.shape[1]); rr = np.arange(3, rmax, 2.0)
    xs = (cxx + rr * np.cos(ang)).astype(int); ys = (cyy + rr * np.sin(ang)).astype(int)
    ok = (xs >= 0) & (xs < img.shape[1]) & (ys >= 0) & (ys < img.shape[0])
    if ok.sum() < 5: return np.nan
    prof = g[ys[ok], xs[ok]]; rr = rr[ok]
    above = np.where(prof > thr)[0]
    return float(rr[above[0]]) if len(above) else float(rr[-1])
rows = []
for i in sts:
    fold, lum, used = [], [], []
    for j, k in enumerate(fr):
        n = f"f{int(k):05d}.png"
        if n not in poses: continue
        if a.frames and not (a.frames[0] <= k <= a.frames[1]): continue
        d = dev[j, i]; ina = d[ARC]; ina = ina[np.isfinite(ina)]; out = d[~ARC]; out = out[np.isfinite(out)]
        if len(ina) < 4 or len(out) < 6: continue
        s = float(np.median(ina) - np.median(out))
        P = S[i] + np.nanmedian(can[i]) * (np.cos(np.radians(a.arc)) * N1[i] + np.sin(np.radians(a.arc)) * N2[i])
        Rw2c = q2R(poses[n][0]); Xc = Rw2c @ (P - (-Rw2c.T @ poses[n][1]))
        if Xc[2] <= 1e-6: continue
        u = fx * Xc[0] / Xc[2] + cx; v = fy * Xc[1] / Xc[2] + cy; ang = np.arctan2(v - cy, u - cx)
        img = cv2.imread(f"{a.images}/images/{n}")
        if img is None: continue
        Ls = [lumen_radius(img, ang + 2 * np.pi * q / a.dirs, cx, cy) for q in range(a.dirs)]
        if not np.isfinite(Ls[0]) or not np.all(np.isfinite(Ls)): continue
        fold.append(s); lum.append(Ls); used.append(int(k))
    if len(fold) < a.min_frames: print(f"station {i}: only {len(fold)} usable frames"); continue
    fold = np.array(fold); lum = np.array(lum)
    cs = [float(np.corrcoef(fold, lum[:, q])[0, 1]) if np.std(lum[:, q]) > 1e-9 else np.nan for q in range(a.dirs)]
    c_own = cs[0]; others = np.array(cs[1:]); rank = int(np.sum(others >= c_own)) + 1
    rows.append(dict(station=i, arclength_R=float(s_st[i] / R), n=len(fold), corr_own_direction=c_own, corr_opposite=float(cs[a.dirs // 2]), corr_other_median=float(np.nanmedian(others)), rank_of_own=rank, n_dirs=a.dirs, fold_p5=float(np.percentile(fold, 5))))
    print(f"station {i:3d} ({s_st[i]/R:.1f} R, n={len(fold)}): own direction {c_own:+.2f} | opposite {cs[a.dirs//2]:+.2f} | median of the other {a.dirs-1} directions {np.nanmedian(others):+.2f} | own ranks {rank} of {a.dirs}")
if rows:
    A = np.array([[q["corr_own_direction"], q["corr_opposite"], q["corr_other_median"], q["rank_of_own"]] for q in rows])
    import math
    p_bin = sum(math.comb(len(rows), t) * (1 / 3) ** t * (2 / 3) ** (len(rows) - t) for t in range(int((A[:, 3] <= max(1, a.dirs // 3)).sum()), len(rows) + 1))
    print(f"[{a.label or os.path.basename(a.run)}] median over {len(rows)} stations: own {np.median(A[:,0]):+.2f}, opposite {np.median(A[:,1]):+.2f}, other directions {np.median(A[:,2]):+.2f}; own beats the other directions by {np.median(A[:,0]-A[:,2]):+.2f}")
    print(f"[{a.label or os.path.basename(a.run)}] own direction lands in the top third at {int((A[:,3] <= max(1, a.dirs//3)).sum())} of {len(rows)} stations (by chance 1 in 3; p = {p_bin:.3f})")
    json.dump(dict(run=a.run, arc=a.arc, width=a.width, stations=rows), open(f"{a.run}/image_sector_check.json", "w"), indent=1)
