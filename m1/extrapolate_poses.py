"""Carry camera poses through frames a rigid model could not register, on a constant-velocity motion prior.

The centre velocity and the angular velocity are fitted over the last --fit registered frames before the gap (least
squares on the centres; mean relative rotation per frame), and every frame from the last reliable one to --hi gets an
extrapolated pose. Frames after --last-good in the input model are dropped (seam frames of a rigid model are the
unreliable ones). Optionally only frames >= --lo are kept, to limit the stereo to the event neighbourhood.
Usage: python m1/extrapolate_poses.py <workspace> <out_workspace> --last-good 1901 --hi 1921 [--lo 1800] [--fit 16]"""
import os, argparse, subprocess, tempfile, shutil, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("workspace"); ap.add_argument("out"); ap.add_argument("--model", default="sparse/0"); ap.add_argument("--last-good", type=int, required=True); ap.add_argument("--hi", type=int, required=True)
ap.add_argument("--lo", type=int, default=0); ap.add_argument("--fit", type=int, default=16); ap.add_argument("--speed-factor", type=float, default=1.0, help="scale the fitted centre velocity (and rotation rate) by this factor for the extrapolated frames"); ap.add_argument("--pattern", default="f{:05d}.png"); a = ap.parse_args(); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap")
def sh(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError(cmd[0] + "\n" + r.stderr[-1500:])
def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
def R2q(R):
    t = np.trace(R)
    if t > 0: s = np.sqrt(t + 1) * 2; return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3; s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2; q = np.zeros(4); q[0] = (R[k, j] - R[j, k]) / s; q[1 + i] = 0.25 * s; q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s; return q
def logR(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-9: return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
def expR(w):
    th = np.linalg.norm(w)
    if th < 1e-12: return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]); return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
td = tempfile.mkdtemp(); sh(["model_converter", "--input_path", f"{a.workspace}/{a.model}", "--output_path", td, "--output_type", "TXT"])
cams = [l for l in open(f"{td}/cameras.txt") if l.strip() and not l.startswith("#")]; L = [l for l in open(f"{td}/images.txt") if l.strip() and not l.startswith("#")]
imgs = {}
for hdr, pts in zip(L[0::2], L[1::2]):
    f = hdr.split(); k = int(os.path.splitext(os.path.basename(f[9]))[0].lstrip("f")); imgs[k] = (hdr, pts, np.array(list(map(float, f[1:5]))), np.array(list(map(float, f[5:8]))), int(f[8]))
keep = {k: v for k, v in imgs.items() if a.lo <= k <= a.last_good}; fitk = sorted(k for k in keep if k > a.last_good - a.fit)
C = {k: -q2R(keep[k][2]).T @ keep[k][3] for k in fitk}; Rw = {k: q2R(keep[k][2]).T for k in fitk}          # c2w rotation
A_ = np.vstack([np.array(fitk, float), np.ones(len(fitk))]).T; vel = np.linalg.lstsq(A_, np.array([C[k] for k in fitk]), rcond=None)[0][0]
wrel = np.mean([logR(Rw[fitk[i + 1]] @ Rw[fitk[i]].T) / (fitk[i + 1] - fitk[i]) for i in range(len(fitk) - 1)], axis=0)
vel = vel * a.speed_factor; wrel = wrel * a.speed_factor
k0 = a.last_good; C0 = C[k0]; R0 = Rw[k0]; print(f"fit on f{fitk[0]}-f{fitk[-1]}: centre speed {np.linalg.norm(vel):.4f} units/frame, rotation rate {np.degrees(np.linalg.norm(wrel)):.3f} deg/frame; extrapolating f{k0 + 1}-f{a.hi}")
newid = max(int(v[0].split()[0]) for v in imgs.values()) + 1; extra = {}
for k in range(k0 + 1, a.hi + 1):
    Ck = C0 + vel * (k - k0); Rk = expR(wrel * (k - k0)) @ R0; Rwc = Rk.T; t = -Rwc @ Ck; extra[k] = (newid, R2q(Rwc), t); newid += 1
# points: keep only tracks through kept images (dropped seam frames would leave dangling references)
kept_ids = {int(v[0].split()[0]) for v in keep.values()}; P_out = []; valid_pts = set()
for l in open(f"{td}/points3D.txt"):
    if l.startswith("#") or not l.strip(): continue
    f = l.split(); tr = f[8:]; tr2 = [x for im, k in zip(tr[0::2], tr[1::2]) if int(im) in kept_ids for x in (im, k)]
    if len(tr2) >= 4: P_out.append(" ".join(f[:8] + tr2) + "\n"); valid_pts.add(int(f[0]))
with open(f"{td}/points3D.txt", "w") as f: f.writelines(P_out)
def clean_pts(line):   # drop 2D observations of points that were removed
    v = line.split(); out = []
    for x, y, pid in zip(v[0::3], v[1::3], v[2::3]): out += [x, y, pid if (int(pid) < 0 or int(pid) in valid_pts) else "-1"]
    return " ".join(out) + "\n"
with open(f"{td}/images.txt", "w") as f:
    for k in sorted(keep): f.write(keep[k][0] + clean_pts(keep[k][1]))
    for k in sorted(extra):
        i, q, t = extra[k]; f.write(f"{i} {' '.join(map(repr, q))} {' '.join(map(repr, t))} {keep[k0][4]} {a.pattern.format(k)}\n0 0 -1\n")
for extra_f in ("frames.txt", "rigs.txt"):                 # COLMAP >= 3.10 rig/frame tables list every original image; let the converter rebuild them
    if os.path.exists(f"{td}/{extra_f}"): os.remove(f"{td}/{extra_f}")
os.makedirs(f"{a.out}/sparse/0", exist_ok=True); sh(["model_converter", "--input_path", td, "--output_path", f"{a.out}/sparse/0", "--output_type", "BIN"]); shutil.rmtree(td)
print(f"{a.out}/sparse/0: {len(keep)} registered frames kept (f{min(keep)}-f{max(keep)}) + {len(extra)} extrapolated (f{min(extra)}-f{max(extra)})")
