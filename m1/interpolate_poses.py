"""Fill camera poses for frames a rigid model dropped, writing a new COLMAP model.

Rigid SfM refuses frames where too much of the wall is moving (the malacia peaks). A streaming system would carry
those frames on its motion prior; here the pose of a dropped frame is interpolated between its nearest registered
neighbours (linear in the camera centre, slerp in rotation) so that M1 can estimate depth and deformation for every
frame. Registered frames keep their poses unchanged. Sparse points are copied.

Usage: python m1/interpolate_poses.py <workspace> <out_workspace> [--n-frames 180] [--pattern f{:05d}.png]
Env:   BRONCHO_COLMAP (default colmap)"""
import sys, os, argparse, subprocess, tempfile, shutil
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("workspace"); ap.add_argument("out"); ap.add_argument("--n-frames", type=int, default=180); ap.add_argument("--pattern", default="f{:05d}.png"); ap.add_argument("--lo", type=int, default=0, help="first frame index (default 0; frames lo..lo+n_frames-1)")
a = ap.parse_args(); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap")


def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R2q(R):
    t = np.trace(R)
    if t > 0: s = np.sqrt(t + 1) * 2; return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3; s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2; q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s; q[i + 1] = 0.25 * s; q[j + 1] = (R[j, i] + R[i, j]) / s; q[k + 1] = (R[k, i] + R[i, k]) / s; return q


def slerp(q0, q1, u):
    d = float(np.dot(q0, q1)); q1 = -q1 if d < 0 else q1; d = abs(d)
    if d > 0.9995: q = q0 + u * (q1 - q0); return q / np.linalg.norm(q)
    th = np.arccos(d); return (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)


with tempfile.TemporaryDirectory() as td:
    subprocess.run([COLMAP, "model_converter", "--input_path", f"{a.workspace}/sparse/0", "--output_path", td, "--output_type", "TXT"], check=True, capture_output=True)
    cam_lines = open(f"{td}/cameras.txt").read(); pts_lines = open(f"{td}/points3D.txt").read()
    L = [l for l in open(f"{td}/images.txt") if l.strip() and not l.startswith("#")]
    imgs = {}
    for hdr, pts in zip(L[0::2], L[1::2]):
        f = hdr.split(); imgs[f[9]] = dict(id=int(f[0]), q=np.array(list(map(float, f[1:5]))), t=np.array(list(map(float, f[5:8]))), cam=int(f[8]), pts=pts.rstrip("\n"))
names = [a.pattern.format(k) for k in range(a.lo, a.lo + a.n_frames)]; have = [k for k, n in enumerate(names) if n in imgs]; missing = [k for k, n in enumerate(names) if n not in imgs]   # k = 0-based index into names
print(f"{len(have)} registered, {len(missing)} missing of {a.n_frames}")
# c2w for registered frames
C = {}; Q = {}
for k in have:
    im = imgs[names[k]]; R = q2R(im["q"]); C[k] = -R.T @ im["t"]; Q[k] = R2q(R.T)          # world-from-camera rotation as quaternion
cam_id = imgs[names[have[0]]]["cam"]; next_id = max(im["id"] for im in imgs.values()) + 1; have_arr = np.array(have)
new = {}
for k in missing:
    prev = have_arr[have_arr < k]; nxt = have_arr[have_arr > k]
    if len(prev) and len(nxt):
        a_, b_ = prev.max(), nxt.min(); u = (k - a_) / (b_ - a_); c = (1 - u) * C[a_] + u * C[b_]; q = slerp(Q[a_], Q[b_], u)
    elif len(prev) >= 2:                                                       # extrapolate at the end with the last velocity
        a_, b_ = prev[-2], prev[-1]; c = C[b_] + (C[b_] - C[a_]) * (k - b_) / (b_ - a_); q = Q[b_]
    else:
        a_, b_ = nxt[0], nxt[1]; c = C[a_] - (C[b_] - C[a_]) * (a_ - k) / (b_ - a_); q = Q[a_]
    Rc2w = q2R(q / np.linalg.norm(q)); Rw2c = Rc2w.T; tvec = -Rw2c @ c; new[k] = (R2q(Rw2c), tvec)
os.makedirs(f"{a.out}/sparse/0", exist_ok=True); td = f"{a.out}/sparse/txt"; os.makedirs(td, exist_ok=True)
if True:
    open(f"{td}/cameras.txt", "w").write(cam_lines); open(f"{td}/points3D.txt", "w").write(pts_lines)
    with open(f"{td}/images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for k in range(a.n_frames):
            n = names[k]
            if n in imgs:
                im = imgs[n]; f.write(f"{im['id']} " + " ".join(f"{v:.10g}" for v in im['q']) + " " + " ".join(f"{v:.10g}" for v in im['t']) + f" {im['cam']} {n}\n{im['pts']}\n")
            else:
                q, tv = new[k]; f.write(f"{next_id} " + " ".join(f"{v:.10g}" for v in q) + " " + " ".join(f"{v:.10g}" for v in tv) + f" {cam_id} {n}\n0 0 -1\n"); next_id += 1        # one dummy 2D point: COLMAP's text reader skips blank lines
    r = subprocess.run([COLMAP, "model_converter", "--input_path", td, "--output_path", f"{a.out}/sparse/0", "--output_type", "BIN"], capture_output=True, text=True)
    if r.returncode != 0: print(r.stderr[-800:]); sys.exit(1)
    if r.stdout.strip() or r.stderr.strip(): print("converter:", (r.stdout + r.stderr).strip()[-600:])
for extra in ("run.log",):
    if os.path.exists(f"{a.workspace}/{extra}"): shutil.copy(f"{a.workspace}/{extra}", f"{a.out}/{extra}")
print(f"wrote {a.out}/sparse/0 with {a.n_frames} images ({len(missing)} interpolated: {missing[:6]}{'...' if len(missing) > 6 else ''})")
