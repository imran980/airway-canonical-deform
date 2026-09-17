"""Bridge two COLMAP models built from ONE database across a gap of unregistered frames, using their common images.

A 3D point of model A and a 3D point of model B that are observed by the same feature (image, point2D index) of a common
image are the same wall point; a RANSAC Sim(3) (Umeyama) on those correspondences maps B into A's frame. B's cameras and
points are transformed and appended (common images keep A's pose). Use when colmap model_merger refuses (too few
common images) and image_registrator finds no 2D-3D matches across the gap.

Usage: python m1/bridge_models.py <modelA> <modelB> <out_model> [--max-err-rel 0.02]"""
import sys, os, argparse, subprocess, tempfile, shutil, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("out"); ap.add_argument("--iters", type=int, default=2000); ap.add_argument("--inlier-rel", type=float, default=0.03, help="inlier threshold as a fraction of the model's point-cloud extent")
a = ap.parse_args(); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap")
def sh(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0: raise RuntimeError(cmd[0] + "\n" + r.stderr[-1500:])
def read(model):
    td = tempfile.mkdtemp(); sh(["model_converter", "--input_path", model, "--output_path", td, "--output_type", "TXT"])
    cams = [l for l in open(f"{td}/cameras.txt") if not l.startswith("#") and l.strip()]
    L = [l for l in open(f"{td}/images.txt") if l.strip() and not l.startswith("#")]; imgs = {}
    for hdr, pts in zip(L[0::2], L[1::2]):
        f = hdr.split(); q = np.array(list(map(float, f[1:5]))); t = np.array(list(map(float, f[5:8]))); cam_id = int(f[8]); name = f[9]
        p = np.array(pts.split(), dtype=float).reshape(-1, 3) if pts.strip() else np.zeros((0, 3))
        imgs[name] = dict(id=int(f[0]), q=q, t=t, cam=cam_id, pts=p)
    P = {}
    for l in open(f"{td}/points3D.txt"):
        if l.startswith("#") or not l.strip(): continue
        f = l.split(); P[int(f[0])] = dict(xyz=np.array(list(map(float, f[1:4]))), rgb=f[4:7], err=f[7], track=f[8:])
    shutil.rmtree(td); return cams, imgs, P
def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
def R2q(R):
    t = np.trace(R)
    if t > 0: s = np.sqrt(t + 1) * 2; return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3; s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2; q = np.zeros(4); q[0] = (R[k, j] - R[j, k]) / s; q[1 + i] = 0.25 * s; q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s; return q
camsA, A, PA = read(a.a); camsB, B, PB = read(a.b); common = sorted(set(A) & set(B)); print(f"A: {len(A)} images, {len(PA)} points; B: {len(B)} images, {len(PB)} points; common images: {common}")
X, Y = [], []
for n in common:
    pa, pb = A[n]["pts"], B[n]["pts"]; m = min(len(pa), len(pb))
    for k in range(m):
        ia, ib = int(pa[k, 2]), int(pb[k, 2])
        if ia >= 0 and ib >= 0 and ia in PA and ib in PB: X.append(PB[ib]["xyz"]); Y.append(PA[ia]["xyz"])
X = np.array(X); Y = np.array(Y); print(f"point correspondences through common features: {len(X)}")
if len(X) < 10: print("too few correspondences"); sys.exit(1)
def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; U, sv, Vt = np.linalg.svd(Yc.T @ Xc / len(X)); d = np.ones(3); d[-1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag(d) @ Vt; s = (sv * d).sum() / (Xc ** 2).sum(1).mean(); t = my - s * R @ mx; return s, R, t
ext = np.linalg.norm(Y.max(0) - Y.min(0)); thr = a.inlier_rel * ext; rng = np.random.default_rng(0); best = (0, None)
for _ in range(a.iters):
    idx = rng.choice(len(X), 4, replace=False); s, R, t = umeyama(X[idx], Y[idx]); res = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1); inl = res < thr
    if inl.sum() > best[0]: best = (inl.sum(), inl)
inl = best[1]; s, R, t = umeyama(X[inl], Y[inl]); res = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1)
print(f"Sim(3): scale {s:.4f}, inliers {inl.sum()}/{len(X)} (threshold {thr:.4f} = {100*a.inlier_rel:.0f}% of extent), inlier residual median {np.median(res[inl]):.4f}, all median {np.median(res):.4f}")
# camera consistency check on the common images: B's pose mapped into A vs A's own pose
for n in common:
    Ra, Ca = q2R(A[n]["q"]), -q2R(A[n]["q"]).T @ A[n]["t"]; Rb, Cb = q2R(B[n]["q"]), -q2R(B[n]["q"]).T @ B[n]["t"]; Cb2 = s * R @ Cb + t; Rb2 = Rb @ R.T
    ang = np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb2.T) - 1) / 2, -1, 1))); print(f"  {n}: camera centre mismatch {np.linalg.norm(Cb2 - Ca):.4f} ({100*np.linalg.norm(Cb2 - Ca)/ext:.1f}% of extent), rotation {ang:.2f} deg")
# write merged model: A + transformed B (B's common images dropped; B's point ids offset)
td = tempfile.mkdtemp(); off = max(PA) + 1; idoff = max(v["id"] for v in A.values()) + 1
with open(f"{td}/cameras.txt", "w") as f: f.writelines(camsA)
with open(f"{td}/images.txt", "w") as f:
    for n, v in A.items(): f.write(f"{v['id']} {' '.join(map(repr, v['q']))} {' '.join(map(repr, v['t']))} {v['cam']} {n}\n" + " ".join(f"{x} {y} {int(p)}" for x, y, p in v["pts"]) + "\n")
    for n, v in B.items():
        if n in A: continue
        Rb = q2R(v["q"]); Cb = -Rb.T @ v["t"]; Rn = Rb @ R.T; Cn = s * R @ Cb + t; tn = -Rn @ Cn; pts = v["pts"].copy(); pts[:, 2] = np.where(pts[:, 2] >= 0, pts[:, 2] + off, -1)
        f.write(f"{v['id'] + idoff} {' '.join(map(repr, R2q(Rn)))} {' '.join(map(repr, tn))} {v['cam']} {n}\n" + " ".join(f"{x} {y} {int(p)}" for x, y, p in pts) + "\n")
with open(f"{td}/points3D.txt", "w") as f:
    for i, p in PA.items(): f.write(f"{i} {' '.join(map(repr, p['xyz']))} {' '.join(p['rgb'])} {p['err']} {' '.join(p['track'])}\n")
    for i, p in PB.items():
        xyz = s * R @ p["xyz"] + t; tr = p["track"]; tr2 = []
        for im, k in zip(tr[0::2], tr[1::2]):
            name = next((n for n, v in B.items() if v["id"] == int(im)), None)
            if name is None or name in A: continue
            tr2 += [str(int(im) + idoff), k]
        f.write(f"{i + off} {' '.join(map(repr, xyz))} {' '.join(p['rgb'])} {p['err']} {' '.join(tr2)}\n")
os.makedirs(a.out, exist_ok=True); sh(["model_converter", "--input_path", td, "--output_path", a.out, "--output_type", "BIN"]); shutil.rmtree(td)
fr = sorted(int(n.split("/")[-1][1:6]) for n in set(A) | set(B)); miss = [k for k in range(fr[0], fr[-1] + 1) if k not in set(fr)]
print(f"merged model -> {a.out}: {len(fr)} images f{fr[0]}-{fr[-1]}, missing {len(miss)}: {miss}")
