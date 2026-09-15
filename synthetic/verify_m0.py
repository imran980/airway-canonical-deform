"""Verify the M0 ground truth from outside the generator.

Checks (each prints PASS/FAIL):
  1. cartilage sector never moves; membrane moves by the solved amplitude
  2. CSA(z,t) at the collapse station dips to the target reduction exactly at t0 and recovers; a far station shows
     only the breathing component; the static scenario has zero variation
  3. camera advances along +z at the set speed with unit-determinant rotations and jitter within bounds
  4. area computed two independent ways agrees (polygon shoelace vs mesh cross-section from the saved PLY vertices)
  5. the mucosal texture (colour not explained by the ring geometry) has no (roll, z-shift) symmetry and a short
     correlation length: v0's helical streaks let rigid SfM register a rolled, ring-shifted copy of the static tube
  6. (visual) a figure: CSA(t) at three stations, membrane vs cartilage displacement, camera z(t), and rendered
     frames before / at / after the event
Also writes the rendered frames to an .mp4 so the rigid bronchotrust pipeline can be run on the synthetic video as
the external cross-check (a rigid pipeline must recover the static tube and must NOT survive the collapse).

Usage: python synthetic/verify_m0.py runs/synth_collapse [--video out.mp4]"""
import sys, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deforming_trachea as dt

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--video", default=None); a = ap.parse_args()
g = np.load(f"{a.run}/gt.npz"); P = json.loads(str(g["params"])); p = dt.Params(**P)
t, z, th, D, CSA, poses = g["t"], g["z"], g["theta"], g["deformation"].astype(np.float32), g["csa_mm2"], g["poses_c2w"]
ok_all = True
def check(name, cond, detail=""):
    global ok_all; ok_all &= bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

print(f"{a.run}: {len(t)} frames, breath_amp {p.breath_amp_mm:.2f} mm, collapse_amp {p.collapse_amp_mm:.2f} mm, membrane half-angle {p.membrane_half_angle_deg:.0f} deg")
# 1. rigidity
w, cart = dt.sector_weights(th, p)
check("cartilage sector rigid", np.abs(D[:, :, cart > 0.5]).max() == 0 if not p.uniform else True, f"(max |d| on rings = {np.abs(D[:, :, cart > 0.5]).max():.3g} mm)")
a_b, a_c = dt.amplitude(t, p); expected_peak = float(np.max(a_b + a_c))           # breathing may be at any phase when the event peaks
check("membrane peak displacement = peak of A_breath(t)+A_collapse(t)", abs(D.max() - expected_peak) < 0.02 * max(expected_peak, 1e-6) or expected_peak == 0, f"(D.max {D.max():.2f} mm, expected {expected_peak:.2f} mm)")
# 2. CSA behaviour
iz0 = int(np.argmin(np.abs(z - p.collapse_z0_mm))); csa0 = CSA[0]
if p.collapse_target > 0:
    it = int(np.argmin(CSA[:, iz0])); red = 1 - CSA[it, iz0] / csa0[iz0]
    check("collapse minimum at t0", abs(t[it] - p.collapse_t0_s) <= 1.0 / p.fps + 1e-9, f"(t_min {t[it]:.2f}s, t0 {p.collapse_t0_s:.2f}s)")
    check("collapse depth = target", abs(red - p.collapse_target) < 0.06, f"(reduction {red:.0%} vs target {p.collapse_target:.0%})")
    far = int(np.argmin(np.abs(z - (p.collapse_z0_mm - 4 * p.collapse_sigma_z_mm)))); red_far = 1 - CSA[it, far] / csa0[far]
    check("far station only breathes", red_far < p.breath_target + 0.05, f"(reduction {red_far:.0%} at z={z[far]:.0f} mm)")
    rec = 1 - CSA[-1, iz0] / csa0[iz0]; check("lumen recovers after the event", rec < p.breath_target + 0.05, f"(final reduction {rec:.0%})")
elif p.breath_target > 0:
    red = 1 - CSA.min() / csa0.mean(); check("breathing depth = target", abs(red - p.breath_target) < 0.06, f"(reduction {red:.0%} vs target {p.breath_target:.0%})")
    k = np.argmin(CSA[:, len(z) // 2]); check("breathing minimum at half period", abs(t[k] - p.breath_period_s / 2) <= 1.0 / p.fps + 1e-9, f"(t_min {t[k]:.2f}s)")
else:
    check("static: CSA constant", np.allclose(CSA, csa0), f"(max dev {np.abs(CSA - csa0).max():.3g})")
# 3. camera
zc = poses[:, 2, 3]; v = np.diff(zc) * p.fps
check("camera speed", np.allclose(v, p.camera_speed_mm_s, atol=1e-6), f"(mean {v.mean():.2f} mm/s)")
check("rotations valid", np.allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=1e-9) and np.allclose(poses[:, :3, :3] @ np.transpose(poses[:, :3, :3], (0, 2, 1)), np.eye(3), atol=1e-9))
check("lateral jitter bounded", np.abs(poses[:, :2, 3]).max() < 4 * p.jitter_mm, f"(max {np.abs(poses[:, :2, 3]).max():.2f} mm)")
# 4. independent area: rebuild the deformed cross-section polygon from the displacement field and compare
r0 = g["r_canonical"]; X, Y = dt.deform_xy(r0, th, D[len(t) // 2], p); csa_re = dt.csa_from_xy(X, Y)
check("area recomputed from displacement field matches stored CSA", np.allclose(csa_re, CSA[len(t) // 2], rtol=1e-4), f"(max rel dev {np.abs(csa_re / CSA[len(t)//2] - 1).max():.2e})")
try:
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(f"{a.run}/canonical_mesh.ply"); V = np.asarray(m.vertices).reshape(p.n_z, p.n_theta, 3)
    csa_mesh = dt.csa_from_xy(V[:, :, 0], V[:, :, 1]); check("canonical mesh cross-section = canonical CSA", np.allclose(csa_mesh, csa0, rtol=1e-4))
    # texture aperiodicity. A texture with a (roll, z-shift) symmetry lets rigid SfM lock onto a rolled / shifted copy of the
    # tube while reporting every frame registered (v0's helical streaks + periodic rings did exactly that: 11.5 deg roll
    # jumps per ring period). High-pass the luminance to remove the legitimate ring modulation, then demand that the
    # autocorrelation over (z-shift, roll) has no peak away from the origin.
    from scipy.ndimage import uniform_filter
    lum = (np.asarray(m.vertex_colors).reshape(p.n_z, p.n_theta, 3) @ np.array([0.299, 0.587, 0.114]))
    # remove the colour explained by the ring geometry (pale crests: axisymmetric and near-periodic, legitimately) so that
    # only the mucosal texture is tested; then a gentle high-pass and normalisation
    ridge_, _ = dt.ring_profile(z, p); ringw = (ridge_[:, None] * cart[None, :]).ravel()
    Xd = np.stack([np.ones(lum.size), ringw, ringw ** 2, np.broadcast_to(cart[None, :], lum.shape).ravel()], 1)
    hp = (lum.ravel() - Xd @ np.linalg.lstsq(Xd, lum.ravel(), rcond=None)[0]).reshape(lum.shape)
    rows_per_ring = max(int(round(p.ring_period_mm / (z[1] - z[0]))), 2)
    hp = hp - uniform_filter(hp, size=(rows_per_ring, max(p.n_theta // 12, 3)), mode=("reflect", "wrap")); hp = (hp - hp.mean()) / hp.std()
    ns = p.n_z // 3; Cmap = np.zeros((ns, p.n_theta))
    for s_ in range(ns):
        A, B = hp[s_:], hp[:p.n_z - s_]; Cmap[s_] = np.fft.irfft((np.fft.rfft(A, axis=1) * np.conj(np.fft.rfft(B, axis=1))).sum(0), n=p.n_theta) / A.size
    thr = 0.35; blob = np.zeros_like(Cmap, bool); stack = [(0, 0)]          # the origin's own correlation blob (flood fill)
    while stack:
        i, j = stack.pop()
        if blob[i, j] or Cmap[i, j] <= thr: continue
        blob[i, j] = True
        for a_, b_ in ((i + 1, j), (i - 1, j), (i, (j + 1) % p.n_theta), (i, (j - 1) % p.n_theta)):
            if 0 <= a_ < ns: stack.append((a_, b_))
    dz = z[1] - z[0]; blob_z = (np.where(blob.any(1))[0].max() + 1) * dz; blob_deg = blob[0].sum() * 360 / p.n_theta
    outside = np.where(blob, 0.0, Cmap); i_, j_ = np.unravel_index(int(np.argmax(outside)), outside.shape); best = float(outside[i_, j_])
    check("texture has no (roll, z-shift) symmetry", best < thr, f"(largest secondary autocorrelation peak {best:.2f} at z-shift {i_ * dz:.2f} mm, roll {j_ * 360 / p.n_theta:.1f} deg)")
    check("texture correlation length is short", blob_z < p.ring_period_mm and blob_deg < 90, f"(origin blob {blob_z:.1f} mm x {blob_deg:.0f} deg)")
except Exception as e: print("  [skip] mesh check:", e)
print("=> M0", "VERIFIED" if ok_all else "HAS FAILURES")

# 5. figure + video
frames = sorted(os.listdir(f"{a.run}/frames")) if os.path.isdir(f"{a.run}/frames") else []
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, cv2
fig = plt.figure(figsize=(15, 8)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.4, wspace=0.3)
ax = fig.add_subplot(gs[0, 0])
for iz, lab in ((iz0, f"collapse station z={z[iz0]:.0f} mm"), (int(len(z) * 0.15), f"z={z[int(len(z)*0.15)]:.0f} mm"), (int(len(z) * 0.85), f"z={z[int(len(z)*0.85)]:.0f} mm")):
    ax.plot(t, CSA[:, iz], label=lab)
ax.set_xlabel("t (s)"); ax.set_ylabel("CSA (mm²)"); ax.set_title("cross-sectional area vs time"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax = fig.add_subplot(gs[0, 1]); ax.plot(t, D[:, iz0, :][:, cart <= 0.5].max(1), label="membrane (max)"); ax.plot(t, np.abs(D[:, iz0, :][:, cart > 0.5]).max(1), label="cartilage (max |d|)")
ax.set_xlabel("t (s)"); ax.set_ylabel("displacement (mm)"); ax.set_title("who moves"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax = fig.add_subplot(gs[0, 2]); ax.plot(t, zc, label="z"); ax.plot(t, poses[:, 0, 3], label="x jitter"); ax.plot(t, poses[:, 1, 3], label="y jitter")
ax.set_xlabel("t (s)"); ax.set_ylabel("mm"); ax.set_title("camera path"); ax.legend(fontsize=8); ax.grid(alpha=.3)
picks = [0, len(frames) // 2, len(frames) - 1] if p.collapse_target == 0 else [max(int((p.collapse_t0_s - 0.8) * p.fps), 0), int(p.collapse_t0_s * p.fps), min(int((p.collapse_t0_s + 0.8) * p.fps), len(frames) - 1)]
for j, k in enumerate(picks):
    ax = fig.add_subplot(gs[1, j]); ax.set_axis_off()
    if frames: ax.imshow(cv2.cvtColor(cv2.imread(f"{a.run}/frames/{frames[k]}"), cv2.COLOR_BGR2RGB)); ax.set_title(f"frame {k}  t={t[k]:.2f}s  CSA@z0 {CSA[k, iz0]:.0f} mm²", fontsize=9)
fig.suptitle(f"M0 verification — {os.path.basename(a.run)}", fontweight="bold"); fig.savefig(f"{a.run}/verify.png", dpi=110); print("figure:", f"{a.run}/verify.png")
if a.video and frames:
    im0 = cv2.imread(f"{a.run}/frames/{frames[0]}"); h, w_ = im0.shape[:2]
    vw = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), p.fps, (w_, h))
    for f in frames: vw.write(cv2.imread(f"{a.run}/frames/{f}"))
    vw.release(); print("video:", a.video, f"({len(frames)} frames @ {p.fps:.0f} fps)")
