"""Synthetic deforming trachea with exact ground truth for pose, deformation and calibre.

Geometry. A tube along +z of length L. Its canonical (undeformed) radius r0(theta, z) is a base profile R(z)
with cartilage RINGS: shallow inward ridges every `ring_period_mm`, present only on the cartilaginous sector
(anterior + lateral, 270 deg). The posterior MEMBRANE (90 deg sector centred on theta = pi) is smooth and is
the only part that moves:

    d(theta, z, t) = w(theta) * [ A_breath(t) + A_collapse(t) * g(z) ]        (mm)

and the membrane moves ANTERIORLY by d (a translation towards the anterior wall, so the lumen goes round ->
crescent -> slit, as seen in 20-V2 and 26-V2), not radially: a radial push of a posterior sector can never remove
more than that sector's share of the area, whereas the real wall bows across the lumen. w(theta) is a raised
cosine over the membrane sector (1 at its centre, 0 at its edges, 0 on the cartilage), A_breath a periodic
breathing excursion, A_collapse a transient Gaussian-in-time event localised in z by g(z) (26-V2, f1908-1921).
Scenario amplitudes are solved so that the peak CROSS-SECTIONAL AREA reduction matches a target (quiet breathing
12%, malacia 45% as in 20-V1, collapse 75% as in 26-V2). In the collapse scenario the lateral walls fold with the
membrane (posterior half-circumference moves); in the others only the 120 deg posterior sector does. Scenarios: static, breathing, collapse, malacia, and `uniform`, in
which w(theta) = 1 everywhere: a uniform radial contraction, which is the case that is NOT identifiable from a
forward-looking monocular camera without an anatomical prior (it looks like camera motion along the axis).
That scenario exists so the identifiability experiment can show the failure, not hide it.

Camera. A bronchoscope advancing along +z at constant speed with low-pass lateral jitter and a small
low-pass tilt; poses are exact. Intrinsics are the pinned OPENCV model of the real scope (pass a
bronchotrust intrinsics json) or a default 1080p, ~98 deg lens. Renders are optional (Open3D offscreen;
distortion is not applied in v0: the real pipeline undistorts anyway).

Ground truth written to <out>/gt.npz: t (N), poses_c2w (N,4,4), z (nz), theta (nt), r_canonical (nz,nt),
deformation (N,nz,nt) float32, csa_mm2 (N,nz), params (json string). Plus canonical_mesh.ply and, with
--save-meshes, one mesh per frame.

Usage:
  python synthetic/deforming_trachea.py --scenario collapse --out runs/synth_collapse [--calib intrinsics.json]
                                        [--fps 30 --duration 6 --render --save-meshes]
"""
import argparse, json, os
from dataclasses import dataclass, asdict, field
import numpy as np

# Scenarios are specified by the peak CROSS-SECTIONAL AREA reduction they must produce (the clinically meaningful
# quantity); the wall displacement amplitude that achieves it is solved numerically in generate().
SCENARIOS = {
    "static":    dict(breath_target=0.00, collapse_target=0.00, uniform=False),
    "breathing": dict(breath_target=0.12, collapse_target=0.00, uniform=False),   # quiet breathing
    "malacia":   dict(breath_target=0.45, collapse_target=0.00, uniform=False),   # 20-V1: 45% expiratory reduction
    "collapse":  dict(breath_target=0.10, collapse_target=0.75, uniform=False, membrane_half_angle_deg=90.0),   # 26-V2: dark lumen 12% -> 3%
    "uniform":   dict(breath_target=0.30, collapse_target=0.00, uniform=True),    # non-identifiable: whole circumference, radial
}


@dataclass
class Params:
    length_mm: float = 60.0
    radius_mm: float = 5.0
    taper_mm_per_mm: float = 0.0            # linear change of R with z (positive = widening distally)
    ring_period_mm: float = 4.0
    ring_depth_mm: float = 0.3
    ring_spacing_jitter: float = 0.15        # per-ring spacing jitter (fraction of ring_period_mm): real rings are irregular
    ring_depth_jitter: float = 0.25          # per-ring depth jitter (fraction); both 0 -> the periodic v0 tube
    membrane_half_angle_deg: float = 60.0    # posterior sector = 120 deg (membrane + folding lateral tips), centred on theta = pi
    n_z: int = 241
    n_theta: int = 180
    fps: float = 30.0
    duration_s: float = 6.0
    breath_target: float = 0.0               # peak CSA reduction from breathing (fraction); amplitude solved from it
    breath_amp_mm: float = 0.0
    breath_period_s: float = 3.0
    collapse_target: float = 0.0             # peak CSA reduction at the collapse event (fraction)
    collapse_amp_mm: float = 0.0
    collapse_t0_s: float = 3.0
    collapse_sigma_s: float = 0.22
    collapse_z0_mm: float = 30.0
    collapse_sigma_z_mm: float = 7.0
    uniform: bool = False
    camera_speed_mm_s: float = 6.0
    camera_start_z_mm: float = -4.0
    jitter_mm: float = 0.35
    tilt_deg: float = 2.0
    seed: int = 0
    width: int = 1920; height: int = 1080
    fx: float = 830.0; fy: float = 830.0; cx: float = 960.0; cy: float = 540.0
    dist: list = field(default_factory=lambda: [-0.09, -0.09, 0.0, 0.0])


def load_calib(p, path):
    d = json.load(open(path)); p.width, p.height = int(d["width"]), int(d["height"])
    p.fx, p.fy, p.cx, p.cy = float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"])
    p.dist = [float(d.get(k, 0.0)) for k in ("k1", "k2", "p1", "p2")]; return p


def sector_weights(theta, p):
    """w(theta): raised cosine on the posterior membrane; 0 on cartilage. cart(theta) = 1 - membrane indicator (smoothed)."""
    half = np.radians(p.membrane_half_angle_deg)
    dth = np.angle(np.exp(1j * (theta - np.pi)))                     # signed distance to the posterior centre
    w = np.where(np.abs(dth) < half, 0.5 * (1 + np.cos(np.pi * dth / half)), 0.0)
    cart = 1.0 - np.where(np.abs(dth) < half, 1.0, 0.0)
    return (np.ones_like(theta) if p.uniform else w), cart


def ring_profile(z, p):
    """Cartilage ring profile along z: ridge(z) in [0, 1] (1 at crests, 0 at troughs) and a per-ring depth scale.
    Spacing and depth are jittered per ring so the tube has no exact periodicity in z for rigid SfM to alias on
    (v0's periodic rings + helical texture let it register a ring-shifted, rolled copy). Deterministic in p.seed."""
    rng = np.random.default_rng(p.seed + 7); troughs = [0.0]
    while troughs[-1] < z.max() + p.ring_period_mm:
        troughs.append(troughs[-1] + p.ring_period_mm * (1 + p.ring_spacing_jitter * rng.uniform(-1, 1)))
    troughs = np.array(troughs); dscale = 1 + p.ring_depth_jitter * rng.uniform(-1, 1, len(troughs))
    k = np.clip(np.searchsorted(troughs, z, side="right") - 1, 0, len(troughs) - 2)
    phase = 2 * np.pi * (z - troughs[k]) / (troughs[k + 1] - troughs[k]) - np.pi
    return 0.5 * (1 + np.cos(phase)), dscale[k]


def canonical_radius(z, theta, p):
    R = p.radius_mm + p.taper_mm_per_mm * (z - z.mean())
    ridge, dscale = ring_profile(z, p)                                   # 1 at ring crests, per-ring depth
    _, cart = sector_weights(theta, p)
    return R[:, None] - (p.ring_depth_mm * dscale * ridge)[:, None] * cart[None, :]


def amplitude(t, p):
    a = p.breath_amp_mm * 0.5 * (1 - np.cos(2 * np.pi * t / p.breath_period_s))
    c = p.collapse_amp_mm * np.exp(-0.5 * ((t - p.collapse_t0_s) / p.collapse_sigma_s) ** 2)
    return a, c


def deformation(t, z, theta, p):
    """d(theta, z, t) >= 0, inward displacement of the wall at time t (nz, nt)."""
    w, _ = sector_weights(theta, p); a, c = amplitude(t, p)
    g = np.exp(-0.5 * ((z - p.collapse_z0_mm) / p.collapse_sigma_z_mm) ** 2)
    return w[None, :] * (a + c * g[:, None])


def deform_xy(r0, theta, d, p):
    """deformed cross-section coordinates (nz, nt) X, Y from canonical radii r0 and displacement field d (mm).
    Membrane: anterior translation (+x; posterior is at theta = pi, x < 0), capped so the wall cannot pass the
    anterior inner wall. Uniform scenario: radial contraction (r0 - d, floored at 0.3 mm)."""
    c, s_ = np.cos(theta)[None, :], np.sin(theta)[None, :]
    if p.uniform:
        r = np.maximum(r0 - d, 0.3); return r * c, r * s_
    X, Y = r0 * c, r0 * s_
    # a displaced posterior point may not pass the anterior wall at its own y (the wall is a circle, not a plane);
    # points with d == 0 are never touched, so the canonical geometry is exact.
    Rs = r0.max(axis=1, keepdims=True); cap = np.sqrt(np.maximum(Rs ** 2 - Y ** 2, 0.0)) - 0.3
    Xd = np.where(d > 0, np.minimum(X + d, cap), X)
    return Xd, Y


def csa_from_xy(X, Y):
    """shoelace area of each z-slice polygon (nz, nt) -> (nz,)"""
    return 0.5 * np.abs(np.sum(X * np.roll(Y, -1, axis=1) - np.roll(X, -1, axis=1) * Y, axis=1))


def cross_section_area(r, theta):
    return csa_from_xy(r * np.cos(theta)[None, :], r * np.sin(theta)[None, :])


def solve_amplitude(target, z, theta, p):
    """displacement amplitude (mm) that reduces the mid-slice CSA by `target` (bisection; 0 if target is 0)."""
    if target <= 0: return 0.0
    r0 = canonical_radius(z, theta, p); w, _ = sector_weights(theta, p); k = len(z) // 2
    a0 = cross_section_area(r0[k:k + 1], theta)[0]
    def f(amp):
        X, Y = deform_xy(r0[k:k + 1], theta, w[None, :] * amp, p); return 1 - csa_from_xy(X, Y)[0] / a0
    lo, hi = 0.0, 4 * p.radius_mm
    if f(hi) < target: print(f"  warning: target reduction {target:.0%} not reachable (max {f(hi):.0%}); using max"); return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


def camera_poses(t, p):
    """camera-to-world 4x4 per frame: advancing along +z, low-pass lateral jitter and tilt, camera looks along +z."""
    rng = np.random.default_rng(p.seed); N = len(t)
    def lowpass(n, scale, tau=12):
        x = rng.standard_normal(n); k = np.exp(-np.arange(0, 4 * tau) / tau); k /= k.sum()
        y = np.convolve(x, k, mode="same"); return scale * y / (y.std() + 1e-9)
    jx, jy = lowpass(N, p.jitter_mm), lowpass(N, p.jitter_mm)
    ax, ay = np.radians(lowpass(N, p.tilt_deg)), np.radians(lowpass(N, p.tilt_deg))
    poses = np.zeros((N, 4, 4)); poses[:, 3, 3] = 1
    for i in range(N):
        cxr, sxr, cyr, syr = np.cos(ax[i]), np.sin(ax[i]), np.cos(ay[i]), np.sin(ay[i])
        Rx = np.array([[1, 0, 0], [0, cxr, -sxr], [0, sxr, cxr]]); Ry = np.array([[cyr, 0, syr], [0, 1, 0], [-syr, 0, cyr]])
        poses[i, :3, :3] = Ry @ Rx                                         # camera z-axis ~ world +z
        poses[i, :3, 3] = [jx[i], jy[i], p.camera_start_z_mm + p.camera_speed_mm_s * t[i]]
    return poses


def surface_points(r, z, theta):
    """(nz, nt, 3) vertices of an undeformed radius field; theta from +x (anterior), posterior centred on -x."""
    X = r * np.cos(theta)[None, :]; Y = r * np.sin(theta)[None, :]; Z = np.broadcast_to(z[:, None], r.shape)
    return np.stack([X, Y, Z], axis=-1)


def surface_points_deformed(r0, z, theta, d, p):
    X, Y = deform_xy(r0, theta, d, p); Z = np.broadcast_to(z[:, None], r0.shape)
    return np.stack([X, Y, Z], axis=-1)


def grid_triangles(nz, nt):
    i = np.arange(nz - 1)[:, None]; j = np.arange(nt)[None, :]; jn = (j + 1) % nt
    a = i * nt + j; b = i * nt + jn; c = (i + 1) * nt + j; d = (i + 1) * nt + jn
    return np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])


def vertex_colors(z, theta, p, rng):
    """procedural mucosa with NO periodicity in theta or z.
    v0 used helical streaks cos(3*theta + 0.15*z) on top of periodic rings: a roll of 0.2 rad with a shift of one ring
    period reproduced the scene exactly, and rigid SfM locked onto that rolled/shifted copy (11.5 deg roll jumps,
    wrong scale) while reporting every frame registered. Real mucosa has no such symmetry, so the texture must not
    either. All fields are random Fourier features evaluated on the cylinder embedding (x, y, z) in mm: aperiodic,
    incommensurate with the ring period, and continuous across the theta seam."""
    _, cart = sector_weights(theta, p); ridge, _ = ring_profile(z, p)
    X = np.stack(np.broadcast_arrays(p.radius_mm * np.cos(theta)[None, :], p.radius_mm * np.sin(theta)[None, :], z[:, None]), -1)

    def rff(n, scale_mm):
        W = rng.standard_normal((n, 3)) / scale_mm; ph = rng.uniform(0, 2 * np.pi, n); amp = rng.standard_normal(n)
        f = (np.cos(X @ W.T + ph) * amp).sum(-1); return f / f.std()

    mottle = 0.5 * rff(48, 3.0) + 0.3 * rff(48, 1.0) + 0.2 * rff(48, 0.45)                 # multi-scale colour mottling
    vessel = np.exp(-(rff(48, 2.5) / 0.12) ** 2) + 0.7 * np.exp(-(rff(48, 1.2) / 0.10) ** 2)  # ridged noise: thin dark branching lines
    vessel = np.clip(vessel, 0, 1); speckle = 0.03 * rng.standard_normal(mottle.shape)
    base = np.array([0.80, 0.45, 0.42]); pale = np.array([0.92, 0.72, 0.66]); dark = np.array([0.50, 0.18, 0.20])
    ring = 0.6 * ridge[:, None] * cart[None, :]                                          # pale crests, subtle as in vivo
    col = base[None, None] * (1 - ring[..., None]) + pale[None, None] * ring[..., None]
    col = col * (1 + 0.18 * mottle[..., None] + speckle[..., None])
    col = col * (1 - 0.65 * vessel[..., None]) + dark[None, None] * 0.65 * vessel[..., None]
    return np.clip(col, 0, 1)


def write_ply(path, V, F, C=None):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V.reshape(-1, 3)), o3d.utility.Vector3iVector(F))
    if C is not None: m.vertex_colors = o3d.utility.Vector3dVector(C.reshape(-1, 3))
    m.compute_vertex_normals(); o3d.io.write_triangle_mesh(path, m)


def render_frames(out, V_all, F, C, poses, p):
    """optional: Open3D offscreen renders with a headlight at the camera. Returns number of frames written."""
    try:
        import open3d as o3d, open3d.visualization.rendering as rd
    except Exception as e:
        print(f"  render skipped (open3d rendering unavailable: {e})"); return 0
    os.makedirs(f"{out}/frames", exist_ok=True)
    try:
        r = rd.OffscreenRenderer(p.width, p.height)
    except Exception as e:
        print(f"  render skipped (no offscreen GL context: {e})"); return 0
    r.scene.set_background([0, 0, 0, 1]); r.scene.scene.set_sun_light([0, 0, 1], [1, 1, 1], 0.0)
    mat = rd.MaterialRecord(); mat.shader = "defaultLit"; mat.base_roughness = 0.6
    intr = o3d.camera.PinholeCameraIntrinsic(p.width, p.height, p.fx, p.fy, p.cx, p.cy)
    n = 0
    for i in range(len(poses)):
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V_all[i].reshape(-1, 3)), o3d.utility.Vector3iVector(F))
        m.vertex_colors = o3d.utility.Vector3dVector(C.reshape(-1, 3)); m.compute_vertex_normals()
        r.scene.clear_geometry(); r.scene.add_geometry("tube", m, mat)
        cam = poses[i]; r.scene.scene.remove_light("head") if i else None
        r.scene.scene.add_point_light("head", [1, 0.95, 0.9], cam[:3, 3].astype(np.float32), 60.0, 40.0, True)
        r.setup_camera(intr, np.linalg.inv(cam).astype(np.float64))
        img = np.asarray(r.render_to_image())
        import cv2; cv2.imwrite(f"{out}/frames/f{i:05d}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR)); n += 1
    return n


def generate(p, out, render=False, save_meshes=False):
    os.makedirs(out, exist_ok=True)
    z = np.linspace(0, p.length_mm, p.n_z); theta = np.linspace(0, 2 * np.pi, p.n_theta, endpoint=False)
    t = np.arange(0, p.duration_s, 1.0 / p.fps); N = len(t)
    p.breath_amp_mm = solve_amplitude(p.breath_target, z, theta, p)
    p.collapse_amp_mm = solve_amplitude(p.collapse_target, z, theta, p) if p.collapse_target > 0 else 0.0
    if p.collapse_target > 0: p.collapse_amp_mm = max(p.collapse_amp_mm - p.breath_amp_mm * 0.5 * (1 - np.cos(2 * np.pi * p.collapse_t0_s / p.breath_period_s)), 0.0)
    r0 = canonical_radius(z, theta, p); F = grid_triangles(p.n_z, p.n_theta)
    rng = np.random.default_rng(p.seed); C = vertex_colors(z, theta, p, rng)
    poses = camera_poses(t, p)
    D = np.zeros((N, p.n_z, p.n_theta), np.float32); CSA = np.zeros((N, p.n_z)); V_all = []
    for i, ti in enumerate(t):
        d = deformation(ti, z, theta, p)
        D[i] = d; X, Y = deform_xy(r0, theta, d, p); CSA[i] = csa_from_xy(X, Y)
        V = surface_points_deformed(r0, z, theta, d, p); V_all.append(V)
        if save_meshes: write_ply(f"{out}/mesh_{i:05d}.ply", V, F, C)
    write_ply(f"{out}/canonical_mesh.ply", surface_points(r0, z, theta), F, C)
    np.savez_compressed(f"{out}/gt.npz", t=t, poses_c2w=poses, z=z, theta=theta, r_canonical=r0, deformation=D.astype(np.float32),
                        csa_mm2=CSA, params=json.dumps(asdict(p)))
    csa0 = CSA[0]; imin = np.unravel_index(np.argmin(CSA), CSA.shape)
    print(f"{out}: {N} frames, z 0-{p.length_mm:.0f} mm, canonical D_CE {2*np.sqrt(csa0.mean()/np.pi):.2f} mm; amplitudes breath {p.breath_amp_mm:.2f} / collapse {p.collapse_amp_mm:.2f} mm; "
          f"min CSA {CSA.min():.1f} mm^2 ({100*(1-CSA.min()/csa0[imin[1]]):.0f}% reduction) at t={t[imin[0]]:.2f}s z={z[imin[1]]:.0f} mm")
    json.dump(dict(model="OPENCV", width=p.width, height=p.height, fx=p.fx, fy=p.fy, cx=p.cx, cy=p.cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0,
                   params_colmap=[p.fx, p.fy, p.cx, p.cy, 0.0, 0.0, 0.0, 0.0],
                   note="renders are pure pinhole (no distortion applied): give THIS file to the rigid pipeline, not the scope's calibration"),
              open(f"{out}/intrinsics_pinhole.json", "w"), indent=1)
    if render:
        n = render_frames(out, np.stack(V_all), F, C, poses, p); print(f"  rendered {n} frames" if n else "  no renders written")
    return dict(t=t, poses=poses, z=z, theta=theta, r0=r0, D=D, CSA=CSA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="collapse"); ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default=None, help="bronchotrust intrinsics json (pinned OPENCV) for the camera model")
    ap.add_argument("--fps", type=float, default=30.0); ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--render", action="store_true"); ap.add_argument("--save-meshes", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    p = Params(fps=a.fps, duration_s=a.duration, seed=a.seed, **SCENARIOS[a.scenario])
    if a.calib: load_calib(p, a.calib)
    generate(p, a.out, render=a.render, save_meshes=a.save_meshes)


if __name__ == "__main__":
    main()
