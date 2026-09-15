"""Synthetic deforming trachea with exact ground truth for pose, deformation and calibre.

Geometry. A tube along +z of length L. Its canonical (undeformed) radius r0(theta, z) is a base profile R(z)
with cartilage RINGS: shallow inward ridges every `ring_period_mm`, present only on the cartilaginous sector
(anterior + lateral, 270 deg). The posterior MEMBRANE (90 deg sector centred on theta = pi) is smooth and is
the only part that moves:

    r(theta, z, t) = r0(theta, z) - w(theta) * [ A_breath(t) + A_collapse(t) * g(z) ]

w(theta) is a raised cosine over the membrane sector (1 at its centre, 0 at its edges, 0 on the cartilage),
A_breath is a periodic breathing excursion, A_collapse a transient Gaussian-in-time event localised in z by
g(z) (modelled on 26-V2, f1908-1921). Scenarios: static, breathing, collapse, malacia, and `uniform`, in
which w(theta) = 1 everywhere: a uniform radial contraction, which is the case that is NOT identifiable from a
forward-looking monocular camera without an anatomical prior (it looks like camera motion along the axis).
That scenario exists so the identifiability experiment can show the failure, not hide it.

Camera. A bronchoscope advancing along +z at constant speed with low-pass lateral jitter and a small
low-pass tilt; poses are exact. Intrinsics are the pinned OPENCV model of the real scope (pass a
bronchotrust intrinsics json) or a default 1080p, ~98 deg lens. Renders are optional (Open3D offscreen;
distortion is not applied in v0: the real pipeline undistorts anyway).

Ground truth written to <out>/gt.npz: t (N), poses_c2w (N,4,4), z (nz), theta (nt), r_canonical (nz,nt),
deformation (N,nz,nt) float16, csa_mm2 (N,nz), params (json string). Plus canonical_mesh.ply and, with
--save-meshes, one mesh per frame.

Usage:
  python synthetic/deforming_trachea.py --scenario collapse --out runs/synth_collapse [--calib intrinsics.json]
                                        [--fps 30 --duration 6 --render --save-meshes]
"""
import argparse, json, os
from dataclasses import dataclass, asdict, field
import numpy as np

SCENARIOS = {
    "static":    dict(breath_amp_mm=0.0, collapse_amp_mm=0.0, uniform=False),
    "breathing": dict(breath_amp_mm=0.6, collapse_amp_mm=0.0, uniform=False),
    "malacia":   dict(breath_amp_mm=2.4, collapse_amp_mm=0.0, uniform=False),      # ~45% area loss at end-expiration
    "collapse":  dict(breath_amp_mm=0.4, collapse_amp_mm=3.6, uniform=False),      # transient near-occlusion, 26-V2 style
    "uniform":   dict(breath_amp_mm=1.0, collapse_amp_mm=0.0, uniform=True),       # the non-identifiable case
}


@dataclass
class Params:
    length_mm: float = 60.0
    radius_mm: float = 5.0
    taper_mm_per_mm: float = 0.0            # linear change of R with z (positive = widening distally)
    ring_period_mm: float = 4.0
    ring_depth_mm: float = 0.3
    membrane_half_angle_deg: float = 45.0    # posterior sector = 90 deg, centred on theta = pi
    n_z: int = 241
    n_theta: int = 180
    fps: float = 30.0
    duration_s: float = 6.0
    breath_amp_mm: float = 0.0
    breath_period_s: float = 3.0
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


def canonical_radius(z, theta, p):
    R = p.radius_mm + p.taper_mm_per_mm * (z - z.mean())
    ridge = 0.5 * (1 + np.cos(2 * np.pi * z / p.ring_period_mm))      # 1 at ring crests
    _, cart = sector_weights(theta, p)
    return R[:, None] - p.ring_depth_mm * ridge[:, None] * cart[None, :]


def amplitude(t, p):
    a = p.breath_amp_mm * 0.5 * (1 - np.cos(2 * np.pi * t / p.breath_period_s))
    c = p.collapse_amp_mm * np.exp(-0.5 * ((t - p.collapse_t0_s) / p.collapse_sigma_s) ** 2)
    return a, c


def deformation(t, z, theta, p):
    """d(theta, z, t) >= 0, inward displacement of the wall at time t (nz, nt)."""
    w, _ = sector_weights(theta, p); a, c = amplitude(t, p)
    g = np.exp(-0.5 * ((z - p.collapse_z0_mm) / p.collapse_sigma_z_mm) ** 2)
    return w[None, :] * (a + c * g[:, None])


def cross_section_area(r, theta):
    """polygon area of each z-slice from radii on a uniform theta grid (nz, nt) -> (nz,)"""
    r2 = np.roll(r, -1, axis=1); dth = theta[1] - theta[0]
    return 0.5 * np.sum(r * r2 * np.sin(dth), axis=1)


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
    """(nz, nt, 3) vertices; theta measured from +x (anterior), posterior membrane centred on -x."""
    X = r * np.cos(theta)[None, :]; Y = r * np.sin(theta)[None, :]; Z = np.broadcast_to(z[:, None], r.shape)
    return np.stack([X, Y, Z], axis=-1)


def grid_triangles(nz, nt):
    i = np.arange(nz - 1)[:, None]; j = np.arange(nt)[None, :]; jn = (j + 1) % nt
    a = i * nt + j; b = i * nt + jn; c = (i + 1) * nt + j; d = (i + 1) * nt + jn
    return np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])


def vertex_colors(z, theta, p, rng):
    """procedural mucosa: pink base, paler cartilage crests, darker vascular streaks (a texture cue for SfM)."""
    _, cart = sector_weights(theta, p); ridge = 0.5 * (1 + np.cos(2 * np.pi * z / p.ring_period_mm))
    base = np.array([0.80, 0.45, 0.42]); pale = np.array([0.92, 0.72, 0.66]); dark = np.array([0.55, 0.20, 0.22])
    ring = ridge[:, None] * cart[None, :]
    ph = rng.uniform(0, 2 * np.pi, 5); streak = np.zeros((len(z), len(theta)))
    for k in range(5): streak += np.maximum(0, np.cos(3 * theta[None, :] + 0.15 * z[:, None] + ph[k]) - 0.85) / 0.15
    streak = np.clip(streak, 0, 1)
    col = base[None, None] * (1 - ring[..., None]) + pale[None, None] * ring[..., None]
    col = col * (1 - 0.6 * streak[..., None]) + dark[None, None] * 0.6 * streak[..., None]
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
    r0 = canonical_radius(z, theta, p); F = grid_triangles(p.n_z, p.n_theta)
    rng = np.random.default_rng(p.seed); C = vertex_colors(z, theta, p, rng)
    poses = camera_poses(t, p)
    D = np.zeros((N, p.n_z, p.n_theta), np.float32); CSA = np.zeros((N, p.n_z)); V_all = []
    for i, ti in enumerate(t):
        d = deformation(ti, z, theta, p); r = np.maximum(r0 - d, 0.3)
        D[i] = d; CSA[i] = cross_section_area(r, theta); V = surface_points(r, z, theta); V_all.append(V)
        if save_meshes: write_ply(f"{out}/mesh_{i:05d}.ply", V, F, C)
    write_ply(f"{out}/canonical_mesh.ply", surface_points(r0, z, theta), F, C)
    np.savez_compressed(f"{out}/gt.npz", t=t, poses_c2w=poses, z=z, theta=theta, r_canonical=r0, deformation=D.astype(np.float16),
                        csa_mm2=CSA, params=json.dumps(asdict(p)))
    csa0 = CSA[0]; imin = np.unravel_index(np.argmin(CSA), CSA.shape)
    print(f"{out}: {N} frames, z 0-{p.length_mm:.0f} mm, canonical D_CE {2*np.sqrt(csa0.mean()/np.pi):.2f} mm; "
          f"min CSA {CSA.min():.1f} mm^2 ({100*(1-CSA.min()/csa0[imin[1]]):.0f}% reduction) at t={t[imin[0]]:.2f}s z={z[imin[1]]:.0f} mm")
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
