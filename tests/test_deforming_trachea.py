"""Ground-truth generator checks. Run: python -m pytest tests/ -q   (or python tests/test_deforming_trachea.py)"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "synthetic"))
import deforming_trachea as dt

SMALL = dict(n_z=61, n_theta=72, fps=10.0, duration_s=4.0, collapse_t0_s=2.0)


def _gen(scenario, tmp):
    p = dt.Params(**SMALL, **dt.SCENARIOS[scenario]); return p, dt.generate(p, os.path.join(tmp, scenario))


def test_static_has_no_deformation(tmp_path):
    p, g = _gen("static", str(tmp_path))
    assert np.all(g["D"] == 0); assert np.allclose(g["CSA"], g["CSA"][0])


def test_cartilage_sector_is_rigid(tmp_path):
    p, g = _gen("collapse", str(tmp_path)); _, cart = dt.sector_weights(g["theta"], p)
    assert np.abs(g["D"][:, :, cart > 0.5]).max() == 0.0, "cartilage rings must not move"
    assert np.abs(g["D"][:, :, cart <= 0.5]).max() > 1.0, "membrane must move"


def test_collapse_is_local_in_time_and_space(tmp_path):
    p, g = _gen("collapse", str(tmp_path)); t, z, CSA = g["t"], g["z"], g["CSA"]
    iz = int(np.argmin(np.abs(z - p.collapse_z0_mm))); it = int(np.argmin(CSA[:, iz]))
    assert abs(t[it] - p.collapse_t0_s) < 0.2, "minimum area at the event time"
    assert CSA[it, iz] < 0.6 * CSA[0, iz], "a near-occlusive event"
    far = int(np.argmin(np.abs(z - (p.collapse_z0_mm - 4 * p.collapse_sigma_z_mm))))
    if 0 <= far < len(z):
        drop_far = 1 - CSA[it, far] / CSA[0, far]; assert drop_far < 0.25, "far from z0 only breathing remains"


def test_uniform_scenario_moves_whole_circumference(tmp_path):
    p, g = _gen("uniform", str(tmp_path))
    assert np.abs(g["D"][len(g["t"]) // 2]).min() > 0.0, "uniform contraction touches every theta"


def test_camera_advances_monotonically(tmp_path):
    p, g = _gen("breathing", str(tmp_path)); zc = g["poses"][:, 2, 3]
    assert np.all(np.diff(zc) > 0); assert np.allclose(np.linalg.det(g["poses"][:, :3, :3]), 1.0, atol=1e-9)


def test_gt_file_roundtrip(tmp_path):
    p, g = _gen("malacia", str(tmp_path)); f = np.load(os.path.join(str(tmp_path), "malacia", "gt.npz"))
    assert f["csa_mm2"].shape == g["CSA"].shape and json.loads(str(f["params"]))["breath_amp_mm"] == p.breath_amp_mm
    red = 1 - f["csa_mm2"].min() / f["csa_mm2"][0].mean(); assert 0.35 < red < 0.6, f"malacia should lose ~45% area, got {red:.2f}"


if __name__ == "__main__":
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            with tempfile.TemporaryDirectory() as d: fn(type("P", (), {"__str__": lambda s: d})()) if False else fn(d)
            print("ok", name)
