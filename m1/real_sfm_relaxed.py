"""Real video: can rigid SfM be carried THROUGH a wall-collapse if registration is relaxed?

The synthetic cross-check showed that COLMAP's poses stay exact through a 76 % posterior collapse because the rigid
cartilage anchors them. On 26-V2 the production pipeline (default thresholds, exhaustive matching) broke into two
models at the collapse (f1908-1921). Here the same frames are extracted exactly as the pipeline does (sequential
decode, bezel mask, CLAHE), features and a sequential matcher are run, and the mapper is run with pinned intrinsics
but relaxed absolute-pose thresholds. If one model spans the event, M1's per-frame depth can then be run across it.

Usage: python m1/real_sfm_relaxed.py --video <mp4> --calib <intrinsics.json> --lo 1600 --hi 2100 --out runs/real_26V2 [--gpu 0]
Env:   BRONCHO_COLMAP (default colmap)"""
import sys, os, json, argparse, subprocess, shutil
import numpy as np, cv2

ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--calib", required=True); ap.add_argument("--lo", type=int, required=True); ap.add_argument("--hi", type=int, required=True)
ap.add_argument("--out", required=True); ap.add_argument("--gpu", default="0"); ap.add_argument("--overlap", type=int, default=20); ap.add_argument("--clahe", type=float, default=3.0)
ap.add_argument("--min-inliers", type=int, default=15); ap.add_argument("--min-inlier-ratio", type=float, default=0.10); ap.add_argument("--skip-extract", action="store_true")
a = ap.parse_args(); COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap"); out = a.out; img, msk = f"{out}/images", f"{out}/masks"; os.makedirs(out, exist_ok=True)
J = json.load(open(a.calib)); pstr = ",".join(str(v) for v in J["params_colmap"])


def sh(cmd, log=None):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True)
    if log: open(log, "a").write(" ".join(cmd[:1]) + "\n" + r.stdout[-3000:] + r.stderr[-3000:])
    if r.returncode != 0: raise RuntimeError(cmd[0] + "\n" + r.stderr[-1500:])


def gpu_opt(cmd):
    """COLMAP renamed the GPU-index options between versions: find the one this binary accepts."""
    h = subprocess.run([COLMAP, cmd, "-h"], capture_output=True, text=True); h = h.stdout + h.stderr
    for line in h.splitlines():
        if "gpu_index" in line: return line.split()[0]
    return None


def bezel(video, n=60):
    cap = cv2.VideoCapture(video); N = int(cap.get(7)); H = int(cap.get(4)); W = int(cap.get(3)); cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, f = cap.read()
        if ok: cum += (f.mean(2) > 8).astype(np.int32)
    cap.release(); m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    return cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((7, 7), np.uint8), iterations=5) > 0


if not a.skip_extract:
    for d in (img, msk): shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    mask = bezel(a.video); bez = mask.astype(np.uint8) * 255; clahe = cv2.createCLAHE(clipLimit=a.clahe, tileGridSize=(8, 8))
    cap = cv2.VideoCapture(a.video); fi = 0; n = 0; dark = 0
    while True:
        ok, fr = cap.read()
        if not ok or fi > a.hi: break
        if a.lo <= fi <= a.hi:
            lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
            if lab[:, :, 0][mask].mean() * 100 / 255 >= 12:
                lab[:, :, 0] = clahe.apply(lab[:, :, 0]); im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR); im[~mask] = 0
                cv2.imwrite(f"{img}/f{fi:05d}.png", im, [cv2.IMWRITE_PNG_COMPRESSION, 1]); cv2.imwrite(f"{msk}/f{fi:05d}.png.png", bez, [cv2.IMWRITE_PNG_COMPRESSION, 9]); n += 1
            else: dark += 1
        fi += 1
    cap.release(); print(f"[extract] {n} frames f{a.lo}-{a.hi} (sequential decode, CLAHE {a.clahe}, {dark} dark skipped)", flush=True)
db = f"{out}/db.db"; log = f"{out}/colmap.log"
if os.path.exists(db): os.remove(db)
sh(["feature_extractor", "--database_path", db, "--image_path", img, "--ImageReader.mask_path", msk, "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1",
    "--ImageReader.camera_params", pstr, "--SiftExtraction.max_image_size", "1600", "--SiftExtraction.max_num_features", "8192", "--SiftExtraction.peak_threshold", "0.005"] + ([gpu_opt("feature_extractor"), a.gpu] if gpu_opt("feature_extractor") else []), log)
print("[features] done", flush=True)
sh(["sequential_matcher", "--database_path", db, "--SequentialMatching.overlap", str(a.overlap), "--SequentialMatching.quadratic_overlap", "0", "--SequentialMatching.loop_detection", "0"] + ([gpu_opt("sequential_matcher"), a.gpu] if gpu_opt("sequential_matcher") else []), log)
print(f"[matching] sequential, overlap {a.overlap}", flush=True)
sp = f"{out}/sparse"; shutil.rmtree(sp, ignore_errors=True); os.makedirs(sp)
sh(["mapper", "--database_path", db, "--image_path", img, "--output_path", sp, "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_refine_principal_point", "0",
    "--Mapper.init_min_tri_angle", "4", "--Mapper.min_num_matches", "8", "--Mapper.abs_pose_min_num_inliers", str(a.min_inliers), "--Mapper.abs_pose_min_inlier_ratio", str(a.min_inlier_ratio),
    "--Mapper.filter_max_reproj_error", "6", "--Mapper.max_reg_trials", "5"], log)
# report models
import tempfile
models = sorted(d for d in os.listdir(sp) if os.path.isdir(f"{sp}/{d}"))
rep = []
for m in models:
    with tempfile.TemporaryDirectory() as td:
        sh(["model_converter", "--input_path", f"{sp}/{m}", "--output_path", td, "--output_type", "TXT"])
        L = [l for l in open(f"{td}/images.txt") if l.strip() and not l.startswith("#")]; fr = sorted(int(l.split()[9][1:6]) for l in L[0::2])
    gaps = [(fr[i], fr[i + 1]) for i in range(len(fr) - 1) if fr[i + 1] - fr[i] > 1]
    rep.append(dict(model=m, n=len(fr), lo=fr[0], hi=fr[-1], gaps=gaps[:10])); print(f"[SfM] model {m}: {len(fr)} frames f{fr[0]}-{fr[-1]}, gaps {gaps[:8]}")
json.dump(dict(video=a.video, lo=a.lo, hi=a.hi, overlap=a.overlap, min_inliers=a.min_inliers, min_inlier_ratio=a.min_inlier_ratio, models=rep), open(f"{out}/sfm_report.json", "w"), indent=1)
best = max(rep, key=lambda r: r["n"]); print(f"[SfM] largest model {best['model']}: {best['n']} frames f{best['lo']}-{best['hi']}")
