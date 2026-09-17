"""Extract a frame range of a real clip for the per-frame pipeline with a chosen photometric treatment.
Modes: n    = per-frame gain (bright wall p95 -> 150) + CLAHE (what runs/real_20V1n used)
       flat = n + local illumination flattening (luminance divided by its 61-px Gaussian mean inside the field of view) before CLAHE
       sat  = n + pixels saturated (> 245) or black (< 12) in the ORIGINAL frame set to 0 so the stereo cannot match on them
Usage: python m2/real_extract.py --video <mp4> --lo 2500 --hi 3000 --out runs/real_20V1_flat --mode flat"""
import argparse, os, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--lo", type=int, required=True); ap.add_argument("--hi", type=int, required=True); ap.add_argument("--out", required=True); ap.add_argument("--mode", default="n", choices=["n", "flat", "sat"])
a = ap.parse_args(); os.makedirs(f"{a.out}/images", exist_ok=True)
cap = cv2.VideoCapture(a.video); N = int(cap.get(7)); H = int(cap.get(4)); W = int(cap.get(3)); cum = np.zeros((H, W), np.int32)
for fi in np.linspace(0, max(N - 1, 0), 60).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, f = cap.read()
    if ok: cum += (f.mean(2) > 8).astype(np.int32)
cap.release(); m = ((cum >= 12).astype(np.uint8)) * 255; mask = cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((7, 7), np.uint8), iterations=5) > 0
maskf = mask.astype(np.float32); blur_mask = cv2.GaussianBlur(maskf, (0, 0), 61)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)); cap = cv2.VideoCapture(a.video); fi = 0; n = 0
while True:
    ok, fr = cap.read()
    if not ok or fi > a.hi: break
    if a.lo <= fi <= a.hi:
        lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB); L = lab[:, :, 0].astype(np.float32); L0 = L.copy()
        med = np.median(L[mask]); p95 = np.percentile(L[mask], 95)
        if med * 100 / 255 >= 12:
            if a.mode == "flat":
                loc = cv2.GaussianBlur(L * maskf, (0, 0), 61) / np.maximum(blur_mask, 1e-3)       # local mean luminance inside the field of view
                L = L / np.maximum(loc, 8.0) * 110.0
                L = np.clip(L * (150.0 / max(np.percentile(L[mask], 95), 1.0)), 0, 255)
            else:
                L = np.clip(L * (150.0 / max(p95, 1.0)), 0, 255)
            lab[:, :, 0] = clahe.apply(L.astype(np.uint8)); im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR); im[~mask] = 0
            if a.mode == "sat": im[(L0 > 245) | (L0 < 12)] = 0
            cv2.imwrite(f"{a.out}/images/f{fi:05d}.png", im, [cv2.IMWRITE_PNG_COMPRESSION, 1]); n += 1
    fi += 1
print(f"[extract {a.mode}] {n} frames -> {a.out}/images")
