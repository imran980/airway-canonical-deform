"""Contact sheet of the raw frames around an event, annotated with the pose-free dark-lumen fraction, so the images can be
compared with the reconstruction. Usage: python m2/eyeball_frames.py runs/real_26V2d/images --frames 1900 1904 ... --out x.png"""
import argparse, os, numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("images"); ap.add_argument("--frames", type=int, nargs="+", required=True); ap.add_argument("--event", type=int, nargs=2, default=None); ap.add_argument("--out", required=True); ap.add_argument("--cols", type=int, default=5)
a = ap.parse_args(); n = len(a.frames); rows = (n + a.cols - 1) // a.cols
fig, axs = plt.subplots(rows, a.cols, figsize=(3.1 * a.cols, 2.5 * rows)); axs = np.atleast_1d(axs).ravel()
for ax, k in zip(axs, a.frames):
    p = f"{a.images}/f{k:05d}.png"; ax.axis("off")
    if not os.path.exists(p): ax.set_title(f"f{k}: missing", fontsize=8); continue
    im = cv2.imread(p); g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(float); m = g > 8; v = g[m]; dark = float((v < 0.25 * np.percentile(v, 95)).mean())
    ax.imshow(im[:, :, ::-1]); inside = a.event and a.event[0] <= k <= a.event[1]
    ax.set_title(f"f{k}{'  (collapse)' if inside else ''}\ndark-lumen fraction {dark:.3f}", fontsize=9, color="tab:red" if inside else "k")
    if inside:
        for sp in ("top", "bottom", "left", "right"): ax.spines[sp].set_visible(True); ax.spines[sp].set_color("tab:red"); ax.spines[sp].set_linewidth(3)
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
for ax in axs[n:]: ax.axis("off")
fig.suptitle(f"raw frames (gain-normalised) around the event — the dark lumen shrinks as the airway closes", fontweight="bold")
fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=110); print("figure:", a.out)
