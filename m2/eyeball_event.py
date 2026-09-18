"""Eyeball figure for a real per-frame run around an event: polar rings at the stations nearest the event (pre-event median,
collapse frames, post-event canonical), the unrolled wall over time at the nearest station, and a 3D view of the canonical
tube with the event-frame wall and the camera path.
Usage: python m2/eyeball_event.py runs/m1_real_26V2d_b3 --event 1908 1921 --pre 1895 1907 --out docs/figures/x.png"""
import argparse, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--event", type=int, nargs=2, required=True); ap.add_argument("--pre", type=int, nargs=2, required=True); ap.add_argument("--out", required=True); ap.add_argument("--frames", type=int, nargs="*", default=None); ap.add_argument("--stations", type=int, nargs="*", default=None)
a = ap.parse_args(); G = np.load(f"{a.run}/m1_real_grid.npz"); fr = G["frames"]; r = G["r_grid"]; rc = G["r_can"]; R = float(G["R"]); S = G["S"]; C = G["C"]; s_st = G["s_st"]; nS, nb = rc.shape; tb = (np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi
ev = (fr >= a.event[0]) & (fr <= a.event[1]); pre = (fr >= a.pre[0]) & (fr <= a.pre[1])
# stations with the most event observations
n_ev = np.isfinite(r[ev]).any(2).sum(0); picks = a.stations or sorted(int(i) for i in np.argsort(-n_ev)[:3] if n_ev[i] >= 5)
frames_show = a.frames or [int(fr[ev][k]) for k in np.linspace(0, ev.sum() - 1, 4).astype(int)]
fig = plt.figure(figsize=(17, 11)); gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
def ring(ax, rr, **kw):
    """markers at every finite sector; lines only between ADJACENT finite sectors (no chords across gaps)"""
    ok = np.isfinite(rr); lab = kw.pop("label", None); ax.plot(tb[ok], rr[ok] / R, linestyle="none", marker=kw.get("marker", "."), ms=kw.get("ms", 4), color=kw.get("color"), label=lab)
    kw.pop("marker", None); kw.pop("ms", None)
    for b in range(nb):
        b2 = (b + 1) % nb
        if ok[b] and ok[b2]: ax.plot([tb[b], tb[b2] if b2 else tb[b] + 2 * np.pi / nb], [rr[b] / R, rr[b2] / R], **kw)
for j, i in enumerate(picks):
    ax = fig.add_subplot(gs[0, j], projection="polar"); ax.set_theta_zero_location("E")
    with np.errstate(all="ignore"): rp = np.nanmedian(r[pre, i], 0)
    ring(ax, rc[i], color="k", lw=2.5, label=f"canonical (post-event median)"); ring(ax, rp, color="tab:blue", lw=1.5, label=f"pre-event median f{a.pre[0]}–{a.pre[1]}")
    cols = plt.cm.autumn(np.linspace(0, 0.85, len(frames_show)))
    for k, c in zip(frames_show, cols):
        jj = np.where(fr == k)[0]
        if len(jj) and np.isfinite(r[jj[0], i]).sum() >= 6: ring(ax, r[jj[0], i], color=c, lw=1.2, marker=".", ms=3, label=f"f{k}")
    ax.set_title(f"station {i} at {s_st[i]/R:.1f} R along the path\n(camera {np.linalg.norm(C[ev]-S[i],axis=1).mean()/R:.1f} R away during the event)", fontsize=9.5); ax.set_ylim(0, 1.6); ax.set_rticks([0.5, 1.0, 1.5]); ax.tick_params(labelsize=7)
    if j == 0: ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.35, -0.2))
# unrolled wall at the nearest station: dev(frame, sector)
i0 = picks[-1] if picks else 0; ax = fig.add_subplot(gs[1, 0:2]); dev = (r[:, i0, :] - rc[i0][None]) / R
m = ax.imshow(dev.T, aspect="auto", origin="lower", extent=[fr.min() - 0.5, fr.max() + 0.5, -180, 180], cmap="RdBu", vmin=-0.5, vmax=0.5, interpolation="nearest")
ax.axvspan(a.event[0] - 0.5, a.event[1] + 0.5, color="k", alpha=0.08); ax.set_xlabel("frame"); ax.set_ylabel("sector angle (deg)"); ax.set_title(f"unrolled wall at station {i0}: deviation from the canonical radius (red = inward, blue = outward), event shaded", fontsize=10); plt.colorbar(m, ax=ax, fraction=0.03, label="deviation (R)")
ax.set_xlim(a.pre[0] - 5, a.event[1] + 60)
# 3D: canonical tube (all stations with a canonical), event-frame rings at the picked stations, camera path
T = np.gradient(S, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True); N1 = np.zeros_like(S); N2 = np.zeros_like(S); n1 = np.cross(T[0], [0, 0, 1.0]); n1 = n1 if np.linalg.norm(n1) > 1e-3 else np.cross(T[0], [0, 1.0, 0]); n1 /= np.linalg.norm(n1)
for i in range(nS):
    n1 = n1 - (n1 @ T[i]) * T[i]; n1 /= np.linalg.norm(n1); N1[i] = n1; N2[i] = np.cross(T[i], n1)
ax3 = fig.add_subplot(gs[1, 2], projection="3d"); i_lo, i_hi = max(0, min(picks) - 4), min(nS - 1, max(picks) + 4)
for i in range(i_lo, i_hi + 1):
    if np.isfinite(rc[i]).mean() < 0.75: continue
    rr = np.where(np.isfinite(rc[i]), rc[i], np.nanmedian(rc[i])); P = S[i][None] + rr[:, None] * (np.cos(tb)[:, None] * N1[i][None] + np.sin(tb)[:, None] * N2[i][None]); P = np.vstack([P, P[:1]]); ax3.plot(P[:, 0] / R, P[:, 1] / R, P[:, 2] / R, color="0.6", lw=0.6)
for i in picks:
    for k, c in zip(frames_show, plt.cm.autumn(np.linspace(0, 0.85, len(frames_show)))):
        jj = np.where(fr == k)[0]
        if not len(jj): continue
        rr = r[jj[0], i]; ok = np.isfinite(rr)
        if ok.sum() < 6: continue
        P = S[i][None] + rr[ok][:, None] * (np.cos(tb[ok])[:, None] * N1[i][None] + np.sin(tb[ok])[:, None] * N2[i][None]); ax3.plot(P[:, 0] / R, P[:, 1] / R, P[:, 2] / R, ".", color=c, ms=3)
near = (fr >= a.pre[0]) & (fr <= a.event[1] + 10); ax3.plot(C[near, 0] / R, C[near, 1] / R, C[near, 2] / R, "k-", lw=1.5, label="camera path (pre → post)"); ax3.plot(C[ev, 0] / R, C[ev, 1] / R, C[ev, 2] / R, "r.", ms=5, label="camera during the event")
ax3.set_title(f"stations {i_lo}–{i_hi}: canonical tube (grey), event-frame wall at stations {picks} (warm), camera", fontsize=9); ax3.legend(fontsize=7, loc="upper left"); ax3.set_box_aspect((1, 1, 1))
Pn = np.vstack([S[i_lo:i_hi + 1], C[near]]) / R; ctr = Pn.mean(0); half = max((Pn.max(0) - Pn.min(0)).max() / 2, 1.5) + 0.3; ax3.set_xlim(ctr[0] - half, ctr[0] + half); ax3.set_ylim(ctr[1] - half, ctr[1] + half); ax3.set_zlim(ctr[2] - half, ctr[2] + half)
Tm = np.mean(T[i_lo:i_hi + 1], 0); az = np.degrees(np.arctan2(Tm[1], Tm[0])) + 90; ax3.view_init(elev=15, azim=az)
fig.suptitle(f"{os.path.basename(a.run)}: what the per-frame reconstruction looks like around the collapse f{a.event[0]}–{a.event[1]} (radii in units of the tube radius R)", fontweight="bold"); fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=115); print("figure:", a.out, "stations", picks, "frames", frames_show)
