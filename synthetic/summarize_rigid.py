"""Run score_rigid.py and pose_diag.py on every rigid workspace and print one markdown results table.

Usage: python synthetic/summarize_rigid.py static collapse breathing malacia   (names <sc>: runs/rigid_<sc> + runs/synth_<sc>_30/gt.npz)
Env:   BRONCHOTRUST, BRONCHO_COLMAP forwarded to the two scripts."""
import sys, os, json, subprocess, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); PY = sys.executable
rows = []
for sc in sys.argv[1:]:
    ws, gt = f"{ROOT}/runs/rigid_{sc}", f"{ROOT}/runs/synth_{sc}_30/gt.npz"
    if not os.path.isdir(f"{ws}/sparse"): print(f"{sc}: no sparse model"); continue
    r1 = subprocess.run([PY, f"{HERE}/score_rigid.py", ws, gt], capture_output=True, text=True); print(r1.stdout.strip()); print(r1.stderr.strip()[-400:] if r1.returncode else "", end="")
    r2 = subprocess.run([PY, f"{HERE}/pose_diag.py", ws, gt], capture_output=True, text=True); print(r2.stdout.strip()); print(r2.stderr.strip()[-400:] if r2.returncode else "", end="")
    S = json.load(open(f"{ws}/score_rigid_rigid_{sc}.json")); g = np.load(gt); P = json.loads(str(g["params"])); CSA = g["csa_mm2"]
    m = max(S["models"], key=lambda r: r["n_registered"]) if S["models"] else None
    relrot = re.search(r"\|est - GT\| relative rotation: median ([\d.]+) deg, max ([\d.]+) deg; first frame exceeding 5 deg: (\S+)", r2.stdout)
    straight = re.search(r"path straightness .*?: est ([\d.]+) \| GT ([\d.]+)", r2.stdout)
    d = S.get("dense", {}); pm = S.get("pipeline_measure_csa", {})
    rows.append(dict(scenario=sc, event=f"{P['collapse_target']:.0%} collapse @ {P['collapse_t0_s']:.0f}s" if P.get("collapse_target", 0) > 0 else (f"{P['breath_target']:.0%} periodic" if P.get("breath_target", 0) > 0 else "none"),
                     n_models=len(S["models"]), registered=f"{m['n_registered']}/{len(g['t'])}" if m else "0", reproj=None,
                     centre_rmse=f"{m['centre_rmse_mm']:.2f}" if m else "-", axis_med=f"{m['axis_err_deg_median']:.1f}" if m else "-",
                     relrot=f"{relrot.group(1)} / {relrot.group(2)} (from {relrot.group(3)})" if relrot else "-", straight=f"{straight.group(1)} vs {straight.group(2)}" if straight else "-",
                     calibre_err=f"{d['calibre_err_mm_mean']:.2f}" if d.get("calibre_err_mm_mean") is not None else "-", calibre_bias=f"{d['calibre_bias_mm']:+.2f}" if d.get("calibre_bias_mm") is not None else "-",
                     slabs=f"{d.get('n_slabs_measured', 0)}/{d.get('n_slabs', 0)}" if d else "-",
                     pipe_pct=f"{pm['pct_obstruction']:.0f}% ({pm['n_accepted']} st., CV {pm['cv']:.2f})" if pm.get("measurable") else (f"not measurable ({pm.get('n_accepted', 0)} st.)" if pm else "-"),
                     gt_pct=f"{S.get('gt_static_pct_obstruction', float('nan')):.0f}%" + (f" static / {S['gt_event_pct_obstruction']:.0f}% at event" if "gt_event_pct_obstruction" in S else "")))
    lg = f"{ws}/run.log"
    if os.path.exists(lg):
        mm = re.search(r"\[model\] \d+ reg, span f\d+-\d+, reproj ([\d.]+)px", open(lg).read()); rows[-1]["reproj"] = mm.group(1) if mm else "-"
hdr = ["scenario", "wall motion", "models", "registered", "reproj px", "centre RMSE mm (Sim3)", "axis err deg (Sim3 median)", "|rel-rot err| median/max deg", "straightness est vs GT", "D_CE err mm (dense, indep.)", "D_CE bias mm", "1-mm slabs measured", "pipeline %obstruction", "GT %obstruction"]
keys = ["scenario", "event", "n_models", "registered", "reproj", "centre_rmse", "axis_med", "relrot", "straight", "calibre_err", "calibre_bias", "slabs", "pipe_pct", "gt_pct"]
print("\n| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
for r in rows: print("| " + " | ".join(str(r[k]) for k in keys) + " |")
json.dump(rows, open(f"{ROOT}/runs/rigid_summary.json", "w"), indent=1)
