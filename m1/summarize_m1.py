"""Collect runs/m1_*/m1_result.json into markdown tables for docs/M1_RESULTS.md.
Usage: python m1/summarize_m1.py runs/m1_static runs/m1_breathing runs/m1_malacia runs/m1_collapse runs/m1_collapse_w1 runs/m1_collapse_w4 ..."""
import sys, os, json
rows = []
for d in sys.argv[1:]:
    f = f"{d}/m1_result.json"
    if not os.path.exists(f): print(f"(missing {f})"); continue
    r = json.load(open(f)); e = r.get("event_station", {})
    def pct(x): return f"{100 * x:+.1f} %" if x is not None else "—"
    rows.append(dict(run=os.path.basename(d), window=r["window"], frames=f"{r['frames_registered']}/{r['frames_total']}", cells=r["cells"], dfrac=f"{100 * r.get('d_estimated_fraction', float('nan')):.0f} %",
                     rigid=f"{pct(r.get('rigid_rel_err_median'))} / {100 * r.get('rigid_rel_err_p90_abs', float('nan')):.1f} %",
                     prior=f"{pct(r.get('csa_model_smoothed_rel_err_median'))} / {100 * r.get('csa_model_smoothed_rel_err_p90_abs', float('nan')):.1f} %",
                     free=f"{pct(r.get('csa_free_rel_err_median'))} / {100 * r.get('csa_free_rel_err_p90_abs', float('nan')):.1f} %",
                     dmm=f"{r.get('membrane_d_smoothed_err_mm_median', float('nan')):.2f} / {r.get('membrane_d_smoothed_err_mm_p90', float('nan')):.2f}",
                     cart=f"{r.get('cartilage_radius_dev_mm_median', float('nan')):.3f} / {r.get('cartilage_radius_dev_mm_p90', float('nan')):.3f}",
                     ev=(f"z={e['z_mm']:.0f}: truth {100 * e['gt_reduction_seen']:.0f} %, M1 {100 * e.get('est_smoothed_reduction', e['est_model_reduction']):.0f} % @ t={e['est_model_min_t'] if 'est_model_min_t' in e else float('nan')}; "
                         f"RMSE M1 {e.get('rmse_smoothed', e['rmse_model']):.1f} vs rigid {e.get('rmse_rigid', float('nan')):.1f} mm²; event-window coverage {e.get('event_window_frames_estimated', '?')}/{e.get('event_window_frames_seen', '?')}") if e else "—"))
hdr = ["run", "window", "frames", "cells", "d estimated", "rigid map err median / p90", "M1 prior err median / p90", "M1 model-free err median / p90", "d err mm median / p90", "cartilage dev mm median / p90", "event station"]
keys = ["run", "window", "frames", "cells", "dfrac", "rigid", "prior", "free", "dmm", "cart", "ev"]
print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
for r in rows: print("| " + " | ".join(str(r[k]) for k in keys) + " |")
