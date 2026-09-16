#!/bin/bash
# M2 estimation loop: compensated sweep with the current d(z,t) estimate -> M1 estimator -> new d -> ...
# usage: bash m2/iterate.sh <scenario> <n_iter> [gpu] [init_grid]
#   scenario: collapse | breathing | malacia | static ; init_grid defaults to the COLMAP-based M1 estimate runs/m1_<sc>/m1_grid.npz
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/home/mi3dr/.conda/envs/colmap-cuda/lib:${LD_LIBRARY_PATH:-}
export BRONCHO_COLMAP=${BRONCHO_COLMAP:-/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap}
PY=${BRONCHO_PY:-/home/mi3dr/.conda/envs/depth-eval/bin/python}
sc=$1; n=$2; gpu=${3:-0}; init=${4:-runs/m1_$sc/m1_grid.npz}
prev=$init
for it in $(seq 1 $n); do
  out=runs/m2_${sc}_it$it
  echo "[m2] $(date +%T) $sc iteration $it: compensating with $prev"
  $PY m2/mc_sweep.py runs/rigid_$sc runs/synth_${sc}_30 --out $out --deform $prev --gpu $gpu 2>&1 | grep -E "^done|Traceback|Error"
  $PY m1/windowed_depth.py runs/rigid_$sc runs/synth_${sc}_30 --out ${out}_eval --window 2 --dense-from $out/dense 2>&1 | grep -v findfont | grep -E "csa_model_smoothed_rel_err_(median|p90)|membrane_d_smoothed_err|d_estimated|event station|Traceback" | sed -E 's/"frames_seen.*//' | cut -c1-360 | sed "s/^/[$sc it$it] /"
  prev=${out}_eval/m1_grid.npz
done
echo "[m2] $(date +%T) $sc done after $n iterations"
