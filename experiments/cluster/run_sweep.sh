#!/usr/bin/env bash
#
# run_sweep.sh — run a sweep level by level in one shell. For a tmux session on a box with
# no scheduler; see sweep.sbatch for the Slurm array version.
#
#     tmux new -s sherpa
#     ./experiments/cluster/run_sweep.sh
#     # detach with C-b d, reattach with: tmux attach -t sherpa
#
# Override anything:
#     SEEDS=25 ./experiments/cluster/run_sweep.sh
#     VALS_ALL="0.05 0.2" SEEDS=100 ./experiments/cluster/run_sweep.sh
#     KEY=thruster_sigma_pct VALS_ALL="0.7 2.0" ./experiments/cluster/run_sweep.sh
#
# Each level writes its own log under logs/, and every rollout checkpoints as it completes,
# so killing this mid-run keeps everything already flown. Re-running skips finished kernels,
# policies and rollouts, so it fills gaps rather than redoing work.

set -uo pipefail          # NOT -e: one level failing should not abandon the rest

KEY="${KEY:-sigma_nav_km}"
# Eight levels, 0.0 to 1.0. The grid skips 0.5 on purpose: solve and rollout cost are not
# monotone in sigma, and at 0.5 the policy came out 435 MB against 5 MB at 0.3, so one POMDP
# rollout took ~15 min instead of ~10 s — every belief update is an argmax over every alpha
# vector. Add it back only with a seed count you are willing to wait for.
VALS_ALL="${VALS_ALL:-0.0 0.015 0.05 0.1 0.2 0.3 0.4 1.0}"
SEEDS="${SEEDS:-100}"
DAYS="${DAYS:-30}"
PLUME="${PLUME:-1.5}"

cd "$(dirname "$0")/../.."     # repo root, wherever this was invoked from
mkdir -p logs

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="logs/sweep_${STAMP}_summary.log"

{
  echo "sweep $KEY = $VALS_ALL"
  echo "$SEEDS seeds, $DAYS d, plume $PLUME"
  echo "host $(hostname), started $(date)"
  echo
} | tee "$SUMMARY"

t_all=$SECONDS
for V in $VALS_ALL; do
  LOG="logs/${KEY}=${V}_${STAMP}.log"
  echo "=== $KEY = $V  ->  $LOG  ($(date +%H:%M:%S)) ===" | tee -a "$SUMMARY"
  t0=$SECONDS

  KEY="$KEY" VALS="$V" SEEDS="$SEEDS" DAYS="$DAYS" PLUME="$PLUME" \
    julia --project=experiments -t auto experiments/sweep.jl > "$LOG" 2>&1
  rc=$?

  mins=$(( (SECONDS - t0) / 60 ))
  n=$(find "artifacts/sweeps/$KEY" -path "*${KEY}=${V}*" -name '*.jld2' 2>/dev/null | wc -l)
  if [ $rc -eq 0 ]; then
    echo "    done in ${mins} min, ${n} rollouts" | tee -a "$SUMMARY"
  else
    # Keep going: the levels are independent, and a re-run resumes this one from disk.
    echo "    FAILED rc=$rc after ${mins} min, ${n} rollouts — see $LOG" | tee -a "$SUMMARY"
    tail -5 "$LOG" | sed 's/^/      /' | tee -a "$SUMMARY"
  fi
done

echo | tee -a "$SUMMARY"
echo "all levels finished in $(( (SECONDS - t_all) / 60 )) min at $(date)" | tee -a "$SUMMARY"
echo "checkpoints: $(find "artifacts/sweeps/$KEY" -name '*.jld2' 2>/dev/null | wc -l)" | tee -a "$SUMMARY"
echo "analyse with: load_sweep(\"artifacts/sweeps/$KEY\")" | tee -a "$SUMMARY"