#!/bin/bash
#SBATCH --job-name=sherpa-nav
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/nav_%A_%a.out
#SBATCH --error=logs/nav_%A_%a.out
#
# One navigation level per array task. Levels are independent and their checkpoints are
# keyed by value, so a task that dies loses only its own level and re-submitting the array
# fills gaps rather than redoing work.
#
#     sbatch --array=0-7 experiments/cluster/sweep.sbatch
#
# Override from the command line:
#     sbatch --array=0-7 --export=ALL,SEEDS=25 experiments/cluster/sweep.sbatch
#     sbatch --array=0-2 --export=ALL,KEY=thruster_sigma_pct,VALS_ALL="0.7 2.0 5.0" ...
#
# NOTE: 12 h is generous on purpose. Solve cost is not monotone in sigma — around
# sigma = 0.5 the policy grew to 435 MB and a single rollout took ~15 min, against ~10 s at
# sigma = 0.3. See experiments/cluster/README.md.

set -euo pipefail

KEY="${KEY:-sigma_nav_km}"
# Eight levels, 0.0 to 1.0. The grid skips 0.5 on purpose: solve and rollout cost are not
# monotone in sigma, and at 0.5 the policy came out 435 MB against 5 MB at 0.3, so one POMDP
# rollout took ~15 min instead of ~10 s — every belief update is an argmax over every alpha
# vector. Add it back only with a seed count you are willing to wait for.
VALS_ALL="${VALS_ALL:-0.0 0.015 0.05 0.1 0.2 0.3 0.4 1.0}"
SEEDS="${SEEDS:-100}"
DAYS="${DAYS:-30}"
PLUME="${PLUME:-1.5}"

# `set -- $VALS_ALL` splits on whitespace in any POSIX shell, so this does not depend on
# bash arrays if the cluster substitutes a different interpreter.
set -- $VALS_ALL
shift "$SLURM_ARRAY_TASK_ID"
V="$1"
[ -n "$V" ] || { echo "no value for array index $SLURM_ARRAY_TASK_ID in: $VALS_ALL" >&2; exit 1; }

mkdir -p logs
export PATH="$HOME/.juliaup/bin:$PATH"

echo "=== task $SLURM_ARRAY_TASK_ID: $KEY = $V, $SEEDS seeds, $DAYS d, plume $PLUME ==="
echo "=== host $(hostname), $(nproc) cpus, started $(date) ==="

# -t auto threads the calibration tree walk. The SARSOP solve is single-threaded regardless.
KEY="$KEY" VALS="$V" SEEDS="$SEEDS" DAYS="$DAYS" PLUME="$PLUME" \
  julia --project=experiments -t auto experiments/sweep.jl

echo "=== finished $(date) ==="