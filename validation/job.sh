#!/bin/bash
# Slurm wrapper for the upstream-vs-new validation (NERSC Perlmutter CPU node shown; adapt for your site).
# Usage: sbatch -A <account> validation/job.sh       (env: N points, UPSTREAM dir, ENV_ACTIVATE script)
#SBATCH -q debug -C cpu -N 1 -t 00:30:00 -J folpsfkpt_validation
#SBATCH -o folpsfkpt_validation_%j.log
set -eu
UPSTREAM=${UPSTREAM:-$SCRATCH/folpsfkpt_validation}   # holds FolpsD_upstream/ and fkptjax_upstream/ checkouts
[ -n "${ENV_ACTIVATE:-}" ] && source "$ENV_ACTIVATE"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
XLA_THREADS=10 validation/run_all.sh "$UPSTREAM/FolpsD_upstream" "$UPSTREAM/fkptjax_upstream" "${N:-100}"
python validation/compare.py
