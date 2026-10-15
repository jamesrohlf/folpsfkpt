#!/bin/bash
# Run every use case through both implementations, one process each, in parallel.
# Usage: validation/run_all.sh UPSTREAM_FOLPSD UPSTREAM_FKPTJAX [N] [CASE ...]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
UP_FOLPS=$1; UP_FKPT=$2; N=${3:-}; shift 3 2>/dev/null || shift $#
CASES=${*:-$(python -c "import sys; sys.path.insert(0, '$HERE'); from usecases import CASES; print(*CASES)")}
mkdir -p "$HERE/data/logs"
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=${XLA_THREADS:-4}"
for c in $CASES; do
  PYTHONPATH=$UP_FOLPS:$UP_FKPT/src python "$HERE/usecases.py" upstream "$c" $N > "$HERE/data/logs/upstream_$c.log" 2>&1 &
  python "$HERE/usecases.py" new "$c" $N > "$HERE/data/logs/new_$c.log" 2>&1 &
done
wait
grep -h "done" "$HERE"/data/logs/*.log
