#!/bin/bash
# Run use cases through both implementations, one process each, at most MAXJOBS at a time.
# Usage: validation/run_all.sh UPSTREAM_FOLPSD UPSTREAM_FKPTJAX [N] [CASE ...]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
UP_FOLPS=$1; UP_FKPT=$2; N=${3:-}; shift 3 2>/dev/null || shift $#
CASES=${*:-$(python -c "import sys; sys.path.insert(0, '$HERE'); from usecases import CASES; print(*CASES)")}
mkdir -p "$HERE/data/logs"
export HERE UP_FOLPS UP_FKPT N PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=${XLA_THREADS:-4}"
for c in $CASES; do echo "upstream $c"; echo "new $c"; done | xargs -P "${MAXJOBS:-10}" -L 1 bash -c '
  if [ "$0" = upstream ]; then export PYTHONPATH=$UP_FOLPS:$UP_FKPT/src; fi
  python "$HERE/usecases.py" "$0" "$1" $N > "$HERE/data/logs/$0_$1.log" 2>&1'
for c in $CASES; do grep -h "done" "$HERE/data/logs/upstream_$c.log" "$HERE/data/logs/new_$c.log"; done
