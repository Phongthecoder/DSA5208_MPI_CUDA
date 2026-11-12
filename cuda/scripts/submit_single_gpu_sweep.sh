#!/bin/bash
set -euo pipefail
if (( $# )); then printf 'Usage: bash cuda/scripts/submit_single_gpu_sweep.sh\n' >&2; exit 2; fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
qstat -Q auto_free >/dev/null
mkdir -p cuda/results
submission="$(mktemp -d "$PWD/cuda/results/vanda-size-submit.XXXXXXXX")"
select='1:ncpus=2:mpiprocs=1:ompthreads=1:ngpus=1:gpu_model=A40:mem=8gb'
job="$(qsub -q auto_free -l "select=$select" -N dsa5208_a40_sizes \
    -o "$submission/pbs.log" cuda/scripts/vanda_single_gpu_sweep.pbs)"
printf 'gpus,submit_queue,mpi_stack,job_id\n1,auto_free,hpcx,%s\n' "$job" |
    tee "$submission/jobs.csv"
printf 'Submission records: %s\n' "$submission" >&2
