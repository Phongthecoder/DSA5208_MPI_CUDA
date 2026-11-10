#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

gpus="${1:-2}"
if (( $# > 1 )) || [[ "$gpus" != 1 && "$gpus" != 2 ]]; then
    printf 'Usage: [PBS_PROJECT=CODE] bash cuda/scripts/submit_vanda.sh [1|2]\n' >&2
    exit 2
fi

stack="${CUDA_MPI_STACK:-hpcx}"
validate_only="${CUDA_BENCH_VALIDATE_ONLY:-0}"
project="${PBS_PROJECT:-}"
queue=auto_free
case "$stack" in hpcx|stock) ;; *) printf 'Unknown MPI stack %s\n' "$stack" >&2; exit 2 ;; esac
case "$validate_only" in 0|1) ;; *) printf 'CUDA_BENCH_VALIDATE_ONLY must be 0 or 1.\n' >&2; exit 2 ;; esac

queue_args=(-q "$queue")
if [[ -n "$project" ]]; then
    queue=auto
    queue_args=(-q "$queue" -P "$project")
fi
qstat -Q "$queue" >/dev/null
mkdir -p cuda/results
submission="$(mktemp -d "$PWD/cuda/results/vanda-submit.XXXXXXXX")"
select="1:ncpus=$((2 * gpus)):mpiprocs=$gpus:ompthreads=1:ngpus=$gpus:gpu_model=A40:mem=$((8 * gpus))gb"
job="$(qsub "${queue_args[@]}" -l "select=$select" -N "dsa5208_cuda_a40_$gpus" \
       -v "CUDA_BENCH_GPUS=$gpus,CUDA_MPI_STACK=$stack,CUDA_BENCH_VALIDATE_ONLY=$validate_only" \
       -o "$submission/pbs.log" cuda/scripts/vanda_cuda.pbs)"
printf 'gpus,submit_queue,mpi_stack,job_id\n%s,%s,%s,%s\n' "$gpus" "$queue" "$stack" "$job" |
    tee "$submission/jobs.csv"
printf 'Submission records: %s\n' "$submission" >&2
