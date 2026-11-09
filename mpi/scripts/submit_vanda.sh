#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mpi_dir="$(pwd -P)"

if (( $# == 0 )); then
    set -- 1 2 4 8 16 32 64 128 256
fi
for ranks in "$@"; do
    if [[ ! "$ranks" =~ ^[1-9][0-9]{0,2}$ ]] ||
        (( ranks > 256 || (ranks > 32 && ranks % 32 != 0) )); then
        printf 'Invalid rank count %s. Use 1 to 32 or a multiple of 32 through 256.\n' "$ranks" >&2
        exit 1
    fi
done

qstat -Q auto_free >/dev/null
mkdir -p results
submit_dir="$(mktemp -d "$mpi_dir/results/vanda-submit.XXXXXXXX")"
printf 'Submission records: %s\n' "$submit_dir" >&2
printf 'ranks,nodes,job_id\n' | tee "$submit_dir/jobs.csv"
for ranks in "$@"; do
    per_node="$ranks"
    if (( per_node > 32 )); then
        per_node=32
    fi
    nodes="$(( ranks / per_node ))"
    job_id="$(qsub -v "MPI_RANKS=$ranks" \
        -l "select=$nodes:ncpus=$per_node:mpiprocs=$per_node:ompthreads=1:mem=8gb" \
        -l place=scatter -N "dsa5208_mpi_p$ranks" -o "$submit_dir/p$ranks.log" \
        scripts/vanda_mpi.pbs)"
    printf '%s,%s,%s\n' "$ranks" "$nodes" "$job_id" | tee -a "$submit_dir/jobs.csv"
done
