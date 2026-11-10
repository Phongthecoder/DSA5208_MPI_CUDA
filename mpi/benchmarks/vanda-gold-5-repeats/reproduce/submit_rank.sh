#!/bin/bash
set -euo pipefail

batch_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$batch_dir/../.."
ranks="${1:?Usage: submit_rank.sh RANKS HOST...}"
shift
case "$ranks" in
    1|2|4|8|16|32|64|128) ;;
    *) printf 'Unsupported rank count: %s\n' "$ranks" >&2; exit 2 ;;
esac
per_node="${MPI_RANKS_PER_NODE:-$((ranks <= 32 ? ranks : 32))}"
if [[ ! "$per_node" =~ ^[1-9][0-9]?$ ]] || (( per_node > 32 || ranks % per_node != 0 )); then
    printf 'MPI_RANKS_PER_NODE must be a divisor of the rank count and at most 32.\n' >&2
    exit 2
fi
nodes=$((ranks / per_node))
if (( $# != nodes )) || [[ "$(printf '%s\n' "$@" | sort -u | wc -l)" -ne "$nodes" ]]; then
    printf 'Expected %d distinct qualified hosts.\n' "$nodes" >&2
    exit 2
fi

chunks=()
for host in "$@"; do
    if ! grep -Fxq "$host" "$batch_dir/qualified_hosts.txt"; then
        printf 'Host %s has not passed the CPU model probe.\n' "$host" >&2
        exit 2
    fi
    chunks+=("1:ncpus=$per_node:mpiprocs=$per_node:ompthreads=1:mem=8gb:node_pool=cpu_free:host=$host")
done
select_spec="$(IFS=+; printf '%s' "${chunks[*]}")"
dependencies=()
if [[ -n "${PBS_DEPENDENCY:-}" ]]; then
    dependencies=(-W "depend=$PBS_DEPENDENCY")
fi

job_table="$batch_dir/jobs.csv"
if [[ ! -f "$job_table" ]]; then
    printf 'ranks,nodes,job_id,hosts\n' > "$job_table"
fi
if awk -F, -v n="$ranks" '$1 == n {found=1} END {exit !found}' "$job_table"; then
    printf 'Rank %s already appears in %s\n' "$ranks" "$job_table" >&2
    exit 1
fi

job_id="$(qsub -q auto_free -N "dsa5208_g5_${batch_dir##*.}_p$ranks" \
    -v "MPI_RANKS=$ranks,MPI_RANKS_PER_NODE=$per_node,MPI_CPU_MODEL_FILE=$batch_dir/expected_cpu_model.txt" \
    -l "select=$select_spec" -l place=scatter \
    "${dependencies[@]}" -o "$batch_dir/p$ranks.log" "$batch_dir/vanda_mpi_same_cpu.pbs")"
host_field="$(IFS=+; printf '%s' "$*")"
printf '%s,%s,%s,%s\n' "$ranks" "$nodes" "$job_id" "$host_field" | tee -a "$job_table"
