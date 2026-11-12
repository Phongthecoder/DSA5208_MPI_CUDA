#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit_vanda.sh [--project CODE] [--layouts FILE] [--dry-run]
                       [--validate-only] [RANKS ...]

The default Xeon Gold sweep is 1 2 4 8 16 32 64 128, plus 256 when a PBS
project is supplied with --project or PBS_PROJECT. The layout file has
three tab-separated columns: total ranks, PBS host, and ranks on that host.
EOF
}

project="${PBS_PROJECT:-}"
layouts=""
dry_run=0
validate_only=0
requested=()
while (( $# )); do
    case "$1" in
        --project|--layouts)
            if (( $# < 2 )) || [[ -z "$2" || "$2" == -* ]]; then
                usage >&2
                exit 2
            fi
            if [[ "$1" == --project ]]; then
                project="$2"
            else
                layouts="$2"
                [[ "$layouts" == /* ]] || layouts="$PWD/$layouts"
            fi
            shift
            ;;
        --dry-run) dry_run=1 ;;
        --validate-only) validate_only=1 ;;
        -h|--help) usage; exit 0 ;;
        -*) usage >&2; exit 2 ;;
        *) requested+=("$1") ;;
    esac
    shift
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mpi_dir="$(pwd -P)"
layouts="${layouts:-$mpi_dir/scripts/vanda_gold_layouts.tsv}"
if [[ ! -r "$layouts" ]]; then
    printf 'Cannot read Gold layouts: %s\n' "$layouts" >&2
    exit 2
fi

if (( ${#requested[@]} == 0 )); then
    requested=(1 2 4 8 16 32 64 128)
    if [[ -n "$project" ]]; then
        requested+=(256)
    fi
fi

selected_layouts=()
host_lists=()
node_counts=()
queues=()
select_specs=()
seen=" "
check_free=0
check_project=0
for ranks in "${requested[@]}"; do
    if [[ ! "$ranks" =~ ^[1-9][0-9]{0,2}$ ]] || (( ranks > 256 )) ||
        [[ "$seen" == *" $ranks "* ]]; then
        printf 'Rank counts must be distinct integers from 1 to 256: %s\n' "$ranks" >&2
        exit 2
    fi
    seen+="$ranks "
    queue=auto_free
    if [[ -n "$project" ]] || (( ranks > 128 )); then
        if [[ ! "$project" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
            printf '%s ranks require a PBS project code (--project CODE or PBS_PROJECT).\n' "$ranks" >&2
            exit 2
        fi
        queue=auto
        check_project=1
    else
        check_free=1
    fi

    # Validate the full request before submitting any jobs.
    if ! layout="$(awk -F '\t' -v ranks="$ranks" '
        $1 == ranks {
            if (NF != 3 || $2 !~ /^[a-z0-9][a-z0-9-]*$/ ||
                $3 !~ /^[1-9][0-9]?$/ || $3 > 64 || seen[$2]++) {
                bad = 1
                next
            }
            print $2 "\t" $3
            total += $3
        }
        END {exit (bad || total != ranks)}
    ' "$layouts")"; then
        printf 'Invalid or missing Gold layout for %s ranks in %s\n' "$ranks" "$layouts" >&2
        exit 2
    fi
    chunks=()
    hosts=()
    while IFS=$'\t' read -r host slots; do
        mem_gb=8
        if [[ "$queue" == auto ]]; then
            # Match the per-core memory request of the recorded project job.
            mem_gb=$(( slots < 4 ? 8 : 2 * slots ))
        fi
        chunks+=("1:ncpus=$slots:mpiprocs=$slots:ompthreads=1:mem=${mem_gb}gb:host=$host")
        hosts+=("$host")
    done <<< "$layout"
    selected_layouts+=("$layout")
    host_lists+=("$(IFS=+; printf '%s' "${hosts[*]}")")
    node_counts+=("${#hosts[@]}")
    queues+=("$queue")
    select_specs+=("$(IFS=+; printf '%s' "${chunks[*]}")")
done

print_plan() {
    printf 'ranks\tnodes\tqueue\tselect\n'
    local i
    for i in "${!requested[@]}"; do
        printf '%s\t%s\t%s\t%s\n' "${requested[i]}" "${node_counts[i]}" "${queues[i]}" "${select_specs[i]}"
    done
}
if (( dry_run )); then
    print_plan
    exit 0
fi
if (( check_free )); then qstat -Q auto_free >/dev/null; fi
if (( check_project )); then qstat -Q auto >/dev/null; fi

mkdir -p results
submit_dir="$(mktemp -d "$mpi_dir/results/vanda-submit.XXXXXXXX")"
relative_dir="results/${submit_dir##*/}"
printf 'Submission records: %s\n' "$submit_dir" >&2
print_plan > "$submit_dir/plan.tsv"
printf 'ranks,nodes,job_id,hosts,queue\n' | tee "$submit_dir/jobs.csv"
for i in "${!requested[@]}"; do
    ranks="${requested[i]}"
    layout="$relative_dir/p$ranks.layout.tsv"
    printf '%s\n' "${selected_layouts[i]}" > "$layout"
    qsub_args=(-q "${queues[i]}" -l "select=${select_specs[i]}" -l place=scatter
               -v "MPI_RANKS=$ranks,MPI_GOLD_LAYOUT=$layout,MPI_VALIDATE_ONLY=$validate_only"
               -N "dsa5208_mpi_p$ranks" -o "$submit_dir/p$ranks.log")
    if [[ "${queues[i]}" == auto ]]; then
        qsub_args+=(-P "$project" -l walltime=01:00:00)
    fi
    job_id="$(qsub "${qsub_args[@]}" scripts/vanda_mpi.pbs)"
    printf '%s,%s,%s,%s,%s\n' "$ranks" "${node_counts[i]}" "$job_id" "${host_lists[i]}" "${queues[i]}" |
        tee -a "$submit_dir/jobs.csv"
done
