# Vanda MPI benchmark on Xeon Gold 6438N

On 10 November 2026, we ran eight MPI rank counts from 1 to 128 on Vanda
with Open MPI 4.1.8 and the `auto_free` submission queue. Each job
confirmed the CPU model on every node before validation and timing. We used
one MPI rank per core, one node through 32 ranks, four nodes at 64 ranks,
and eight nodes at 128 ranks. The MPI source was `8c6646967ece`.

Each filter setting has five measured runs at 256 steps. Strong scaling
uses 4,194,304 total particles and weak scaling uses 16,384 per rank. Both
use distributed resampling at an ESS threshold of 0.5. The range-bearing
model uses data seed 42 and filter seed 1729. Filter time is the maximum
elapsed time across MPI ranks. Speedup and efficiency use the one-rank
median. Resampling and ESS comparisons use 1,048,576 particles. Each
resampling sample averages 20 iterations, with five samples per setting.

| Ranks | Nodes | Strong median (s) | Strong range (s) | Strong speedup | Weak median (s) | Weak efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 210.968 | 210.212–213.075 | 1.00× | 0.815 | 100.0% |
| 2 | 1 | 89.693 | 87.848–89.987 | 2.35× | 0.982 | 83.0% |
| 4 | 1 | 47.262 | 45.162–48.006 | 4.46× | 0.760 | 107.2% |
| 8 | 1 | 18.083 | 18.056–19.116 | 11.67× | 0.566 | 144.2% |
| 16 | 1 | 11.391 | 11.302–12.185 | 18.52× | 0.761 | 107.2% |
| 32 | 1 | 5.769 | 5.767–6.592 | 36.57× | 0.770 | 105.9% |
| 64 | 4 | 3.367 | 3.352–3.373 | 62.66× | 1.218 | 67.0% |
| 128 | 8 | 1.722 | 1.709–1.769 | 122.51× | 1.066 | 76.5% |

The single-rank baseline was 210.968 s (210.212–213.075 s across five runs). Two ranks took
89.693 s (87.848–89.987 s across five runs), for 2.352× speedup.
Particle propagation and weight updates took 193.341 s at one rank
and 82.192 s at two ranks. The runs used separate Gold nodes
(IN-007 and IN-008) in the shared queue.

At 128 ranks, strong efficiency was 95.7% and weak efficiency was
76.5%. With 4,194,304 particles on the same eight nodes, centralized
resampling took 16.716 s and distributed resampling took 1.722 s for
the filter. All eight correctness checks passed. The files contain 135
filter measurements and 140 resampling measurements.

## Data and reproduction

- [Filter measurements](mpi_runs.csv) and [filter summary](mpi_summary.csv)
- [Resampling measurements](resampling_runs.csv) and [resampling summary](resampling_summary.csv)
- [Scaling plot](mpi_scaling.png) and [resampling and ESS plot](mpi_resampling.png)
- [CPU by node](cpu_model_by_node.tsv), [validation results](validation_summary.tsv), and [validation output](validation.txt)

From the repository root on Vanda, copy the recorded scripts into a fresh
results directory and submit the recorded host layout. PBS may queue jobs
while those hosts are occupied.

~~~bash
bench=mpi/benchmarks/vanda-gold-5-repeats
mkdir -p mpi/results
run_dir=$(mktemp -d "$PWD/mpi/results/vanda-gold5-replay.XXXXXXXX")
cp "$bench"/reproduce/* "$run_dir/"
while IFS=, read -r ranks nodes previous_job hosts; do
    [[ "$ranks" == ranks ]] && continue
    IFS=+ read -r -a selected_hosts <<< "$hosts"
    MPI_RANKS_PER_NODE="$((ranks / nodes))" bash "$run_dir/submit_rank.sh" "$ranks" "${selected_hosts[@]}"
done < "$bench/jobs.csv"
~~~

The replay records PBS jobs in `$run_dir/jobs.csv`. Each job writes
measurements to `mpi/results/<job_id>/`. To rebuild the plots from
the published measurements, run

~~~bash
python3 mpi/scripts/summarize_mpi.py "$bench/mpi_runs.csv" \
  --resampling "$bench/resampling_runs.csv" --output-dir mpi/results/vanda-gold5-summary \
  --plot-title "CPU MPI on Vanda | Xeon Gold 6438N | Five repeats"
~~~
