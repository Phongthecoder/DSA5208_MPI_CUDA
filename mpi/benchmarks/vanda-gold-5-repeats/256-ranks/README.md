# Vanda 256-rank Xeon Gold follow-up

PBS job `1536089.stdct-mgmt-02` ran on 10 November 2026 with Open MPI
4.1.8. The allocation used eight Xeon Gold 6438N hosts and 256 distinct
cores. [CPU probes](cpu_models.tsv), [core bindings](rank_bindings.txt),
[host layout](layout.tsv) and [PBS job details](pbs_job.tsv) record the
allocation. The [validation log](validation.txt) shows that all 256-rank
checks passed.

- [Filter timings](mpi_runs.csv): 15 rows, five each for strong distributed,
  weak distributed and strong centralized. At 256 ranks, both scaling
  workloads use 4,194,304 total particles.
- [Resampling timings](resampling_runs.csv): 20 rows, five for each method
  and weight shape, with 20 resampling iterations per measurement.
- [Timing summary](summary.tsv): 256-rank medians and ranges with the
  corresponding 128-rank medians. Efficiencies use the original one-rank
  distributed baselines.
- [128 versus 256 analysis](../README.md#128-versus-256-ranks) and
  [scaling figure](mpi_scaling.png).

The [original PBS script and benchmark runner](reproduce/) are archived
with the [source commit](source_commit.txt). To submit the same layout
from the repository root on Vanda using the maintained scripts, run

~~~bash
bash mpi/scripts/submit_vanda.sh --project PROJECT_CODE 256
~~~

Replace `PROJECT_CODE` with your Vanda PBS project code. The script checks
the allocated layout and CPU models and validates the MPI filter before
benchmarking.
