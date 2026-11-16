# CPU MPI particle filter

This implementation tracks a 2D target using the constant-velocity model and
linear or range-bearing measurements from the serial reference. It supports
centralized and distributed global systematic resampling. [Tracking model](../tracking_problem.md).

## Build and validate

Use a C++17 compiler, MPI with `mpicxx` and `mpiexec`, and `make`. The benchmark
runner requires Python 3.7 or newer. From the checkout root, run

```sh
cd mpi
make
make check
mpiexec -n 4 ./build/mpi_pf --particles 65536 --steps 256 --resampler distributed --resample 0.5
```

For the Anaconda MPICH wrapper on macOS, use `MPICH_CXX=c++ make` when it needs
the system C++ compiler. `make check` uses 1, 3, and 8 ranks by default.
`CHECK_RANKS` overrides these counts when the MPI allocation has fewer slots.
`--trajectory FILE` writes the same columns as the serial trajectory CSV.
`--resample always` resamples at every step, and `--resample 0` disables it.

The validation suite checks the Philox known-answer vector, exact systematic
ancestry and payload on small examples, and agreement with a one-rank MPI
reference. Cases include uneven particle counts, empty ranks, zero weights,
and a sample on a CDF boundary. Both methods are compared step by step for ESS,
means, covariance, and log likelihood. A separate Kalman comparison uses six
linear-Gaussian sequences at 512 and 65,536 particles. Six nonlinear sequences
compare MPI, the serial filter, and the raw measurements against the same truth.

## Algorithm and communication

Rank zero simulates and broadcasts the observations. The first observation
follows the first propagation at `t=1`, using the same convention as the serial
implementation. Each rank owns a contiguous block of global particle slots.
The block sizes differ by at most one. Philox4x32-10 and Box–Muller map the
filter seed, step, and global slot to the Gaussian draws. A separate Philox key
selects the shared systematic offset. The MPI RNG stream is independent of the
serial RNG stream. [Random123](https://github.com/DEShawResearch/random123).

An `MPI_Allreduce` finds the maximum log weight. A second reduction combines
the shifted weight sum, sum of squared weights, and four weighted state sums.
Each rank can then compute the same ESS, posterior mean, and log likelihood
increment. Validation can also reduce covariance about the global mean.

- The centralized method gathers states and shifted weights to rank zero,
  applies the serial systematic sampler, and scatters the resulting slots.
- The distributed method uses `MPI_Allgather` on one mass per rank and builds
  shared CDF boundaries in rank order. Each rank computes the slots selected
  by its local CDF interval, packs those slots in order, exchanges counts with
  `MPI_Alltoall`, and sends particles with `MPI_Alltoallv`. Receive displacements
  preserve global slot order and the original block sizes. Shared boundaries
  give every slot exactly one owner, including when some ranks have zero mass.

The systematic points are `(k + u) / N`. A point belongs to the first cumulative
weight strictly above it. After resampling, all log weights are reset to
`-log(N)`. Floating-point sums can group weights differently across ranks,
so numerical comparisons use tolerances.

## Vanda experiments

Submit from a Vanda checkout that contains both `mpi/` and `serial/`.

```sh
# Full sweep, ranks 1 to 256
bash mpi/scripts/submit_vanda.sh --project PROJECT_CODE

# Repeat just the 64-rank experiment
bash mpi/scripts/submit_vanda.sh --project PROJECT_CODE 64
```

Pass your Vanda PBS project code with `--project` or set `PBS_PROJECT`.
The script submits one job per rank count to the `auto` project queue, which
routes them to `batch_cpu`. Without a project code, jobs of up to 128 ranks
go to `auto_free`, whose node pool has few Platinum hosts. The
[default Platinum layouts](scripts/vanda_platinum_layouts.tsv) match the
measured configurations: one node up to 32 ranks, four hosts with 16 ranks
each at 64, and the same eight hosts with 16 and 32 ranks each at 128 and
256. The layouts pin named hosts, so PBS queues a job until those hosts are
free; edit the layout file to use other hosts. Before choosing a host, check
`pbsnodes -v "cn-XXX[0]"` for the comment "Unable to clean up one or more
cgroups", which makes every job on that host fail to start.

Use `--dry-run` to print the allocation without contacting PBS. To choose a
different set of hosts, pass `--layouts FILE`; the file contains total
ranks, PBS host, and ranks on that host, separated by tabs. Every job checks
its actual host and rank layout against the request and probes every host
for the exact Intel Xeon Platinum 8452Y model before building or timing. It uses
one MPI rank per core and runs `make check` at the allocated rank count.
Use `--validate-only` to run these checks without benchmarking.

Each job saves its CPU models, layout, core bindings, validation log and
timings under `mpi/results/<job_id>/`. The submission directory contains
`plan.tsv`, per-job layout files, `jobs.csv` and PBS logs.

The workload uses 256 time steps, data seed 42, filter seed 1729, one short
warm-up, and five measured repetitions. Strong scaling holds the total at
2^22 particles. Weak scaling holds 2^14 particles per rank. Both use ESS 0.5.
The strong sweep also compares the two resamplers from two ranks onward.
A separate resampling experiment uses 2^20 particles with uniform weights
and with 90% of mass in the first 1% of particles. The ESS sweep uses 32
ranks and 2^20 particles with thresholds 0.25, 0.5, and 0.75, plus
resampling every step.

The filter timing is the maximum elapsed time over ranks, from particle
allocation through the last filter step. Workflow time also covers observation
simulation, broadcast and output. The Python driver records the launch wall
time including MPI startup. The maximum fraction of rank time in MPI includes
waiting inside the collectives. Each phase maximum can come from a different
rank. The resampling payload counts logical off-rank state and weight bytes,
with collective metadata and transport overhead outside that count. The
packing ratio divides the most particles packed by any rank in any step by
the average particles per rank.

After the jobs finish, replace `JOB_TABLE` below with the path to the printed
`jobs.csv`. Use one submission table so retries and other sweeps do not
contribute duplicate runs.

```bash
cd mpi
JOB_TABLE=results/vanda-submit.XXXXXXXX/jobs.csv
filter_csv=()
resampling_csv=()
while IFS=, read -r ranks nodes job_id hosts queue; do
    test "$ranks" = ranks && continue
    filter_csv+=("results/$job_id/mpi_runs.csv")
    if [[ -f "results/$job_id/resampling_runs.csv" ]]; then
        resampling_csv+=("results/$job_id/resampling_runs.csv")
    fi
done < "$JOB_TABLE"
summary_inputs=("${filter_csv[@]}")
if (( ${#resampling_csv[@]} )); then
    summary_inputs+=(--resampling "${resampling_csv[@]}")
fi
python3 scripts/summarize_mpi.py "${summary_inputs[@]}" --output-dir results/vanda-summary
```

Add `--plot-title "CPU MPI on Vanda"` when running the summary command with
Matplotlib installed. Strong speedup and both efficiencies use the corresponding
one-rank run of the same MPI binary.

## Local benchmark

The following commands run the same experiment design on one host with up
to eight ranks. The strong and ESS workloads are reduced for the local test.
On macOS with MPICH over libfabric, set `FI_TCP_IFACE=lo0 HYDRA_IFACE=lo0`
in the local environment to keep the one-host runs on loopback during VPN
interface changes. Results are written to `results/local/`.

```sh
python3 scripts/benchmark_mpi.py --binary build/mpi_pf \
  --resampling-binary build/resampling_bench --output-dir results/local \
  --strong-particles 1048576 --ess-particles 262144 --ess-rank 8 --expected-nodes 1
python3 scripts/summarize_mpi.py results/local/mpi_runs.csv \
  --resampling results/local/resampling_runs.csv --output-dir results/local \
  --plot-title "Local CPU MPI benchmark"
```

The [Vanda Platinum results](benchmarks/vanda-platinum-5-repeats/) are the
ones used in the report. They cover 1 to 256 MPI ranks on Xeon Platinum 8452Y
nodes, with five measurements per setting and validation on every allocation;
`jobs.csv` lists the nine jobs, `jobs/<job_id>/` holds each job's raw output,
and `figures/` holds the report figures. Rebuild the figures from the
repository root with

```sh
python3 mpi/scripts/plot_report_figures.py mpi/benchmarks/vanda-platinum-5-repeats
```

## AI use

OpenAI Codex assisted with implementation, test development, benchmark
automation, and figures drawn from measured output.
