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
bash mpi/scripts/submit_vanda.sh
```

The script submits one `auto_free` job per rank count
(`1 2 4 8 16 32 64 128 256`). You can pass individual counts as arguments.
The allocation uses one rank per core, at most 32 ranks per host, and eight
hosts at 256 ranks. The job records host slots, core binding, compiler, and MPI
version. Each job saves its files under `mpi/results/<job_id>/`, and the
submission directory contains `jobs.csv` and PBS logs.

The workload uses 256 time steps, data seed 42, filter seed 1729, one short
warm-up, and three measured repetitions. Strong scaling holds the total at
2^22 particles. Weak scaling holds 2^14 particles per rank. Both use ESS 0.5.
The strong sweep also compares the two resamplers. A separate resampling
experiment uses 2^20 particles with uniform weights and with 90% of mass in
the first 1% of particles. The ESS sweep uses 32 ranks and 2^20 particles with
thresholds 0.25, 0.5, and 0.75, plus resampling every step.

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
while IFS=, read -r ranks nodes job_id; do
    test "$ranks" = ranks && continue
    filter_csv+=("results/$job_id/mpi_runs.csv")
    if (( ranks > 1 )); then
        resampling_csv+=("results/$job_id/resampling_runs.csv")
    fi
done < "$JOB_TABLE"
python3 scripts/summarize_mpi.py "${filter_csv[@]}" --resampling "${resampling_csv[@]}" \
  --output-dir results/vanda-summary
```

Add `--plot-title "CPU MPI on Vanda"` when running the summary command with
Matplotlib installed. Strong speedup and both efficiencies use the corresponding
one-rank run of the same MPI binary.

## Local benchmark

The following commands run the same experiment design on one host with up
to eight ranks. The strong and ESS workloads are reduced for the local test.
On macOS with MPICH over libfabric, set `FI_TCP_IFACE=lo0 HYDRA_IFACE=lo0`
in the local environment to keep the one-host runs on loopback during VPN
interface changes.

```sh
python3 scripts/benchmark_mpi.py --binary build/mpi_pf \
  --resampling-binary build/resampling_bench --output-dir results/local \
  --strong-particles 1048576 --ess-particles 262144 --ess-rank 8 --expected-nodes 1
python3 scripts/summarize_mpi.py results/local/mpi_runs.csv \
  --resampling results/local/resampling_runs.csv --output-dir results/local \
  --plot-title "Local CPU MPI benchmark"
```

The [local results](benchmarks/local/) are from one Apple M5 Mac with 10 CPU
cores, 32 GiB of memory, Apple Clang 17, and MPICH 4.3.2. Ranks were not pinned.
Times below are medians of three measurements at 256 steps. The weak workload
uses 16,384 particles per rank.

| MPI ranks | Strong: distributed (s) | Strong: centralized (s) | Strong speedup: distributed | Weak: distributed (s) | Weak efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.613 | shared baseline | 1.00× | 0.203 | 100.0% |
| 2 | 6.846 | 7.304 | 1.84× | 0.208 | 97.7% |
| 4 | 4.237 | 5.033 | 2.98× | 0.278 | 73.0% |
| 8 | 3.368 | 4.364 | 3.74× | 0.444 | 45.7% |

At eight ranks, strong efficiency was 46.8% with distributed resampling.
The highest share of rank time spent inside MPI calls was 20.9% for that run
configuration and 44.5% for the centralized equivalent (medians).
In the isolated resampling test, distributed resampling took 0.670 ms with
uniform weights and 1.088 ms with skewed weights. Skew raised its logical
off-rank payload from zero to 6.90 MiB per resample and the busiest rank packed
7.29 times the average number of particles. The centralized sampler took
2.182 ms and 2.224 ms, respectively, with 15.75 MiB of payload in both cases.

On the fixed observation sequence in the eight-rank ESS test, threshold 0.5
resampled on 79 of 256 steps and took 0.881 s with a position RMSE of 8.376 m.
Resampling every step took 1.008 s with an RMSE of 8.351 m. The threshold
reduced runtime by 12.6% on this workload; the RMSE difference is descriptive for
this sequence. The [validation log](benchmarks/local/validation.txt) shows
that the methods agree on the deterministic tests and that the MPI filter
approaches the Kalman reference as the particle count increases.

These are one-host results. The Vanda submission on 9 November 2026 was
blocked by a PBS connection error, so cross-node scaling is pending.

## AI use

OpenAI Codex assisted with implementation, test development, benchmark
automation, and figures drawn from measured output.
