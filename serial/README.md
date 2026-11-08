# Serial C++ particle filter

The serial C++17 reference uses sequential Monte Carlo for 2D tracking under a
constant-velocity model. It supports Cartesian observations for validation and
noisy range and bearing for the main tracking problem. [Tracking specification](../tracking_problem.md).

## Run the serial reference

Use a C++17 compiler and `make` on Linux or macOS. The benchmark runner requires
Python 3.7 or newer. Start in the root of a checkout of this repository.

```sh
cd serial
make
make check
mkdir -p results
./build/serial_pf --model range-bearing --particles 8192 --steps 200 \
  --data-seed 42 --filter-seed 1729 --resample 0.5 \
  --trajectory results/range_bearing_track.csv
```

`serial_pf` writes a one-row CSV summary to stdout. It includes position and velocity
RMSE against the simulated truth, cumulative log likelihood, the resampling count,
and timings. `--trajectory` saves the truth, observations, estimates, ESS and
resampling decisions at every time step. Use `--model linear` for Cartesian
observations, `--resample always` to resample every step, or `--resample 0` to disable
resampling. The default rule resamples when `ESS < 0.5 * N`.

## Model and implementation

- State order is `(px, py, vx, vy)` in metres and metres per second. The independent
  Gaussian initial components have means `(-250, -60, 2, 3)` and standard deviations
  `(20, 20, 1.5, 1.5)`. The prior is at `t=0`. The first observation follows the first
  propagation at `t=1`.
- The step is 1 second and the acceleration spectral density is 0.04 m²/s³. A
  Cholesky factor samples the correlated position and velocity innovations of each
  axis. The radar is at the origin. The observation standard deviations are
  10 m per Cartesian component or 8 m in range and 0.04 rad in bearing. The
  range-bearing likelihood uses the shortest angular residual in `(-pi, pi]`.
- Each update adds the observation log likelihood to the previous log weight,
  normalizes by log-sum-exp and computes ESS. The estimate uses the normalized
  weights before resampling. Systematic resampling copies exactly `N` ancestors
  and resets the next step's weights to `1/N`. The sum of the log normalizing
  increments estimates `log p(y[1:T])`.
- `simulate` gives the same observations for a fixed model and data seed within
  a given build, and longer runs preserve the shorter observation prefix. Truth,
  measurement noise, particle motion and resampling use separate random streams.
  Save the trajectory CSV when comparing across standard libraries, since their
  normal distribution implementations can produce different samples. [Shared C++ interface](src/particle_filter.hpp).

For the parallel resampling comparison, use points `(k + u) / N` for `u` in
`[0, 1)`. Select the first cumulative weight strictly greater than the point.
With this convention, the number of points strictly below a normalized CDF `C`
is `clamp(ceil(N * C - u), 0, N)`. `make check` includes an exact eight-particle
case and cases with zero weights and points on interval boundaries.

## Correctness checks

`make check` verifies the resampling ancestry, angular wrapping, weight sums,
ESS bounds and stability with small likelihoods. The linear test compares the
particle filter to an independent Kalman recursion on six 80-step synthetic
sequences. The position and velocity metrics are Euclidean RMSE across all
steps and sequences. Covariance error is the RMSE over all matrix entries after
dividing each error by the product of the two relevant Kalman posterior standard
deviations. Log likelihood error is the mean absolute difference of the cumulative
log likelihoods divided by 80.

| Particles | Kalman position RMSE (m) | Kalman velocity RMSE (m/s) | Standardized covariance RMSE | Log likelihood error per step (nats) |
|---:|---:|---:|---:|---:|
| 512 | 3.420 | 0.409 | 0.251 | 0.07711 |
| 4,096 | 1.117 | 0.174 | 0.080 | 0.00949 |
| 65,536 | 0.286 | 0.048 | 0.025 | 0.00183 |

The nonlinear test uses six 200-step trajectories and 8,192 particles. The
position error is relative to simulated truth. The raw measurement baseline
converts each range and bearing into Cartesian coordinates directly.

| Method | Position RMSE (m) | Velocity RMSE (m/s) | Mean resampling steps out of 200 |
|---|---:|---:|---:|
| Raw measurements | 22.112 | | |
| ESS threshold 0.5 | 7.755 | 0.984 | 60.3 |
| Resample every step | 7.761 | 0.984 | 200 |

These results are from a one-core Vanda run on an Intel Xeon Platinum 8452Y
with GCC 11.4.1 at `-O3`. The simulated trajectory uses the same motion model
as the filter, so the error figures describe this controlled setting.
[Validation log](benchmarks/vanda/serial_validation.txt).

## Serial scaling benchmark

Run the benchmark from `serial/`. To plot the results, first generate the
trajectory with the command above and install Matplotlib in your Python environment.

```sh
make benchmark
python3 scripts/plot_serial.py  # optional, writes results/serial_validation.png
```

The benchmark uses fixed range-bearing observations, a fixed filter seed, the
0.5 ESS rule and posterior means. It changes `N` at `T=256` and changes `T` at
`N=16384`. Each configuration has one warm-up and five measured runs. The raw
runs and medians are in `results/serial_benchmark_runs.csv` and
`results/serial_benchmarks.csv`. `filter_ms` includes allocation, prior sampling
and every filter step. `workflow_ms` also includes simulation, error calculation
and trajectory export when requested. Phase columns separate initialization,
propagation and weighting, normalization and ESS, estimation, and resampling.
`particle_buffer_bytes` counts the allocated particle, weight and ancestor arrays.
It excludes the trajectory and per-step output, so it measures particle storage.

On Vanda, increasing `N` from 1,024 to 65,536 at 256 steps increased median
filter time from 35.8 to 2,352.8 ms. Increasing `T` from 64 to 512 at 16,384 particles
increased it from 147.8 to 1,165.7 ms. This is consistent with linear scaling in each
dimension over the tested sizes. [Timing summary](benchmarks/vanda/serial_benchmarks.csv)
· [40 timed runs](benchmarks/vanda/serial_benchmark_runs.csv)
· [Compute environment](benchmarks/vanda/environment.txt)
· [PBS allocation](benchmarks/vanda/pbs_job.txt).

## Run on Vanda

The PBS script requests one CPU core and 2 GB of memory via `auto_free`.
Ensure a C++17 compiler and Python 3.7 or newer are on the job's `PATH`. The job
saves its environment, validation log, and timing CSVs under
`serial/results/<job_id>/`. [NUS PBS batch guide](https://nusit.nus.edu.sg/services/getting-started/how-to-run-batch-job/).

Submit the job from `serial/` using the following command.

```sh
qsub scripts/serial_vanda.pbs
```

From the checkout root, use `qsub serial/scripts/serial_vanda.pbs`.

## AI use and references

OpenAI Codex assisted with the serial implementation, validation and benchmark
code, and figures derived from the measured output.

- Arulampalam, Maskell, Gordon and Clapp (2002). [A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking](https://mediawiki.isr.ist.utl.pt/images/7/7d/Arulampalam%2C02.pdf).
- Särkkä and Svensson (2023). [Bayesian Filtering and Smoothing, second edition](https://doi.org/10.1017/9781108917407).
