# DSA5208: parallel particle filter for 2D target tracking

This repository implements a bootstrap particle filter (sequential Monte
Carlo) for tracking an object in 2D from noisy radar measurements, in three
versions:

| Version | Directory | Parallelism |
|---|---|---|
| Serial reference | [`serial/`](serial/) | one CPU core, C++17 |
| CPU MPI | [`mpi/`](mpi/) | particles split across MPI ranks, 1 to 256 cores |
| MPI + CUDA | [`cuda/`](cuda/) | one GPU per MPI rank, 1 to 2 NVIDIA A40 GPUs |

The model is a constant-velocity motion model with either linear (position)
observations, used to validate against an exact Kalman filter, or nonlinear
range-bearing observations from a radar at the origin. The
[tracking problem](tracking_problem.md) specifies the model, and the
[sequential Monte Carlo notes](sequential_mc.md) describe the algorithm and its
parallel design. The CPU MPI and CUDA versions reuse the serial model code and
draw each particle's random numbers from a counter-based Philox generator keyed
on its global index, so their results do not depend on the number of ranks.

## Repository layout

```
serial/   serial filter, validation tests, benchmark and plotting scripts
mpi/      CPU MPI filter, validation tests, Vanda PBS scripts, results
cuda/     MPI + CUDA filter, validation, Vanda PBS scripts, results
  */src/          source code
  */tests/        correctness tests run by `make check`
  */scripts/      benchmark drivers, PBS job scripts, summaries and plots
  */benchmarks/   archived measurements used in the report
tracking_problem.md   model specification
sequential_mc.md      algorithm and parallel design
environment.yml       conda environment for a workstation
```

Each directory has its own README with every option, the benchmark protocol
and the recorded results.

## Set up

### On a workstation (Linux or macOS)

The serial and CPU MPI versions need a C++17 compiler, `make` and Open MPI.
The scripts need Python 3.8 or newer, and Matplotlib for figures. The conda
environment provides all of these:

```sh
conda env create -f environment.yml
conda activate dsa5208-pf
```

The CUDA version also needs the CUDA 12 toolkit and an NVIDIA GPU, which the
environment does not install.

### On Vanda (NUS HPC)

Use the cluster modules for compilers, MPI and CUDA:

```sh
module load openmpi/gcc/64/4.1.8           # serial and CPU MPI (GCC 11.4)
module load cuda12.4/toolkit/12.4.1        # CUDA build
module load nvhpc-hpcx-2.20-cuda12/26.1    # CUDA-aware MPI for the device path
```

The PBS scripts load these modules themselves. Python 3 is on the system path.
For figures, install Matplotlib with `pip install --user matplotlib` or load
the `matplotlib/3.10.3-gfbf-2025a` module.

## Build, test and run

Run every command from the root of the repository.

### Serial

```sh
make -C serial
make -C serial check
serial/build/serial_pf --model range-bearing --particles 8192 --steps 200 \
  --data-seed 42 --filter-seed 1729 --resample 0.5
```

`serial_pf` prints a one-row CSV with the RMSE, log likelihood, resampling
count and timings. Add `--trajectory FILE` to save every time step.

### CPU MPI

```sh
make -C mpi
make -C mpi check
mpiexec -n 4 mpi/build/mpi_pf --model range-bearing --particles 65536 \
  --steps 256 --resampler distributed --resample 0.5
```

`make check` validates at 1, 3 and 8 ranks; set `CHECK_RANKS` to change them.
`--resampler centralized` gathers all particles on rank 0 instead, for
comparison.

### MPI + CUDA

```sh
make -C cuda
mpiexec -n 1 cuda/build/cuda_pf --validate --transport staged
mpiexec -n 1 cuda/build/cuda_pf --transport staged --particles 1048576 --steps 256
```

The default `CUDA_ARCH=86` targets the A40. `--transport device` passes GPU
buffers straight to MPI and needs a CUDA-aware MPI such as HPC-X. Use `-n 2`
on a node with two GPUs.

## Reproduce the Vanda experiments

Each command submits PBS jobs. Every job checks its CPU or GPU model and runs
the validation suite before timing, then writes its results to
`<version>/results/<job_id>/`.

| Experiment | Command |
|---|---|
| Serial particle and step sweeps | `qsub serial/scripts/serial_vanda.pbs` |
| Serial extended sweep and profile | `qsub serial/scripts/serial_extended_vanda.pbs` |
| CPU MPI, 1 to 256 ranks | `bash mpi/scripts/submit_vanda.sh --project PROJECT_CODE` |
| MPI + CUDA, 1 and 2 GPUs | `bash cuda/scripts/submit_vanda.sh 2` |
| Single-GPU particle sweep | `bash cuda/scripts/submit_single_gpu_sweep.sh` |

`PROJECT_CODE` is your Vanda PBS project. The CPU MPI jobs run on fixed
Xeon Platinum 8452Y hosts listed in `mpi/scripts/vanda_platinum_layouts.tsv`;
edit that file if those hosts are busy, as the [MPI README](mpi/README.md)
explains. After the jobs finish, the summary scripts in each `scripts/`
directory compute medians, speedup and efficiency and draw the figures. Each
README gives the exact command.

## Results

| Version | Archived results |
|---|---|
| Serial | [`serial/benchmarks/vanda/`](serial/benchmarks/vanda/) |
| CPU MPI | [`mpi/benchmarks/vanda-platinum-5-repeats/`](mpi/benchmarks/vanda-platinum-5-repeats/) |
| MPI + CUDA | [`cuda/benchmarks/vanda-a40-20261110/`](cuda/benchmarks/vanda-a40-20261110/) and [`cuda/benchmarks/vanda-a40-particle-sweep-20261111/`](cuda/benchmarks/vanda-a40-particle-sweep-20261111/) |

Unless a README says otherwise, timings use 256 range-bearing steps, data
seed 42, filter seed 1729, an ESS threshold of 0.5 and the median of five
runs. `mpi/benchmarks/vanda-gold-5-repeats/` holds earlier CPU MPI results
on Xeon Gold 6438N nodes.
