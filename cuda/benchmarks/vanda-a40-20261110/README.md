# Vanda A40 MPI + CUDA experiments

PBS job `1537080.stdct-mgmt-02` completed on 10 November 2026 at
23:41 Singapore time in the free GPU queue, with exit status zero.
It allocated four CPU cores, 16 GB host memory and two NVIDIA A40 GPUs on
`GN-A40-004`. Each GPU reported 46,068 MiB of memory. The node used Intel Xeon
Platinum 8452Y CPUs, driver 575.57.08, CUDA 12.4.131, GCC 11.5.0 and
HPC-X 2.20 with Open MPI 4.1.7a1.

Each MPI rank selected a distinct GPU. Both one-GPU and two-GPU
measurements ran in this allocation. Strong scaling used 4,194,304 total
particles. Weak scaling used 1,048,576 particles per GPU. Every filter
used 256 range-bearing observations, data seed 42, filter seed 1729 and
ESS threshold 0.5. Each configuration had an eight-step warm-up and five
full measurements. Filter times include initialization of the particle
arrays through the final step and use the maximum across MPI ranks.

| Configuration | Strong median (s) | Strong min–max (s) | Strong speedup | Strong efficiency | Weak median (s) | Weak efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 GPU | 1.411 | 1.4106–1.4112 | 1.00× | 100.0% | 0.366 | 100.0% |
| 2 GPUs, host staging | 2.212 | 2.186–2.246 | 0.64× | 31.9% | 1.119 | 32.7% |
| 2 GPUs, device buffers | 0.755 | 0.755–0.756 | 1.87× | 93.4% | 0.396 | 92.4% |

At two GPUs, device-buffer exchange made the strong workload 2.93 times
faster than host staging. In the staged strong run, the phase maxima
included 0.664 s of explicit staging and 0.816 s inside resampling MPI
calls. The device-buffer run spent 0.037 s inside those MPI calls.
MPI timings include waiting and transfers performed internally by MPI.
Individual phase maxima can come from different ranks.

All three validation configurations passed exact systematic ancestry tests
and comparisons against fixed-seed CPU MPI estimates and ESS decisions.
The recorded timing repeats and both two-GPU transports produced matching
accuracy outputs. The filter uses a fixed-order hierarchical scan to keep
the CDF calculation reproducible.

## Files

- [Raw measurements](gpu_runs.csv), [timing summary](gpu_summary.csv) and
  [scaling figure](cuda_scaling.png).
- [CPU comparison](cpu_gpu_comparison.csv), using the published
  [Xeon Gold CPU MPI measurements](../../../mpi/benchmarks/vanda-gold-5-repeats/).
- [One-GPU validation](validation_p1_staged.txt),
  [two-GPU staged validation](validation_p2_staged.txt) and
  [two-GPU device-buffer validation](validation_p2_device.txt).
- [One-GPU binding](gpu_bindings_p1.tsv),
  [two-GPU bindings](gpu_bindings_p2.tsv) and
  [device-transport status](device_transport.tsv).
- [Hardware and source fingerprints](environment.txt), [source hashes](source_sha256.txt),
  [source commit](source_commit.txt), [PBS allocation](pbs_job.tsv) and
  [actual launcher flags](launcher_flags.txt).

The original PBS wrapper printed `--bind-to core` in `environment.txt`.
The timing driver applied `--bind-to none` under HPC-X to inherit PBS CPU
affinity. `launcher_flags.txt` records the flags actually used. The current
submission script expresses the same policy explicitly.

## CPU comparison and scope

The same 4,194,304-particle workload took 1.722 s on 128 Xeon Gold 6438N
CPU MPI ranks and 0.755 s on two A40s with device-buffer exchange, giving
a CPU/GPU time ratio of 2.28. The comparison matches the workload, seeds,
ESS threshold and filter timing scope. Hardware and process counts differ.
GPU scaling in this archive covers one to two GPUs within one node.

An [independent one-GPU check](single-gpu-check/) on `GN-A40-101`
completed in PBS job `1537259.stdct-mgmt-02`. It passed validation and
recorded five strong and five weak timings, with medians of 1.408 s and
0.362 s. The scaling table above uses the one-GPU baseline from the
two-GPU allocation.

See the [CUDA README](../../README.md) for submission and summary commands.
