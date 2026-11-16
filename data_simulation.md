# How the tracking data is simulated

All experiments use synthetic data: a simulated object moving in the plane and
the noisy radar measurements it produces. One function generates it,
`simulate()` in [`serial/src/particle_filter.cpp`](serial/src/particle_filter.cpp),
and every version of the filter calls that same function. The model it follows
is specified in [tracking_problem.md](tracking_problem.md).

## What `simulate()` returns

```cpp
Scenario simulate(Sensor sensor, std::size_t steps, std::uint64_t seed);
```

| Argument | Meaning | Value in the experiments |
|---|---|---|
| `sensor` | `Sensor::linear` (position measurements) or `Sensor::range_bearing` (radar) | range-bearing for all timings; linear for the Kalman validation |
| `steps` | number of time steps $T$ | 256 for timings, 80 or 200 in validation |
| `seed` | data seed | 42 for timings; 11, 29, 47, 65, 83 and 101 in validation |

The returned `Scenario` holds two arrays of length $T$:

- `truth`: the true state $x_t = (p^x_t, p^y_t, v^x_t, v^y_t)$ at $t = 1, \dots, T$, in metres and metres per second;
- `observations`: the measurement $y_t$ at $t = 1, \dots, T$.

The filters only see `observations`. `truth` is used afterwards to measure
accuracy (position and velocity RMSE).

## Steps

The parameters are fixed constants in
[`serial/src/particle_filter.hpp`](serial/src/particle_filter.hpp).

1. **Initial state ($t = 0$).** Draw the true starting state from the prior,
   independently in each component:
   $$x_0 \sim \mathcal{N}\big(\mu_0, \operatorname{diag}(\sigma_0^2)\big), \quad
   \mu_0 = (-250, -60, 2, 3), \quad \sigma_0 = (20, 20, 1.5, 1.5).$$
   The object therefore starts about 260 m from the radar and moves at about
   3.6 m/s.

2. **Move the object ($t = 1, \dots, T$).** Apply the constant-velocity model
   with random accelerations,
   $$x_t = F x_{t-1} + s_t, \qquad s_t \sim \mathcal{N}(0, Q),$$
   with time step $\Delta t = 1$ s and acceleration density $q = 0.04$
   m²/s³. For each axis, $Q$ couples position and velocity:
   $$Q_{\text{axis}} = q \begin{pmatrix} \Delta t^3/3 & \Delta t^2/2 \\ \Delta t^2/2 & \Delta t \end{pmatrix}.$$
   The code samples $s_t$ exactly through the Cholesky factor of
   $Q_{\text{axis}}$: two standard normal draws per axis give the correlated
   position and velocity noise.

3. **Measure it.** Compute the noise-free measurement of the new state and add
   Gaussian noise:
   - linear: $y_t = (p^x_t, p^y_t) + v_t$, with $v_t \sim \mathcal{N}(0, 10^2 I_2)$ (m);
   - range-bearing, from a radar at the origin:
     $y_t = \big(\sqrt{(p^x_t)^2 + (p^y_t)^2},\ \operatorname{atan2}(p^y_t, p^x_t)\big) + v_t$,
     with $v_t \sim \mathcal{N}\big(0, \operatorname{diag}(8^2, 0.04^2)\big)$
     (m, rad). The noisy bearing is wrapped to $(-\pi, \pi]$.

4. Store $x_t$ and $y_t$, and repeat steps 2 and 3 until $t = T$.

The first measurement is taken after the first move, at $t = 1$. The filters
use the same convention: they start from the prior at $t = 0$ and propagate
before weighting the first observation.

## Random numbers and reproducibility

`simulate()` uses two independent `std::mt19937_64` generators with
`std::normal_distribution`:

- the **motion stream**, seeded with the data seed, draws the initial state
  and the process noise;
- the **measurement stream**, seeded with `seed ^ 0x9e3779b97f4a7c15`, draws
  the measurement noise.

Consequences:

- The same `sensor`, `steps` and `seed` always produce the same truth and
  observations within one build.
- A longer run starts with the same observations as a shorter one, because each
  step draws a fixed number of values from each stream.
- The linear and range-bearing data share the same true trajectory for a given
  seed, because the motion stream does not depend on the sensor.
- The data seed is separate from the filter seed (1729 in the experiments),
  which drives the particles' random numbers. Changing one does not change the
  other.

The C++ standard does not fix the algorithm behind `std::normal_distribution`,
so another standard library (for example libc++ with Clang on macOS) can give
different data from the same seed. All Vanda builds use GCC 11.4 and libstdc++,
so every version and every job saw identical data. To compare results across
systems, save the data with `serial_pf --trajectory FILE`.

## How each version uses it

The data is regenerated on every run; it is never read from a file. Each
program calls `simulate()` once at start-up, before the filter timer starts.

| Version | Where | What happens |
|---|---|---|
| Serial | [`serial/src/serial.cpp`](serial/src/serial.cpp) | `simulate()` is called directly with `--model`, `--steps` and `--data-seed`. |
| CPU MPI | [`mpi/src/main.cpp`](mpi/src/main.cpp) via `simulate_on_root()` in [`mpi/src/mpi_filter.cpp`](mpi/src/mpi_filter.cpp) | Rank 0 calls `simulate()`; one `MPI_Bcast` then sends the $2T$ observation values to every rank. Only rank 0 keeps the truth, for the RMSE. |
| MPI + CUDA | [`cuda/src/main.cpp`](cuda/src/main.cpp) | Uses the same `simulate_on_root()` with the range-bearing model. The validation in `cuda_filter.cu` also uses it. |
| Tests | [`serial/tests/`](serial/tests/), [`mpi/tests/`](mpi/tests/) | Simulate six sequences with seeds 11, 29, 47, 65, 83 and 101 to compare against the Kalman filter and the serial reference. |

Simulation time is not part of the reported filter time (`filter_ms`). It is
included in the separate `workflow_ms` column, and the serial program also
reports it on its own as `simulate_ms`. It is small: $T$ steps of one object,
done on one rank.
