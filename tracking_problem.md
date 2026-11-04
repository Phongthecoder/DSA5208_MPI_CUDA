# Target Problem: 2D Object Tracking with a Particle Filter

## 1. Synthetic problem

An object moves in the plane. We observe it through noisy measurements and estimate its state over time.

### State variables

$$
x_t = (p^x_t,\; p^y_t,\; v^x_t,\; v^y_t)^\top \in \mathbb{R}^4
$$

| Symbol | Meaning |
|---|---|
| $p^x_t, p^y_t$ | position at time step $t$ |
| $v^x_t, v^y_t$ | velocity at time step $t$ |
| $\Delta t$ | time between steps |
| $T$ | number of time steps |
| $y_t$ | observation at time $t$ |

### Motion model (constant velocity)

$$
x_t = F x_{t-1} + w_t, \qquad w_t \sim \mathcal{N}(0, Q)
$$

$$
F = \begin{pmatrix}
1 & 0 & \Delta t & 0\\
0 & 1 & 0 & \Delta t\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{pmatrix},
\qquad
Q = q \begin{pmatrix}
\tfrac{\Delta t^3}{3} I_2 & \tfrac{\Delta t^2}{2} I_2\\
\tfrac{\Delta t^2}{2} I_2 & \Delta t\, I_2
\end{pmatrix}
$$

$q$ sets the strength of random accelerations (white-noise acceleration model).

### Observation model A: linear (validation)

The sensor measures position directly:

$$
y_t = H x_t + v_t, \qquad
H = \begin{pmatrix} 1 & 0 & 0 & 0\\ 0 & 1 & 0 & 0 \end{pmatrix}, \qquad
v_t \sim \mathcal{N}(0, \sigma^2 I_2)
$$

The model is linear-Gaussian, so the **Kalman filter gives the exact posterior**. We use it as ground truth to validate the particle filter: estimated means, covariances and the log-likelihood $\log p(y_{1:T})$ should converge to the Kalman values as the number of particles $N$ grows.

### Observation model B: nonlinear (range-bearing)

A sensor at position $s = (s^x, s^y)$ measures distance and angle to the object:

$$
y_t =
\begin{pmatrix}
\sqrt{(p^x_t - s^x)^2 + (p^y_t - s^y)^2}\\[2pt]
\operatorname{atan2}(p^y_t - s^y,\; p^x_t - s^x)
\end{pmatrix}
+ v_t, \qquad
v_t \sim \mathcal{N}\!\left(0, \operatorname{diag}(\sigma_r^2, \sigma_\theta^2)\right)
$$

The Kalman filter is no longer exact here. This is the case that motivates the particle filter. Bearing residuals are wrapped to $(-\pi, \pi]$.

### Synthetic data generation

1. Draw $x_0$ from a chosen initial distribution.
2. Simulate the true trajectory $x_{1:T}$ with the motion model.
3. Generate observations $y_{1:T}$ with model A or model B.

All random draws use fixed seeds for reproducibility.

### Problem size

- $N$: number of particles
- $T$: number of time steps

## 2. Real data application: AIS ship tracking

This is optional, depends on whether we have enough time until submission of the project.
### The dataset

The **Automatic Identification System (AIS)** is a system in which ships broadcast their GPS position, speed, heading and identity. NOAA publishes AIS records for US waters through [Marine Cadastre](https://marinecadastre.gov/ais). Each record holds a ship ID (MMSI), timestamp, latitude, longitude, speed over ground and course over ground.

### How we use it

The AIS positions are accurate GPS fixes, so we treat each recorded track as the **true trajectory**:

1. Convert latitude and longitude to local planar coordinates in metres.
2. Simulate noisy range-bearing observations from a virtual coastal radar (model B).
3. Run the particle filter on those observations.
4. Compare the estimates against the AIS track, using the AIS speed and course to check velocity estimates too.

This keeps the motion real while giving the observations a known noise model.

### Scaling the problem

- **More ships ($M$):** track many vessels at once, increasing total work to $N \times M$ particles per step.
- **More particles per ship ($N$):** improves estimate accuracy at higher cost.
- **Longer time windows ($T$):** more time steps, such as a full day of traffic instead of an hour.

AIS also brings challenges absent from the synthetic problem:

- irregular time gaps between reports
- manoeuvres (turns, port approaches) that break the constant-velocity assumption
- uneven track lengths across ships, which can cause load imbalance
