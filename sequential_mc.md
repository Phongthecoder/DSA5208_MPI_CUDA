# The sequential Monte Carlo/particle filter for 2D tracking problem

## General form:
Assume that we have a state space model, where $x \in \mathbb{R}^d$ represents the hidden state and $y \in \mathbb{R}^k$ represents the observation. And this evolves according to time $t$.

At first, we have that $x_0 \sim p_0$ for some simple distribution $p_0$.

Two models we have are:
1. The propogation of state model:
$x_{t+1} = f(x_t) + s_t$

2. The observation model:
$y_{t+1} = g(x_{t+1}) + \epsilon_{t+1}$

where $s_t$ is the noise from propagation and $\epsilon_t$ is the observation noise.

## Quantities of interest
In this state space model, people are interested in 2 main things:
1. The filtering distribution: given $y_{0:t}$, what is the distribution of hidden state $x_t$, we denote this distribution with $p(x_{t}|y_{0:t})$.
2. The prediction distribution: given $y_{0:t}$, what is the distribution of next hidden state $x_{t+1}$, denote this distribution $p(x_{t}|y_{0:t})$.

**Note:** In the easy case where the dynamics are all linear and the noise terms are Gaussian. The system above has close form solution, and people call it the Kalman filter

But if one small deviation happens (non-linear dynamics or non-Gaussian noise), analytic solution is not available, and hence we need to perform computation.

## Approximation of the system

The main idea of approximating this system is that at any time $t$, we have $N$ particles tracking the distribution of $x_t$. And propogate these particles according to the dynamics, then compare these tracked particles with observation, to "correct for errors" in the tracking phase.

### The cloud of particles
Assume at time $t$, we have $N$ particles $X_t^{1:N}$, with their associated normalised weights $W_t^{1:N}$ tracking the distribution $p(x_t|y_{0:t})$.


In the boostrap particle filters, we do the following steps:

1. First check if the ESS is below the ESS_threshold (meaning that most weight are concentrated in a few number of particles), to decide whether we should do resampling or not (have a categorical sampling algo for this)

2. After the step 1, we have a new cloud of particles $X^{1:N}_{t+1}$ which can be the same, or actually has been resampled from the previous cloud. Then, what we do next is to propagate these particles according to the hidden state dynamics. This step is actually just:

$$X_{i}^{t+1} \leftarrow g(X_{i}^{t+1}) + S^{t+1}$$
where the $S^{t+1}$ is generated according to the specified distribution

3. After the previous step, the weight hasnt been change. Next, we update it according to the observation. Let say at time $t+1$, we observe $Y_{t+1}$, then clearly there are hidden states where this $Y_{t+1}$ is not probable and states where it is more probable. We will use this idea to update. Clearly, from the observation model with specified distribution on $\epsilon_{t+1}$, we can derive density of $Y_{t+1} | X_{t+1}$. The weight is updated as:
$$W^i_{t+1} \leftarrow p(Y_{t+1} | X_{t+1}^i) W^{i}_{t}$$

4. Then in the final step, we just need to normalise these weights (divide by the sum of all the weights).


Right. These are the only two steps where ranks have to communicate, and they differ a lot in difficulty:

Renormalise: simple. Each rank sums its own weights, and one MPI_Allreduce gives every rank the global total. It’s one number per time step and costs almost nothing.

Resample: the advanced part.

MPI_Exscan gives each rank its offset in the global cumulative sum of weights.

MPI_Bcast shares the single random $u$.

MPI_Alltoallv moves resampled particles between ranks, because the copies of a heavy particle may belong on other ranks. This is the irregular, expensive communication, and it is the core of your scaling analysis.

## The parallel algorithm

### Overview

We use two levels of parallelism:

$$\underbrace{N \text{ particles}}_{\text{total}} \;\xrightarrow{\text{MPI}}\; \underbrace{N_{\text{loc}} = N/p \text{ particles per rank}}_{\text{1 rank} \leftrightarrow \text{1 GPU}} \;\xrightarrow{\text{CUDA}}\; \underbrace{1 \text{ thread per particle}}_{\text{inside the GPU}}$$

- **MPI** distributes the particles across $p$ ranks. Rank $r$ owns the global particle indices $[\,r N_{\text{loc}},\ (r+1) N_{\text{loc}})$.
- **CUDA** does all per-particle work inside each rank, with one GPU per rank and one thread per particle.
- Ranks communicate only when global information is needed: **normalising the weights**, **computing the ESS**, and **resampling**.
- We use **CUDA-aware MPI**, so MPI calls take GPU pointers directly and no host copies are needed.

### Parameters

- $p$: number of MPI ranks (one GPU per rank)
- $N$: total number of particles, with $N_{\text{loc}} = N/p$ per rank
- $\text{ESS}_{\min}$: resample only when the ESS falls below this threshold (e.g. $N/2$)
- $y_{0:T}$: observed data, broadcast once to all ranks at the start with `MPI_Bcast`

### Data layout (per rank, on the GPU)

- Particle states $x^{(i)}$, stored as a structure of arrays for coalesced memory access
- Log-weights $\ell^{(i)} = \log w^{(i)}$. We work with log-weights to avoid underflow.
- A second particle buffer for resampling (double buffering)
- A random number generator: counter-based cuRAND Philox, seeded by (global particle index, time step), so results do not depend on the number of ranks

### Algorithm

**Initialisation ($t = 0$)**

1. Each thread $i$ samples $x_0^{(i)}$ from the prior and sets $\ell^{(i)} = \log g(y_0 \mid x_0^{(i)})$.

**For each time step $t = 1, \dots, T$:**

**Step 1: Propagate and weight (CUDA, fused kernel, no communication)**

Each thread $i$:
$$x_t^{(i)} \sim f(\cdot \mid x_{t-1}^{(i)}), \qquad \ell_t^{(i)} = \ell_{t-1}^{(i)} + \log g(y_t \mid x_t^{(i)})$$

Propagate and weight are fused into one kernel. Each thread keeps $x_t^{(i)}$ in a register and computes its weight straight away, which saves a kernel launch and a round trip to global memory.

**Step 2: Global maximum of the log-weights (CUDA + MPI)**

- CUDA: a block-level reduction in shared memory gives the local maximum $m_{\text{loc}}$.
- MPI: `MPI_Allreduce(m_loc, m, MPI_MAX)`, so every rank receives the global maximum $m$.

This is for numerical stability (log-sum-exp trick).

**Step 3: Normalise the weights and compute the ESS (CUDA + MPI)**

- CUDA: each thread computes $\hat w^{(i)} = e^{\ell^{(i)} - m}$, and a block reduction gives two local sums:
$$S_1^{\text{loc}} = \sum_{i \in \text{rank}} \hat w^{(i)}, \qquad S_2^{\text{loc}} = \sum_{i \in \text{rank}} \big(\hat w^{(i)}\big)^2$$
- MPI: **one** `MPI_Allreduce` with `count = 2` and `MPI_SUM` gives the global sums $S_1, S_2$ on every rank.
- Then:
$$w^{(i)} = \frac{\hat w^{(i)}}{S_1}, \qquad \text{ESS} = \frac{S_1^2}{S_2} = \frac{1}{\sum_i \big(w^{(i)}\big)^2}$$

Packing both sums into one call halves the latency of this step. Every rank now holds the same ESS, so all ranks make the same resampling decision without further communication.

Each `MPI_Allreduce` takes $O(\log_2 p)$ rounds with recursive doubling, and every rank ends with the result.

**Step 4: Resampling decision**

- If $\text{ESS} \ge \text{ESS}_{\min}$: skip resampling and go to the next time step.
- If $\text{ESS} < \text{ESS}_{\min}$: do systematic resampling (Step 5), then reset $\ell^{(i)} = -\log N$.

**Step 5: Systematic resampling**

*Idea:* the normalised weights sum to 1, so they cut $[0,1]$ into $N$ intervals, with particle $i$ owning an interval of length $w^{(i)}$. We place $N$ evenly spaced points $u_k = (k + u)/N$, $k = 0, \dots, N-1$, with a single $u \sim U[0,1)$. The number of points falling in particle $i$'s interval is its number of copies.

*5a. Local cumulative sum (CUDA).* A prefix scan (Hillis–Steele within each block, followed by adding block offsets) gives the local cumulative weights $W_{\text{loc}}^{(i)}$, accumulated in `double` for precision.

*5b. Global offset (MPI).* `MPI_Exscan(local_total, offset, MPI_SUM)` gives each rank the total weight of all earlier ranks (rank 0 sets offset $= 0$). Each thread then has its global cumulative weight:
$$W^{(i)} = \text{offset} + W_{\text{loc}}^{(i)}$$
Only one number per rank is communicated.

*5c. Shared random number.* All ranks need the same $u$. Instead of `MPI_Bcast`, every rank generates $u$ from the same seed and time step, which needs no communication.

*5d. Copy counts and output slots (CUDA, one thread per particle).*
$$\text{lo}_i = \big\lfloor N W^{(i-1)} + u \big\rfloor, \qquad \text{hi}_i = \big\lfloor N W^{(i)} + u \big\rfloor, \qquad o^{(i)} = \text{hi}_i - \text{lo}_i$$
Particle $i$ has $o^{(i)}$ copies, which go to the output slots $[\text{lo}_i, \text{hi}_i)$ of a new global array of $N$ slots.

*5e. Destination ranks (CPU, cheap).* Output slot $k$ belongs to rank $\lfloor k / N_{\text{loc}} \rfloor$, so each rank again ends with exactly $N_{\text{loc}}$ particles. Because the slots increase with $i$, each rank's output is a single contiguous range $[\text{lo}_{\text{first}}, \text{hi}_{\text{last}})$. The number sent to each destination rank is the overlap of this range with that rank's slots, which takes $O(p)$ arithmetic. `MPI_Alltoall` then tells every rank how many particles it will receive from each sender.

*5f. Pack (CUDA).* Each thread writes its $o^{(i)}$ copies into the send buffer at positions $[\text{lo}_i - \text{lo}_{\text{first}},\ \text{hi}_i - \text{lo}_{\text{first}})$. The buffer comes out already grouped by destination rank, so no sorting is needed.

*5g. Exchange (MPI).* `MPI_Alltoallv` on GPU buffers (CUDA-aware MPI) sends each group to its destination rank. Copies that stay on the same rank are copied directly on the GPU. The received particles become the particle array for the next time step (swap the buffers).

**Example** ($N = 8$, $p = 2$, $N_{\text{loc}} = 4$): rank 0 owns new slots 0–3 and rank 1 owns slots 4–7.

| Rank | Particle | Copies | Slots | Destination |
|---|---|---|---|---|
| 0 | A | 0 | none | dropped |
| 0 | B | 3 | 0, 1, 2 | rank 0 |
| 0 | C | 2 | 3, 4 | rank 0 and rank 1 |
| 0 | D | 1 | 5 | rank 1 |
| 1 | E | 0 | none | dropped |
| 1 | F | 1 | 6 | rank 1 |
| 1 | G | 0 | none | dropped |
| 1 | H | 1 | 7 | rank 1 |

After the exchange: rank 0 holds {B, B, B, C} and rank 1 holds {C, D, F, H}, so each rank again holds 4 particles.

### Communication summary (per time step)

| Step | MPI call | Volume | Cost |
|---|---|---|---|
| Max of log-weights | `MPI_Allreduce` (`MPI_MAX`) | 1 double | latency-bound, $O(\log p)$ |
| Weight sum + ESS | `MPI_Allreduce` (`MPI_SUM`, count 2) | 2 doubles | latency-bound, $O(\log p)$ |
| Scan offset (if resampling) | `MPI_Exscan` | 1 double | latency-bound, $O(\log p)$ |
| Receive counts (if resampling) | `MPI_Alltoall` | $p$ integers | small |
| Particle redistribution (if resampling) | `MPI_Alltoallv` | depends on the data, up to $O(N_{\text{loc}} \cdot d)$ per rank | **dominant**, bandwidth-bound |

Here $d$ is the dimension of the particle state. The redistribution volume grows when the weights are uneven, because one rank's heavy particles must then be spread across many ranks. This step is the main focus of the communication analysis.

### Possible optimisations

- **Send each particle once with its count:** send $(x^{(i)}, o^{(i)})$ instead of $o^{(i)}$ duplicates, and let the receiver expand them. This reduces the volume when a few particles dominate.
- **Adaptive resampling:** resampling only when $\text{ESS} < \text{ESS}_{\min}$ reduces how often the expensive `MPI_Alltoallv` runs.
- **CUDA streams:** overlap local GPU copies with the MPI exchange.


