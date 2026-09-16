# Running a sweep on a cluster

Nothing here is Sherpa-specific beyond the sweep command — the point is that
`SARSOP_jll` ships the solver binary as a Julia artifact, so `Pkg.instantiate()`
installs everything including the solver. No system package, no compilation.

## One-time setup on the cluster

```bash
git clone git@github.com:gkim65/sherpa-orbital.git
cd sherpa-orbital

# Julia 1.12+ — via juliaup if the cluster has no module for it
curl -fsSL https://install.julialang.org | sh -s -- --yes
export PATH="$HOME/.juliaup/bin:$PATH"

julia --project=experiments -e 'using Pkg; Pkg.instantiate()'   # pulls SARSOP_jll too
julia --project=experiments -e 'using SherpaOrbital, SARSOP; println("ok")'
```

The committed `Manifest.toml` pins every version, so the cluster resolves the same
environment this was developed against.

## What a sweep costs

Per navigation level, at 10 seeds:

| stage | cost | notes |
|---|---|---|
| calibrate | 1–3 min | threaded; `-t auto` |
| solve | 9–40 min | single-threaded, the long pole |
| rollouts | 1–150 min | scales with the policy size, see below |

`sigma_nav_km` is a recalibration axis, so every level pays all three.

**Solve and rollout cost are NOT monotone in sigma.** Measured policy sizes:

| sigma | policy | one POMDP rollout |
|---|---|---|
| 0.1 | 6.9 MB | ~10 s |
| 0.3 | 5.3 MB | ~10 s |
| 0.4 | 41 MB | ~40 s |
| 0.5 | **435 MB** | **~900 s** |
| 1.0 | 24 MB | ~40 s |

Intermediate navigation error is the worst case: observations are blurry enough that
beliefs spread over many states but still informative enough that the solver has to track
the distinctions, so the reachable belief set explodes. A rollout then pays for it on every
belief update, which is an argmax over every alpha vector. Budget generously around
sigma ≈ 0.4–0.7, or expect to cap `precision`.

## Submitting

One level per job is the right granularity: levels are independent, the checkpoints are
keyed by value, and a job that dies loses only its own level.

```bash
sbatch --array=0-7 experiments/cluster/sweep.sbatch
```

Edit `VALS_ALL` in the script to change the grid. Re-submitting the same array is safe:
kernels, policies and rollouts already on disk are skipped, so a re-run fills gaps rather
than redoing work.

## Collecting

Checkpoints land in `artifacts/sweeps/<key>/<arm>/<key>=<value>/rollout_NNNNN.jld2`. Pull
that tree back and analyse locally:

```julia
using SherpaOrbital, CairoMakie
rows = load_sweep("artifacts/sweeps/sigma_nav_km")
plot_sweep_bars(rows, :sigma_nav_km; path = "figures/nav_bars")
```

The solved policies under `artifacts/solver/` are large and only needed to fly more seeds
at a level that already ran; leave them on the cluster unless you want that.