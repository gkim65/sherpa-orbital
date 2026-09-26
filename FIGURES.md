# Regenerating the figures

Every figure is derived from a sweep's per-rollout checkpoints, so reproducing one means
reproducing the sweep that fed it. Nothing here needs a solver rerun if the artifacts are
already on disk.

Two toolchains, split by what each is good at:

| | tool | why |
|---|---|---|
| 2D panels, bands, sweeps | Julia + CairoMakie | Makie is the better 2D plotter and the figures live beside the model |
| 3D trajectory views | Python + matplotlib | CairoMakie has no depth buffer, so a surface cannot occlude a line behind it, and any alpha below 1 on a surface emits a PDF soft mask per quad — measured at 2204 masks and 2.3 MB for one moon, against 204 KB from mplot3d drawing three times the paths |

## 0. The sweep

All figures read `artifacts/sweeps/<key>/`, written by `experiments/sweep.jl`. On a
recalibration axis each level calibrates its own kernels, solves its own policy, then flies
the rollouts:

```bash
KEY=sigma_nav_km VALS=0,0.015,0.05,0.1,0.2,0.3,0.4,1.0 SEEDS=1000 \
  julia --project=experiments -t auto experiments/sweep.jl
```

Budget ~3 min calibrate + 9–40 min solve per level, plus seconds per rollout. Everything is
keyed by value and skipped if present, so a killed run resumes and more seeds fly only the
new ones. See `experiments/cluster/` for running it across processes.

Collapse the checkpoints into one file to move them between machines:

```julia
using SherpaOrbital
pack_sweep("artifacts/sweeps/sigma_nav_km", "nav_sweep_1000.jld2")
```

`load_sweep` accepts a packed file wherever it accepts a directory, so the figure commands
below work either way.

## 1. Navigation sensitivity — `paper_3panel`

Science among survivors, survival rate, and mean survival time, against navigation error.

```julia
using SherpaOrbital, CairoMakie
rows = load_sweep("nav_sweep_1000.jld2")
keep = filter(r -> Float64(r["sigma_nav_km"]) in (0.0, 0.1, 0.2, 0.3, 0.4), rows)

plot_sweep(keep, :sigma_nav_km;
           path      = "figures/paper_3panel",
           xlabel    = "Navigation noise σ (km)",
           arms      = ["POMDP", "Safety", "MPC hold", "Greedy",
                        "Cyclic k=2", "Cyclic k=3"],
           panels    = (:science_survivors, :survival, :days_mean),
           bar_panels = (:days_mean,),
           legend_below = true,
           size      = (470, 720))
```

Science is reported over SURVIVING runs. Pooling survivors with failures averages two
populations — a run that dies on day 3 banks almost nothing — so the pooled mean tracks the
survival rate rather than the science a working controller collects. Measured at sigma = 0.3,
`Greedy` survives 25.8% of runs but its survivors bank 91.8, against the policy's 103.9 over
all 1000 — pooling would report Greedy at roughly a quarter of its science and read as a
science gap rather than the survival gap it is. The survival panel carries the risk instead.

## 2. Policy behaviour — `policy_panel`

Action bands beside the flown geometry, one row per navigation level. This is the composite
figure; the bands are a 1000-rollout statistic and the trajectory is a single rollout, so
the two halves of a row answer different questions.

Three steps, because the 3D half is drawn in Python:

```bash
# a. one rollout's flown geometry per level -> figures/arcs.npz
julia --project=experiments experiments/export_arcs.jl 0.0,0.05,0.1,0.2

# b. the band matrices -> figures/bands.npz
julia --project=experiments -e '
using SherpaOrbital, NPZ
rows = load_sweep("nav_sweep_1000.jld2")
ACTS = ["EXCURSE_HIGH", "CORRECT", "EXCURSE_MID", "EXCURSE_LOW"]
pay = Dict{String,Any}("action_order_utf8" => Vector{UInt8}(join(ACTS, ",")),
                       "sigmas" => [0.0, 0.05, 0.1, 0.2])
for v in (0.0, 0.05, 0.1, 0.2)
    c = action_bands(rows, :sigma_nav_km, v; actions = ACTS)
    k = replace(string(v), "." => "p")
    pay["M_$k"], pay["tdays_$k"] = c.M, c.t_days
    pay["alive_$k"], pay["nroll_$k"] = c.alive, c.nroll
end
npzwrite("figures/bands.npz", pay)'

# c. the composite
uv run python scripts/plot_policy_panel.py --sigma 0.0 0.05 0.1 0.2
```

`export_arcs.jl` needs a solved policy AND matching kernels per level, so run the sweep for
those levels first. The arcs come from `run_rollout`'s `n_arc` keyword: nothing else stored
can reconstruct the flown path, and re-propagating from a saved state gives an uncontrolled
coast that escapes to ~476,000 km, because the burns are not in that state.

The insets carry no axes on purpose. Coordinates there are Enceladus-CENTRED, so the z
values near the pass run −157 to −297 km while the altitudes are 23 to 120 km — printing
"−400" beside trajectories the figure labels "24 km" invites the wrong reading. Caption the
scale instead: the sphere is Enceladus, R = 252 km.

## 3. Standalone 3D views — `rollout_3d`

One panel per level, full orbit or cropped:

```bash
uv run python scripts/plot_rollout_3d.py              # whole orbit
uv run python scripts/plot_rollout_3d.py --zoom       # close approach
uv run python scripts/plot_rollout_3d.py --sigma 0.0 0.1
```

## 4. Single-run comparison — `baseline_comparison`

Cumulative science and the fraction of passes flown degraded, for one rollout per arm. Reads
rollout results directly rather than a sweep:

```julia
using SherpaOrbital, CairoMakie
plot_baseline_comparison(["POMDP" => res_pomdp, "MPC hold" => res_mpc], config;
                         path = "figures/baseline_comparison")
```

## Conventions

Every figure writes PDF, SVG and PNG, transparent, so one file reads on a light page or a
dark slide; the Julia figures take `theme = :dark` for the latter. Type is Computer Modern
serif — via LaTeX when it is on PATH, mathtext-cm otherwise.

`figures/` is gitignored: the figures are build products, and the checkpoints they come from
are the record.