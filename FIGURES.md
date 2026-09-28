# Regenerating the figures

See [GUIDE.md](GUIDE.md) for the model itself and [README.md](README.md) for installation.

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
# The sweep carries eight levels and seven arms; the paper figure uses five levels and drops
# `Cyclic k=1`, which tracks `k=2` closely enough to add a line without adding a finding.
keep = filter(r -> Float64(r["sigma_nav_km"]) in (0.0, 0.1, 0.2, 0.3, 0.4) &&
                   r["arm"] != "Cyclic k=1", rows)

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

Three columns per row: the action bands, a close view of the pass under the moon, and the
whole revolution at a tilted angle. The bands are a 1000-rollout statistic and the geometry
is a single rollout, so the columns of a row answer different questions.

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
uv run python scripts/plot_policy_panel.py --sigma 0.0 0.1 0.2
```

THREE levels, not four. The measured action mix at sigma = 0, 0.015 and 0.05 is the same to
within a percent (the deepest band is chosen on 6.7% of passes at all three), and sigma = 0.1
is where it goes to 0.000 and CORRECT jumps from 0.57 to 0.91. A fourth row samples one
regime twice and costs a page.

Sized for a two-column placement (10.5 x 4.2 in, 2:5); `--width` and `--aspect` change that,
and `FONT_BASE` at the top of the script sets every point size in one place. The layout
constants below it (`BANDS_RIGHT`, `ZOOM_LEFT`, `CTX_LEFT`, the `*_PAD_*` terms) place the
two 3D columns explicitly rather than through a gridspec, because mplot3d pads inside its
own axes box by a fraction no layout setting exposes -- `wspace` shifts that padding around
instead of removing it.

`export_arcs.jl` needs a solved policy AND matching kernels per level, so run the sweep for
those levels first. The arcs come from `run_rollout`'s `n_arc` keyword: nothing else stored
can reconstruct the flown path, and re-propagating from a saved state gives an uncontrolled
coast that escapes to ~476,000 km, because the burns are not in that state.

The insets carry no axes on purpose. Coordinates there are Enceladus-CENTRED, so the z values
near the pass run -157 to -297 km while the altitudes are 23 to 120 km -- printing "-400"
beside trajectories the figure labels "24 km" invites the wrong reading. Caption the scale
instead: the sphere is Enceladus, R = 252 km.

The radial scale is TRUE. `--r-draw` shrinks the drawn body and stretches altitudes by
`--gain` to pull the four bands apart, which reads well in isolation but lifts the passes to
several times their real altitude and detaches them from the surface. The bands column
already carries the altitudes exactly, so the insets keep true scale and the honest orbit
shape.

Arcs are RESAMPLED by arclength before drawing. They are stored at 60 points of uniform time
and periapsis is where the spacecraft is fastest, so the raw sampling steps ~1.5 km per point
inbound against ~30 km outbound; drawn directly, the sharpest part of the turn becomes a few
straight chords with a visible kink that reads as a wobble in the orbit rather than as the
sampling artifact it is.

The dotted rules are placed from MEASURED INK, not from a projection: `_ink_bbox` rasterises
the figure and scans the alpha channel inside each axes. Projecting sampled points cannot
bound a panel, because matplotlib renders a surface quad or line segment whenever any part of
it falls inside the view window, so the drawn extent always reaches past the last point that
passed a visibility test. The same measurement sets the right-edge crop on save.

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