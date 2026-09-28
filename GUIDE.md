# SherpaOrbital reference

How the model is built, what every parameter does, and how to reproduce each result. The
[README](README.md) covers installation and a one-paragraph summary; everything else is here.

**Contents**

- [POMDP formulation](#pomdp-formulation)
- [Hierarchical planning](#hierarchical-planning)
- [Measured transition kernels](#measured-transition-kernels)
- [`StationkeepingPOMDP` parameters](#stationkeepingpomdp-parameters)
- [Examples](#examples)
- [Figures](#figures)
- [Repository layout](#repository-layout)
- [Tests](#tests)
- [Physics conventions](#physics-conventions)
- [Key references](#key-references)

## POMDP formulation

One decision epoch is one periapsis pass (11.996 hr).

|            | |
|------------|--|
| **State**  | `(h, c, i, ρ)`. \|S\| = **5627** |
| **Action** | `CORRECT`, `EXCURSE_{LOW,MID,HIGH}`. \|A\| = **4** |
| **Obs**    | noisy altitude region. \|O\| = **7** |
| **Reward** | `+0.5` surviving pass, `−200` crash/loss, `+20 × f(c,c′) u(i′) y(ρ′)` science |
| **γ**      | 0.95 |

**State.** `h` is the achieved periapsis-altitude region: five live regions (below 20,
20–27 LOW, 27–34 MID, 34–44, above 44 HIGH km) plus absorbing `CRASHED` and `LOST`, so
orbital safety needs no separate state variable. Three are science regions; 34–44 km holds
the `CORRECT` limit cycle at 37.2 km and is not one, so science requires a deliberate
maneuver. `c` counts samples per science region, saturating at `visit_cap`. `i` is the plume
intensity the last pass returned. `ρ` bins the planner's targeting residual at 15 and 25 km
into `R_OK` / `R_DEGRADED` / `R_CRITICAL`. Only `h` is partially observed.

Conditioned on altitude alone, the transition model averages fresh and degraded departures
from the same region and reports `P(lost) ≈ 0` for the maneuver that loses the vehicle.
Conditioned on `ρ`, correcting from LOW at `R_CRITICAL` measures `P(lost) = 0.82` against
`0.0` at moderate and high altitude.

**Actions.** Every action commands a maneuver; there is no null action. One impulse does not
deliver a band, so reaching one takes several passes of the same `EXCURSE_*` — each solving
afresh from the state the previous burn produced, driven by the policy re-choosing it.

**Reward.** `f` is the coverage increment, falling to `repeat_factor` once a region saturates
or a pass lands outside every science region. `u(i′) ∈ {0.30, 0.65, 1.00}` scales with
realized intensity; `y(ρ′) ∈ {1.00, 0.65, 0.30}` discounts samples from a degraded orbit.
Both the science and loss terms are expectations under `T`, so risk enters as measured
probability rather than a tuned penalty. Science is credited on the *observed* region and
only on a surviving pass.

### Hierarchical planning

Representing maneuvers in the POMDP would require a continuous action space. A discrete menu
of burn directions is not a substitute: the stationkeeping burn is approximately
orbit-normal, and a prograde-only or fixed-magnitude menu loses the vehicle within 1–2 days.

The POMDP therefore chooses *what* objective to attempt; an onboard planner chooses *how*,
solving a continuous ΔV against a CR3BP model of the Saturn–Enceladus system. Actions encode
intent, not burn commands.

```
estimate ──►  belief ──π──►  objective ──►  planner (CR3BP) ──ΔV──►  spacecraft (CR3BP + J₂)
   ▲                                              │ residual                │
   └──────────── navigation ◄─────────────────────┴────────────────────────┘
```

The planner's targeting residual is returned upward and binned as `ρ`, so the damage variable
costs no extra sensing. Trajectories are propagated under a truth model that adds Enceladus's
J2, while the planner uses CR3BP alone; that discrepancy is the dynamical model uncertainty
under study.

### Measured transition kernels

`T` is measured from the truth model, one kernel per action, rows keyed on the joint `(h, ρ)`
and columns on the joint successor. One artifact per (thruster noise, navigation noise) pair, 
and kernels measured at one value are not valid for a rollout flown at another.

```
artifacts/tables.json                           noise-free
artifacts/tables_noisy_gaussian0.7.json         0.7% thruster, planning from truth
artifacts/tables_noisy_gaussian0.7_nav0.1.json  0.7% thruster, planning at σ = 0.1 km
```

```julia
tables = load_tables(tables_path_for(config; nav_sigma_km = 0.1))
validate_tables(tables)
```

## `StationkeepingPOMDP` parameters

The struct is the configuration ([src/StationkeepingPOMDP.jl](src/StationkeepingPOMDP.jl)).

* **constructor:** `StationkeepingPOMDP(; kwargs...)`
* **altitude discretization**
  * `alt_edges::NTuple{4,Float64}` — region boundaries (km), default `(20, 27, 34, 44)`
  * `alt_rep_km::Dict{Symbol,Float64}` — per-region nav observation mean (km)
* **science regions**
  * `band_names::NTuple{3,Symbol}` — default `(:LOW, :MID, :HIGH)`
  * `band_bins::NTuple{3,Symbol}` — region per band, default `(:A20_27, :A27_34, :ABOVE_44)`
  * `band_target_km::Dict{Symbol,Float64}` — commanded altitude (km), default
    `23.5 / 30.5 / 46.0`. HIGH is bounded above by the orbit: 55 and 60 km escape by pass 3
  * `visit_cap::Int` — samples counted per region, default `4`. \|S\| grows as `(cap+1)^3`
  * `correct_bin::Symbol` — region the `CORRECT` cycle settles in, default `:A34_44`
* **uncertainty (sweep axes)**
  * `sigma_nav_km::Float64` — 1σ navigation error (km), default `2.0`. Drives both the
    observation model and planner noise. MacKenzie Exhibit C-8 implies ~0.1 km.
    **Recalibration axis**
  * `noisy_thruster::Bool` — default `true`. **Recalibration axis**
  * `thruster_sigma_pct::Float64` — 1σ burn-magnitude error (%), default `0.7` (Exhibit B-24
    Model 1; Model 2 is `2.0`). **Recalibration axis**
  * `plume_gradient::Float64` — plume altitude-gradient strength, default `0.0`. Analytic, so
    no recalibration
  * `plume_levels::Int` — intensity levels, default `3`. Changes \|S\|, not a sweep axis
* **reward**
  * `r_science::Float64` = `20.0`, `r_step_ok::Float64` = `0.5`,
    `r_crashed` / `r_lost::Float64` = `-200.0`
  * `repeat_factor::Float64` — yield after a region saturates, default `0.2`. Must be
    nonzero, or the policy goes indifferent once every region caps
  * `intensity_value_min::Float64` — weakest intensity level's value, default `0.3`
  * `damage_yield::NTuple{3,Float64}` — multiplier per damage bin, default
    `(1.0, 0.65, 0.3)`. A modelling choice, not measured; `(1,1,1)` disables it
  * `fuel_weight::Float64` — default `0.0`, so `action_dv_cost` is inert
* **solver**
  * `discount::Float64` — default `0.95`
  * `tables_path::Union{Nothing,String}` — `nothing` resolves from the noise settings

## Examples

### Solve and inspect

```julia
using Pkg
Pkg.activate("experiments")
Pkg.instantiate()

using SherpaOrbital, SARSOP, POMDPs

config = StationkeepingPOMDP(; sigma_nav_km = 0.1, plume_gradient = 1.5)
print_model_summary(config)

tables = load_tables(tables_path_for(config; nav_sigma_km = config.sigma_nav_km))
policy = solve(SARSOP.SARSOPSolver(; precision = 1e-3, timeout = 1800.0,
                                   pomdp_filename  = "artifacts/solver/nav0.1.pomdpx",
                                   policy_filename = "artifacts/solver/nav0.1.out"),
               build_pomdp(config; tables = tables))

print_policy_table(policy, config)
```

Roughly 9 min to solve at |S| = 5627. Reload without re-solving:

```julia
policy = SARSOP.load_policy(build_pomdp(config; tables = tables),
                            "artifacts/solver/nav0.1.out")
```

Name `policy_filename` and `pomdp_filename` explicitly — SARSOP.jl otherwise writes
`policy.out` and `model.pomdpx` into the working directory, so concurrent solves overwrite
each other. `export_policy` writes a self-describing JSON instead, but it carries the dense
`T[s][a][s']` (~1.6 GB here), so it is an archive format rather than the reload path.


### Fly a policy against the truth model

```julia
ic  = nondim_to_cr3bp(collect(PERIOD1_SOUTH_IC_ND))
res = run_rollout(SARSOPController(policy, config; ref_ic = ic, tables = tables),
                  ic, cr3bp_j2_eom!, PERIOD1_TRIPLE_PERIOD_S, 30 * 24 * 3600.0)

res.outcome, res.n_bands, res.total_dv_ms
```

About 10 s per 30-day rollout.

### Baselines

Each baseline replaces the policy with a rule and shares everything below it — the same
planner, band targets, navigation noise, and stop-once-saturated rule — so a gap is
attributable to the decision layer.

```julia
core = scripted_core(config; ref_ic = ic)

CyclicController(core, 2)                          # LOW, CORRECT×2, MID, ... then hold
GreedyController(core)                             # excurse to the least-sampled band
SafetyController(core)                             # excurse from a clean orbit, never deep
MPCController(; ref_ic = ic, mode = :position, nav_sigma_km = config.sigma_nav_km)
```

`MPCController` defaults to `nav_sigma_km = 0.0`, which plans from the true state and is an
oracle — pass the config's σ for a fair comparison.

### Sweeps

```bash
KEY=sigma_nav_km       VALS=0,0.1,0.3,1,2  SEEDS=3  julia --project=experiments -t auto experiments/sweep.jl
KEY=thruster_sigma_pct VALS=0.7,2          SEEDS=3  julia --project=experiments -t auto experiments/sweep.jl
KEY=plume_gradient     VALS=0,1.5,4        SEEDS=3  julia --project=experiments -t auto experiments/sweep.jl
```

Other knobs: `ARMS`, `DAYS`, `PLUME`, `OUT`. On a recalibration axis each level calibrates,
solves and flies — about 13 min per level.

Everything is keyed by value and skipped if present, so a sweep can be extended without
redoing any of it:

```bash
# fill in the curve: only the new levels calibrate, solve and fly
KEY=sigma_nav_km VALS=0,0.05,0.1,0.2,0.3,0.5,1,2 SEEDS=3 julia --project=experiments -t auto experiments/sweep.jl

# add seeds: nothing re-solves
KEY=sigma_nav_km VALS=0,0.1,0.3,1,2 SEEDS=10 julia --project=experiments -t auto experiments/sweep.jl
```

Each rollout checkpoints on completion with its full per-step trace, so a killed sweep keeps
what it flew and a new metric needs no re-run:

```julia
rows = load_sweep("artifacts/sweeps/sigma_nav_km")
rows[1]["actions"]                          # action per pass
rows[1]["peri_alts_km"]                     # where each pass went
rows[1]["residuals_km"]                     # planner residual per pass
rows[1]["obs_bins"], rows[1]["true_bins"]   # observed vs. actual region
```

### Regenerate the kernels

```bash
julia --project=experiments -t auto experiments/calibrate.jl     # ~3-7 min
```

The artifact path follows the config, so a run never clobbers kernels measured under
different noise. Use `-t 1` for an artifact meant to be reproduced exactly — the threaded
walk shares an RNG, so `rng_seed` does not pin it.

## Figures

```julia
using SherpaOrbital, CairoMakie      # the caller supplies the plotting package

rows = load_sweep("artifacts/sweeps/sigma_nav_km")
plot_sweep(rows, :sigma_nav_km; path = "figures/nav_sweep",
           xlabel = "Navigation noise σ (km)")
plot_action_bands(rows, :sigma_nav_km; path = "figures/actions",
                  values = (0.0, 0.05, 0.1), label = "σ", unit = "km")
```

All write PDF, SVG and PNG, transparent, with a `theme = :dark` option. Science and risk are
plotted separately because one scalar return conflates them — a run that survives the full
horizon can still score badly for expected loss it never incurred.

```julia
science_trace(res, config)    # (t_days, cumulative)
damage_trace(res)             # residual, level, n_degraded, frac_degraded, lost_day
delivery_trace(res, config)   # commanded vs. achieved periapsis, per excursion
sweep_summary(rows, :sigma_nav_km)
action_bands(rows, :sigma_nav_km, 0.1)   # fraction choosing each action, per pass
```

3D trajectory views are drawn in Python, from arcs exported by
`experiments/export_arcs.jl`. CairoMakie has no depth buffer, so a surface cannot occlude a
line behind it, and any alpha below 1 on a surface emits a PDF soft mask per quad — 2.3 MB
for one moon against matplotlib's 204 KB. The 2D figures stay in Julia.

**See [FIGURES.md](FIGURES.md)** for the exact commands behind each paper figure, including
which sweep has to exist first.

## Repository layout

```
src/
  SherpaOrbital.jl        module + exports
  StationkeepingPOMDP.jl  the @kwdef config struct
  states.jl               (h, c, i, ρ) space, altitude and residual binning
  actions.jl              action set, excursion -> region mapping
  observations.jl         O[s,o] — analytic Gaussian nav model
  plume.jl                P_θ(intensity | region)
  tables.jl               load/write/validate the measured kernels
  transition.jl           T[s,a,s'] — coverage banking + intensity draw
  rewards.jl              r(s,a)
  model.jl                build_pomdp
  export.jl               solved policy -> JSON archive, θ-keyed paths
  calibration/            measure the kernels from the truth model
  dynamics/               CR3BP (onboard) + J2 variants (truth), kept separate
  planner.jl              onboard burn planner (CR3BP only)
  baselines/mpc.jl        MPC hold baseline
  baselines/scripted.jl   cyclic / greedy / safety baselines
  spacecraft/             thruster + nav models (explicit rng)
  common/simulate.jl      unified rollout harness
  common/figures.jl       comparison and sweep figures (CairoMakie at call time)
  common/checkpoint.jl    per-rollout sweep checkpoints
  common/report.jl        model + policy pretty-printing
experiments/              own Project.toml — isolates the solver dependency
  example.jl              solve, inspect, export
  calibrate.jl            regenerate a kernel artifact
  sweep.jl                calibrate, solve and fly across a swept axis
  export_arcs.jl          flown trajectory arcs -> .npz, for the Python 3D figures
  cluster/                tmux and Slurm drivers for a sharded sweep
scripts/                  Python figures (matplotlib 3D; see FIGURES.md)
artifacts/                tables*.json committed; solver output and sweeps gitignored
test/                     runtests.jl
doc/                      committed README assets
legacy/                   frozen Python, deliberately not ported
```

`legacy/` holds the pre-Julia prototypes, kept only as a provenance record: the
`russell-lara/` scripts reproduce a published figure that validated the non-spherical
gravity model, and `figures-reference/` holds the matplotlib versions of early figures. No
Julia code path calls into it, and nothing in it is maintained.

The library declares no solver dependency — SARSOP lives only in `experiments/Project.toml`.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                                 # full, ~5 min
julia --project=. -e 'using Pkg; Pkg.test(test_args=["plume","rewards"])'     # ~8 s
```

The orbit-geometry, family-continuation and rollout testsets take minutes; pass substrings to
run only matching ones.

## Physics conventions

- **Frame:** Saturn–Enceladus CR3BP rotating frame. Saturn at x = −μ, Enceladus at x = 1−μ.
- **Units:** km, km/s, s. ΔV costs in m/s.
- **Truth vs. onboard:** truth model includes J2, the onboard planner is CR3BP only. The gap
  between them is the model uncertainty being studied; the two stay separate.
- **Stability:** all six Floquet multipliers lie on the unit circle, so the orbit is
  marginally stable rather than hyperbolically unstable. An uncontrolled orbit diverges
  because of the J2 model gap, not orbital instability.

> **This is a period-1 orbit, not MacKenzie's period-3.** We fly a period-1 member of the
> Saturn–Enceladus L1 halo family: 11.996 hr, one periapsis per revolution, south-polar
> periapsis, commanded between ~23 and ~46 km. MacKenzie §B.2.3 specifies a period-3 member,
> whose three geometrically distinct periapses buy ground-track diversity. Reaching it is a
> branch-switching problem — no period-tripling bifurcation has been shown here at these
> altitudes. The altitude/risk tradeoff is the same either way.

## Key references

- MacKenzie, S. M. et al. (2020). *Enceladus Orbilander Mission Concept Study* — orbit
  parameters (§B.2.3), thruster error (Exhibit B-24), navigation (Exhibit C-8).
- Ershova, A. et al. (2024). Modeling the Enceladus dust plume based on in situ measurements
  performed with the Cassini Cosmic Dust Analyzer. *A&A* 689, A114.
- Kurniawati, H., Hsu, D., Lee, W. S. (2008). SARSOP: Efficient point-based POMDP planning by
  approximating optimally reachable belief spaces. *Robotics: Science and Systems*.
- Howell, K. C. (1984). Three-Dimensional, Periodic, Halo Orbits. *Celestial Mechanics* 32(1).
- Iess, L. et al. (2014). The Gravity Field and Interior Structure of Enceladus. *Science*.
- JPL Three-Body Periodic Orbits Catalog (DE440 ephemeris).
