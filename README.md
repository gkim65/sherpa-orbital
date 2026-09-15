# SherpaOrbital.jl

Offline POMDP stationkeeping for the Enceladus Orbilander mission concept
(MacKenzie et al. 2020).

Enceladus, one of Saturn's moons, is thought to harbor a subsurface ocean that may provide
conditions suitable for life, making it an important target in the search for
extraterrestrial habitability. Material from its subsurface ocean is ejected through plumes
at the moon's south pole, providing an opportunity to sample ocean-derived material without
directly accessing the subsurface ocean. The **Enceladus Orbilander mission concept**
(MacKenzie et al. 2020) is designed to investigate this environment through an orbital
science phase followed by a landed phase.

This repository focuses on autonomous stationkeeping and science planning during the orbital
phase. The spacecraft operates in an unstable L1 halo orbit with periapsis over Enceladus's
south pole, allowing repeated passes through the plume region. These passes create a direct
tradeoff between **science** and **orbital safety**. Lower-altitude plume crossings encounter
more concentrated plume material, motivating excursions to different periapsis altitudes,
while the spacecraft must simultaneously maintain a trajectory that can be recovered without
impact or escape.

Because the halo orbit is dynamically unstable, these decisions are coupled across orbital
passes. A science excursion changes the trajectory from which future stationkeeping maneuvers
must be performed, and an excursion that is recoverable from a healthy orbit may become
unsafe after the trajectory has already degraded. The spacecraft must make these decisions
autonomously because the communication delay to Saturn prevents real-time ground control.

The problem is also partially observable. Navigation provides an uncertain estimate of the
spacecraft's orbital condition, and maneuver execution and dynamical modeling introduce
additional uncertainty. The spacecraft must therefore decide when to collect science and when
to prioritize orbit recovery while reasoning about both its uncertain current state and the
consequences of its actions on future passes.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/gkim65/sherpa-orbital")
```

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
orbital safety requires no separate state variable. Three regions are science regions;
34–44 km holds the `CORRECT` limit cycle at 37.2 km and is not one of them, so science
requires a deliberate maneuver rather than passive holding. `c` is the per-region sample
count, saturating at `visit_cap`. `i` is the plume intensity the last pass returned. `ρ` is
the orbit-damage bin, `R_OK` / `R_DEGRADED` / `R_CRITICAL` at residual boundaries of 15 and
25 km. Only `h` is partially observed; `c`, `i` and `ρ` are known exactly.

`ρ` is what makes the coupling across passes representable. Conditioned on altitude alone,
the transition model averages fresh and degraded departures from the same region and reports
`P(lost) ≈ 0` for the maneuver that loses the vehicle. Conditioned on `ρ`, correcting from
LOW at `R_CRITICAL` measures `P(lost) = 0.82`, against `0.0` at moderate and high altitude.

**Actions.** Every action commands a maneuver; there is no null action, since an uncorrected
pass is not a decision but a step toward losing the vehicle. `EXCURSE_*` sets a persistent
reference held until a `CORRECT` clears it, so a band is reached by settling over several
passes rather than in one impulse.

**Observations.** A noisy read of the achieved altitude region, computed analytically by
integrating a Gaussian nav error over the region boundaries. Science is credited on the
*observed* region and only on a surviving pass, so ~11% of passes are attributed to the
wrong region on average (16% in the outer, one-sided regions). Delivery error is
sub-kilometre, so coverage error is a sensing problem rather than a control one.

**Reward.** `f` is the coverage increment, falling to `repeat_factor` once a region saturates
or a pass lands outside every science region. `u(i′) ∈ {0.30, 0.65, 1.00}` scales with
realized intensity; `y(ρ′) ∈ {1.00, 0.65, 0.30}` discounts samples taken from a degraded
orbit. Both the science and loss terms are expectations under `T`, so maneuver risk enters
as measured probability rather than a tuned penalty.

### Hierarchical decision and maneuver planning

Representing maneuvers directly in the POMDP would require reasoning over a continuous space
of burn vectors. A discrete menu of burn directions is not a workable substitute: the
stationkeeping burn is approximately orbit-normal, and a prograde-only or fixed-magnitude
menu loses the vehicle within 1–2 days.

Mission-level decisions and maneuver generation are therefore separated. The POMDP determines
*what* objective to attempt over the discrete state above; an onboard planner determines
*how* to execute it, solving for a continuous ΔV against a CR3BP model of the
Saturn–Enceladus system. POMDP actions encode maneuver intent, not burn commands.

```
belief  ──π──►  objective  ──►  planner (CR3BP)  ──ΔV──►  spacecraft (CR3BP + J₂)
   ▲                                   │ residual              │ obs, intensity
   └───────────────────────────────────┴───────────────────────┘
```

The planner's targeting residual — the extent to which the orbit has drifted beyond
single-impulse correction — is returned upward and binned as `ρ`. It requires no additional
sensing, because the solver computes it while solving for the burn.

Trajectories are propagated under a truth model that adds Enceladus's J2 oblateness, while
the planner uses CR3BP alone. That discrepancy is the dynamical model uncertainty under
study, and the two models are kept separate.

### Measured transition kernels

`T` is measured from the truth model rather than derived in closed form: one kernel per
action, rows keyed on the joint `(h, ρ)` and columns on the joint successor. Keying on both
matters — pooling across regions makes the excursions identical in `T`, and keying on
altitude alone loses the conditional risk described above. Kernels are stored in
[artifacts/tables.json](artifacts/tables.json) with their provenance, loaded at build time,
and committed so a re-measurement appears in a diff.

```julia
tables = load_tables()      # validates row-stochasticity and altitude ordering
validate_tables(tables)
```

## `StationkeepingPOMDP` parameters

The struct is the configuration ([src/StationkeepingPOMDP.jl](src/StationkeepingPOMDP.jl)).

* **constructor:** `StationkeepingPOMDP(; kwargs...)`
* **altitude discretization**
  * `alt_edges::NTuple{4,Float64}` — region boundaries (km), default `(20, 27, 34, 44)`
  * `alt_rep_km::Dict{Symbol,Float64}` — per-region nav observation mean (km), default
    `18 / 23.5 / 30.5 / 37.2 / 46`
* **science regions**
  * `band_names::NTuple{3,Symbol}` — default `(:LOW, :MID, :HIGH)`
  * `band_bins::NTuple{3,Symbol}` — region per band, default `(:A20_27, :A27_34, :ABOVE_44)`
  * `band_target_km::Dict{Symbol,Float64}` — commanded periapsis altitude (km), default
    `23.5 / 30.5 / 46.0`. HIGH is bounded above by the orbit: 55 and 60 km escape by pass 3
  * `visit_cap::Int` — samples counted per region, default `4`. \|S\| grows as
    `(cap+1)^n_bands`
  * `correct_bin::Symbol` — region the `CORRECT` cycle settles in, default `:A34_44`
* **uncertainty (sweep axes)**
  * `sigma_nav_km::Float64` — 1σ nav noise (km), default `2.0`. Conservative; MacKenzie
    Exhibit C-8 implies ~0.1 km
  * `noisy_thruster::Bool` — default `true`. Recalibration axis
  * `thruster_sigma_pct::Float64` — 1σ burn-magnitude error (%), default `0.7` (Exhibit B-24
    Model 1; Model 2 is `2.0`). Recalibration axis
  * `plume_gradient::Float64` — plume altitude-gradient strength, default `0.0`. Enters `T`
    analytically, so a sweep needs no recalibration
  * `plume_levels::Int` — intensity levels, default `3`. Changes \|S\|, so not a sweep axis
* **reward**
  * `r_science::Float64` — default `20.0`
  * `r_step_ok::Float64` — surviving pass, default `0.5`
  * `r_crashed`, `r_lost::Float64` — default `-200.0`
  * `repeat_factor::Float64` — yield after a region saturates, default `0.2`. Must be
    nonzero, or the policy goes indifferent once every region caps
  * `intensity_value_min::Float64` — value of the weakest intensity level, default `0.3`
  * `damage_yield::NTuple{3,Float64}` — multiplier per damage bin, default
    `(1.0, 0.65, 0.3)`. A modelling choice, not measured; `(1,1,1)` disables it
  * `fuel_weight::Float64` — default `0.0`, so `action_dv_cost` is inert. Raising it restores
    the fuel tradeoff with no code change
  * `action_dv_cost::Dict{Symbol,Float64}` — measured per-pass median ΔV (m/s)
* **solver**
  * `discount::Float64` — default `0.95`
  * `tables_path::Union{Nothing,String}` — `nothing` resolves from `noisy_thruster` /
    `thruster_sigma_pct`

## Examples

### Solve and inspect a policy

```julia
using Pkg
Pkg.activate("experiments")     # from the repo root
Pkg.instantiate()

using SherpaOrbital, SARSOP, POMDPs

config = StationkeepingPOMDP()          # baseline: noisy at 0.7% (B-24 Model 1)
print_model_summary(config)

pomdp  = build_pomdp(config)

# `policy_filename`/`pomdp_filename` default to `policy.out`/`model.pomdpx` in the WORKING
# directory — name them so concurrent solves cannot overwrite each other, and so the policy
# can be reloaded later without re-solving.
policy = solve(SARSOP.SARSOPSolver(; precision = 1e-3, timeout = 900.0,
                                   pomdp_filename  = "model.pomdpx",
                                   policy_filename = "policy.out"), pomdp)

print_policy_table(policy, config)
export_policy(policy, config)           # -> artifacts/policy.json
```

```bash
julia --project=experiments experiments/example.jl
```


### Fly a policy against the truth model

```julia
ic  = nondim_to_cr3bp(collect(PERIOD1_SOUTH_IC_ND))
res = run_rollout(SARSOPController(load_policy(); ref_ic = ic),
                  ic, cr3bp_j2_eom!, PERIOD1_TRIPLE_PERIOD_S, 30 * 24 * 3600.0)

res.outcome, res.n_bands, res.total_dv_ms
discounted_return(res, config)
```

About 10 s per 30-day rollout. This is the truth-model return, not the discrete-model one
from `POMDPs.simulate` with a `RolloutSimulator` (milliseconds, states drawn from `T`). Both
are valid and they are not interchangeable.

### Baselines

Each baseline replaces the policy with a rule and shares everything below it — the same
planner, band targets, observed-altitude coverage banking, and stop-once-saturated rule — so
a performance gap is attributable to the decision layer.

```julia
core = scripted_core(config; ref_ic = ic)

CyclicController(core, 2)                          # LOW, CORRECT×2, MID, ... then hold
GreedyController(core)                             # excurse to the least-sampled band
ThresholdController(core; max_residual = "R_OK")   # excurse only from a clean orbit
MPCController(; ref_ic = ic, mode = :position)     # hold only, no science
```

### Sweeps

```bash
KEY=sigma_nav_km       VALS=2,4,6,8  SEEDS=3  julia --project=experiments experiments/sweep.jl
KEY=thruster_sigma_pct VALS=0.7,2    SEEDS=5  julia --project=experiments experiments/sweep.jl
KEY=plume_gradient     VALS=0,1.5,4  DAYS=15  julia --project=experiments experiments/sweep.jl
```

Other knobs: `ARMS`, `DAYS`, `PLUME`, `OUT`. Each rollout is checkpointed on completion, so a
killed sweep keeps what it already flew and rerunning the same command resumes from disk. The
per-step trace is stored, not just the return, so a new metric or figure needs no re-run:

```julia
rows = load_sweep("artifacts/sweeps/sigma_nav_km")
rows[1]["actions"]                          # action per pass
rows[1]["peri_alts_km"]                     # where each pass actually went
rows[1]["residuals_km"]                     # planner residual per pass
rows[1]["obs_bins"], rows[1]["true_bins"]   # observed vs. actual region
```

A `plume_gradient` sweep needs one solved policy per value, since the reward changes. Nav and
thruster sweeps reuse a single policy while the world varies, which is the deployment
question: a precomputed policy meeting conditions it was not solved for.

### Regenerate the kernels

```bash
julia --project=experiments -t auto experiments/calibrate.jl     # ~7 min
```

The artifact path follows the config, so a noisy run never clobbers the noise-free
`tables.json`. Use `-t 1` for a noisy artifact meant to be reproduced exactly — the threaded
walk draws from a shared RNG, so `rng_seed` does not pin it.

## Figures

```julia
using SherpaOrbital, CairoMakie      # the caller supplies the plotting package

runs = ["POMDP" => res_pomdp, "Threshold" => res_thresh, "MPC hold" => res_mpc]
plot_baseline_comparison(runs, config; path = "figures/baselines", theme = :light)
```

Writes PDF, SVG and PNG, transparent so one figure reads on either background. Two panels:
cumulative science earned, and the fraction of passes flown degraded or worse, with a marker
where a run was lost. They are separate because a single scalar return conflates them — a run
that survives the full horizon can still score badly for expected loss it never incurred.

```julia
science_trace(res, config)    # (t_days, cumulative)
damage_trace(res)             # (t_days, residual_km, level, n_degraded, frac_degraded, lost_day)
delivery_trace(res, config)   # commanded vs. achieved periapsis altitude, per excursion
```

## Measured results

30-day horizon, seeds 0–2, CR3BP + Enceladus J2 truth. Both rows are matched: kernels
calibrated and rollout flown under the same thruster law, which is what `thruster_sigma_pct`
exists to guarantee.

| thruster | outcome | return (mean ± sd) | ΔV (m/s) | bands | samples |
|---|---|---|---|---|---|
| **0.7%, B-24 Model 1** (default) | **holds 30 d, 3/3** | **135.4 ± 1.4** | 116–126 | 3 | 12 |
| noise-free (optimistic corner) | holds 30 d, 3/3 | 142.5 ± 0.9 | 90–95 | 3 | 11–12 |

Execution error costs return but not the mission: every seed holds the full 30 days and banks
all three bands either way. The ~7-point gap is the policy correcting more often and earlier,
so at γ = 0.95 the science it defers is science discounted.

The mechanism is failed repair rather than crashes. `EXCURSE_HIGH` from `A34_44|R_DEGRADED`
arrives `R_OK` 100% of the time noise-free, 52% at 0.7%, and 40% at 2.0%; damage then
accumulates into the `R_CRITICAL` states where `P(lost)` really is ~0.7. Per-row
`P(lost or crashed)` moves by at most +0.10 between noise-free and 2%, while total-variation
distance over the successor distribution exceeds 0.10 on 24 of 60 rows.

Two behaviours worth noting, neither engineered. The policy corrects on roughly half of all
passes and interleaves excursions between corrections rather than chaining them. And it runs
LOW as a bounded campaign: `EXCURSE_LOW` fires a few times, spaced three passes apart, then
never again — the "excurse, then correct twice" pattern the measured kernels require.

`outcome = :idle` means the horizon was reached with no crash and no escape but the controller
stopped triggering before the end: survival, not a claim of active hold to the last second.

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
  export.jl               solved policy -> JSON, θ-keyed paths
  calibration/            measure the kernels from the truth model
  dynamics/               CR3BP (onboard) + J2 variants (truth), kept separate
  planner.jl              onboard burn planner (CR3BP only)
  baselines/mpc.jl        MPC hold baseline
  baselines/scripted.jl   cyclic / greedy / threshold baselines
  spacecraft/             thruster + nav models (explicit rng)
  common/simulate.jl      unified rollout harness
  common/figures.jl       comparison figures (CairoMakie, resolved at call time)
  common/checkpoint.jl    per-rollout sweep checkpoints
  common/report.jl        model + policy pretty-printing
experiments/              own Project.toml — isolates the solver dependency
  example.jl              solve, inspect, export
  calibrate.jl            regenerate artifacts/tables.json
  sweep.jl                sweep an axis, checkpointing per rollout
artifacts/                tables.json committed; policies and sweeps gitignored
test/                     runtests.jl
legacy/                   frozen Python, deliberately not ported
```

The library declares no solver dependency — SARSOP lives only in `experiments/Project.toml`.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                                 # full, ~5 min
julia --project=. -e 'using Pkg; Pkg.test(test_args=["plume","rewards"])'     # ~8 s
```

The orbit-geometry, family-continuation and rollout testsets take minutes, which is too slow
for an edit loop on the model layer; pass substrings to run only matching testsets.

## Physics conventions

- **Frame:** Saturn–Enceladus CR3BP rotating frame. Saturn at x = −μ, Enceladus at x = 1−μ.
- **Units:** km, km/s, s. ΔV costs in m/s.
- **Truth vs. onboard:** the truth model includes J2, the onboard planner is CR3BP only. The
  gap between them is the model uncertainty being studied, and the two are kept separate.
- **Stability:** all six Floquet multipliers lie on the unit circle, so the orbit is
  marginally stable rather than hyperbolically unstable. The fast divergence of an
  uncontrolled orbit is the J2 model gap, not orbital instability.

> **The orbit studied here is period-1, not MacKenzie's period-3.** We fly a period-1 member
> of the Saturn–Enceladus L1 halo family: 11.996 hr, one periapsis per revolution, south-polar
> periapsis, commanded between ~23 and ~46 km. MacKenzie §B.2.3 specifies a period-3 member
> instead, whose three geometrically distinct periapses buy ground-track diversity. That is a
> different orbit in the same family but the
> altitude/risk tradeoff the POMDP solves is the same either way.

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

## License

MIT — see [LICENSE](LICENSE).