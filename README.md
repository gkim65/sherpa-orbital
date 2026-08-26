# SherpaOrbital — Enceladus Orbilander POMDP stationkeeping

Offline POMDP stationkeeping for the Enceladus Orbilander mission concept
(MacKenzie et al. 2020). The spacecraft holds a period-3 L1 halo orbit around Enceladus
and must stationkeep without ground contact, trading **science** (sampling a range of
periapsis altitudes) against **safety** (not crashing into Enceladus or escaping an
unstable orbit).

This is SHERPA-RPA Direction 3.

---

## Quick start

```julia
using Pkg
Pkg.activate("experiments")     # from the repo root
Pkg.instantiate()

using SherpaOrbital, NativeSARSOP, POMDPs

config = StationkeepingPOMDP()          # baseline scenario
pomdp  = build_pomdp(config)
policy = solve(SARSOPSolver(; precision = 1e-3, max_time = 30.0), pomdp)

print_policy_table(policy, config)
export_policy(policy, config)           # -> artifacts/policy.json
```

Or just run the worked example:

```bash
julia --project=experiments experiments/example.jl
```

---

## The model

|              | |
|--------------|--|
| **State**    | `(dev, cov)`. `dev` = apse-position deviation bin (`OK`/`DRIFT`/`FAR`, plus terminal `LOST`/`CRASHED`) — the safety variable. `cov` = 3-bit mask over science altitude bands (`LOW`/`MID`/`HIGH`) — the science variable. \|S\| = 3·2³ + 2 = 26 |
| **Actions**  | `OBSERVE`, `CORRECT`, `EXCURSE_LOW`, `EXCURSE_MID`, `EXCURSE_HIGH`. \|A\| = 5 |
| **Obs**      | Noisy read of the dev bin (Gaussian nav noise on the measured deviation). `cov` is known exactly. \|O\| = 5 |
| **Reward**   | `+r_science` per newly sampled band, −fuel per burn, large − on crash/escape |

Two things worth knowing about the formulation:

**Actions encode intent, not a burn vector.** A fixed menu of burn directions was measured
and shown to fail; the burn direction is solved live by the planner against the onboard
model, so the POMDP only chooses *what to attempt*.

**Science is banked only on survival.** An excursion adds its band to `cov` only if the
pass did not go terminal. That coupling is what forces the policy to sequence excursions
rather than attempt everything at once — and it is visible in the solved policy, which
excurses toward whichever band is unsampled from `OK`, but always `CORRECT`s from
`DRIFT`/`FAR`.

### Editing the scenario

Every hyperparameter is a keyword field with a literal default — the struct *is* the
config (see [src/StationkeepingPOMDP.jl](src/StationkeepingPOMDP.jl)):

```julia
StationkeepingPOMDP(; r_science = 40.0)                     # value science more
StationkeepingPOMDP(; dev_edges = (10.0, 50.0, 150.0))      # tighter safety bins
StationkeepingPOMDP(; discount = 0.99, fuel_weight = 2.0)   # patient, fuel-conscious
```

---

## Measured tables, not analytic guesses

The dev-transition kernels are **measured** from CR3BP+J2 experiments, not derived in
closed form. They live in [artifacts/tables.json](artifacts/tables.json) alongside their
provenance (which experiment produced each row, and how much to trust it), and are loaded
at model-build time:

```julia
tables = load_tables()          # validates: row-stochastic, correct dev ordering
validate_tables(tables)
```

`artifacts/` is committed on purpose. These are a scientific provenance record — being
able to see a probability change in a diff is how a re-measurement gets noticed.

> **Current status:** the kernels are still hand-transcribed from the Python experiment
> output rather than machine-generated, and the `DRIFT`/`FAR` rows rest on few trials.
> See the `meta.caveats` field in the artifact. Replacing this with a Julia calibration
> step that measures them directly is the next milestone.

---

## Repository layout

```
Project.toml            Julia package (SherpaOrbital)
Manifest.toml           committed — reproducible environment
src/
  SherpaOrbital.jl      module + exports
  StationkeepingPOMDP.jl  the @kwdef config struct
  states.jl             (dev, cov) space, binning, coverage bitmask
  actions.jl            action set, excursion -> band mapping
  observations.jl       O[s,o] — analytic Gaussian nav model
  tables.jl             load/write/validate the measured kernels
  transition.jl         T[s,a,s'] — includes the science-banking coupling
  rewards.jl            r(s,a) — science / fuel / expected terminal cost
  model.jl              build_pomdp
  export.jl             solved policy -> JSON
  common/report.jl      model + policy pretty-printing
experiments/            own Project.toml — isolates the solver dependency
  example.jl            worked end-to-end example
artifacts/              measured tables + exported policy (committed)
test/                   runtests.jl
legacy/                 frozen Python, NOT part of the pipeline (see below)
  russell-lara/         Russell & Lara (2009) Hill-problem study, frozen
  figures-reference/    matplotlib originals kept as Makie-port specs
```

The library declares **no solver dependency** — `NativeSARSOP` lives only in
`experiments/Project.toml`, so anyone who wants the model without the solver can have it.

---

## Status: port complete — the pipeline is 100% Julia

`python-legacy/` was **deleted** in Session 5. Every module in the live pipeline (CR3BP
dynamics, the halo-orbit differential corrector, the planner, the MPC baseline, the
spacecraft models, the unified rollout harness, calibration, the POMDP model and solver
pipeline) is Julia.

The Python reference is preserved as **frozen JSON dumps** in `scratch/compare/ref_*.json`
(gitignored, local). The `compare_*.jl` scripts diff Julia against those dumps and read
nothing else; the `ref_*.py` dumpers are retained as provenance and no longer run. See
`scratch/compare/README.md` for the pre-deletion audit and the list of retired rows.

What remains in `legacy/` is deliberately NOT ported: the Russell & Lara Hill-problem
study (a separate dynamical model, frozen, numpy-only) and the matplotlib figure scripts,
kept as specifications for a Makie port.

---

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'    # 59/59
```

The suite covers the **POMDP model layer** (state space, dev binning, actions, measured
tables, transition/observation matrices, rewards, config plumbing). The physics layer is
cross-checked by `scratch/compare/*.jl` against the frozen Python dumps rather than by
unit tests — see the Session-5 log for why that split is deliberate and when it changes.

---

## Physics conventions

- **Frame**: Saturn–Enceladus CR3BP rotating frame. Saturn at x = −μ, Enceladus at x = 1−μ.
- **Units**: km, km/s, s. ΔV costs in m/s.
- **Truth vs. onboard**: the truth model includes J2; the onboard model is CR3BP only.
  The gap between them is the model uncertainty being studied — the two are deliberately
  kept separate.

## Key references

- MacKenzie et al. (2020). *Enceladus Orbilander Mission Concept Study*, §B.2.3 — orbit
  parameters, thruster specs, stationkeeping strategy.
- Kim et al. (2025) —  Life Detection POMDP 
- Howell (1984). Three-Dimensional, Periodic, Halo Orbits. *Celestial Mechanics* 32(1).
- Iess et al. (2014). The Gravity Field and Interior Structure of Enceladus. *Science*.
- JPL Three-Body Periodic Orbits Catalog (DE440 ephemeris).

## License

MIT — see [LICENSE](LICENSE).