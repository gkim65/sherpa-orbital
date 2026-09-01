# SherpaOrbital — Enceladus Orbilander POMDP stationkeeping

Offline POMDP stationkeeping for the Enceladus Orbilander mission concept
(MacKenzie et al. 2020). The spacecraft holds an L1 halo orbit with periapsis over
Enceladus's south pole and must stationkeep without ground contact, trading **science**
(sampling a range of periapsis altitudes, where the plumes are denser lower down) against
**safety** (not crashing into Enceladus or escaping the orbit).

> **The orbit: a period-1 L1 halo, studied for the altitude problem.**
>
> We fly a period-1 member of the Saturn–Enceladus L1 halo family — 11.996 hr, one periapsis
> per revolution, south-polar periapsis, commanded between ~23 and ~46 km. It poses the
> problem this repo is about: an orbit that must be actively held, where lower passes pay
> more science and cost more risk, and where the controller has to choose how deep to go.
>
> MacKenzie §B.2.3 specifies a period-3 member instead, whose three geometrically distinct
> periapses buy ground-track diversity. That is a different orbit in the same family, and
> reaching it is a branch-switching problem — no period-tripling bifurcation has been shown
> to exist here at these altitudes. The altitude/risk tradeoff the POMDP solves is the same
> either way.
>
> Stationkeeping is genuinely required: uncontrolled, the orbit runs away within a few
> passes under the truth model. All six Floquet multipliers lie on the unit circle, so it is
> **marginally** stable rather than hyperbolically unstable — the fast divergence is the
> **J2 model gap** between the truth and onboard models, not orbital instability.

---

## Quick start

```julia
using Pkg
Pkg.activate("experiments")     # from the repo root
Pkg.instantiate()

using SherpaOrbital, NativeSARSOP, POMDPs

config = StationkeepingPOMDP()          # baseline scenario
pomdp  = build_pomdp(config)

# NOTE: `use_binning = false` is required, not a tuning choice — NativeSARSOP's
# `entropy(::SparseVector)` throws `InexactError` on this model otherwise.
policy = solve(SARSOPSolver(; precision = 1e-3, max_time = 120.0,
                            use_binning = false), pomdp)

print_policy_table(policy, config)
export_policy(policy, config)           # -> artifacts/policy.json
```

Or just run the worked example:

```bash
julia --project=experiments experiments/example.jl
```

### Using it as a library

The package is consumed by [ClusterPolicyGen](https://github.com/gkim65/ClusterPolicyGen)
for regret-based policy reuse:

```julia
Pkg.add(url = "https://github.com/gkim65/sherpa-orbital")
```

It satisfies that project's env-spec contract directly — `make_pomdp(spec, θ)` is
`build_pomdp(StationkeepingPOMDP(; θ...))`, and `POMDPs.RolloutSimulator` runs on the
resulting `QuickPOMDP`, so the discounted return regret needs is available for free.

---

## Sweeping the environment (θ)

The struct **is** the environment, so a parameterized POMDP family is one loop:

```julia
for θ in [(plume_gradient = g,) for g in (0.0, 1.0, 2.0, 4.0)]
    cfg = StationkeepingPOMDP(; θ...)

    tbl = if needs_recalibration(θ)        # dynamics axis -> re-measure (~7 min)
        rows, diag = calibrate_tables(cfg)
        t = tables_from_rows(rows, diag)
        write_tables(t; path = theta_path("tables", θ)); t
    else                                    # analytic axis -> reuse kernels (~0 s)
        nothing
    end

    policy = solve(solver, build_pomdp(cfg; tables = tbl))
    export_policy(policy, cfg; path = theta_path("policy", θ))
end
```

Three axes, all shaped identically as scalar fields on the config:

| axis | field | paper category | cost per θ |
|---|---|---|---|
| nav noise | `sigma_nav_km` | `O_θ` | ~2 s — analytic (`observation_matrix`) |
| plume gradient | `plume_gradient` | `T_θ` | ~2 s — analytic (`transition_matrix`) |
| thruster error | `noisy_thruster`, `thruster_kwargs` | `T_θ` | **~7 min — re-measures the kernels** |

`needs_recalibration(θ)` reports which of the two regimes a θ falls into. `theta_path`
keys artifacts by θ so a sweep cannot overwrite its own output.

**Choosing `plume_gradient` values — θ REDISTRIBUTES, it does not inflate.** θ is the slope
of a linear tilt in band depth, centered on the mean depth over `ALT_BINS`, so bins deeper
than the mean gain exactly what shallower bins lose. Expected science per pass
(`r_science × E[value]`), measured:

| bin | depth | θ=0 | θ=1 | θ=2 | θ=4 | θ=8 |
|---|---|---|---|---|---|---|
| `BELOW_20` | 1.00 | 13.00 | 15.17 | 17.35 | 17.67 | 17.67 |
| `A20_27` (LOW) | 0.80 | 13.00 | 14.26 | 15.51 | 17.67 | 17.67 |
| `A27_34` (MID) | 0.55 | 13.00 | 13.09 | 13.18 | 13.36 | 13.72 |
| `A34_44` (cycle) | 0.31 | 13.00 | 11.97 | 10.95 | 8.89 | 8.33 |
| `ABOVE_44` (HIGH) | 0.00 | 13.00 | 10.51 | 8.33 | 8.33 | 8.33 |
| **five-bin mean** | | **0.6500** | **0.6500** | **0.6532** | **0.6592** | **0.6572** |

θ = 0 is the null hypothesis and the structural sanity gate: every tilt vanishes, so every
bin is worth the same. The invariant five-bin mean is what makes returns comparable across
θ — regret compares matrix columns, so an axis that inflated total value would confound
"better policy" with "richer world".

**Usable range is θ ∈ [0, 4].** The tilt is clamped to keep probabilities non-negative, so
the shallowest bin pins around θ ≈ 1.9 and the deepest two by θ ≈ 4; past that θ buys
almost nothing and the clamp drifts the mean ~1.4%. **`(0, 1, 2, 4)`** gives four distinct
models inside the range.

**Note:** θ reprices altitude, it does not change the physics. The kernels are altitude
dynamics and are shared across θ, so a high θ makes low passes more valuable, not more
survivable.

`S`, `A`, `O` and `γ` stay fixed across a sweep, as the formulation requires — note this
makes `plume_levels` (which changes `|S|`) **not** a legal θ.

### Evaluating a policy: the discounted return

Regret is a difference of value functions, `R_ij = V^{π_i}_{θ_i}(b₀) − V^{π_j}_{θ_i}(b₀)`
with `V^π_θ(b₀) = E[Σ γᵗ R(sₜ,aₜ)]`, so an evaluator needs a **discounted return**.
`run_rollout` reports survival, ΔV, band visits and achieved altitudes — none of which is
one. [`discounted_return`](src/common/simulate.jl) supplies it:

```julia
res = run_rollout(SARSOPController(load_policy(p_i), ref_ic = ic), ic,
                  cr3bp_j2_eom!, PERIOD1_TRIPLE_PERIOD_S, 30*24*3600.0)

V = discounted_return(res, StationkeepingPOMDP(; plume_gradient = θ_j))   # V^{π_i}_{θ_j}
```

Scoring is **post-hoc over the trace**, deliberately: the reward belongs to a θ, the
trajectory belongs to the physics, and one trajectory is legitimately scored under several θ
to fill a matrix row. So an N×N matrix costs **N rollouts, not N²**.

**Note:** this is the **truth-model** return (~10 s per 30-day rollout, real CR3BP
dynamics), not the discrete-model one from `POMDPs.simulate` with a `RolloutSimulator`
(milliseconds, states sampled from `T`). Both are valid, they are not interchangeable, and
a regret matrix must use one or the other throughout.

---

## The model

|              | |
|--------------|--|
| **State**    | `(alt, visits, intensity, residual)`. `alt` = achieved periapsis-**altitude** bin (`BELOW_20`/`A20_27`/`A27_34`/`A34_44`/`ABOVE_44`, plus terminal `CRASHED`/`LOST` — there is no separate safety variable). `visits` = per-band sample **count**, saturating at `visit_cap`. `intensity` = the plume sample intensity the last pass yielded, `1:plume_levels`. `residual` = **orbit-damage** bin (`R_OK`/`R_DEGRADED`/`R_CRITICAL`) — see below. \|S\| = 5·(cap+1)³·k·3 + 2 = **5627** at cap 4, k 3 |
| **Actions**  | `CORRECT`, `EXCURSE_LOW`, `EXCURSE_MID`, `EXCURSE_HIGH`. \|A\| = **4**. Every action burns — **there is no `OBSERVE`** (removed 2026-08-30: on this orbit a no-burn coast is not a decision, it is a slow loss of the vehicle) |
| **Obs**      | Noisy read of the achieved periapsis altitude (Gaussian nav noise, σ = `sigma_nav_km`), binned. Visit counts, intensity and residual are all known exactly given the observed bin. \|O\| = 7 |
| **Reward**   | `r_science × visit_factor × value(intensity) × damage_yield(residual)`, −fuel per burn, large − on crash/escape. **Every pass collects** — sampling is passive |

### The residual (orbit-damage) dimension

`residual` bins the **onboard `solve_burn` residual**: how badly the six apse constraints
failed to be satisfiable by one impulse, i.e. how far the orbit has drifted from what the
controller can fix in a single burn. It is the model's damage variable, and it exists
because without it *the policy cannot represent how degraded the orbit is*, so it cannot
learn that a LOW excursion needs two corrections before the next one.

Bin edges are **measured, not chosen** — over 3671 pooled pass-to-pass transitions, scored
on whether the *next* pass loses the apse pair (the failure that flies an uncontrolled pass
and loses the vehicle):

| bin | residual | n | P(lose apse pair next pass) |
|---|---|---|---|
| `R_OK` | < 15 km | 2363 | **0.000** |
| `R_DEGRADED` | 15–25 km | 661 | 0.009 |
| `R_CRITICAL` | ≥ 25 km | 647 | 0.062 |

The hard zero on `R_OK` is the load-bearing property: it is what lets the policy tell a safe
excursion from a dangerous one. Without it the kernel reports `P(loss) = 0.0` for the
transition that actually kills the vehicle — correct for a *fresh* excursion, but the danger
is conditional on damage the state could not see. Neither more calibration nor larger loss
penalties fix that.

**It is observed exactly**, unlike altitude: the onboard solver computes it rather than a
sensor measuring it, so modelling it as noisy would invent uncertainty that does not exist.

What the measured kernels say about recovery (tree depth 9):

```
CORRECT  A27_34   R_DEGRADED   P(damage decreases) = 1.000   P(LOST) = 0.000   n=2412
CORRECT  A27_34   R_CRITICAL                        0.928             0.072   n=2452
CORRECT  ABOVE_44 R_DEGRADED                        1.000             0.000   n=2792
CORRECT  A20_27   R_CRITICAL                        0.179             0.821   n=5373
```

Correcting reliably repairs a degraded orbit **everywhere except low**, where it mostly
fails and the vehicle is usually lost. That is the "LOW is special" result, measured.

Three properties of the formulation worth knowing:

- **Actions encode intent, not a burn vector.** The direction is solved live by the planner
  against the onboard model; the POMDP only chooses *what to attempt*.
- **Science is banked on the OBSERVED altitude, and only on survival.** A band pays when the
  pass lands in its bin and did not go terminal, so a missed excursion earns nothing and
  `CORRECT` banks its own bin passively. Coverage carries the observation model's ~15–20%
  edge-driven misbin rate — report the science product with that attached.
- **The altitude bins bracket the controller's limit cycle.** `CORRECT` settles at 37.17 km
  in `A34_44`, deliberately not a science band. Otherwise bands are banked by doing nothing
  and the tradeoff is vacuous.

### Editing the scenario

Every hyperparameter is a keyword field with a literal default — the struct *is* the
config (see [src/StationkeepingPOMDP.jl](src/StationkeepingPOMDP.jl)):

```julia
StationkeepingPOMDP(; r_science = 40.0)                     # value science more
StationkeepingPOMDP(; plume_gradient = 4.0)                 # steep plume altitude gradient
StationkeepingPOMDP(; discount = 0.99, fuel_weight = 2.0)   # patient, fuel-conscious
```

`fuel_weight` defaults to **0** — the study is science yield under environmental
uncertainty, not fuel feasibility. The fuel machinery is intact, so raising the weight
restores the tradeoff with no code change.

---

## Measured tables, not analytic guesses

The altitude-transition kernels are **measured** from the Julia truth model (CR3BP + J2),
not derived in closed form. They live in [artifacts/tables.json](artifacts/tables.json)
alongside their provenance — θ (which environment), effort (how hard it was sampled),
per-row trial counts — and are loaded at model-build time:

```julia
tables = load_tables()          # validates: row-stochastic, correct altitude ordering
validate_tables(tables)
```

Regenerate them with:

```bash
julia --project=experiments experiments/calibrate.jl     # ~7 min
```

`artifacts/tables.json` is committed on purpose: it is a scientific provenance record, and
being able to see a probability change in a diff is how a re-measurement gets noticed.
Exported policies are **not** committed — they are large and derived, and `.gitignore`
excludes `artifacts/policy.json` and the θ-keyed `artifacts/policy_*` a sweep writes.

**One kernel per action, conditioned on orbit damage.** Rows are keyed
`[action][(alt_bin, residual_bin)]`, columns are the joint successor `(alt, residual)`.
Both halves matter: pooling across bands makes the excursions identical in `T`, and keying
on altitude alone averages fresh and degraded departures into a row that reports
`P(loss) = 0.0` for the transition that loses the vehicle. `load_tables` rejects older
artifact formats rather than remapping them.

> ### Read `meta.trials` before quoting a policy behaviour
> Of the 60 rows in the committed artifact, **44 clear `MIN_TRIALS_TRUSTED = 20`, 4 are thin
> (`0 < n < 20`), and 12 are unmeasured (`n = 0`) and filled** — `BELOW_20` at degraded or
> critical damage, plus `A34_44/R_CRITICAL`, combinations the vehicle does not reach. A
> filled row is not evidence; `meta.unmeasured_rows` names them, and `_fill_unmeasured_row`
> inherits the nearest less-degraded measured row rather than inventing a risk-free
> self-transition.
>
> Kernels are noise-free unless calibrated with `noisy_thruster = true`, so any survival
> number derived from them is an **upper bound**, not feasibility.

---

## Repository layout

```
Project.toml            Julia package (SherpaOrbital)
Manifest.toml           committed — reproducible environment
src/
  SherpaOrbital.jl      module + exports
  StationkeepingPOMDP.jl  the @kwdef config struct — holds ALL of θ
  states.jl             (alt, visits, intensity, residual) space, altitude binning
  actions.jl            action set, excursion -> band mapping
  observations.jl       O[s,o] — analytic Gaussian nav model
  plume.jl              P_θ(intensity | band) — the plume altitude gradient
  tables.jl             load/write/validate the measured kernels
  transition.jl         T[s,a,s'] — science banking + the intensity draw
  rewards.jl            r(s,a) — science / fuel / expected terminal cost
  model.jl              build_pomdp
  export.jl             solved policy -> JSON, θ-keyed artifact paths
  calibration/          MEASURE the kernels from the truth model
  dynamics/             CR3BP (onboard) + J2 variants (truth) — kept separate
  planner.jl            onboard burn planner (CR3BP only)
  baselines/mpc.jl      MPC baseline
  spacecraft/           thruster + nav models (explicit rng)
  common/simulate.jl    unified rollout harness
  common/report.jl      model + policy pretty-printing
experiments/            own Project.toml — isolates the solver dependency
  example.jl            worked end-to-end example
  calibrate.jl          regenerates artifacts/tables.json
artifacts/              tables.json committed; exported policies gitignored
test/                   runtests.jl
legacy/                 frozen Python, deliberately not ported
  russell-lara/         Russell & Lara (2009) Hill-problem study, frozen
  figures-reference/    matplotlib originals kept as Makie-port specs
```

The library declares **no solver dependency** — `NativeSARSOP` lives only in
`experiments/Project.toml`, so anyone who wants the model without the solver can have it.

---

## Current result: the policy holds the orbit for 30 days

Rolling the solved θ = 0 policy against the CR3BP + Enceladus J2 truth model, 30-day
horizon, seeds 0–2:

| thruster | outcome | return (mean ± sd) | ΔV (m/s) | bands | samples |
|---|---|---|---|---|---|
| **noise-free** (matched) | **holds 30 d, 5/5** | **152.1 ± 9.6** | 90–95 | 3 | 11–12 |
| noisy (mismatched) | holds 30 d, 4/5 | 115.4 ± 67.7 | 73–187 | 3 | 7–12 |

**Noise-free is the matched experiment and the one to quote.** The committed kernels are
calibrated with `noisy_thruster = false`, so a noisy rollout of a policy solved against them
is testing a model mismatch, not the policy — and `run_rollout` therefore defaults to
`noisy_thruster = false`.

The mismatch is not cosmetic. Under noise the return spread is **145% of the mean** against
15% noise-free, one seed in five crashes (return −3.7), an `EXCURSE_LOW` commanded at 23.5 km
lands at 14.65–16.94 km — in `BELOW_20`, not the LOW band — and the residual runs 45–60 km,
deep in `R_CRITICAL`. Note also that the noisy path uses the legacy `:uniform` error law
unless told otherwise (0–20% underburn, mean 10% short, never over), which is ~10×
MacKenzie Exhibit B-24's symmetric 0.7–2.0% and therefore a worst case rather than a
realistic thruster.

**A regret matrix built on noisy rollouts would be mostly noise.** At sd ≈ 68, resolving a
plausible ~20-point θ effect needs ~50 seeds per cell; noise-free at sd ≈ 10 needs a
handful. Either sweep noise-free, or calibrate *and* fly noisy with
`(model = :gaussian_pct, sigma_pct = 2.0)`.

Two behaviours worth noting, neither engineered:

- **It corrects heavily** — roughly half of all passes — and interleaves excursions between
  corrections rather than chaining them.
- **It runs LOW as a bounded campaign.** `EXCURSE_LOW` fires four times, spaced three passes
  apart, then never again. That spacing is "excurse, then correct twice", the pattern the
  measured kernels say is required — found from the kernel, not from being told.

**Note:** `outcome = :idle` means the horizon was reached with no crash and no escape but
the controller stopped triggering before the end — survival, not a claim of active hold to
the last second.

---

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'    # 281/281, ~5 min
```

The orbit-geometry, family-continuation and rollout testsets take minutes, which is too
slow for an edit loop on the model layer. Pass substrings to run only matching testsets:

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["plume","rewards"])'   # ~8 s
```

The suite covers the POMDP model layer (state space, binning, actions, measured tables,
transition/observation matrices, rewards, the plume gradient, config plumbing) plus orbit
geometry and the controller.

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