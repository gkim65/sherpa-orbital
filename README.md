# SherpaOrbital — Enceladus Orbilander POMDP stationkeeping

Offline POMDP stationkeeping for the Enceladus Orbilander mission concept
(MacKenzie et al. 2020). The spacecraft holds an L1 halo orbit with periapsis over
Enceladus's south pole and must stationkeep without ground contact, trading **science**
(sampling a range of periapsis altitudes, where the plumes are denser lower down) against
**safety** (not crashing into Enceladus or escaping an unstable orbit).

> **On the orbit: this is a TEST SCENARIO, not MacKenzie's exact science orbit.**
>
> We fly a **period-1** L1 halo — true period **11.996 hr**, one periapsis per revolution,
> periapsis over the south pole at ~23–46 km depending on the commanded band. It is in the same
> family and altitude range as the mission concept's orbit and poses the same control problem:
> an orbit that must be actively held, where lower passes pay more science and cost more risk.
> That is what the POMDP formulation is being tested on, and it is sufficient for the policy
> question this repo is asking.
>
> MacKenzie §B.2.3 specifies a **period-3** orbit whose three periapses are geometrically
> distinct and grouped over the south pole, chosen for ground-track diversity. We do not have
> that orbit. Getting it is a branch-switching problem — locate a period-tripling bifurcation
> along the family and continue off it — and no such crossing has been shown to exist in this
> family at these altitudes. Future work, not a blocker.
>
> Reading older material in this repo: text before 2026-08-29 (including this README)
> described the orbit as period-3. That was wrong, and it survived because the 11.996 hr period
> happens to reproduce MacKenzie's ~12 hr periapsis spacing. It closes at `T/3` to 0.009 km —
> better than at the nominal 35.988 hr — its three apparent "passes" are the same point
> revisited to 3+ decimals in altitude, latitude and longitude, and it matches the JPL
> Three-Body Periodic Orbit Catalog at 11.99604 hr, whose L1 halo periods span 7.80–16.14 hr.
> `PERIOD1_TRIPLE_PERIOD_S` is 3x the true period, kept only because every controller
> measurement to date used it as the rollout horizon.
>
> On "unstable": stationkeeping is genuinely required — uncontrolled, the orbit runs
> 31 → 52 → 113 → 950 → 43,819 km in five passes under the truth model, and a 1 m error grows
> ~183x over 8 revolutions. The precise form is worth knowing: all six Floquet multipliers lie
> on the unit circle, so it is **marginally** stable rather than hyperbolically unstable, and
> the fast divergence is driven by the **J2 model gap** between the truth and onboard models.
> Growth is polynomial, not exponential.

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

# ⚠️ `use_binning = false` is REQUIRED, not a tuning choice — NativeSARSOP's
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

θ = 0 is the null hypothesis — no altitude gradient, every bin worth the same — and is the
structural sanity gate (every tilt vanishes, so every bin is exactly uniform).

The invariant five-bin mean is what makes returns comparable **across** θ: a high-θ
environment repriced altitude, it did not get richer. Regret compares columns of the matrix,
so an axis that inflated total value would confound "better policy" with "richer world".

**Usable range is θ ∈ [0, 4].** The tilt is clamped to keep probabilities non-negative, and
with the default `alt_rep_km` the mean depth is ≈ 0.534, so the shallowest bin pins at
θ ≈ 1.87 and the deepest two pin together by θ ≈ 4. Past that θ buys almost nothing — LOW−MID
stops growing — and the clamp lets the five-bin mean drift ~1.4% off its θ = 0 value. An
evenly spaced **`(0, 1, 2, 4)`** stays inside the usable range and gives four distinct models.

⚠️ `plume_gradient` reprices altitude but does NOT change the physics: the measured kernels
are altitude dynamics and are shared across θ. A high θ makes low passes more valuable, not
more survivable.

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

⚠️ This is the **truth-model** return (~10 s per 30-day rollout, real CR3BP dynamics), not
the discrete-model one from `POMDPs.simulate` with a `RolloutSimulator` (milliseconds,
states sampled from `T`). Both are valid; they are **not** interchangeable, and a regret
matrix must use one or the other throughout.

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
excursion from a dangerous one. Before this dimension existed the kernel reported
`P(loss) = 0.0` for the transition that actually kills the vehicle — correct for a *fresh*
excursion, but the danger is conditional on damage the state could not see. Neither more
calibration (which averages fresh and degraded together) nor larger loss penalties (which
multiply `P(loss)`, and anything × 0.0 is 0.0) can fix that.

**It is observed exactly**, unlike altitude. The residual is computed by the onboard solver
from the onboard model, not measured by a sensor, so the spacecraft knows it to machine
precision; modelling it as noisy would invent uncertainty that does not exist.

What the measured kernels say about recovery (tree depth 9):

```
CORRECT  A27_34   R_DEGRADED   P(damage decreases) = 1.000   P(LOST) = 0.000   n=2412
CORRECT  A27_34   R_CRITICAL                        0.928             0.072   n=2452
CORRECT  ABOVE_44 R_DEGRADED                        1.000             0.000   n=2792
CORRECT  A20_27   R_CRITICAL                        0.179             0.821   n=5373
```

Correcting reliably repairs a degraded orbit **everywhere except low**, where it mostly
fails and the vehicle is usually lost. That is the "LOW is special" result, measured.

Four things worth knowing about the formulation:

**Actions encode intent, not a burn vector.** A fixed menu of burn directions was measured
and shown to fail; the burn direction is solved live by the planner against the onboard
model, so the POMDP only chooses *what to attempt*.

**Science is banked on the OBSERVED altitude, not the commanded one.** A band pays when
the pass actually lands in its bin, so a missed excursion earns nothing and `CORRECT` banks
its own bin passively. Coverage therefore carries the observation model's ~15–20%
edge-driven misbin rate; report the science product with that attached.

**Science is banked only on survival.** A pass banks its band only if it did not go
terminal. That coupling is what forces the policy to sequence excursions rather than
attempt everything at once.

**The altitude bins bracket the controller's limit cycle.** `CORRECT` settles at 37.17 km,
which sits in `A34_44` — deliberately **not** a science band. Without that, two of three
bands were banked by doing nothing and the science/safety tradeoff was vacuous.

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
`[action][(alt_bin, residual_bin)]` and their columns are the joint successor
`(alt, residual)` — artifact `format: 3`. Both halves matter: pooling the kernels across
bands made `EXCURSE_LOW/MID/HIGH` identical in `T`, and keying on altitude alone averaged
fresh and degraded departures into one row that reported `P(loss) = 0.0` for the transition
that actually loses the vehicle. `load_tables` **rejects** older formats rather than
remapping them, since neither distinction is recoverable from the file.

> ### Read `meta.trials` before quoting a policy behaviour
> Of the 60 rows in the committed artifact, **44 clear `MIN_TRIALS_TRUSTED = 20`, 4 are thin
> (`0 < n < 20`), and 12 are unmeasured (`n = 0`) and filled**. The filled rows are
> `BELOW_20` at degraded or critical damage, plus `A34_44/R_CRITICAL` — combinations the
> vehicle does not reach. A filled row is not evidence, and the policy's behaviour there
> means nothing; `meta.unmeasured_rows` names them, and `_fill_unmeasured_row` inherits the
> same action's measured behaviour at the nearest less-degraded damage bin rather than
> inventing a risk-free self-transition.
>
> Also: kernels are noise-free unless calibrated with `noisy_thruster = true`, so any
> survival number derived from them is an **upper bound**, not feasibility.

---

## Repository layout

```
Project.toml            Julia package (SherpaOrbital)
Manifest.toml           committed — reproducible environment
src/
  SherpaOrbital.jl      module + exports
  StationkeepingPOMDP.jl  the @kwdef config struct — holds ALL of θ
  states.jl             (alt, visits, intensity) space, altitude binning
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
legacy/                 frozen Python, NOT part of the pipeline (see below)
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

⚠️ **A regret matrix built on noisy rollouts would be mostly noise.** At sd ≈ 68, resolving a
plausible ~20-point θ effect needs on the order of 50 seeds per cell; noise-free at sd ≈ 10
needs a handful. Either sweep noise-free, or calibrate *and* fly noisy with the cited
`(model = :gaussian_pct, sigma_pct = 2.0)` law.

Before the residual dimension and the reward fix this policy **escaped at 3.8 d** having
chosen `CORRECT` once in six passes.

Two behaviours worth noting, neither of which was engineered:

- **It corrects heavily** — roughly half of all passes — and interleaves excursions between
  corrections rather than chaining them.
- **It runs LOW as a bounded campaign.** `EXCURSE_LOW` fires at passes 11, 14, 17, 20 —
  four times, spaced three apart — and then never again in the remaining 40 passes. The
  3-pass spacing is "excurse, then correct twice", which is exactly the pattern the measured
  kernels say is required; the policy found it from the kernel, not from being told.

⚠️ `outcome = :idle` means the horizon was reached with no crash and no escape, but the
controller stopped triggering before the end — survival, not a claim of active hold to the
last second. And every number here is one policy at one θ; at θ = 8 one noisy seed crashes
at 11.2 d, in a run where `CORRECT` was chosen only 2 of 23 times.

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
julia --project=. -e 'using Pkg; Pkg.test()'    # 281/281, ~5 min
```

The orbit-geometry, family-continuation and rollout testsets take minutes, which is too
slow for an edit loop on the model layer. Pass substrings to run only matching testsets:

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["plume","rewards"])'   # ~8 s
```

The suite covers the **POMDP model layer** (state space, altitude binning, actions, measured
tables, transition/observation matrices, rewards, the plume gradient, config plumbing) plus
orbit geometry and the controller. The physics layer is additionally cross-checked by
`scratch/compare/*.jl` against frozen Python dumps — see the Session-5 log for why that
split is deliberate.

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