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

using SherpaOrbital, SARSOP, POMDPs

config = StationkeepingPOMDP()          # baseline: NOISY at 0.7% (B-24 Model 1)
pomdp  = build_pomdp(config)

# NOTE: SARSOP.jl, not NativeSARSOP — the latter stops after 4 iterations on this model
# with no error raised, giving the same 22-alpha-vector policy at every θ. Details in
# `experiments/example.jl`.
# `policy_filename`/`pomdp_filename` default to `policy.out`/`model.pomdpx` in the
# WORKING directory — name them so concurrent solves cannot overwrite each other, and so
# the policy can be reloaded later without re-solving.
policy = solve(SARSOP.SARSOPSolver(; precision = 1e-3, timeout = 900.0,
                                   pomdp_filename  = "model.pomdpx",
                                   policy_filename = "policy.out"), pomdp)

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
| thruster error | `noisy_thruster`, `thruster_sigma_pct` | `T_θ` | **~6 min — re-measures the kernels** |

`needs_recalibration(θ)` reports which of the two regimes a θ falls into. `theta_path`
keys artifacts by θ so a sweep cannot overwrite its own output.

### Protocol: config -> kernels -> solve -> rollout

The transition kernels are **measured**, so `noisy_thruster` / `thruster_sigma_pct` are not
free parameters: a policy solved against kernels measured at a different noise level is a
model-mismatch measurement, not a policy measurement. The five steps below keep the law
used to MEASURE and the law used to FLY the same thing.

**1. The config picks the kernels.** The artifact path is derived, never typed, so the file
a calibration writes is the file a solve reads:

| config | artifact (`resolve_tables_path`) |
|---|---|
| `StationkeepingPOMDP()` — noisy 0.7%, the DEFAULT | `artifacts/tables_noisy_gaussian0.7.json` |
| `thruster_sigma_pct = 2.0` | `artifacts/tables_noisy_gaussian2.0.json` |
| `noisy_thruster = false` | `artifacts/tables.json` |

The default is **noisy at B-24 Model 1 (0.7%)**: execution error is the deployment
environment, and a matched noisy setup holds the orbit for 30 days (see below), so there is
no reason for the optimistic corner to be the default. `noisy_thruster = false` stays
reachable as the mismatch-free reference.

**2. Measure the kernels, if that σ has none yet.** The depth is argument 1, the 1σ
burn-magnitude error in percent is argument 2; omit it for noise-free. B-24 Model 2 is 2.0,
Model 1 is 0.7 (`thruster.jl`).

```bash
julia --project=experiments -t auto experiments/calibrate.jl 7        # -> tables.json
julia --project=experiments -t 1    experiments/calibrate.jl 7 2.0    # -> ..._gaussian2.0.json
julia --project=experiments -t 1    experiments/calibrate.jl 7 0.7    # -> ..._gaussian0.7.json
```

NOTE: `-t 1` for a NOISY calibration. Threaded, the tree draws from a shared RNG, so
`rng_seed` does not pin the artifact — at 2.0% / depth 7, `-t auto` and `-t 1` left 11 and
8 rows unmeasured from the same seed. Noise-free is thread-invariant, so `-t auto` is fine
there and several times faster (365 s serial vs 181 s threaded at depth 7).

**3. Load and CHECK the kernels.** Missing kernels throw and name the command to run; a
θ mismatch does not, so check it when you override `tables_path`:

```julia
cfg = StationkeepingPOMDP(; noisy_thruster = true, thruster_sigma_pct = 2.0)
tbl = load_tables(cfg)
tbl.meta["theta"]            # noisy_thruster => true, thruster_sigma_pct => 2.0
tbl.meta["unmeasured_rows"]  # FABRICATED rows — behaviour there is not evidence
```

**4. Solve with SARSOP.jl.** Name the two output files: the defaults (`model.pomdpx`,
`policy.out`) land in the WORKING directory, so concurrent solves silently overwrite each
other. The names are free — keep the `.out` extension, which is what downstream code keys
on to tell a SARSOP policy from a serialized one.

```julia
solver = SARSOP.SARSOPSolver(; precision = 1e-3, timeout = 3600.0,
                             pomdp_filename  = "model_sigma2.0.pomdpx",
                             policy_filename = "policy_sigma2.0.out")
policy = solve(solver, build_pomdp(cfg))
```

NOTE: check the solve CONVERGED rather than hitting `timeout` — a capped solve returns a
truncated policy with no error. Solver cost varies enormously with how much of the model is
measured at depth 7:

| kernels | fabricated rows | alpha vectors | solver time |
|---|---|---|---|
| noise-free | 12 | 136 | 31 s |
| 0.7% (B-24 Model 1) | 12 | 741 | 213 s |
| 2.0% (B-24 Model 2) | 8 | 12757 | >900 s, did not converge |

Cost is steeply nonlinear in σ, and σ and coverage are not separable: more execution error
drives the vehicle into critically-degraded states more often, so those rows get MEASURED
rather than fabricated, and a real measurement there is sharp where a fabricated row —
inherited from the nearest less-degraded row — is smooth. A cheap low-σ solve is therefore
partly cheap because the model knows less about the dangerous states.

Most of the WALL clock is pomdpx serialisation, not search (534 s total for that 31 s
solve), so cache the pomdpx rather than re-solving to re-run a rollout.

**5a. Rollout in the SAME environment.** `thruster_sigma_pct` must equal what the kernels
were measured at:

```julia
c = SARSOPController(policy, cfg; ref_ic = ic)          # solved policy directly, no files
r = run_rollout(c, ic, cr3bp_j2_eom!, period_s / 3, 30 * 86400.0;
                rng = Xoshiro(0),
                noisy_thruster = true, thruster_sigma_pct = 2.0)
```

Check `r.n_failed_solves` before reading `r.outcome` — a non-converged solve returns ΔV = 0,
which is indistinguishable in the summary from choosing to coast.

**5b. Rollout in a DIFFERENT environment (a regret cell).** The policy supplies the alpha
vectors; the config supplies `T`/`O`, so the belief filter runs on θ_j's dynamics:

```julia
cfg_j = StationkeepingPOMDP(; plume_gradient = 4.0)     # theta_j
c     = SARSOPController(policy, cfg_j; ref_ic = ic)    # pi_i flown in theta_j
```

NOTE: if θ_j moves a RECALIBRATION axis (`needs_recalibration`), θ_j needs its own measured
kernels — `sigma_nav_km` and `plume_gradient` are analytic and reuse the existing ones. Only
`|S|` is checked, so a config that enumerates the same `|S|` differently misindexes
silently; sharing S/A/O/γ across the family, which the regret formulation requires anyway,
is what makes this safe.

**Reloading a policy from an earlier session** — the solver already wrote it:

```julia
pol = SARSOP.load_policy(build_pomdp(cfg), "policy_sigma2.0.out")   # qualify it
c   = SARSOPController(pol, cfg; ref_ic = ic)
```

NOTE: prefer either of the above over [`export_policy`](src/export.jl) +
[`load_policy`](src/common/simulate.jl) for anything you are going to FLY. That JSON carries
the dense `T[s][a][s']` — |S|^2 |A| floats, gigabytes at |S| = 5627 — so writing and
re-parsing it dominates a solve-then-fly script: 19 s via the policy object against ~1 hour
via the JSON, measured. It stays the right choice as a committed, self-describing ARCHIVE
and for a θ-keyed policy library that outlives the process that solved it.

If the artifact for a config does not exist, `load_tables` **throws and names the exact
command to run**. It does not fall back to `tables.json` — that fallback used to solve a
noisy θ against noise-free measurements with no error anywhere.

Pass `tables_path` to override the derived path, or `tables_path_for(cfg; dir = ...)` to
keep a sweep's artifacts outside the package. They only need to live in `artifacts/` if a
git-URL dependent must resolve them from the Julia package cache.

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
# π_i flown in θ_j: the policy comes from θ_i, the T/O the belief filter runs on come
# from the config passed here. That substitution IS the regret cell.
res = run_rollout(SARSOPController(pol_i, StationkeepingPOMDP(; plume_gradient = θ_j);
                                   ref_ic = ic), ic,
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

The library declares **no solver dependency** — `SARSOP` lives only in
`experiments/Project.toml`, so anyone who wants the model without a solver can have it.

---

## Current result: the policy holds the orbit for 30 days

Rolling the solved θ = 0 policy against the CR3BP + Enceladus J2 truth model, 30-day
horizon, seeds 0–2:

Both rows below are MATCHED — kernels calibrated and rollout flown under the same
thruster law, which is what `thruster_sigma_pct` exists to guarantee:

| thruster | outcome | return (mean ± sd) | ΔV (m/s) | bands | samples |
|---|---|---|---|---|---|
| **0.7%, B-24 Model 1** (default) | **holds 30 d, 3/3** | **135.4 ± 1.4** | 116–126 | 3 | 12 |
| noise-free (optimistic corner) | holds 30 d, 3/3 | 142.5 ± 0.9 | 90–95 | 3 | 11–12 |

**Execution error costs return but not the mission.** Every seed holds the full 30 days and
banks all three bands either way; the ~7-point gap is the policy correcting more often (34
`CORRECT` against 30) and correcting EARLIER, so at γ = 0.95 the science it defers is
science discounted. Note `fuel_weight = 0.0` by default, so this is not a fuel penalty.

The mechanism is visible in the kernels: `EXCURSE_HIGH` from `A34_44|R_DEGRADED` arrives
`R_OK` 100% of the time noise-free, 52% at 0.7%, and 40% at 2.0%. Burns stop REPAIRING the
orbit rather than crashing the vehicle, so damage accumulates and more corrections are
needed. Per-row `P(lost or crashed)` moves by at most +0.10.

NOTE: 2.0% (Model 2) is calibrated and committed but its 30-day rollout is not measured
here — that model needs more than 900 s to solve (see the solve-cost table above).

**What execution error actually costs, measured at B-24 Model 2 (2.0% 1σ).** Comparing the
noise-free kernels against `tables_noisy_gaussian2.0.json`:

- **Survival barely moves.** Per-row `P(lost or crashed)` shifts by −0.008 to +0.104.
  `EXCURSE_LOW` from `A20_27|R_CRITICAL` is 0.707 noise-free and 0.716 at 2% — that ~0.7 is
  a property of the **orbit geometry**, not of execution error.
- **But the transition model moves a lot.** Total-variation distance over the successor
  distribution exceeds 0.10 on 24 of 60 rows (median 0.041, mean 0.151).
- **The mechanism is failed repair, not crashes.** `EXCURSE_HIGH` from `A34_44|R_DEGRADED`
  arrives `R_OK` 100% of the time noise-free but only 40% at 2% (51.4% `R_DEGRADED`, 8.6%
  `R_CRITICAL`). Damage then accumulates into the `R_CRITICAL` states where `P(lost)` really
  is ~0.7. Healthy origins barely move (`EXCURSE_LOW / A20_27|R_OK` has TV 0.013).

**A regret matrix built on noisy rollouts would be mostly noise.** At sd ≈ 68, resolving a
plausible ~20-point θ effect needs ~50 seeds per cell; noise-free at sd ≈ 10 needs a
handful. Either sweep noise-free, or calibrate *and* fly at the same `thruster_sigma_pct`.

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