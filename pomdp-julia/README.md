# pomdp-julia — toy Enceladus stationkeeping POMDP

A tiny, self-contained POMDP that proves the **POMDPs.jl + NativeSARSOP**
toolchain and the stationkeeping formulation for the SHERPA Direction-3 project.
It is deliberately NOT a fidelity model — the dynamics are a crude drift+burn
placeholder, isolated behind one function so the real CR3BP can be swapped in
later without touching the POMDP.

## Layout
```
pomdp-julia/
├── Project.toml                 # isolated Julia env (does not touch Python src/)
└── src/
    ├── dynamics.jl              # crude drift+burn model + MC table builders  <-- the swappable seam
    ├── stationkeeping_pomdp.jl  # the QuickPOMDP definition (states/actions/obs/reward)
    ├── solve.jl                 # solve with NativeSARSOP + simulate + print policy
    └── export_policy.jl         # solve + EXPORT policy/tables to JSON for the Python rollout
```

## Run
```bash
cd pomdp-julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # first time only
julia --project=. src/solve.jl            # solve + print the greedy policy
julia --project=. src/export_policy.jl    # solve + write ../policy/sarsop_policy.json
```

## Exporting the policy to Python (the high-fidelity rollout seam)
The toy is solved here in Julia, but the policy is TESTED against the real CR3BP
truth model in Python — that is where the accurate integrator and the Phase-2
noise models live. `src/export_policy.jl` is the one-way, file-based seam between
the two languages:

- It solves the toy with NativeSARSOP and writes **`../policy/sarsop_policy.json`**
  (repo-level `policy/` dir), containing: the SARSOP alpha vectors + their action
  indices, the toy transition/observation tables `T[s,a,s']` / `O[s',o]`, and all
  labels + bin edges. (No JSON.jl dependency — a tiny hand-rolled writer is used.)
- The Python side, **`baselines/pomdp_rollout.py`**, loads that JSON, reproduces
  `AlphaVectorPolicy` (argmax α·b) + the discrete Bayes belief filter, and rolls
  the policy out against `cr3bp_j2_eom` / `cr3bp_saturn_enc_j2_eom` with realistic
  burn (`thruster.apply_dv_noisy`) and nav (`nav.observe_altitude`) noise. Run it:
  `python -m scripts.pomdp_rollout_feasibility`.
- The policy crosses the boundary EXACTLY ONCE, as this file — no live PyCall
  bridge. Rationale + full results are in
  `../docs/session-log/2026-07-06-pomdp-rollout.md`.

### Key finding from the rollout (why this matters)
Tested against the REAL dynamics and compared to the deterministic MPC baseline
(MacKenzie Strategy 3, apse position-vector targeting):
- **CR3BP+EncJ2: the orbit IS holdable** — MPC-S3 holds it 30 days, but this TOY
  policy fails in ~1 day. The gap is the toy's crude **scalar periapsis-raise
  action** (it cannot do the apse-POSITION targeting MPC uses), NOT the orbit.
- **CR3BP+SaturnJ2: both fail fast** — that rung is instability/cadence-limited;
  the lever there is orbit choice / control cadence, not the belief model.
This closed a circularity: previously the policy was only re-tested against the
same crude model that generated its tables (self-consistency, not real-orbit
holding). See the session log for the corrected Strategy-3 numbers.

## Formulation (toy)
- **State**: periapsis-altitude bin — CRASHED (<5 km, terminal) / LOW (5–25) /
  NOMINAL (25–60) / HIGH (60–120) / ESCAPED (>120 km, terminal). Bin edges from
  the MacKenzie apse bands + crash/escape thresholds in `../src/constants.py`.
- **Action**: NO_BURN / SMALL_BURN / LARGE_BURN — fixed ΔV raises applied with
  random burn efficiency η_eff ~ Uniform(0.8, 1.0).
- **Observation**: noisy altitude bin from 2 km Gaussian nav noise (partial
  observability).
- **Reward**: +hold NOMINAL, −LOW, −−CRASH/ESCAPE, −fuel.
- **Transition table**: Monte-Carlo of the crude drift+burn model (PLACEHOLDER
  for the real CR3BP). SARSOP solves offline from the precomputed tables and
  never calls the dynamics.

## Swapping in the real dynamics later
Replace the body of `step_altitude` in `dynamics.jl` with a call into the CR3BP
truth model (PyCall into `../src/dynamics/`, or a native Julia CR3BP) that
propagates one control period and returns the resulting periapsis altitude.
Nothing else in the model changes.

## Known toy artifacts (not bugs — intentional simplifications)
- No "burn down" action, and drift never lowers HIGH → HIGH is a mild dead-end;
  SARSOP's `HIGH → LARGE_BURN` choice is near-indifferent. Add a retrograde
  action or let HIGH drift back down when this matters.
- Scalar altitude only (real burns are 3-D and couple periapsis/apoapsis).
- No fuel-remaining state dimension yet (planned optional extension).