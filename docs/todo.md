# SHERPA-RPA Direction 3 — Task List

## Phase 1 — Orbital Simulator ✅ Session 1 goal

- [x] Project scaffold: all folders + empty modules + `constants.py`
- [x] `cr3bp.py`: CR3BP equations of motion, physical units (km/km/s/s)
- [x] `integrator.py`: RK45 wrapper with apoapsis/periapsis/altitude/crash event detection
- [x] `cr3bp_j2.py`: Enceladus J2 perturbation added to truth model
- [x] `orbital_elements.py`: physical ↔ non-dim CR3BP normalizer, Keplerian elements helper
- [ ] **Differential correction** to find a true closed period-3 L1 halo orbit IC
  - Target: x₀ ≈ [1.0-μ + δ, 0, z_amp, 0, ẏ₀, 0] with single-shooting corrector
  - Confirm ~12-hour orbit period in physical units
- [ ] Validation test: reproduce period-3 halo with periapsis 20–70 km, apoapsis >400 km
- [ ] `test_cr3bp.py`: Jacobi conservation + period ballpark + J2 divergence (written, needs passing ICs)
- [ ] Matplotlib plot: trajectory in rotating frame (x-z plane, multiple revolutions)
- [ ] CR3BP vs CR3BP+J2 divergence plot over 30 days

## Phase 2 — Spacecraft Models

- [ ] `thruster.py`: random-walk degradation model, noisy ΔV execution (MR-106E)
- [ ] `nav.py`: Gaussian position observation model (σ_r = 2 km, §C.1)
- [ ] `test_thruster.py`: statistical checks over N=10,000 samples
- [ ] `test_nav.py`: verify noise distribution

## Phase 3 — Gymnasium Environment

- [ ] `env.py`: Gym environment wrapping the simulator
  - `reset()`: sample halo IC + η_eff ~ Uniform(0.8, 1.0)
  - `step(action)`: propagate one orbit (apoapsis→apoapsis), execute burn
  - `done`: periapsis altitude < 5 km (crash) or fuel = 0
- [ ] `reward.py`: R = −|δr_peri| − |δr_apo| − λ·fuel_cost
- [ ] `test_env.py`: 100-step random rollout survives

## Phase 4 — Deterministic MPC Baseline

- [ ] `baselines/mpc.py`: Strategy 3 from MacKenzie §B.2.3
  - At each 600-km altitude crossing: solve for burn targeting apse bounds ≤ 1 km
  - Multiple-shooting over N_m = 2–3 revolutions
- [ ] Evaluate: 100 × 30-day rollouts → survival rate + fuel use

## Phase 5 — RL Baselines

- [ ] `baselines/deep_rl.py`: Stable-Baselines3 SAC, ~1M steps
- [ ] `baselines/random_policy.py`: sanity check floor
- [ ] Compare all on survival rate, periapsis deviation, fuel

## Phase 6 — POMDP (Core Contribution)

- [ ] `belief/particle_filter.py`: particle filter over 5D compressed belief state
  - State: {δr_peri, δr_apo, δv_peri, η_eff, fuel_fraction}
- [ ] `belief/compression.py`: 6D Cartesian → compressed state mapping
- [ ] `methods/offline_pomdp.py`: interface with `pomdp-py` / SARSOP
- [ ] Ablation: vary compression fidelity, measure performance drop

## Open Questions / Blockers

- Need a reliable halo orbit initial condition generator (differential corrector or
  existing dataset). Session 2 priority.
- Stationkeeping event timing: Strategy 3 fires at "600-km altitude crossing".
  Need to confirm whether this is measured from Enceladus surface (= 852.1 km
  from Enceladus centre) or from the barycentre.  From context in §B.2.3 and
  Exhibit B-23, it is **altitude above Enceladus surface**.
