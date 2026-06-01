# SHERPA-RPA Direction 3 — Task List

## Phase 1 — Orbital Simulator ✅ Session 1 goal

- [x] Project scaffold: all folders + empty modules + `constants.py`
- [x] `cr3bp.py`: CR3BP equations of motion, physical units (km/km/s/s)
- [x] `integrator.py`: RK45 wrapper with apoapsis/periapsis/altitude/crash event detection
- [x] `cr3bp_j2.py`: Enceladus J2 perturbation added to truth model
- [x] `orbital_elements.py`: physical ↔ non-dim CR3BP normalizer, Keplerian elements helper

### Session 1 Audit (Session 2) ✅ verified (no code changed)
- [x] **Code audit**: rotating frame convention, J2 formula, event direction flags — all correct
- [x] **Numerical validation**: 69 tests pass, Jacobi drift 1.9e-13 over 30 days
- [x] **Confirm J2 source**: J2=5.435e-3 and R_Enc=252.1 km confirmed (Iess 2014)
- ⚠️  **L1/L2 values will shift**: constants updated to JPL DE440 in Session 3 — recheck
      L1 ≈ L* × (1−μ−γ1) and L2 with new MU/LU if needed

### Orbit Generation ✅ Session 3
- [x] **`src/utils/halo_ic.py`**: nondim CR3BP EOM+STM, multi-crossing half-period
      propagator, 2×2 Newton corrector, seed scan, Richardson approx, characterise_orbit
- [x] **`src/constants.py`**: updated to JPL DE440 (LU=238529 km, TU=18913.28 s,
      MU=1.9011097e-7) to match the periodic orbit catalog
- [x] **Period-3 IC found**: `PERIOD3_IC_ND` in halo_ic.py — JPL catalog seed corrected
      in our system. Periapsis 31 km ✓, apoapsis 1065 km ✓, period 36 hr ✓, 3 periapsis
      passes per orbit ✓, south polar ✓. Closure 0.0017 km. 76 tests pass.
- [x] **Validation tests**: `test_period3_ic_closes`, `test_period3_ic_three_periapses`

### Plots (Session 2) — `scripts/` folder
- [ ] `scripts/plot_orbit.py`: reusable plotting functions:
      - 3D trajectory in rotating frame (x, y, z in km)
      - x-z and x-y projections with Enceladus sphere drawn to scale
      - Altitude vs time (show periapsis/apoapsis bounds as dashed lines)
      - Period-3 closure check: overlay 3 successive revolutions to verify they stack
- [ ] **Reproduce Exhibit B-21** (MacKenzie §B.2.3): three-panel figure matching the paper:
      - Panel (a): x-z plane trajectory (CR3BP black, initial guess)
      - Panel (b): x-y groundtrack with periapsis 20–70 km band highlighted in red/blue
      - Panel (c): periapsis altitude vs elapsed days over 24 revolutions (~12 days)
- [ ] CR3BP vs CR3BP+J2 divergence plot over 30 days (position error km vs time)

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
