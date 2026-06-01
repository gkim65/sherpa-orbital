# SHERPA-RPA Direction 3 — Enceladus Orbilander POMDP Stationkeeping

Offline POMDP-based autonomous stationkeeping simulator for the Enceladus Orbilander
mission (MacKenzie et al. 2020). The spacecraft holds a period-3 L1 halo orbit around
Enceladus — periapsis 31 km, apoapsis 1065 km, period 36 hr, 3 south-polar periapsis
passes per orbit — and must stationkeep without ground contact.

This is **SHERPA-RPA Direction 3**, targeting a POMDP controller that outperforms
deterministic MPC and deep-RL baselines on long-duration (30-day) stationkeeping
survival rate and fuel efficiency.

---

## What's implemented

### Phase 1 — CR3BP Orbital Simulator ✅

| Module | Description |
|--------|-------------|
| `src/constants.py` | All physical constants (JPL DE440 unit system: LU=238529 km, TU=18913 s, μ=1.9011e-7) |
| `src/dynamics/cr3bp.py` | Saturn-Enceladus CR3BP equations of motion, Jacobi constant, libration points |
| `src/dynamics/cr3bp_j2.py` | CR3BP + Enceladus J2 oblateness perturbation (truth model) |
| `src/dynamics/integrator.py` | RK45 wrapper with apoapsis/periapsis/altitude/crash event detection |
| `src/utils/orbital_elements.py` | Physical ↔ non-dimensional state conversion, Keplerian elements |
| `src/utils/halo_ic.py` | Differential corrector (Howell 1984) for period-3 southern L1 halo orbit; hardcoded verified IC (`PERIOD3_IC_ND`) |

The period-3 science orbit is verified against MacKenzie §B.2.3:
- Periapsis altitude: **31.0 km** (band: 19.8–64.3 km ✓)
- Apoapsis altitude: **1064.8 km** (band: 1000–1110 km ✓)
- Period: **35.99 hr** (~36 hr ✓)
- Closure residual: **0.0017 km** after 1 Newton iteration

### Figures ✅

| Script | Output | Description |
|--------|--------|-------------|
| `scripts/plot_orbit.py` | `figures/exhibit_b21.png` | Reproduces MacKenzie Exhibit B-21: X-Z trajectory, Y-Z polar flyby view, periapsis altitude vs time (CR3BP vs CR3BP+J2, 24 revolutions) |

---

## Setup

```bash
# Clone
git clone git@github.com:gkim65/sherpa-orbital.git
cd sherpa-orbital

# Install dependencies (Python 3.10+)
pip install numpy scipy matplotlib pytest
```

No package installation needed — scripts import `src/` directly via `sys.path`.

---

## Running the code

### Generate the orbit figure (Exhibit B-21)

```bash
python scripts/plot_orbit.py
```

Outputs `figures/exhibit_b21.png` — a three-panel figure:
- **(a) X-Z trajectory**: full 36-hr period in the CR3BP rotating frame, coloured by
  elapsed time. All 3 loops of the period-3 orbit are geometrically superimposed in X-Z.
- **(b) Y-Z view**: looking along −X (from Saturn toward Enceladus). Shows the
  spacecraft skimming 31 km above the north pole — the clearance that is invisible
  in the standard X-Y top-down projection.
- **(c) Periapsis altitude vs time**: CR3BP (flat, periodic) vs CR3BP+J2 truth model
  (escapes within ~2 revolutions without stationkeeping). This instability motivates
  the POMDP controller.

### Run tests

```bash
pytest                        # all tests (~2 s)
pytest -m "not slow"          # skip slow differential-corrector tests
pytest tests/dynamics/        # dynamics tests only
pytest tests/utils/           # orbit IC tests only
```

**76 tests, 0 failures.**

---

## Repository structure

```
sherpa-orbital/
├── README.md
├── CLAUDE.md                  # AI session protocol and code conventions
├── docs/
│   ├── project-brief.md
│   ├── todo.md                # task list by phase
│   └── session-log/           # per-session technical notes
├── figures/
│   └── exhibit_b21.png        # MacKenzie Exhibit B-21 reproduction
├── scripts/
│   └── plot_orbit.py          # orbit visualisation (run directly)
├── src/
│   ├── constants.py
│   ├── dynamics/
│   │   ├── cr3bp.py           # onboard model EOM
│   │   ├── cr3bp_j2.py        # truth model EOM (adds J2)
│   │   └── integrator.py      # RK45 wrapper + event detection
│   ├── spacecraft/            # Phase 2 — thruster + nav models (upcoming)
│   ├── environment/           # Phase 3 — Gymnasium env (upcoming)
│   └── utils/
│       ├── halo_ic.py         # differential corrector + verified IC
│       └── orbital_elements.py
└── tests/
    ├── dynamics/
    └── utils/
```

---

## Physics conventions

- **Frame**: Saturn-Enceladus CR3BP rotating frame. Saturn at x = −μ, Enceladus at x = 1−μ.
- **Units**: km, km/s, s throughout. Non-dimensional via LU/TU/VU from `constants.py`.
- **Truth model**: CR3BP + Enceladus J2 (J2 = 5.435×10⁻³, Iess et al. 2014).
- **Onboard model**: CR3BP only (the gap to truth is the uncertainty being studied).
- **Integrator**: `scipy.solve_ivp` RK45. Truth: rtol=1e-10, atol=1e-12. Onboard: rtol=1e-8.

---

## Key references

- MacKenzie et al. (2020). *Enceladus Orbilander Mission Concept Study*. §B.2.3.
- Kim et al. (2025). POMDP/Bayesian network stationkeeping formulation.
- Howell (1984). Three-Dimensional, Periodic, Halo Orbits. *Celestial Mechanics* 32(1).
- Iess et al. (2014). The Gravity Field and Interior Structure of Enceladus. *Science*.
- JPL Three-Body Periodic Orbits Catalog (DE440 ephemeris).