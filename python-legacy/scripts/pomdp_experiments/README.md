# POMDP stationkeeping experiments

De-risking experiments for the higher-fidelity stationkeeping POMDP (Phase 6).
These measure the real CR3BP+EncJ2 physics to justify the toy's action/state
design before building SARSOP tables. See `docs/session-log/2026-07-13-hifi-sk-pomdp.md`.

Run all from the repo root with the package on the path:
```
PYTHONPATH=. python scripts/pomdp_experiments/<file>.py
```
(or `python -m scripts.pomdp_experiments.<name>` after adding `__init__.py`).

All use the SAME trusted code paths as the MPC baseline (`baselines/mpc.py`,
`src/dynamics/`, `src/utils/halo_ic.py`) — no hand-rolled dynamics.

| # | Script | Question it answers |
|---|--------|---------------------|
| 01 | `01_verify_escape_and_mpc.py` | Does the uncontrolled period-3 orbit really escape under EncJ2 (via MPC's own events)? What altitudes/ΔV does the WORKING MPC-S3 run visit? → escape @ 2.27 d; MPC holds 30 d at ~1.28 m/s/burn, 600-km shell. |
| 02 | `02_calibrate_shell_cadence.py` | Calibration numbers at the 600-km-shell cadence: uncontrolled dev growth per step; dev after a solve_burn CORRECT. → CORRECT attractor ~7 km @~1.3 m/s; uncontrolled → escape in ~3 steps. |
| 03 | `03_quantize_burn.py` | Does a DISCRETIZED burn still hold? Compares exact solve_burn vs direction-exact/magnitude-quantized vs prograde-only vs fixed-1 m/s. → **direction matters (cross-track), magnitude precision doesn't**: dir+mag-quantized holds 30 d; prograde-only escapes 1.74 d. |
| 04 | `04_fixed_direction_menu.py` | Does a SMALL FIXED menu of burn DIRECTIONS hold the orbit? → **NO.** Direction is ~95% orbit-normal, median within 2.5° of a fixed vector, but max swings of 30–41°; a single fixed dir crashes @1.72 d, a 4-axis menu escapes @3.77 d. Conclusion: the POMDP MUST keep solve_burn in the loop for direction; the discrete action owns INTENT (when/how-hard to correct), not geometry. |
| 05 | `05_target_altitude.py` | Can we command a burn to hold a CHOSEN target altitude? → You CAN hit any target altitude (peri 22/42/60 km) but scalar-ALTITUDE targeting ESCAPES in ~3 d (only POSITION mode holds — reproduces MacKenzie Strategy 1/2 fails, Strategy 3 holds). So a distinct altitude regime needs a real reference orbit, not a scalar target. |
| 06 | `06_shift_and_restabilize.py` | Can we do a ONE-PASS excursion to a different altitude and RECOVER (for the GP/active-sensing science idea)? → **YES.** Excursions to 45/60/22/55 km every 4 passes all held 10 d, recovered to nominal the very next pass, ΔV comparable to pure hold (21–28 vs 24.6 m/s). Caveat: one shell-burn only PARTIALLY reaches the commanded altitude (55→43, 60→45); precise altitude needs a truer target orbit or a 2-burn excursion. Mechanism (excurse+recover, no loss) proven. |
| 07 | `07_reachable_spread.py` | What periapsis altitudes can ONE shell-burn actually reach (the real GP action menu)? → A COMPRESSED, MONOTONIC, SAFE map: commanded 15/31/60/90/120 → achieved 33/37/45/54/65 km, ΔV 1.2–4.2 m/s. So ~6–8 distinguishable bands over ~33–65 km, all recovering. Floor ~33 km (a late shell burn RAISES, can't dip below nominal). This IS the single-burn action menu. |
| 08 | `08_two_burn_excursion.py` | Does a 2-burn (apoapsis→periapsis) excursion reach WIDER/more precise altitudes? → MIXED / cautionary (FIRST attempt, altitude-mode). Targets 60/90 hit PRECISELY (60.3/90.3 km) and recovered; but 15/20/40/120 OVERSHOOT and ESCAPE (target 40 → 511 km @ 392 m/s). NOTE: burn #1 used the weak ALTITUDE mode + n_revs=2 — see 08b for a proper retry. |
| 08b| `08b_two_burn_retry.py` | Retry the escaped cases with a DAMPED + magnitude-CAPPED (≤5 m/s) apoapsis burn. → Mostly a SOLVER ARTIFACT: targets 15/20/120 km now REACHED (15.4/21.5/120.3 km) at 5–10 m/s; only target 40 km still escapes (ill-conditioned apoapsis-burn geometry there). Revised story: two-burn reaches a WIDE 15–120 km range precisely, but needs per-target damping and has occasional bad spots. The exp 08 "fragile/escapes" claim was overstated. |
| 09 | `09_burn_geometry.py` (+ `scripts/plot_burn_geometry.py`) | WHERE do burns fire and WHICH direction is the ΔV? → All 6 burns fire at the 600-km shell on the inbound descent; ΔV is prograde≈0, orbit-normal ~1.2–1.4 m/s + some in-plane cross-track. Confirms the holding burn is OUT-OF-PLANE, not a prograde speed change. Figure: figures/burn_geometry.{png,pdf,svg}. |
| 06b| `06b_multi_altitude_timeline.py` | Multi-altitude touch-and-go: cycle excursion targets (achieved 41/47/52/62 km) while holding nominal between — feeds panel (d) of plot_pomdp_findings. Holds + recovers over 16 d, min peri 20.9 km. |
| 10 | `10_two_burn_damping_sweep.py` (+ `scripts/plot_two_burn_damping.py`) | The two-burn cap/damping story as a HEATMAP (target × per-burn cap → achieved alt; escapes hatched). → NUANCED: target 15 robust; 120 needs cap≥5; 20 only cap 3–5; 40 unreachable (overshoots 700–840 km or escapes). No single universal cap — two-burn excursions need PER-TARGET tuning. (Corrects the oversimplified "cap≤5 fixes it".) Figure: figures/two_burn_damping.{png,pdf,svg}. |

Note: an earlier probe (`calib_probe2.py`, NOT kept) wrongly reported the orbit as
bounded — its periapsis finder re-found the same periapsis each call. Corrected by
01. For the period-3 orbit's 3 distinct periapses, deviation must be phase-matched.
