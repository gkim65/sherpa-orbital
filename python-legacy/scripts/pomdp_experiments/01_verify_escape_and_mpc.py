"""
Verification of the two diagnostic claims, using ONLY trusted code paths
(baselines.mpc.run_mpc and the production integrator/events) — no hand-rolled
cadence. Answers:
  A) Does the uncontrolled period-3 IC actually escape under EncJ2 (the SAME
     escape/crash events MPC uses), and when? -> is the toy's 1.17 d "escape" real?
  B) What true periapsis altitudes does the WORKING MPC-S3 run actually visit,
     and what is its apse-position deviation at its own control cadence?
"""
import numpy as np
from baselines.mpc import (
    run_mpc, _altitude_km, _make_escape_event, nominal_apse_positions,
    predict_apse_states,
)
from src.dynamics.cr3bp import cr3bp_eom
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.integrator import (
    propagate, make_periapsis_event, make_crash_event, RTOL_TRUTH, ATOL_TRUTH,
)
from src.constants import PERIAPSIS_CRASH_ALT
from src.utils.halo_ic import PERIOD3_IC_ND, PERIOD3_PERIOD_S, _nd_to_phys

ONE_REV_S = PERIOD3_PERIOD_S / 3.0
ic = _nd_to_phys(PERIOD3_IC_ND)

# ── A) UNCONTROLLED under EncJ2, with MPC's OWN terminal events ──────────────
print("=== A) uncontrolled EncJ2, using MPC's crash+escape+periapsis events ===")
peri = make_periapsis_event(terminal=False)       # NON-terminal: log every peri
crash = make_crash_event(PERIAPSIS_CRASH_ALT)
escape = _make_escape_event()                      # alt > 5550 km ascending
sol = propagate(cr3bp_j2_eom, ic, (0.0, 30 * 86400.0),
                events=[peri, crash, escape], rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
print(f" periapsis passes logged: {len(sol.t_events[0])}")
for i, (t, y) in enumerate(zip(sol.t_events[0][:15], sol.y_events[0][:15])):
    print(f"   peri {i:2d} @ t={t/86400:6.3f} d  alt={_altitude_km(y):9.1f} km")
if len(sol.t_events[1]) > 0:
    print(f" CRASH @ t={sol.t_events[1][0]/86400:.3f} d")
if len(sol.t_events[2]) > 0:
    print(f" ESCAPE (alt>5550) @ t={sol.t_events[2][0]/86400:.3f} d")
if len(sol.t_events[1]) == 0 and len(sol.t_events[2]) == 0:
    print(f" NO crash/escape in 30 d — orbit BOUNDED under EncJ2 (integrated to t={sol.t[-1]/86400:.1f} d)")

# ── B) WORKING MPC-S3: what altitudes + deviations does it actually visit? ────
print("\n=== B) MPC-S3 (the run that HOLDS 30 d) — its burns/altitudes ===")
res = run_mpc(ic, truth_eom=cr3bp_j2_eom, period_s=ONE_REV_S, horizon_s=30 * 86400.0,
              n_revs=3, mode="position", ref_ic=ic)
print(f" outcome={res['outcome']}  n_burns={res['n_burns']}  ΔV={res['total_dv_ms']:.2f} m/s"
      f"  min_peri={res['min_peri_alt_km']:.1f} km  surv={res['survival_time_s']/86400:.2f} d")
alts = [b["alt_km"] for b in res["burns"][:10]]
print(" first 10 burn trigger altitudes (should be ~600 km shell):",
      [f"{a:.0f}" for a in alts])
dvs = [b["dv_ms"] for b in res["burns"][:10]]
print(" first 10 burn ΔV (m/s):", [f"{d:.2f}" for d in dvs])
