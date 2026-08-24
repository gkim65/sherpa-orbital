"""
Experiment 08b — retry the two-burn excursions that ESCAPED in exp 08.

Exp 08's escapes (targets 15/20/40/120 km) came from the APOAPSIS burn (altitude
mode) OVER-correcting — a huge single impulse (e.g. 392 m/s for target 40). Question:
were those escapes a SOLVER ARTIFACT (over-correction) rather than a real dynamical
limit? Here we DAMP the apoapsis burn: apply only a fraction of the solved ΔV and/or
cap its magnitude, so periapsis is moved GRADUALLY toward the target over a couple of
apoapsis passes, then recover to nominal. If the damped version reaches the target
altitude and survives, the escapes were an artifact.

Per excursion attempt (for a given target + damping):
  - up to K apoapsis passes, each applying damp * solve_burn(altitude, peri=target),
    with |ΔV| capped, watching for crash/escape;
  - record the periapsis altitude actually reached and the outcome.

Run: PYTHONPATH=. python scripts/pomdp_experiments/08b_two_burn_retry.py
"""
import numpy as np
from baselines.mpc import (
    solve_burn, nominal_apse_positions, _altitude_km, _make_escape_event,
)
from src.dynamics.cr3bp import cr3bp_eom, X_ENCELADUS
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.integrator import (
    propagate, make_periapsis_event, make_apoapsis_event, make_crash_event,
    RTOL_TRUTH, ATOL_TRUTH,
)
from src.constants import R_ENCELADUS, PERIAPSIS_CRASH_ALT
from src.utils.halo_ic import PERIOD3_IC_ND, PERIOD3_PERIOD_S, _nd_to_phys

ONE_REV_S = PERIOD3_PERIOD_S / 3.0
ic = _nd_to_phys(PERIOD3_IC_ND)
r_peri_nom, r_apo_nom = nominal_apse_positions(ic, ONE_REV_S, eom=cr3bp_eom)
APO_NOM = np.linalg.norm((r_apo_nom - np.array([X_ENCELADUS, 0, 0]))) - R_ENCELADUS


def coast_to(maker, s, horizon=4 * ONE_REV_S):
    ev = maker(terminal=True); crash = make_crash_event(PERIAPSIS_CRASH_ALT); esc = _make_escape_event()
    sol = propagate(cr3bp_j2_eom, s, (0.0, horizon), events=[ev, crash, esc],
                    rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
    if len(sol.t_events[1]): return "crash"
    if len(sol.t_events[2]): return "escape"
    if not len(sol.t_events[0]): return None
    return sol.y_events[0][0].copy()

# settle onto held orbit (2 nominal periapsis passes via apoapsis-position holding)
s = ic.copy()
for _ in range(2):
    ap = coast_to(make_apoapsis_event, s)
    if isinstance(ap, str) or ap is None: break
    b = solve_burn(ap, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                   r_peri_nom=r_peri_nom, r_apo_nom=r_apo_nom)
    ap2 = ap.copy(); ap2[3:] += b["dv"]
    s = coast_to(make_periapsis_event, ap2)
if isinstance(s, str) or s is None:
    s = ic.copy()


def excursion(target, damp, cap_ms=5.0, K=3):
    """Damped multi-apoapsis-burn excursion toward peri=target, then report reached
    periapsis altitude and outcome. Returns (achieved_alt, total_dv_ms, outcome)."""
    st = s.copy(); total = 0.0
    for k in range(K):
        ap = coast_to(make_apoapsis_event, st)
        if isinstance(ap, str): return np.nan, total, ap
        if ap is None: return np.nan, total, "no-apo"
        b = solve_burn(ap, ONE_REV_S, n_revs=2, eom=cr3bp_eom, mode="altitude",
                       peri_target_km=float(target), apo_target_km=APO_NOM)
        dv = b["dv"] * damp
        mag = np.linalg.norm(dv) * 1e3
        if mag > cap_ms:
            dv = dv * (cap_ms / mag); mag = cap_ms
        total += mag
        ap2 = ap.copy(); ap2[3:] += dv
        pr = coast_to(make_periapsis_event, ap2)
        if isinstance(pr, str): return np.nan, total, pr
        if pr is None: return np.nan, total, "no-peri"
        alt = _altitude_km(pr)
        st = pr
        if abs(alt - target) < 3.0:      # close enough
            return alt, total, "reached"
    return _altitude_km(st), total, "partial"


print(f"nominal apo alt used = {APO_NOM:.1f} km;  targets that escaped in exp 08: 15,20,40,120")
print(f"{'target':>7} {'damp':>5} {'achieved':>9} {'ΔV(m/s)':>8}  outcome")
for target in [15, 20, 40, 120]:
    for damp in [1.0, 0.5, 0.25]:
        alt, dv, out = excursion(target, damp)
        astr = f"{alt:9.1f}" if np.isfinite(alt) else "      nan"
        print(f"{target:7.0f} {damp:5.2f} {astr} {dv:8.2f}  {out}")
