"""
Experiment 08 — does a TWO-BURN excursion reach a wider / more precise altitude?

Exp 07 maps what ONE shell-burn reaches (geometry-limited undershoot). A classic way
to change periapsis altitude deliberately is a Hohmann-like PAIR: burn #1 at APOAPSIS
to move the opposite apse (periapsis), then burn #2 at the new PERIAPSIS to stabilize.
This tests whether splitting the excursion across apoapsis+periapsis hits target
altitudes the single late shell-burn can't, then recovers to nominal.

Mechanism per excursion:
  1. From a held nominal state, coast to the next APOAPSIS.
  2. Burn #1 there: solve_burn(mode="altitude") targeting (peri=excursion_alt,
     apo=nominal) — an apoapsis burn primarily moves periapsis.
  3. Coast to the resulting PERIAPSIS; record achieved altitude (the science pass).
  4. Recover: from there run one nominal position-mode hold burn at the shell.
Compare achieved excursion altitude + total ΔV vs the single-burn result (exp 07).

Run: PYTHONPATH=. python scripts/pomdp_experiments/08_two_burn_excursion.py
"""
import numpy as np
from baselines.mpc import (
    solve_burn, nominal_apse_positions, _altitude_km, _make_escape_event,
    CONTROL_ALT_KM,
)
from src.dynamics.cr3bp import cr3bp_eom, X_ENCELADUS
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.integrator import (
    propagate, make_altitude_event, make_periapsis_event, make_apoapsis_event,
    make_crash_event, RTOL_TRUTH, ATOL_TRUTH,
)
from src.constants import R_ENCELADUS, PERIAPSIS_CRASH_ALT
from src.utils.halo_ic import PERIOD3_IC_ND, PERIOD3_PERIOD_S, _nd_to_phys

ONE_REV_S = PERIOD3_PERIOD_S / 3.0
ic = _nd_to_phys(PERIOD3_IC_ND)
r_peri_nom, r_apo_nom = nominal_apse_positions(ic, ONE_REV_S, eom=cr3bp_eom)
APO_NOM = np.linalg.norm((r_apo_nom - np.array([X_ENCELADUS, 0, 0]))) - R_ENCELADUS


def coast_to(event_maker, s, horizon=4 * 86400.0):
    ev = event_maker(terminal=True); crash = make_crash_event(PERIAPSIS_CRASH_ALT)
    esc = _make_escape_event()
    sol = propagate(cr3bp_j2_eom, s, (0.0, horizon), events=[ev, crash, esc],
                    rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
    if len(sol.t_events[1]): return "crash"
    if len(sol.t_events[2]): return "escape"
    if not len(sol.t_events[0]): return None
    return sol.y_events[0][0].copy()


def coast_to_shell(s, horizon=4 * 86400.0):
    ctrl = make_altitude_event(CONTROL_ALT_KM, terminal=True)
    crash = make_crash_event(PERIAPSIS_CRASH_ALT); esc = _make_escape_event()
    sol = propagate(cr3bp_j2_eom, s, (0.0, horizon), events=[ctrl, crash, esc],
                    rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
    if len(sol.t_events[1]): return "crash"
    if len(sol.t_events[2]): return "escape"
    if not len(sol.t_events[0]): return None
    return sol.y_events[0][0].copy()

# settle onto held orbit
s = ic.copy()
for _ in range(2):
    sc = coast_to_shell(s)
    if isinstance(sc, str) or sc is None: break
    b = solve_burn(sc, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                   r_peri_nom=r_peri_nom, r_apo_nom=r_apo_nom)
    sp = sc.copy(); sp[3:] += b["dv"]
    s = coast_to(make_periapsis_event, sp)

print("two-burn excursion (apoapsis burn #1 -> periapsis science pass -> recover)")
print("target_alt -> achieved_peri_alt   ΔV_burn1   recover_ΔV   outcome")
for tgt in [15, 20, 40, 60, 90, 120]:
    apo_s = coast_to(make_apoapsis_event, s)
    if isinstance(apo_s, str) or apo_s is None:
        print(f"  {tgt:6.0f} -> (no apoapsis: {apo_s})"); break
    # Burn #1 at apoapsis: target excursion periapsis altitude (altitude mode).
    b1 = solve_burn(apo_s, ONE_REV_S, n_revs=2, eom=cr3bp_eom, mode="altitude",
                    peri_target_km=float(tgt), apo_target_km=APO_NOM)
    ap = apo_s.copy(); ap[3:] += b1["dv"]
    # Coast to periapsis (the science pass).
    yp = coast_to(make_periapsis_event, ap)
    if isinstance(yp, str) or yp is None:
        print(f"  {tgt:6.0f} -> ({yp})  ΔV1={b1['dv_mag_ms']:.2f}")
        s = coast_to(make_apoapsis_event, s) if not isinstance(s, str) else s
        continue
    achieved = _altitude_km(yp)
    # Recover: nominal hold burn at next shell.
    sc = coast_to_shell(yp)
    rec_dv = np.nan; outcome = "ok"
    if isinstance(sc, str): outcome = sc
    elif sc is not None:
        b2 = solve_burn(sc, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                        r_peri_nom=r_peri_nom, r_apo_nom=r_apo_nom)
        rec_dv = b2["dv_mag_ms"]
        sp = sc.copy(); sp[3:] += b2["dv"]
        s = coast_to(make_periapsis_event, sp)
        if isinstance(s, str): outcome = f"recover->{s}"
    print(f"  {tgt:6.0f} -> {achieved:12.1f}     {b1['dv_mag_ms']:7.2f}     "
          f"{rec_dv:7.2f}     {outcome}")
    if isinstance(s, str) or s is None:
        print("   (chain ended)"); break
