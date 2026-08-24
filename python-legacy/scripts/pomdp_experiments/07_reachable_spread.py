"""
Experiment 07 — what periapsis altitudes can ONE shell-burn actually reach?

Exp 06 showed one-pass excursions recover cleanly but UNDERSHOOT the commanded
altitude (target 55 -> reached 43 km). The undershoot is GEOMETRY, not fuel: the
burn fires at the 600-km descending shell, close in time to the next periapsis, so a
single impulse has limited leverage on that periapsis. This script maps the REAL
menu: for a swept excursion target, what periapsis altitude is ACTUALLY reached on
the excursion pass (starting from the held nominal orbit)? That reachable set is the
action space the future GP/active-sensing POMDP would actually get.

We first coast a couple passes to settle onto the held orbit, then from a nominal
periapsis approach command a one-pass excursion toward each swept target and record
the achieved periapsis altitude + ΔV.

Run: PYTHONPATH=. python scripts/pomdp_experiments/07_reachable_spread.py
"""
import numpy as np
from baselines.mpc import (
    solve_burn, nominal_apse_positions, _altitude_km, _make_escape_event,
    CONTROL_ALT_KM,
)
from src.dynamics.cr3bp import cr3bp_eom, X_ENCELADUS
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.integrator import (
    propagate, make_altitude_event, make_periapsis_event, make_crash_event,
    RTOL_TRUTH, ATOL_TRUTH,
)
from src.constants import R_ENCELADUS, PERIAPSIS_CRASH_ALT
from src.utils.halo_ic import PERIOD3_IC_ND, PERIOD3_PERIOD_S, _nd_to_phys

ONE_REV_S = PERIOD3_PERIOD_S / 3.0
ic = _nd_to_phys(PERIOD3_IC_ND)
r_peri_nom, r_apo_nom = nominal_apse_positions(ic, ONE_REV_S, eom=cr3bp_eom)


def enc_rel(r):
    r = r.copy(); r[0] -= X_ENCELADUS; return r


def scale_to_alt(r_nom, alt):
    rr = enc_rel(r_nom); tgt = R_ENCELADUS + alt
    rr2 = rr * (tgt / np.linalg.norm(rr))
    out = rr2.copy(); out[0] += X_ENCELADUS; return out


def coast_to_shell(s, horizon=4 * 86400.0):
    ctrl = make_altitude_event(CONTROL_ALT_KM, terminal=True)
    crash = make_crash_event(PERIAPSIS_CRASH_ALT); esc = _make_escape_event()
    sol = propagate(cr3bp_j2_eom, s, (0.0, horizon), events=[ctrl, crash, esc],
                    rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
    if not len(sol.t_events[0]): return None
    return sol.y_events[0][0].copy()


def next_peri(s, horizon=4 * 86400.0):
    peri = make_periapsis_event(terminal=True); crash = make_crash_event(PERIAPSIS_CRASH_ALT)
    esc = _make_escape_event()
    sol = propagate(cr3bp_j2_eom, s, (0.0, horizon), events=[peri, crash, esc],
                    rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
    if len(sol.t_events[1]): return "crash"
    if len(sol.t_events[2]): return "escape"
    if not len(sol.t_events[0]): return None
    return sol.y_events[0][0].copy()

# Settle onto the held orbit: two nominal held passes.
s = ic.copy()
for _ in range(2):
    sc = coast_to_shell(s)
    if sc is None: break
    b = solve_burn(sc, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                   r_peri_nom=r_peri_nom, r_apo_nom=r_apo_nom)
    sp = sc.copy(); sp[3:] += b["dv"]
    s = next_peri(sp)

print("target_alt_km ->  achieved_peri_alt_km   ΔV(m/s)   note")
for tgt in [15, 20, 25, 31, 40, 50, 60, 70, 90, 120]:
    sc = coast_to_shell(s)
    if sc is None:
        print(f"  {tgt:6.0f} -> (no shell)"); continue
    rp = scale_to_alt(r_peri_nom, tgt); ra = scale_to_alt(r_apo_nom,
        np.linalg.norm(enc_rel(r_apo_nom)) - R_ENCELADUS)
    b = solve_burn(sc, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                   r_peri_nom=rp, r_apo_nom=ra)
    sp = sc.copy(); sp[3:] += b["dv"]
    yp = next_peri(sp)
    if isinstance(yp, str):
        print(f"  {tgt:6.0f} -> {yp}");
        s = coast_to_shell(s);  # try to keep going from nominal
        continue
    if yp is None:
        print(f"  {tgt:6.0f} -> (lost)"); break
    achieved = _altitude_km(yp)
    print(f"  {tgt:6.0f} -> {achieved:12.1f}          {b['dv_mag_ms']:6.2f}")
    # recover to nominal for the next probe
    sc2 = coast_to_shell(yp)
    if sc2 is None: break
    b2 = solve_burn(sc2, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                    r_peri_nom=r_peri_nom, r_apo_nom=r_apo_nom)
    sp2 = sc2.copy(); sp2[3:] += b2["dv"]
    s = next_peri(sp2)
    if isinstance(s, str) or s is None:
        print(f"     (recovery ended: {s})"); break
