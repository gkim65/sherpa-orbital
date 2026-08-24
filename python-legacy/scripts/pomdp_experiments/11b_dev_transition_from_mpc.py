"""
Experiment 11b — measure the dev transition table FROM THE SUSTAINED MPC LOOP.

Exp 11 measured single isolated CORRECT burns from randomly-nudged periapsis states
and found CORRECT barely helps — which CONTRADICTS the known fact that MPC-S3 (solve_burn
at the shell every pass) HOLDS the orbit 30 d. The likely bug: an isolated single burn
from an arbitrary phase is not the sustained loop, and the 3-periapsis phase may not
match the single nominal apse. So here we instrument the ACTUAL controlled loop:

  CORRECT rows: run the real sustained shell-cadence CORRECT loop (as in exp 03's
    'exact' controller that held 30 d) and log dev_before -> dev_after each step.
  OBSERVE rows: from the SAME visited states, coast one pass with NO burn and log
    dev_before -> dev_after (a counterfactual "what if we had observed instead").

This yields the true T[dev_bin, action, next] consistent with the holding controller.

Run: PYTHONPATH=. python scripts/pomdp_experiments/11b_dev_transition_from_mpc.py
"""
import numpy as np
from collections import defaultdict
from baselines.mpc import (
    solve_burn, nominal_apse_positions, _altitude_km, _make_escape_event,
    CONTROL_ALT_KM,
)
from src.dynamics.cr3bp import cr3bp_eom
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.integrator import (
    propagate, make_altitude_event, make_periapsis_event, make_crash_event,
    RTOL_TRUTH, ATOL_TRUTH,
)
from src.constants import PERIAPSIS_CRASH_ALT
from src.utils.halo_ic import PERIOD3_IC_ND, PERIOD3_PERIOD_S, _nd_to_phys

ONE_REV_S = PERIOD3_PERIOD_S / 3.0
ic = _nd_to_phys(PERIOD3_IC_ND)
r_peri_nom, r_apo_nom = nominal_apse_positions(ic, ONE_REV_S, eom=cr3bp_eom)

DEV_EDGES = (15.0, 60.0, 200.0)
DEV_NAMES = ("OK", "DRIFT", "FAR", "LOST")
ALL_NEXT = ("OK", "DRIFT", "FAR", "LOST", "CRASHED")


def dev_bin(dev):
    if not np.isfinite(dev): return "LOST"
    if dev < DEV_EDGES[0]: return "OK"
    if dev < DEV_EDGES[1]: return "DRIFT"
    if dev < DEV_EDGES[2]: return "FAR"
    return "LOST"


def dev_of(p): return float(np.linalg.norm(p[:3] - r_peri_nom))


def to_shell(s, horizon):
    ctrl = make_altitude_event(CONTROL_ALT_KM, terminal=True); crash = make_crash_event(PERIAPSIS_CRASH_ALT)
    esc = _make_escape_event()
    sol = propagate(cr3bp_j2_eom, s, (0.0, horizon), events=[ctrl, crash, esc],
                    rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
    if len(sol.t_events[1]): return "CRASHED", None
    if len(sol.t_events[2]): return "LOST", None
    if not len(sol.t_events[0]): return "LOST", None
    return "ok", sol.y_events[0][0].copy()


def to_peri(s, horizon):
    peri = make_periapsis_event(terminal=True); crash = make_crash_event(PERIAPSIS_CRASH_ALT)
    esc = _make_escape_event()
    sol = propagate(cr3bp_j2_eom, s, (0.0, horizon), events=[peri, crash, esc],
                    rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
    if len(sol.t_events[1]): return "CRASHED", None
    if len(sol.t_events[2]): return "LOST", None
    if not len(sol.t_events[0]): return "LOST", None
    return "ok", sol.y_events[0][0].copy()


# Run the sustained CORRECT loop for many steps; log dev_before -> dev_after,
# and at each visited shell state also compute the OBSERVE counterfactual.
T = {a: defaultdict(lambda: defaultdict(int)) for a in ("OBSERVE", "CORRECT")}
s = ic.copy(); t = 0.0; horizon = 25 * 86400.0; nsteps = 0
# get onto the orbit at a periapsis
st, p = to_peri(s, 4 * ONE_REV_S)
s = p if st == "ok" else ic.copy()
while t < horizon and nsteps < 120:
    db = dev_bin(dev_of(s))                      # dev entering this pass
    st, sc = to_shell(s, horizon - t)
    if st != "ok":
        T["CORRECT"][db][st] += 1; break
    # OBSERVE counterfactual from this shell state (no burn) -> next peri dev
    st_o, p_o = to_peri(sc, 4 * ONE_REV_S)
    nb_o = st_o if st_o != "ok" else dev_bin(dev_of(p_o))
    if db in DEV_NAMES[:3]:
        T["OBSERVE"][db][nb_o] += 1
    # CORRECT (the sustained loop actually taken) -> next peri dev
    b = solve_burn(sc, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                   r_peri_nom=r_peri_nom, r_apo_nom=r_apo_nom)
    scb = sc.copy(); scb[3:] += b["dv"]
    st_c, p_c = to_peri(scb, 4 * ONE_REV_S)
    nb_c = st_c if st_c != "ok" else dev_bin(dev_of(p_c))
    if db in DEV_NAMES[:3]:
        T["CORRECT"][db][nb_c] += 1
    if st_c != "ok":
        break
    s = p_c; t += ONE_REV_S; nsteps += 1

print(f"sustained CORRECT loop: {nsteps} steps logged")
print("\nT[dev_bin, action, next] (from the holding MPC loop + OBSERVE counterfactual):")
for a in ("OBSERVE", "CORRECT"):
    print(f"\n  action = {a}")
    for b in DEV_NAMES[:3]:
        counts = T[a][b]; tot = sum(counts.values())
        if not tot:
            print(f"    {b:6s}: (not visited)"); continue
        row = "  ".join(f"{k}={counts[k]/tot:.2f}" for k in ALL_NEXT if counts[k])
        print(f"    {b:6s} (n={tot:3d}) -> {row}")
