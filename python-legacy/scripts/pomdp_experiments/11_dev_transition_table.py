"""
Experiment 11 — measure the per-bin, per-action dev transition table (Stage-1 POMDP).

The Stage-1 state is the periapsis apse-position deviation dev = |r_peri - r_peri_nom|
(km), binned OK/DRIFT/FAR/LOST(+CRASHED). This measures, from a spread of starting
states in each dev bin, where dev lands after ONE shell-step under each action:
  - OBSERVE : no burn (dev grows — instability)
  - CORRECT : solve_burn(mode=position) toward nominal (dev shrinks toward attractor)

Output: for each (dev bin, action), the distribution of NEXT dev bin — i.e. the rows
of the SARSOP transition table T[s, a, s'], measured from the real CR3BP+EncJ2 truth.
We create off-nominal start states by nudging the velocity at a periapsis and, for
larger deviations, letting the uncontrolled orbit drift a pass or two.

Run: PYTHONPATH=. python scripts/pomdp_experiments/11_dev_transition_table.py
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

# Stage-1 dev bins (km). OK/DRIFT/FAR non-terminal; LOST/CRASHED terminal.
DEV_EDGES = (15.0, 60.0, 200.0)          # -> OK[0,15) DRIFT[15,60) FAR[60,200) LOST[200,inf)
DEV_NAMES = ("OK", "DRIFT", "FAR", "LOST")


def dev_bin(dev):
    if not np.isfinite(dev): return "LOST"
    if dev < DEV_EDGES[0]: return "OK"
    if dev < DEV_EDGES[1]: return "DRIFT"
    if dev < DEV_EDGES[2]: return "FAR"
    return "LOST"


def dev_of(peri_state):
    return float(np.linalg.norm(peri_state[:3] - r_peri_nom))


def next_peri(state, eom=cr3bp_j2_eom, horizon=4 * ONE_REV_S):
    peri = make_periapsis_event(terminal=True); crash = make_crash_event(PERIAPSIS_CRASH_ALT)
    esc = _make_escape_event()
    sol = propagate(eom, state, (0.0, horizon), events=[peri, crash, esc],
                    rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
    if len(sol.t_events[1]): return "CRASHED"
    if len(sol.t_events[2]): return "LOST"
    if not len(sol.t_events[0]): return "LOST"
    return sol.y_events[0][0].copy()


def coast_to_shell(state, horizon=4 * ONE_REV_S):
    ctrl = make_altitude_event(CONTROL_ALT_KM, terminal=True); crash = make_crash_event(PERIAPSIS_CRASH_ALT)
    esc = _make_escape_event()
    sol = propagate(cr3bp_j2_eom, state, (0.0, horizon), events=[ctrl, crash, esc],
                    rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
    if len(sol.t_events[1]): return "CRASHED"
    if len(sol.t_events[2]): return "LOST"
    if not len(sol.t_events[0]): return "LOST"
    return sol.y_events[0][0].copy()


def make_start_states(rng, n=40):
    """Generate periapsis start states spanning a range of dev, by nudging velocity
    and drifting. Returns list of (dev, peri_state)."""
    starts = []
    for _ in range(n):
        s = ic.copy()
        s[3:] += rng.normal(0, rng.uniform(1e-4, 8e-4), 3)   # 0.1-0.8 m/s nudge
        p = next_peri(s)
        if isinstance(p, str):
            continue
        starts.append((dev_of(p), p))
        # optionally drift one more uncontrolled pass to reach larger dev
        if rng.random() < 0.5:
            p2 = next_peri(p)
            if not isinstance(p2, str):
                starts.append((dev_of(p2), p2))
    return starts


def step(peri_state, action):
    """One shell-step under an action. Returns next dev-bin label (or terminal)."""
    sc = coast_to_shell(peri_state)
    if isinstance(sc, str): return sc
    if action == "CORRECT":
        b = solve_burn(sc, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                       r_peri_nom=r_peri_nom, r_apo_nom=r_apo_nom)
        sc = sc.copy(); sc[3:] += b["dv"]
    p = next_peri(sc)
    if isinstance(p, str): return p
    return dev_bin(dev_of(p))


rng = np.random.default_rng(20260715)
starts = make_start_states(rng, n=60)
print(f"generated {len(starts)} start states")
by_bin = defaultdict(list)
for dev, p in starts:
    by_bin[dev_bin(dev)].append(p)
for b in DEV_NAMES:
    print(f"  {b:6s}: {len(by_bin[b])} states")

ALL_NEXT = ("OK", "DRIFT", "FAR", "LOST", "CRASHED")
print("\nTransition rows  T[dev_bin, action, next]  (measured, EncJ2 truth):")
for action in ("OBSERVE", "CORRECT"):
    print(f"\n  action = {action}")
    for b in DEV_NAMES[:3]:   # non-terminal source bins
        states = by_bin[b]
        if not states:
            print(f"    {b:6s}: (no start states)"); continue
        counts = defaultdict(int)
        for p in states:
            counts[step(p, action)] += 1
        tot = sum(counts.values())
        row = "  ".join(f"{k}={counts[k]/tot:.2f}" for k in ALL_NEXT if counts[k])
        print(f"    {b:6s} (n={tot:2d}) -> {row}")
