"""
Experiment 10 — the two-burn damping/cap story, as a sweep for plotting.

Shows WHY the two-burn excursion escaped at first and HOW capping/damping fixes it:
for each escaped-in-exp-08 target, sweep the per-burn magnitude cap and record the
achieved periapsis altitude + outcome. Uncapped (large cap) -> over-correction ->
escape; a modest cap -> reaches the target. Writes
figures/two_burn_damping_data.npz for scripts/plot_two_burn_damping.py.

Run: PYTHONPATH=. python scripts/pomdp_experiments/10_two_burn_damping_sweep.py
"""
import numpy as np
from pathlib import Path
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
OUT = Path(__file__).resolve().parents[2] / "figures" / "two_burn_damping_data.npz"

TARGETS = [15, 20, 40, 120]
CAPS = [2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 1e9]   # 1e9 = effectively uncapped
DAMP = 1.0
K = 3


def coast_to(maker, s, horizon=4 * ONE_REV_S):
    ev = maker(terminal=True); crash = make_crash_event(PERIAPSIS_CRASH_ALT); esc = _make_escape_event()
    sol = propagate(cr3bp_j2_eom, s, (0.0, horizon), events=[ev, crash, esc],
                    rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
    if len(sol.t_events[1]): return "crash"
    if len(sol.t_events[2]): return "escape"
    if not len(sol.t_events[0]): return None
    return sol.y_events[0][0].copy()

# settle onto held orbit
s0 = ic.copy()
for _ in range(2):
    ap = coast_to(make_apoapsis_event, s0)
    if isinstance(ap, str) or ap is None: break
    b = solve_burn(ap, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                   r_peri_nom=r_peri_nom, r_apo_nom=r_apo_nom)
    ap2 = ap.copy(); ap2[3:] += b["dv"]
    s0 = coast_to(make_periapsis_event, ap2)
if isinstance(s0, str) or s0 is None:
    s0 = ic.copy()


def excursion(target, cap_ms):
    st = s0.copy()
    for k in range(K):
        ap = coast_to(make_apoapsis_event, st)
        if isinstance(ap, str): return np.nan, ap
        if ap is None: return np.nan, "no-apo"
        b = solve_burn(ap, ONE_REV_S, n_revs=2, eom=cr3bp_eom, mode="altitude",
                       peri_target_km=float(target), apo_target_km=APO_NOM)
        dv = b["dv"] * DAMP; mag = np.linalg.norm(dv) * 1e3
        if mag > cap_ms:
            dv = dv * (cap_ms / mag)
        ap2 = ap.copy(); ap2[3:] += dv
        pr = coast_to(make_periapsis_event, ap2)
        if isinstance(pr, str): return np.nan, pr
        if pr is None: return np.nan, "no-peri"
        alt = _altitude_km(pr); st = pr
        if abs(alt - target) < 3.0: return alt, "reached"
    return _altitude_km(st), "partial"


achieved = np.full((len(TARGETS), len(CAPS)), np.nan)
outcome = np.empty((len(TARGETS), len(CAPS)), dtype=object)
print(f"{'target':>7} " + " ".join(f"{c:>7.0f}" for c in CAPS[:-1]) + "   uncapped")
for ti, tgt in enumerate(TARGETS):
    row = []
    for ci, cap in enumerate(CAPS):
        alt, out = excursion(tgt, cap)
        achieved[ti, ci] = alt; outcome[ti, ci] = out
        row.append("ESC" if out in ("escape", "crash") else f"{alt:.0f}")
    print(f"{tgt:7.0f} " + " ".join(f"{v:>7}" for v in row))

np.savez(OUT, targets=np.array(TARGETS), caps=np.array(CAPS), achieved=achieved,
         outcome=outcome.astype(str))
print(f"wrote {OUT}")
