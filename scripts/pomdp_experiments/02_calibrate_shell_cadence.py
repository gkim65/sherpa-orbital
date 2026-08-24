"""
Calibration at MPC's 600-km-shell cadence (the cadence that HOLDS the orbit).

One control step = coast under truth to the descending 600-km shell, act
(OBSERVE or CORRECT via solve_burn mode=position), then coast past the shell to
the next descending crossing. We record, per step:
  - periapsis altitude reached on the leg (the observable),
  - apse-position deviation dev = |r_peri - r_peri_nom(phase)| where r_peri_nom is
    the phase-matched nominal (nearest of the reference orbit's periapses).

Outputs the numbers the surrogate needs:
  (1) natural dev growth per shell-step under NO burn (instability), and
  (2) dev after a CORRECT, from a range of starting devs.
"""
import numpy as np
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


def coast_to_shell(state, horizon):
    """Coast under truth to the next descending 600-km crossing; also detect
    periapsis (for altitude/dev) and crash/escape en route. Returns dict."""
    ctrl = make_altitude_event(CONTROL_ALT_KM, terminal=True)
    peri = make_periapsis_event(terminal=False)
    crash = make_crash_event(PERIAPSIS_CRASH_ALT)
    esc = _make_escape_event()
    sol = propagate(cr3bp_j2_eom, state, (0.0, horizon),
                    events=[ctrl, crash, peri, esc], rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
    peri_alt = _altitude_km(sol.y_events[2][0]) if len(sol.t_events[2]) else np.nan
    peri_dev = (np.linalg.norm(sol.y_events[2][0][:3] - r_peri_nom)
                if len(sol.t_events[2]) else np.nan)
    return {
        "crashed": len(sol.t_events[1]) > 0,
        "escaped": len(sol.t_events[3]) > 0,
        "at_shell": len(sol.t_events[0]) > 0,
        "shell_state": sol.y_events[0][0].copy() if len(sol.t_events[0]) else None,
        "t_shell": float(sol.t_events[0][0]) if len(sol.t_events[0]) else np.nan,
        "peri_alt": peri_alt, "peri_dev": peri_dev,
    }

# (1) UNCONTROLLED per-shell dev growth.
print("=== (1) uncontrolled: dev + peri_alt per 600-km shell step ===")
s = ic.copy(); t = 0.0
for k in range(8):
    leg = coast_to_shell(s, 30 * 86400.0 - t)
    print(f" step {k}: peri_alt={leg['peri_alt']:9.1f}  dev_peri={leg['peri_dev']:9.2f} km"
          f"  {'CRASH' if leg['crashed'] else ''}{'ESCAPE' if leg['escaped'] else ''}")
    if leg["crashed"] or leg["escaped"] or not leg["at_shell"]:
        break
    # coast past the shell to next periapsis then onward (no burn) — emulate MPC re-arm
    s = leg["shell_state"]; t += leg["t_shell"]
    # nudge past shell: propagate a tiny bit inbound to next periapsis
    peri = make_periapsis_event(terminal=True); crash = make_crash_event(PERIAPSIS_CRASH_ALT)
    esc = _make_escape_event()
    sol = propagate(cr3bp_j2_eom, s, (0.0, 4 * ONE_REV_S),
                    events=[peri, crash, esc], rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
    if len(sol.t_events[1]) or len(sol.t_events[2]) or not len(sol.t_events[0]):
        print("   -> lost past shell"); break
    s = sol.y_events[0][0].copy(); t += float(sol.t_events[0][0])

# (2) CORRECT residual: at the shell, apply solve_burn, then measure next-peri dev.
print("\n=== (2) CORRECT (solve_burn @ shell) -> next periapsis dev ===")
s = ic.copy()
for k in range(6):
    leg = coast_to_shell(s, 30 * 86400.0)
    if not leg["at_shell"]:
        print(" no shell"); break
    sc = leg["shell_state"]
    burn = solve_burn(sc, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                      r_peri_nom=r_peri_nom, r_apo_nom=r_apo_nom)
    s_post = sc.copy(); s_post[3:] += burn["dv"]
    # coast to next periapsis, measure dev
    peri = make_periapsis_event(terminal=True); crash = make_crash_event(PERIAPSIS_CRASH_ALT)
    esc = _make_escape_event()
    sol = propagate(cr3bp_j2_eom, s_post, (0.0, 4 * ONE_REV_S),
                    events=[peri, crash, esc], rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
    if not len(sol.t_events[0]):
        print(" lost after burn"); break
    yp = sol.y_events[0][0]
    dev_after = np.linalg.norm(yp[:3] - r_peri_nom)
    print(f" step {k}: dev_before={leg['peri_dev']:8.2f} -> dev_after={dev_after:7.2f} km"
          f"  peri_alt={_altitude_km(yp):7.1f}  ΔV={burn['dv_mag_ms']:5.2f}  conv={burn['converged']}")
    s = yp.copy()
