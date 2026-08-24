"""
Does a DISCRETIZED burn still hold the orbit? (the POMDP viability question)

We take the exact solve_burn ΔV at each 600-km shell crossing (the burn MPC uses)
and DEGRADE it three ways, then run 30 days and see if the orbit still holds:

  A) EXACT           : the full 3-D solve_burn ΔV (control — this is MPC-S3).
  B) DIRECTION-EXACT, MAG-QUANTIZED : keep solve_burn's DIRECTION, but snap the
     magnitude to the nearest of a small discrete menu {0, 0.5, 1.0, 1.5, 2.0} m/s.
  C) PROGRADE-ONLY, MAG-QUANTIZED   : ignore solve_burn's direction; fire along
     +velocity with a quantized magnitude (what the OLD scalar toy did).
  D) FIXED 1.0 m/s in solve_burn direction (your "what if it's just 1 every step").

Also prints, for the first few steps, the solve_burn ΔV COMPONENTS + how it splits
vs the velocity direction, to answer "how much ΔV in each direction, does it matter".
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
MENU = np.array([0.0, 0.5, 1.0, 1.5, 2.0])  # m/s discrete magnitude menu


def quantize(mag_ms, menu=MENU):
    return float(menu[np.argmin(np.abs(menu - mag_ms))])


def run(mode, horizon=30 * 86400.0, verbose_steps=0):
    """mode in {'exact','dir_quant','prograde_quant','fixed1'}."""
    s = ic.copy(); t = 0.0; total_dv = 0.0; nb = 0; min_alt = np.inf; step = 0
    while t < horizon:
        ctrl = make_altitude_event(CONTROL_ALT_KM, terminal=True)
        crash = make_crash_event(PERIAPSIS_CRASH_ALT); esc = _make_escape_event()
        peri = make_periapsis_event(terminal=False)
        sol = propagate(cr3bp_j2_eom, s, (0.0, horizon - t),
                        events=[ctrl, crash, peri, esc], rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
        for yp in sol.y_events[2]:
            min_alt = min(min_alt, _altitude_km(yp))
        if len(sol.t_events[3]): return dict(outcome="escape", t=(t+sol.t_events[3][0])/86400, nb=nb, dv=total_dv, min_alt=min_alt)
        if len(sol.t_events[1]): return dict(outcome="crash", t=(t+sol.t_events[1][0])/86400, nb=nb, dv=total_dv, min_alt=min_alt)
        if not len(sol.t_events[0]): break
        t += float(sol.t_events[0][0]); sc = sol.y_events[0][0].copy()

        burn = solve_burn(sc, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                          r_peri_nom=r_peri_nom, r_apo_nom=r_apo_nom)
        dv = burn["dv"]; mag = np.linalg.norm(dv) * 1e3
        vhat = sc[3:] / np.linalg.norm(sc[3:])
        if mode == "exact":
            dv_cmd = dv
        elif mode == "dir_quant":
            qm = quantize(mag)
            dv_cmd = (dv / (mag/1e3) * (qm/1e3)) if mag > 0 else np.zeros(3)
        elif mode == "prograde_quant":
            qm = quantize(mag)
            dv_cmd = vhat * (qm/1e3)
        elif mode == "fixed1":
            dv_cmd = (dv / (mag/1e3) * (1.0/1e3)) if mag > 0 else np.zeros(3)
        applied = np.linalg.norm(dv_cmd) * 1e3
        total_dv += applied; nb += 1 if applied > 0 else 0

        if step < verbose_steps:
            comp_v = float(dv @ vhat) * 1e3
            perp = dv - (dv @ vhat) * vhat
            print(f"   step {step}: exact ΔV={mag:.3f} m/s  along-v={comp_v:+.3f}  "
                  f"perp={np.linalg.norm(perp)*1e3:.3f}  components(m/s)={dv*1e3}")
        step += 1

        s_post = sc.copy(); s_post[3:] += dv_cmd
        # coast past shell to next periapsis
        peri2 = make_periapsis_event(terminal=True); crash2 = make_crash_event(PERIAPSIS_CRASH_ALT); esc2 = _make_escape_event()
        c = propagate(cr3bp_j2_eom, s_post, (0.0, horizon - t),
                      events=[peri2, crash2, esc2], rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
        if len(c.t_events[2]): return dict(outcome="escape", t=(t+c.t_events[2][0])/86400, nb=nb, dv=total_dv, min_alt=min_alt)
        if len(c.t_events[1]): return dict(outcome="crash", t=(t+c.t_events[1][0])/86400, nb=nb, dv=total_dv, min_alt=min_alt)
        if len(c.t_events[0]):
            s = c.y_events[0][0].copy(); min_alt = min(min_alt, _altitude_km(s)); t += float(c.t_events[0][0])
        else:
            t = horizon
    return dict(outcome="held", t=horizon/86400, nb=nb, dv=total_dv, min_alt=min_alt)


print("=== solve_burn ΔV components for first 4 steps (direction breakdown) ===")
run("exact", horizon=3*86400.0, verbose_steps=4)

print("\n=== 30-day hold under discretized burns ===")
for mode in ["exact", "dir_quant", "prograde_quant", "fixed1"]:
    r = run(mode)
    print(f" {mode:16s}: outcome={r['outcome']:6s}  t={r['t']:6.2f} d  "
          f"burns={r['nb']:3d}  ΔV={r['dv']:7.2f} m/s  min_alt={r['min_alt']:.1f} km")
