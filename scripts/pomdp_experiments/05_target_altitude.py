"""
Experiment 05 — can we command a burn to hold a CHOSEN target altitude, and switch
between altitude regimes (science-low vs safe-high) across passes?

This is the physical foundation for the future science objective: the POMDP action
could be a TARGET-ALTITUDE choice (HOLD / GO-LOW-for-science / GO-SAFE-away-from-crash),
each realized by solve_burn toward that target, rather than a raw ΔV. We test whether
that capability actually exists in the truth model before building actions on it.

Three questions, all at the 600-km-shell cadence, EncJ2 truth, using trusted paths:

  A) POSITION mode toward the NOMINAL orbit (baseline, = exp 03/04): holds ~31 km peri.
  B) ALTITUDE mode toward a chosen (peri_target, apo_target): can we hold a DIFFERENT
     periapsis altitude than nominal? Try a SAFE-HIGH target (peri ~60 km, top of the
     MacKenzie band) and a LOW-SCIENCE target (peri ~20 km, bottom of the band).
  C) SWITCHING: hold nominal for a few days, then command GO-SAFE (raise peri target)
     and confirm periapsis actually rises and stays — the "different pass to stabilize"
     idea. Reports peri altitude over time so we can see the regime change.

MacKenzie band (constants): periapsis 19.8-64.3 km, apoapsis 1000-1110 km.

Run: PYTHONPATH=. python scripts/pomdp_experiments/05_target_altitude.py
"""
import numpy as np
from baselines.mpc import (
    run_mpc, solve_burn, nominal_apse_positions, _altitude_km, _make_escape_event,
    CONTROL_ALT_KM, PERIAPSIS_ALT_TARGET, APOAPSIS_ALT_TARGET,
)
from src.dynamics.cr3bp import cr3bp_eom
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.integrator import (
    propagate, make_altitude_event, make_periapsis_event, make_crash_event,
    RTOL_TRUTH, ATOL_TRUTH,
)
from src.constants import (
    PERIAPSIS_CRASH_ALT, PERIAPSIS_ALT_MIN, PERIAPSIS_ALT_MAX,
    APOAPSIS_ALT_MIN, APOAPSIS_ALT_MAX,
)
from src.utils.halo_ic import PERIOD3_IC_ND, PERIOD3_PERIOD_S, _nd_to_phys

ONE_REV_S = PERIOD3_PERIOD_S / 3.0
ic = _nd_to_phys(PERIOD3_IC_ND)
APO_NOM = 1055.0  # ~mid apoapsis band; the period-3 orbit's apoapsis


def run_altitude(peri_tgt, apo_tgt, horizon=15 * 86400.0, switch=None, log_every=1):
    """Run MPC-style loop in ALTITUDE mode toward (peri_tgt, apo_tgt).
    If switch=(t_switch_d, new_peri_tgt): change the peri target at that time."""
    s = ic.copy(); t = 0.0; total_dv = 0.0; nb = 0; min_alt = np.inf
    p_tgt = peri_tgt; peris = []
    while t < horizon:
        if switch and t >= switch[0] * 86400.0 and p_tgt != switch[1]:
            p_tgt = switch[1]
            peris.append((t/86400.0, None, f"SWITCH peri_target -> {p_tgt} km"))
        ctrl = make_altitude_event(CONTROL_ALT_KM, terminal=True)
        crash = make_crash_event(PERIAPSIS_CRASH_ALT); esc = _make_escape_event()
        peri = make_periapsis_event(terminal=False)
        sol = propagate(cr3bp_j2_eom, s, (0.0, horizon - t),
                        events=[ctrl, crash, peri, esc], rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
        for yp in sol.y_events[2]:
            a = _altitude_km(yp); min_alt = min(min_alt, a)
        if len(sol.t_events[3]): return dict(outcome="escape", t=(t+sol.t_events[3][0])/86400, nb=nb, dv=total_dv, min_alt=min_alt, peris=peris)
        if len(sol.t_events[1]): return dict(outcome="crash", t=(t+sol.t_events[1][0])/86400, nb=nb, dv=total_dv, min_alt=min_alt, peris=peris)
        if not len(sol.t_events[0]): break
        t += float(sol.t_events[0][0]); sc = sol.y_events[0][0].copy()
        burn = solve_burn(sc, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="altitude",
                          peri_target_km=p_tgt, apo_target_km=apo_tgt)
        s_post = sc.copy(); s_post[3:] += burn["dv"]
        total_dv += burn["dv_mag_ms"]; nb += 1 if burn["dv_mag_ms"] > 0 else 0
        peri2 = make_periapsis_event(terminal=True); crash2 = make_crash_event(PERIAPSIS_CRASH_ALT); esc2 = _make_escape_event()
        c = propagate(cr3bp_j2_eom, s_post, (0.0, horizon - t),
                      events=[peri2, crash2, esc2], rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
        if len(c.t_events[2]): return dict(outcome="escape", t=(t+c.t_events[2][0])/86400, nb=nb, dv=total_dv, min_alt=min_alt, peris=peris)
        if len(c.t_events[1]): return dict(outcome="crash", t=(t+c.t_events[1][0])/86400, nb=nb, dv=total_dv, min_alt=min_alt, peris=peris)
        if len(c.t_events[0]):
            s = c.y_events[0][0].copy(); pa = _altitude_km(s); min_alt = min(min_alt, pa)
            peris.append((t/86400.0, pa, f"tgt={p_tgt:.0f}")); t += float(c.t_events[0][0])
        else:
            t = horizon
    return dict(outcome="held", t=horizon/86400, nb=nb, dv=total_dv, min_alt=min_alt, peris=peris)


print(f"MacKenzie band: peri {PERIAPSIS_ALT_MIN}-{PERIAPSIS_ALT_MAX} km, "
      f"apo {APOAPSIS_ALT_MIN}-{APOAPSIS_ALT_MAX} km")
print(f"default targets: peri {PERIAPSIS_ALT_TARGET:.1f}, apo {APOAPSIS_ALT_TARGET:.1f} km\n")

print("=== A) POSITION mode toward nominal (baseline, 15 d) ===")
r = run_mpc(ic, truth_eom=cr3bp_j2_eom, period_s=ONE_REV_S, horizon_s=15*86400.0,
            n_revs=3, mode="position", ref_ic=ic)
print(f"  outcome={r['outcome']:6s} t={r['survival_time_s']/86400:5.2f} d  "
      f"burns={r['n_burns']:3d} ΔV={r['total_dv_ms']:6.2f}  min_peri={r['min_peri_alt_km']:.1f} km")

print("\n=== B) ALTITUDE mode toward chosen targets (15 d) ===")
for label, ptgt in [("SAFE-HIGH peri=60", 60.0), ("MID peri=42", 42.0), ("LOW-SCI peri=22", 22.0)]:
    r = run_altitude(ptgt, APO_NOM)
    held_peris = [p for (_, p, _) in r["peris"] if p is not None]
    med = np.median(held_peris) if held_peris else np.nan
    print(f"  {label:20s}: outcome={r['outcome']:6s} t={r['t']:5.2f} d  "
          f"burns={r['nb']:3d} ΔV={r['dv']:6.2f}  min_peri={r['min_alt']:5.1f}  "
          f"median held peri={med:.1f} km")

print("\n=== C) SWITCHING: hold peri=42 for 5 d, then GO-SAFE peri=60 (12 d total) ===")
r = run_altitude(42.0, APO_NOM, horizon=12*86400.0, switch=(5.0, 60.0))
print(f"  outcome={r['outcome']:6s} t={r['t']:5.2f} d burns={r['nb']} ΔV={r['dv']:.2f} min_peri={r['min_alt']:.1f}")
print("  periapsis altitude timeline (day : peri_alt : note):")
for (d, p, note) in r["peris"]:
    ps = f"{p:7.1f}" if p is not None else "   ----"
    print(f"    {d:6.2f} d : {ps} km : {note}")
