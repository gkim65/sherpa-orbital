"""
Experiment 06 — can we do a ONE-PASS EXCURSION to a different altitude and RECOVER?

Physical prerequisite for the GP/active-sensing science idea (single-excursion form):
the policy wants to SAMPLE a science field at a chosen altitude on ONE pass, then
RETURN to the nominal orbit — a "touch and go", NOT dwelling at the off-nominal
altitude. This is more forgiving than holding an off-nominal orbit: we only need to
(a) command a burn that brings the NEXT periapsis to a target altitude, (b) get there
without crashing, then (c) re-target the nominal orbit and recover without escaping.

Exp 05 showed scalar-ALTITUDE targeting escapes in ~3 d if held. Here we do NOT hold
the excursion altitude — we visit it for one pass then go back to position-mode
nominal targeting. Question: can we excurse + recover, over repeated excursions,
without loss?

We realize the excursion by targeting a radially-scaled nominal peri POSITION (about
Enceladus) for ONE pass, then reverting r_peri_nom/r_apo_nom to the true nominal.

Run: PYTHONPATH=. python scripts/pomdp_experiments/06_shift_and_restabilize.py
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


def enc_relative(r):
    r = r.copy(); r[0] -= X_ENCELADUS; return r


def scale_to_alt(r_nom, target_alt_km):
    """Scale a nominal apse position vector radially (about Enceladus) so its
    altitude becomes target_alt_km, keeping direction. Returns barycentre-frame."""
    rr = enc_relative(r_nom)
    cur = np.linalg.norm(rr)
    tgt = R_ENCELADUS + target_alt_km
    rr_scaled = rr * (tgt / cur)
    out = rr_scaled.copy(); out[0] += X_ENCELADUS
    return out


NOM_PERI_ALT = np.linalg.norm(enc_relative(r_peri_nom)) - R_ENCELADUS
NOM_APO_ALT = np.linalg.norm(enc_relative(r_apo_nom)) - R_ENCELADUS
print(f"nominal peri alt = {NOM_PERI_ALT:.1f} km, apo alt = {NOM_APO_ALT:.1f} km")


def run(horizon_d, excursion_period=None, excursion_alt=None, apo_alt=None):
    """Hold position-mode toward the NOMINAL orbit. Every `excursion_period` passes,
    do ONE pass targeting a scaled peri altitude `excursion_alt` (touch-and-go), then
    revert to nominal. excursion_period=None -> pure nominal hold (control).

    Records per pass (day, peri_alt, is_excursion, target_alt)."""
    horizon = horizon_d * 86400.0
    apo_alt = NOM_APO_ALT if apo_alt is None else apo_alt
    ra_nom_scaled = scale_to_alt(r_apo_nom, apo_alt)
    s = ic.copy(); t = 0.0; total_dv = 0.0; nb = 0; min_alt = np.inf; peris = []
    pass_i = 0
    while t < horizon:
        is_exc = (excursion_period is not None and pass_i > 0
                  and pass_i % excursion_period == 0)
        if is_exc:
            rp = scale_to_alt(r_peri_nom, excursion_alt); ra = ra_nom_scaled; tgt = excursion_alt
        else:
            rp = r_peri_nom; ra = r_apo_nom; tgt = NOM_PERI_ALT

        ctrl = make_altitude_event(CONTROL_ALT_KM, terminal=True)
        crash = make_crash_event(PERIAPSIS_CRASH_ALT); esc = _make_escape_event()
        peri = make_periapsis_event(terminal=False)
        sol = propagate(cr3bp_j2_eom, s, (0.0, horizon - t),
                        events=[ctrl, crash, peri, esc], rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
        for yp in sol.y_events[2]:
            min_alt = min(min_alt, _altitude_km(yp))
        if len(sol.t_events[3]): return dict(outcome="escape", t=(t+sol.t_events[3][0])/86400, nb=nb, dv=total_dv, min_alt=min_alt, peris=peris)
        if len(sol.t_events[1]): return dict(outcome="crash", t=(t+sol.t_events[1][0])/86400, nb=nb, dv=total_dv, min_alt=min_alt, peris=peris)
        if not len(sol.t_events[0]): break
        t += float(sol.t_events[0][0]); sc = sol.y_events[0][0].copy()
        burn = solve_burn(sc, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                          r_peri_nom=rp, r_apo_nom=ra)
        s_post = sc.copy(); s_post[3:] += burn["dv"]
        total_dv += burn["dv_mag_ms"]; nb += 1 if burn["dv_mag_ms"] > 0 else 0
        peri2 = make_periapsis_event(terminal=True); crash2 = make_crash_event(PERIAPSIS_CRASH_ALT); esc2 = _make_escape_event()
        c = propagate(cr3bp_j2_eom, s_post, (0.0, horizon - t),
                      events=[peri2, crash2, esc2], rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
        if len(c.t_events[2]): return dict(outcome="escape", t=(t+c.t_events[2][0])/86400, nb=nb, dv=total_dv, min_alt=min_alt, peris=peris)
        if len(c.t_events[1]): return dict(outcome="crash", t=(t+c.t_events[1][0])/86400, nb=nb, dv=total_dv, min_alt=min_alt, peris=peris)
        if len(c.t_events[0]):
            s = c.y_events[0][0].copy(); pa = _altitude_km(s); min_alt = min(min_alt, pa)
            peris.append((t/86400.0, pa, is_exc, tgt)); t += float(c.t_events[0][0])
        else:
            t = horizon
        pass_i += 1
    return dict(outcome="held", t=horizon/86400, nb=nb, dv=total_dv, min_alt=min_alt, peris=peris)


print("\n=== A) control: pure nominal hold, no excursions (10 d) ===")
r = run(10.0)
print(f"  outcome={r['outcome']:6s} t={r['t']:5.2f} d burns={r['nb']:3d} ΔV={r['dv']:6.2f} min_peri={r['min_alt']:.1f}")

print("\n=== B) ONE-PASS EXCURSIONS every 4 passes, then recover (10 d) ===")
for exc_alt in [45.0, 60.0, 22.0]:
    r = run(10.0, excursion_period=4, excursion_alt=exc_alt)
    exc_peris = [p for (_, p, e, _) in r["peris"] if e]
    print(f"  excursion peri={exc_alt:5.1f}: outcome={r['outcome']:6s} t={r['t']:5.2f} d "
          f"burns={r['nb']:3d} ΔV={r['dv']:6.2f}  min_peri={r['min_alt']:5.1f}  "
          f"excursion peris={[f'{p:.0f}' for p in exc_peris[:6]]}")

print("\n=== C) detailed timeline: excursion to 55 km every 4 passes (10 d) ===")
r = run(10.0, excursion_period=4, excursion_alt=55.0)
print(f"  outcome={r['outcome']:6s} t={r['t']:5.2f} d burns={r['nb']} ΔV={r['dv']:.2f} min_peri={r['min_alt']:.1f}")
print("  pass (day : peri_alt : E=excursion/H=hold : target):")
for (d, p, e, tgt) in r["peris"]:
    print(f"    {d:6.2f} d : {p:7.1f} km : {'E' if e else 'H'} : tgt={tgt:.0f}")
