"""
Experiment 06b — touch-and-go timeline VISITING MULTIPLE distinct altitudes.

Extends exp 06 (single excursion altitude) to CYCLE through several excursion
targets, so the timeline shows the spacecraft sampling a RANGE of altitudes and
recovering to nominal between each — the direct picture of the GP/active-sensing
"sample a range of altitudes" idea. Writes figures/excursion_timeline_data.npz for
scripts/plot_pomdp_findings.py panel (d).

Cadence: hold nominal (position-mode) each pass; every `excursion_period` passes do a
one-pass excursion to the NEXT altitude in a cycling menu, then revert to nominal.

Run: PYTHONPATH=. python scripts/pomdp_experiments/06b_multi_altitude_timeline.py
"""
import numpy as np
from pathlib import Path
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
OUT = Path(__file__).resolve().parents[2] / "figures" / "excursion_timeline_data.npz"

# Cycling excursion menu (commanded targets; achieved will be compressed per exp 07).
EXC_MENU = [45.0, 70.0, 120.0, 90.0]
EXC_PERIOD = 3


def enc_rel(r):
    r = r.copy(); r[0] -= X_ENCELADUS; return r


def scale_to_alt(r_nom, alt):
    rr = enc_rel(r_nom); rr2 = rr * ((R_ENCELADUS + alt) / np.linalg.norm(rr))
    out = rr2.copy(); out[0] += X_ENCELADUS; return out


ra_nom = scale_to_alt(r_apo_nom, np.linalg.norm(enc_rel(r_apo_nom)) - R_ENCELADUS)
s = ic.copy(); t = 0.0; horizon = 16 * 86400.0
days, peri, is_exc, cmd = [], [], [], []
pass_i = 0; exc_k = 0
while t < horizon:
    do_exc = (pass_i > 0 and pass_i % EXC_PERIOD == 0)
    if do_exc:
        tgt = EXC_MENU[exc_k % len(EXC_MENU)]; exc_k += 1
        rp = scale_to_alt(r_peri_nom, tgt); ra = ra_nom
    else:
        tgt = np.nan; rp = r_peri_nom; ra = r_apo_nom
    ctrl = make_altitude_event(CONTROL_ALT_KM, terminal=True)
    crash = make_crash_event(PERIAPSIS_CRASH_ALT); esc = _make_escape_event()
    sol = propagate(cr3bp_j2_eom, s, (0.0, horizon - t), events=[ctrl, crash, esc],
                    rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
    if len(sol.t_events[1]) or len(sol.t_events[2]) or not len(sol.t_events[0]):
        print("terminal:", "crash" if len(sol.t_events[1]) else
              ("escape" if len(sol.t_events[2]) else "no-shell")); break
    t += float(sol.t_events[0][0]); sc = sol.y_events[0][0].copy()
    b = solve_burn(sc, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                   r_peri_nom=rp, r_apo_nom=ra)
    sp = sc.copy(); sp[3:] += b["dv"]
    peri2 = make_periapsis_event(terminal=True); crash2 = make_crash_event(PERIAPSIS_CRASH_ALT); esc2 = _make_escape_event()
    c = propagate(cr3bp_j2_eom, sp, (0.0, horizon - t), events=[peri2, crash2, esc2],
                  rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
    if len(c.t_events[1]) or len(c.t_events[2]) or not len(c.t_events[0]):
        print("terminal after burn"); break
    s = c.y_events[0][0].copy(); t += float(c.t_events[0][0])
    days.append(t / 86400.0); peri.append(_altitude_km(s))
    is_exc.append(do_exc); cmd.append(tgt)
    pass_i += 1

days = np.array(days); peri = np.array(peri); is_exc = np.array(is_exc); cmd = np.array(cmd)
np.savez(OUT, days=days, peri=peri, is_exc=is_exc, cmd=cmd)
print(f"passes={len(days)}  excursions={int(is_exc.sum())}  "
      f"min_peri={peri.min():.1f}  max_peri={peri.max():.1f}")
for d, p, e, cq in zip(days, peri, is_exc, cmd):
    tag = f"E cmd={cq:.0f}" if e else "H"
    print(f"  {d:6.2f} d : {p:7.1f} km : {tag}")
print(f"wrote {OUT}")
