"""
Experiment 09 — WHERE are the maneuvers, and in WHAT direction is the ΔV?

Captures the controlled trajectory (MPC Strategy 3, EncJ2 truth) over a few revs and,
at each 600-km-shell burn, records:
  - the spacecraft POSITION (Enceladus-relative, km),
  - the local orbit frame {v_hat prograde, c_hat in-plane cross-track, h_hat normal},
  - the exact solve_burn ΔV and its decomposition into (prograde, cross-track, normal).

Writes a .npz the plotting script (scripts/plot_burn_geometry.py) turns into a figure:
the orbit around Enceladus with ΔV arrows at each burn, showing the burns fire on the
INBOUND leg (~600 km shell) and point ~orbit-normal, NOT prograde.

Run: PYTHONPATH=. python scripts/pomdp_experiments/09_burn_geometry.py
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
from pathlib import Path

ONE_REV_S = PERIOD3_PERIOD_S / 3.0
ic = _nd_to_phys(PERIOD3_IC_ND)
r_peri_nom, r_apo_nom = nominal_apse_positions(ic, ONE_REV_S, eom=cr3bp_eom)
OUT = Path(__file__).resolve().parents[2] / "figures" / "burn_geometry_data.npz"


def enc_rel(state):
    r = state[:3].copy(); r[0] -= X_ENCELADUS; return r


def local_frame(state):
    r = enc_rel(state); v = state[3:].copy()
    v_hat = v / np.linalg.norm(v)
    h = np.cross(r, v); h_hat = h / np.linalg.norm(h)
    c_hat = np.cross(h_hat, v_hat)
    return v_hat, c_hat, h_hat


# Collect a dense trajectory (Enceladus-relative) + burn records over ~3 revs.
traj = []         # (x,y,z) enc-relative, km, sampled
burns = []        # dicts: pos, dv (km/s), decomposition (m/s), alt
s = ic.copy(); t = 0.0; horizon = 6.2 * ONE_REV_S
n_burn = 0
while t < horizon and n_burn < 12:
    ctrl = make_altitude_event(CONTROL_ALT_KM, terminal=True)
    crash = make_crash_event(PERIAPSIS_CRASH_ALT); esc = _make_escape_event()
    sol = propagate(cr3bp_j2_eom, s, (0.0, horizon - t), events=[ctrl, crash, esc],
                    rtol=RTOL_TRUTH, atol=ATOL_TRUTH, dense_output=True)
    # sample the leg for the trajectory line
    tt = np.linspace(0, sol.t[-1], 400)
    ys = sol.sol(tt)
    for col in ys.T:
        traj.append(enc_rel(col))
    if len(sol.t_events[1]) or len(sol.t_events[2]) or not len(sol.t_events[0]):
        break
    t += float(sol.t_events[0][0]); sc = sol.y_events[0][0].copy()
    burn = solve_burn(sc, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                      r_peri_nom=r_peri_nom, r_apo_nom=r_apo_nom)
    dv = burn["dv"]; v_hat, c_hat, h_hat = local_frame(sc)
    burns.append(dict(
        pos=enc_rel(sc), dv=dv,
        prograde=float(dv @ v_hat) * 1e3, cross=float(dv @ c_hat) * 1e3,
        normal=float(dv @ h_hat) * 1e3, mag=float(np.linalg.norm(dv)) * 1e3,
        alt=_altitude_km(sc),
    ))
    n_burn += 1
    s_post = sc.copy(); s_post[3:] += dv
    peri2 = make_periapsis_event(terminal=True); crash2 = make_crash_event(PERIAPSIS_CRASH_ALT); esc2 = _make_escape_event()
    c = propagate(cr3bp_j2_eom, s_post, (0.0, horizon - t), events=[peri2, crash2, esc2],
                  rtol=RTOL_TRUTH, atol=ATOL_TRUTH, dense_output=True)
    tt = np.linspace(0, c.t[-1], 400); ys = c.sol(tt)
    for col in ys.T: traj.append(enc_rel(col))
    if len(c.t_events[1]) or len(c.t_events[2]) or not len(c.t_events[0]): break
    s = c.y_events[0][0].copy(); t += float(c.t_events[0][0])

traj = np.array(traj)
bp = np.array([b["pos"] for b in burns])
bdv = np.array([b["dv"] for b in burns])
print(f"captured {len(burns)} burns over ~3 revs")
print(f"{'#':>2} {'alt(km)':>8} {'|ΔV|':>7} {'prograde':>9} {'cross':>7} {'normal':>7}  (m/s)")
for i, b in enumerate(burns):
    print(f"{i:2d} {b['alt']:8.1f} {b['mag']:7.3f} {b['prograde']:+9.3f} "
          f"{b['cross']:+7.3f} {b['normal']:+7.3f}")

np.savez(OUT, traj=traj, burn_pos=bp, burn_dv=bdv,
         prograde=[b["prograde"] for b in burns],
         cross=[b["cross"] for b in burns],
         normal=[b["normal"] for b in burns],
         mag=[b["mag"] for b in burns],
         r_enceladus=R_ENCELADUS)
print(f"wrote {OUT}")
