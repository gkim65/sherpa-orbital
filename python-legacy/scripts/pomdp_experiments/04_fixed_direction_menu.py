"""
Experiment 04 — can a SMALL FIXED menu of burn DIRECTIONS hold the orbit?

Motivation
----------
Experiment 03 showed the holding burn is ~cross-track and that MAGNITUDE can be
coarsely quantized without losing the 30-day hold — BUT it kept solve_burn's EXACT
DIRECTION each step. The open question for a fully-discrete-action POMDP: can the
DIRECTION also come from a small fixed menu, or does the required direction rotate
rev-to-rev (forcing solve_burn to stay in the loop)?

This script does two things at the 600-km descending shell cadence (EncJ2 truth):

  PART A — DIAGNOSE how much the exact solve_burn direction rotates.
    At each shell crossing, record the exact solve_burn unit vector, expressed in
    two frames:
      - the rotating (CR3BP) frame (raw x/y/z components), and
      - a LOCAL orbit frame at the control point: {v_hat (prograde),
        h_hat (orbit normal = r×v), c_hat (in-plane cross-track = h×v)}.
    If the direction is ~constant in the local frame, one body-relative menu entry
    could hold it. Report the pairwise angles between consecutive unit vectors.

  PART B — TEST fixed-direction menus over 30 days.
    Replace solve_burn's direction with the NEAREST entry from a candidate menu
    (magnitude still taken from solve_burn, then optionally quantized), and see if
    the orbit holds. Menus tried:
      - single mean direction (the average local-frame direction from Part A),
      - a small set of local-frame axes (+/- h_hat, +/- c_hat),
      - control: exact direction (must hold, = experiment 03's dir_quant baseline).

Run: PYTHONPATH=. python scripts/pomdp_experiments/04_fixed_direction_menu.py
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
from src.constants import PERIAPSIS_CRASH_ALT
from src.utils.halo_ic import PERIOD3_IC_ND, PERIOD3_PERIOD_S, _nd_to_phys

ONE_REV_S = PERIOD3_PERIOD_S / 3.0
ic = _nd_to_phys(PERIOD3_IC_ND)
r_peri_nom, r_apo_nom = nominal_apse_positions(ic, ONE_REV_S, eom=cr3bp_eom)
MENU_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0])  # m/s


def local_frame(state):
    """Local orbit frame at a state: (v_hat, h_hat, c_hat).
    v_hat = velocity dir (prograde); h_hat = orbit normal (r x v);
    c_hat = in-plane cross-track (h x v). r is Enceladus-relative position."""
    r = state[:3].copy(); r[0] -= X_ENCELADUS
    v = state[3:].copy()
    v_hat = v / np.linalg.norm(v)
    h = np.cross(r, v); h_hat = h / np.linalg.norm(h)
    c_hat = np.cross(h_hat, v_hat)      # completes right-handed {v, c, h} in-plane cross-track
    return v_hat, h_hat, c_hat


def to_local(u, frame):
    v_hat, h_hat, c_hat = frame
    return np.array([u @ v_hat, u @ c_hat, u @ h_hat])  # [prograde, cross-track, normal]


def quant_mag(m):
    return float(MENU_MAG[np.argmin(np.abs(MENU_MAG - m))])


def run(direction_mode, quantize_mag=True, horizon=30 * 86400.0, collect=False):
    """direction_mode:
        'exact'          -> use solve_burn's exact unit vector
        'mean_local'     -> fixed unit vector = MEAN_LOCAL (set below) in local frame
        callable(frame, dv_hat_exact) -> returns a chosen UNIT vector
    Returns result dict; if collect, also returns per-step exact local unit vectors.
    """
    s = ic.copy(); t = 0.0; total_dv = 0.0; nb = 0; min_alt = np.inf
    locals_log = []
    while t < horizon:
        ctrl = make_altitude_event(CONTROL_ALT_KM, terminal=True)
        crash = make_crash_event(PERIAPSIS_CRASH_ALT); esc = _make_escape_event()
        peri = make_periapsis_event(terminal=False)
        sol = propagate(cr3bp_j2_eom, s, (0.0, horizon - t),
                        events=[ctrl, crash, peri, esc], rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
        for yp in sol.y_events[2]:
            min_alt = min(min_alt, _altitude_km(yp))
        if len(sol.t_events[3]): return dict(outcome="escape", t=(t+sol.t_events[3][0])/86400, nb=nb, dv=total_dv, min_alt=min_alt, locals=locals_log)
        if len(sol.t_events[1]): return dict(outcome="crash", t=(t+sol.t_events[1][0])/86400, nb=nb, dv=total_dv, min_alt=min_alt, locals=locals_log)
        if not len(sol.t_events[0]): break
        t += float(sol.t_events[0][0]); sc = sol.y_events[0][0].copy()

        burn = solve_burn(sc, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                          r_peri_nom=r_peri_nom, r_apo_nom=r_apo_nom)
        dv = burn["dv"]; mag = np.linalg.norm(dv) * 1e3
        frame = local_frame(sc)
        if mag > 0:
            dv_hat_exact = dv / (mag / 1e3)
            if collect:
                locals_log.append(to_local(dv_hat_exact, frame))
            if direction_mode == "exact":
                u = dv_hat_exact
            elif callable(direction_mode):
                u = direction_mode(frame, dv_hat_exact)
            else:
                raise ValueError(direction_mode)
            use_mag = quant_mag(mag) if quantize_mag else mag
            dv_cmd = u * (use_mag / 1e3)
        else:
            dv_cmd = np.zeros(3)
        applied = np.linalg.norm(dv_cmd) * 1e3
        total_dv += applied; nb += 1 if applied > 0 else 0

        s_post = sc.copy(); s_post[3:] += dv_cmd
        peri2 = make_periapsis_event(terminal=True); crash2 = make_crash_event(PERIAPSIS_CRASH_ALT); esc2 = _make_escape_event()
        c = propagate(cr3bp_j2_eom, s_post, (0.0, horizon - t),
                      events=[peri2, crash2, esc2], rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
        if len(c.t_events[2]): return dict(outcome="escape", t=(t+c.t_events[2][0])/86400, nb=nb, dv=total_dv, min_alt=min_alt, locals=locals_log)
        if len(c.t_events[1]): return dict(outcome="crash", t=(t+c.t_events[1][0])/86400, nb=nb, dv=total_dv, min_alt=min_alt, locals=locals_log)
        if len(c.t_events[0]):
            s = c.y_events[0][0].copy(); min_alt = min(min_alt, _altitude_km(s)); t += float(c.t_events[0][0])
        else:
            t = horizon
    return dict(outcome="held", t=horizon/86400, nb=nb, dv=total_dv, min_alt=min_alt, locals=locals_log)


# ── PART A: how much does the exact direction rotate? ─────────────────────────
print("=== PART A: exact solve_burn direction in LOCAL frame [prograde, cross-track, normal] ===")
diag = run("exact", quantize_mag=False, horizon=10 * 86400.0, collect=True)
L = np.array(diag["locals"])
print(f" collected {len(L)} shell-burn directions over ~10 d (outcome={diag['outcome']})")
if len(L):
    mean_local = L.mean(axis=0); mean_local /= np.linalg.norm(mean_local)
    print(f" per-component mean unit dir  [v, c, h] = {L.mean(axis=0)}")
    print(f" per-component std            [v, c, h] = {L.std(axis=0)}")
    print(f" normalized MEAN direction    [v, c, h] = {mean_local}")
    # angle of each exact dir to the mean
    angs = np.degrees(np.arccos(np.clip(L @ mean_local, -1, 1)))
    print(f" angle of each exact dir to the mean:  min={angs.min():.1f}  "
          f"median={np.median(angs):.1f}  max={angs.max():.1f} deg")
    # consecutive rotation
    cons = np.degrees(np.arccos(np.clip(np.sum(L[1:] * L[:-1], axis=1), -1, 1)))
    print(f" consecutive-step rotation:  median={np.median(cons):.1f}  max={cons.max():.1f} deg")
else:
    mean_local = np.array([0.0, 1.0, 0.0])

# ── PART B: fixed-direction menus over 30 days ────────────────────────────────
print("\n=== PART B: 30-day hold under FIXED-direction menus (magnitude quantized) ===")

def dir_mean_local(frame, dv_hat_exact):
    v_hat, h_hat, c_hat = frame
    u = mean_local[0]*v_hat + mean_local[1]*c_hat + mean_local[2]*h_hat
    return u / np.linalg.norm(u)

def make_axis_menu():
    """Menu = {+/- c_hat, +/- h_hat} in local frame; pick nearest to exact dir."""
    axes_local = [np.array([0,1,0.]), np.array([0,-1,0.]),
                  np.array([0,0,1.]), np.array([0,0,-1.])]
    def choose(frame, dv_hat_exact):
        v_hat, h_hat, c_hat = frame
        cands = [a[1]*c_hat + a[2]*h_hat + a[0]*v_hat for a in axes_local]
        cands = [c/np.linalg.norm(c) for c in cands]
        dots = [dv_hat_exact @ c for c in cands]
        return cands[int(np.argmax(dots))]
    return choose

for label, mode in [
    ("exact (control)", "exact"),
    ("single mean-local dir", dir_mean_local),
    ("4-axis local menu", make_axis_menu()),
]:
    r = run(mode, quantize_mag=True)
    print(f" {label:24s}: outcome={r['outcome']:6s}  t={r['t']:6.2f} d  "
          f"burns={r['nb']:3d}  ΔV={r['dv']:7.2f} m/s  min_alt={r['min_alt']:.1f} km")
