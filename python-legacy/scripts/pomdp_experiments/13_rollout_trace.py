"""
Experiment 13 — diagnostic rollout trace of the science+safety SARSOP policy vs the
real CR3BP+EncJ2 truth, recording EVERYTHING the deck plots need.

Re-runs the same control logic as baselines/pomdp_rollout.run_pomdp_rollout, but also
logs, per control step:
  - the DENSE trajectory arc (Enceladus-relative x,y,z) for both legs (to-shell +
    post-burn to-periapsis), so we can draw the actual orbit over time;
  - the SHELL burn position (where the maneuver fired) and the exact solve_burn ΔV
    VECTOR (km/s) + its magnitude — the maneuver the SARSOP action produced;
  - the achieved periapsis position + altitude, the TRUE apse deviation, and the NOISY
    OBSERVED deviation (so we can show how real the observation was vs truth);
  - the action label + resulting coverage.

Saves figures/rollout_trace_data.npz for scripts/plot_rollout_trace.py.

Run: PYTHONPATH=. python scripts/pomdp_experiments/13_rollout_trace.py
"""
import numpy as np
from pathlib import Path
from baselines.pomdp_rollout import SarsopPolicy, ONE_REV_S, _enc_rel, _scale_to_alt
from baselines.mpc import (
    solve_burn, nominal_apse_positions, _altitude_km, _make_escape_event, CONTROL_ALT_KM,
)
from src.dynamics.cr3bp import cr3bp_eom, X_ENCELADUS
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.integrator import (
    propagate, make_periapsis_event, make_crash_event, make_altitude_event,
    RTOL_TRUTH, ATOL_TRUTH,
)
from src.constants import R_ENCELADUS, PERIAPSIS_CRASH_ALT, SIGMA_NAV_POS
from src.spacecraft import thruster
from src.utils.halo_ic import PERIOD3_IC_ND, _nd_to_phys

OUT = Path(__file__).resolve().parents[2] / "figures" / "rollout_trace_data.npz"
ic = _nd_to_phys(PERIOD3_IC_ND)
policy = SarsopPolicy.load()
r_peri_nom, r_apo_nom = nominal_apse_positions(ic, ONE_REV_S, eom=cr3bp_eom)
apo_nom_alt = np.linalg.norm(_enc_rel(r_apo_nom)) - R_ENCELADUS
rng = np.random.default_rng(0)

HORIZON = 8 * 86400.0     # 8 days — enough to show all excursions + the hold that follows
truth = cr3bp_j2_eom


def prop(events, s, horizon):
    return propagate(truth, s, (0.0, horizon), events=events,
                     rtol=RTOL_TRUTH, atol=ATOL_TRUTH, dense_output=True)


def dev_of(p): return float(np.linalg.norm(p[:3] - r_peri_nom))
def arc(sol, n=200):
    tt = np.linspace(0.0, sol.t[-1], n)
    return _enc_rel_cols(sol.sol(tt))
def _enc_rel_cols(ys):
    out = ys.copy(); out[0, :] -= X_ENCELADUS; return out[:3, :].T   # (n,3) enc-rel pos


state = ic.copy(); t_now = 0.0; cov = 0
belief = policy.initial_belief()
traj = []                 # list of (n,3) arcs, concatenated later
burns = []                # dicts per maneuver
peris = []                # per-pass periapsis records
while t_now < HORIZON:
    # to shell
    sol = prop([make_altitude_event(CONTROL_ALT_KM, terminal=True),
                make_crash_event(PERIAPSIS_CRASH_ALT), _make_escape_event()],
               state, HORIZON - t_now)
    traj.append(arc(sol))
    if len(sol.t_events[1]) or len(sol.t_events[2]) or not len(sol.t_events[0]):
        break
    t_now += float(sol.t_events[0][0]); sc = sol.y_events[0][0].copy()

    action = policy.action(belief)
    dv_cmd = np.zeros(3); band = 0
    if action == "CORRECT":
        b = solve_burn(sc, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                       r_peri_nom=r_peri_nom, r_apo_nom=r_apo_nom); dv_cmd = b["dv"]
    elif action.startswith("EXCURSE_"):
        band = policy.excurse_band[action]; bn = policy.band_names[band - 1]
        rp = _scale_to_alt(r_peri_nom, policy.band_target_km[bn])
        ra = _scale_to_alt(r_apo_nom, apo_nom_alt)
        b = solve_burn(sc, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                       r_peri_nom=rp, r_apo_nom=ra); dv_cmd = b["dv"]

    dv_app, eta = thruster.apply_dv_noisy(dv_cmd, rng=rng)
    burns.append(dict(pos=_enc_rel(sc)[:3], dv=dv_cmd.copy(), dv_ms=float(np.linalg.norm(dv_app)*1e3),
                      action=action, t=t_now))
    s_post = sc.copy(); s_post[3:] += dv_app

    sol2 = prop([make_periapsis_event(terminal=True),
                 make_crash_event(PERIAPSIS_CRASH_ALT), _make_escape_event()],
                s_post, HORIZON - t_now)
    traj.append(arc(sol2))
    if len(sol2.t_events[1]) or len(sol2.t_events[2]) or not len(sol2.t_events[0]):
        break
    t_now += float(sol2.t_events[0][0]); pr = sol2.y_events[0][0].copy()

    dev_true = dev_of(pr)
    obs = abs(dev_true + rng.normal(0, SIGMA_NAV_POS))
    if band != 0: cov |= (1 << (band - 1))
    belief = policy.update_belief(belief, action, policy.dev_bin(obs))
    peris.append(dict(pos=_enc_rel(pr)[:3], alt=_altitude_km(pr), dev_true=dev_true,
                      dev_obs=obs, t=t_now, cov=cov))
    state = pr

traj = np.vstack(traj)
np.savez(
    OUT,
    traj=traj,
    burn_pos=np.array([b["pos"] for b in burns]),
    burn_dv=np.array([b["dv"] for b in burns]),
    burn_dv_ms=np.array([b["dv_ms"] for b in burns]),
    burn_action=np.array([b["action"] for b in burns]),
    burn_t=np.array([b["t"] for b in burns]),
    peri_pos=np.array([p["pos"] for p in peris]),
    peri_alt=np.array([p["alt"] for p in peris]),
    dev_true=np.array([p["dev_true"] for p in peris]),
    dev_obs=np.array([p["dev_obs"] for p in peris]),
    peri_t=np.array([p["t"] for p in peris]),
    r_enceladus=R_ENCELADUS, control_alt=CONTROL_ALT_KM,
    dev_edges=np.array(policy.dev_edges),
)
print(f"{len(burns)} maneuvers, {len(peris)} periapsis passes over "
      f"{peris[-1]['t']/86400:.1f} d (final cov={cov:03b})")
print(f"{'t(d)':>6} {'action':>13} {'ΔV(m/s)':>8} {'peri_alt':>9} "
      f"{'dev_true':>9} {'dev_obs':>8}")
for b, p in zip(burns, peris):
    print(f"{p['t']/86400:6.2f} {b['action']:>13} {b['dv_ms']:8.2f} "
          f"{p['alt']:9.1f} {p['dev_true']:9.1f} {p['dev_obs']:8.1f}")
print(f"wrote {OUT}")
