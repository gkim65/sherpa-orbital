"""
Deterministic MPC stationkeeping baseline (Strategy 3, MacKenzie et al. 2020 §B.2.3).

This is the feasibility controller for the Enceladus Orbilander period-3 L1 halo
orbit. It answers the project question "can we even stationkeep this orbit?" before
investing further in propagator fidelity (external review 2026-06-22, Decision D3).

Control concept (MacKenzie §B.2.3, "Strategy 3"):
  - The spacecraft coasts under the TRUTH dynamics. At each descending crossing of a
    fixed altitude shell (600 km above the Enceladus surface), the controller solves
    for an impulsive burn ΔV that re-targets the next periapsis and apoapsis altitudes
    to the centre of the mission bands.
  - The burn is planned with a MULTIPLE-SHOOTING prediction over N_m = 2–3 revolutions
    of the ONBOARD model (CR3BP, no perturbations). The world then propagates the
    resulting state forward under the TRUTH model until the next control trigger.

CRITICAL truth/onboard split (CLAUDE.md rule, do NOT collapse):
  - `truth_eom` (passed in) integrates the real world: cr3bp_j2_eom,
    cr3bp_saturn_enc_j2_eom, or later the SPICE inertial model. Tight tolerances.
  - The burn solver plans ONLY with `cr3bp_eom` (onboard) at looser tolerances. The
    controller never sees the truth perturbations — that model-gap is the uncertainty
    the POMDP is meant to absorb.

Burn solver (confirmed design, session 2026-06-22):
  - Decision variable: full 3-D impulsive ΔV (km/s) applied to velocity at the
    control point.
  - Targets: periapsis altitude and apoapsis altitude (2 residuals) at the centre of
    the mission bands.
  - 3 controls vs 2 targets is underdetermined → minimum-norm Gauss-Newton
    (pseudo-inverse step) selects the smallest-ΔV correction each iteration.

Fuel metric:
  - Reported as ΔV (m/s). Conversion to propellant mass (kg) via the rocket equation
    is deferred: it needs Isp and spacecraft wet mass from MacKenzie et al. 2020 §3.5
    (MR-106E), which must be sourced and added to `src/constants.py` first.
    TODO(constants): add ISP_MR106E and M_SPACECRAFT_WET, then expose delta-m here.

References:
  MacKenzie, S. M. et al. (2020). Enceladus Orbilander: A Flagship Mission Concept
    for Astrobiology. §B.2.3 (stationkeeping Strategy 3), §3.5 (propulsion).
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional

from src.constants import (
    R_ENCELADUS,
    PERIAPSIS_ALT_MIN, PERIAPSIS_ALT_MAX,
    APOAPSIS_ALT_MIN, APOAPSIS_ALT_MAX,
    PERIAPSIS_CRASH_ALT,
)
from src.dynamics.cr3bp import cr3bp_eom, X_ENCELADUS
from src.dynamics.integrator import (
    propagate,
    make_altitude_event,
    make_crash_event,
    make_apoapsis_event,
    make_periapsis_event,
    RTOL_TRUTH, ATOL_TRUTH,
    RTOL_ONBOARD, ATOL_ONBOARD,
)

# ── Control configuration ─────────────────────────────────────────────────────
CONTROL_ALT_KM: float = 600.0          # Strategy 3 trigger shell, above Enceladus surface
PERIAPSIS_ALT_TARGET: float = 0.5 * (PERIAPSIS_ALT_MIN + PERIAPSIS_ALT_MAX)  # 42.05 km
APOAPSIS_ALT_TARGET:  float = 0.5 * (APOAPSIS_ALT_MIN + APOAPSIS_ALT_MAX)    # 1055 km
TARGET_TOL_KM: float = 1.0             # apse-targeting tolerance (MacKenzie ≤ 1 km)

# Escape threshold: if the spacecraft reaches this altitude above Enceladus it has left
# the ~1065-km-apoapsis science orbit and the controller can no longer recover it (the
# 600-km descending control shell never re-arms). ~5× the nominal apoapsis altitude.
# This is a controller/feasibility parameter, not a physical constant.
ESCAPE_ALT_KM: float = 5.0 * APOAPSIS_ALT_MAX  # 5550 km


def _altitude_km(state: np.ndarray) -> float:
    """Altitude above the Enceladus surface (km) for a barycentre-frame state."""
    dx = state[0] - X_ENCELADUS
    r = np.sqrt(dx**2 + state[1]**2 + state[2]**2)
    return r - R_ENCELADUS


def _make_escape_event(altitude_km: float = ESCAPE_ALT_KM) -> Callable:
    """
    Terminal event firing when the spacecraft ascends through ``altitude_km`` above the
    Enceladus surface — i.e. it has left the science orbit and is escaping. Direction
    +1 (ascending) so it does not fire on the normal inbound/outbound apoapsis arc.
    """
    target_r = R_ENCELADUS + altitude_km

    def escape(t: float, state: np.ndarray) -> float:
        dx = state[0] - X_ENCELADUS
        return np.sqrt(dx**2 + state[1]**2 + state[2]**2) - target_r

    escape.terminal = True
    escape.direction = +1.0
    return escape


def predict_apses(
    state0: np.ndarray,
    n_revs: int,
    period_s: float,
    eom: Callable = cr3bp_eom,
    rtol: float = RTOL_ONBOARD,
    atol: float = ATOL_ONBOARD,
) -> tuple[float, float]:
    """
    Predict periapsis and apoapsis altitude over the next ``n_revs`` revolutions.

    This is the multiple-shooting prediction used by the burn solver. It propagates
    the ONBOARD model (default ``cr3bp_eom``) for ``n_revs`` orbital periods and
    records every periapsis/apoapsis passage (non-terminal events), returning the
    altitudes of the FIRST periapsis and FIRST apoapsis encountered.

    Parameters
    ----------
    state0 : np.ndarray, shape (6,)
        Barycentre-frame state [x,y,z,vx,vy,vz] in km, km/s (post-burn).
    n_revs : int
        Number of revolutions to span (MacKenzie N_m = 2–3).
    period_s : float
        Single-revolution period estimate (s); sets the propagation horizon.
    eom : Callable
        Equations of motion for the prediction. Defaults to the onboard CR3BP model.
    rtol, atol : float
        Integration tolerances (onboard defaults).

    Returns
    -------
    peri_alt_km, apo_alt_km : float
        Altitude (km) of the first periapsis and first apoapsis over the horizon.
        Returns np.nan for an apse not reached within the horizon.

    Notes
    -----
    Using the FIRST apse of each type (rather than the minimum/maximum over the whole
    span) keeps the residual a smooth function of ΔV, which the Gauss-Newton Jacobian
    needs. The multi-rev horizon exists so the solver "sees" several apses and the
    corrector stays well-conditioned even when the first apse is close to the burn.
    """
    t_max = n_revs * period_s
    peri_ev = make_periapsis_event(terminal=False)
    apo_ev = make_apoapsis_event(terminal=False)
    sol = propagate(
        eom, state0, (0.0, t_max),
        events=[peri_ev, apo_ev],
        rtol=rtol, atol=atol,
    )
    peri_alt = (
        _altitude_km(sol.y_events[0][0]) if len(sol.t_events[0]) > 0 else np.nan
    )
    apo_alt = (
        _altitude_km(sol.y_events[1][0]) if len(sol.t_events[1]) > 0 else np.nan
    )
    return peri_alt, apo_alt


def predict_apse_states(
    state0: np.ndarray,
    n_revs: int,
    period_s: float,
    eom: Callable = cr3bp_eom,
    rtol: float = RTOL_ONBOARD,
    atol: float = ATOL_ONBOARD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict the FIRST periapsis and apoapsis 6-states over ``n_revs`` revolutions.

    Same propagation as ``predict_apses`` but returns the full Cartesian states
    (not just altitudes) so the controller can bound the apse POSITION VECTORS —
    MacKenzie et al. 2020 Strategy 3, which bounds |r_apse − r_apse,nominal|.

    Returns
    -------
    peri_state, apo_state : np.ndarray, shape (6,)
        First periapsis and first apoapsis state (km, km/s). An array of NaNs if
        that apse is not reached within the horizon.
    """
    t_max = n_revs * period_s
    peri_ev = make_periapsis_event(terminal=False)
    apo_ev = make_apoapsis_event(terminal=False)
    sol = propagate(
        eom, state0, (0.0, t_max),
        events=[peri_ev, apo_ev],
        rtol=rtol, atol=atol,
    )
    nan6 = np.full(6, np.nan)
    peri_state = sol.y_events[0][0] if len(sol.t_events[0]) > 0 else nan6
    apo_state = sol.y_events[1][0] if len(sol.t_events[1]) > 0 else nan6
    return peri_state, apo_state


def nominal_apse_positions(
    ref_ic: np.ndarray,
    period_s: float,
    eom: Callable = cr3bp_eom,
    rtol: float = RTOL_ONBOARD,
    atol: float = ATOL_ONBOARD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Periapsis and apoapsis POSITION vectors of a nominal (reference) orbit.

    Propagates the uncontrolled reference IC one revolution and records its first
    periapsis/apoapsis positions. These are the r_apse,nominal targets for
    MacKenzie Strategy 3 (apse-position bounding). Barycentre frame, km.

    Returns
    -------
    r_peri_nom, r_apo_nom : np.ndarray, shape (3,)
    """
    peri_state, apo_state = predict_apse_states(
        ref_ic, n_revs=1, period_s=period_s, eom=eom, rtol=rtol, atol=atol
    )
    return peri_state[:3].copy(), apo_state[:3].copy()


def _apse_residual(
    state0: np.ndarray,
    dv: np.ndarray,
    n_revs: int,
    period_s: float,
    eom: Callable,
    peri_target_km: float = PERIAPSIS_ALT_TARGET,
    apo_target_km: float = APOAPSIS_ALT_TARGET,
    mode: str = "altitude",
    r_peri_nom: np.ndarray | None = None,
    r_apo_nom: np.ndarray | None = None,
) -> np.ndarray:
    """
    Residual r(ΔV) used by the Gauss-Newton burn solver.

    Two targeting modes:

    ``mode="altitude"`` (default — our original behaviour, MacKenzie-like
        Strategy 1/2): 2-vector [peri_alt − peri_target, apo_alt − apo_target] (km).
        ``peri_target_km`` / ``apo_target_km`` default to the MacKenzie period-3
        halo bands; pass an orbit's own apse altitudes to hold that orbit.

    ``mode="position"`` (MacKenzie Strategy 3 proper): 6-vector
        [r_peri − r_peri_nom, r_apo − r_apo_nom] (km), bounding the full apse
        POSITION VECTORS against the nominal orbit. ``r_peri_nom`` / ``r_apo_nom``
        (3-vectors, barycentre frame) are required in this mode.

    The burn ΔV (km/s) is added to the velocity of ``state0`` and the apses are
    predicted with the ONBOARD model.
    """
    s = state0.copy()
    s[3:] = s[3:] + dv

    if mode == "position":
        if r_peri_nom is None or r_apo_nom is None:
            raise ValueError("mode='position' requires r_peri_nom and r_apo_nom")
        peri_state, apo_state = predict_apse_states(s, n_revs, period_s, eom=eom)
        return np.concatenate([
            peri_state[:3] - r_peri_nom,
            apo_state[:3] - r_apo_nom,
        ])

    peri_alt, apo_alt = predict_apses(s, n_revs, period_s, eom=eom)
    return np.array([
        peri_alt - peri_target_km,
        apo_alt - apo_target_km,
    ])


def solve_burn(
    state0: np.ndarray,
    period_s: float,
    n_revs: int = 3,
    eom: Callable = cr3bp_eom,
    max_iter: int = 20,
    fd_step: float = 1e-6,
    damp: float = 0.8,
    tol_km: float = TARGET_TOL_KM,
    peri_target_km: float = PERIAPSIS_ALT_TARGET,
    apo_target_km: float = APOAPSIS_ALT_TARGET,
    mode: str = "altitude",
    r_peri_nom: np.ndarray | None = None,
    r_apo_nom: np.ndarray | None = None,
) -> dict:
    """
    Solve for the impulsive ΔV that re-targets the next periapsis/apoapsis apses.

    Minimum-norm Gauss-Newton on a 3-control (ΔV ∈ ℝ³) problem. The residual is
    2-D in ``mode="altitude"`` (peri/apo altitudes) or 6-D in ``mode="position"``
    (peri/apo position vectors, MacKenzie Strategy 3). The Jacobian J = ∂r/∂ΔV
    (m×3) is built by forward finite differences; the pseudo-inverse step
    ΔV ← ΔV − J⁺ r selects the minimum-norm correction, so the solver prefers cheap
    burns and the (under- or over-determined) system stays well posed via pinv.

    Planning uses the ONBOARD model (``eom`` default ``cr3bp_eom``); the truth model is
    NOT used here — that split is enforced by the caller.

    Parameters
    ----------
    state0 : np.ndarray, shape (6,)
        Barycentre-frame state at the control point (pre-burn), km / km/s.
    period_s : float
        Single-revolution period estimate (s).
    n_revs : int
        Multiple-shooting horizon in revolutions (N_m, MacKenzie 2–3).
    eom : Callable
        Onboard planning EOM.
    max_iter : int
        Maximum Gauss-Newton iterations.
    fd_step : float
        Finite-difference step on ΔV components (km/s) for the Jacobian.
    damp : float
        Step damping ∈ (0,1].
    tol_km : float
        Convergence tolerance on the apse residual (km).
    peri_target_km, apo_target_km : float
        Periapsis / apoapsis altitude targets (km), used in ``mode="altitude"``.
        Default to the MacKenzie period-3 halo bands; set to an orbit's own apse
        altitudes to hold that orbit (e.g. the Russell-Lara halo: ~24 / ~1053 km).
    mode : str
        "altitude" (default) or "position" (MacKenzie Strategy 3 proper).
    r_peri_nom, r_apo_nom : np.ndarray (3,), optional
        Nominal periapsis / apoapsis position vectors (km, barycentre frame),
        required when ``mode="position"`` (see ``nominal_apse_positions``).

    Returns
    -------
    dict with keys:
        'dv'         : np.ndarray (3,) — solved ΔV (km/s).
        'dv_mag_ms'  : float — |ΔV| in m/s.
        'converged'  : bool — residual below ``tol_km``.
        'residual_km': float — final ‖r‖ (km).
        'iterations' : int.
    """
    res_kw = dict(mode=mode, r_peri_nom=r_peri_nom, r_apo_nom=r_apo_nom)
    dv = np.zeros(3)
    r = _apse_residual(state0, dv, n_revs, period_s, eom,
                       peri_target_km, apo_target_km, **res_kw)

    iters = 0
    for iters in range(1, max_iter + 1):
        if not np.all(np.isfinite(r)):
            break
        if np.linalg.norm(r) < tol_km:
            break

        # Finite-difference Jacobian J (m×3): columns = ∂r/∂ΔV_i (m = len(r)).
        J = np.zeros((r.size, 3))
        for i in range(3):
            dv_p = dv.copy()
            dv_p[i] += fd_step
            r_p = _apse_residual(state0, dv_p, n_revs, period_s, eom,
                                 peri_target_km, apo_target_km, **res_kw)
            if not np.all(np.isfinite(r_p)):
                J = None
                break
            J[:, i] = (r_p - r) / fd_step
        if J is None:
            break

        # Minimum-norm Gauss-Newton step: ΔV -= J⁺ r  (J⁺ = pinv, min-norm solution).
        step = np.linalg.pinv(J) @ r
        dv = dv - damp * step
        r = _apse_residual(state0, dv, n_revs, period_s, eom,
                           peri_target_km, apo_target_km, **res_kw)

    residual = float(np.linalg.norm(r)) if np.all(np.isfinite(r)) else np.inf
    return {
        "dv": dv,
        "dv_mag_ms": float(np.linalg.norm(dv) * 1.0e3),  # km/s → m/s
        "converged": residual < tol_km,
        "residual_km": residual,
        "iterations": iters,
    }


def run_mpc(
    state0: np.ndarray,
    truth_eom: Callable,
    period_s: float,
    horizon_s: float,
    n_revs: int = 3,
    rtol_truth: float = RTOL_TRUTH,
    atol_truth: float = ATOL_TRUTH,
    verbose: bool = False,
    peri_target_km: float = PERIAPSIS_ALT_TARGET,
    apo_target_km: float = APOAPSIS_ALT_TARGET,
    mode: str = "altitude",
    ref_ic: np.ndarray | None = None,
    max_burns: int = 2000,
) -> dict:
    """
    Run the event-driven MPC stationkeeping loop against a TRUTH model.

    The world propagates under ``truth_eom``; at each descending 600-km altitude
    crossing the controller plans a burn with the onboard model (``solve_burn``) and
    applies the impulsive ΔV. The loop ends when the horizon is reached or the
    spacecraft crashes (periapsis altitude < ``PERIAPSIS_CRASH_ALT`` = 5 km).

    TRUTH/ONBOARD SPLIT: this function is the ONLY place ``truth_eom`` is integrated.
    All burn planning inside ``solve_burn`` uses the onboard CR3BP model.

    Parameters
    ----------
    state0 : np.ndarray, shape (6,)
        Initial barycentre-frame state, km / km/s (typically the period-3 IC).
    truth_eom : Callable
        World dynamics: cr3bp_j2_eom, cr3bp_saturn_enc_j2_eom, or SPICE inertial.
        Passed in so the SAME controller runs at every fidelity rung.
    period_s : float
        Single-revolution period estimate (s).
    horizon_s : float
        Total mission horizon to simulate (s).
    n_revs : int
        Multiple-shooting horizon for the burn solver (N_m).
    rtol_truth, atol_truth : float
        Truth-model integration tolerances.
    verbose : bool
        Print per-burn diagnostics.
    peri_target_km, apo_target_km : float
        Periapsis / apoapsis altitude targets (km), used when ``mode="altitude"``.
        Default to the MacKenzie period-3 halo bands (42.05 / 1055 km). Set to an
        orbit's own apse altitudes to stationkeep a different orbit — e.g. the
        Russell-Lara halo (~24 / ~1053 km) — without editing the module constants.
    mode : str
        "altitude" (default; apse-altitude targeting, MacKenzie Strategy 1/2-like)
        or "position" (MacKenzie Strategy 3 proper: bound the apse POSITION
        vectors against the nominal orbit). ``mode="position"`` requires ``ref_ic``.
    ref_ic : np.ndarray (6,), optional
        Nominal (uncontrolled) orbit IC; its first periapsis/apoapsis positions
        become the Strategy-3 targets. Defaults to ``state0`` when ``mode="position"``
        and ``ref_ic`` is not given.
    max_burns : int
        Hard cap on executed burns (default 2000) — a safety guard so the loop can
        never run unbounded. A normal 30-day run uses ~60 burns; hitting the cap
        returns outcome ``'max_burns'`` and signals something is wrong upstream.

    Returns
    -------
    dict with keys:
        'survived'        : bool — reached the horizon without crashing or escaping.
        'outcome'         : str — 'held' (actively stationkept to horizon), 'idle'
                            (no crash but controller stopped triggering), 'crash'
                            (periapsis < 5 km), 'escape' (left the science orbit),
                            or 'max_burns' (hit the safety burn cap).
        'survival_time_s' : float — time of loss (crash/escape), or ``horizon_s``.
        'controller_active_until_s' : float — last time the controller was triggering.
        'n_burns'         : int — number of burns executed.
        'total_dv_ms'     : float — cumulative ΔV (m/s).
        'burns'           : list[dict] — per-burn records
                            {t_s, alt_km, dv_ms, converged, residual_km}.
        'min_peri_alt_km' : float — smallest periapsis altitude seen (km).
    """
    state = state0.copy()
    t_now = 0.0
    total_dv_ms = 0.0
    burns: list[dict] = []
    min_peri_alt = np.inf

    # Strategy 3: compute the nominal apse position targets once, from the
    # reference orbit (defaults to the initial state). Planning uses the onboard
    # model so the targets live in the same model the burn solver predicts in.
    r_peri_nom = r_apo_nom = None
    if mode == "position":
        ref = state0 if ref_ic is None else ref_ic
        r_peri_nom, r_apo_nom = nominal_apse_positions(ref, period_s, eom=cr3bp_eom)

    while t_now < horizon_s:
        # Safety guard: the loop advances t_now each iteration by construction, but
        # cap the burn count so a pathological no-progress state can never spin
        # forever (a real run holds ~2 burns/period, so 2000 ≈ 1000 periods).
        if len(burns) >= max_burns:
            return {
                "survived": False,
                "outcome": "max_burns",
                "survival_time_s": t_now,
                "controller_active_until_s": t_now,
                "n_burns": len(burns),
                "total_dv_ms": total_dv_ms,
                "burns": burns,
                "min_peri_alt_km": float(min_peri_alt) if np.isfinite(min_peri_alt) else np.nan,
            }

        ctrl_ev = make_altitude_event(CONTROL_ALT_KM, terminal=True)
        crash_ev = make_crash_event(PERIAPSIS_CRASH_ALT)
        peri_ev = make_periapsis_event(terminal=False)
        escape_ev = _make_escape_event()

        sol = propagate(
            truth_eom, state, (0.0, horizon_s - t_now),
            events=[ctrl_ev, crash_ev, peri_ev, escape_ev],
            rtol=rtol_truth, atol=atol_truth,
        )

        # Track the minimum periapsis altitude across this leg.
        if len(sol.t_events[2]) > 0:
            for ys in sol.y_events[2]:
                min_peri_alt = min(min_peri_alt, _altitude_km(ys))

        # Escape check (event index 3) — orbit lost, controller cannot recover.
        if len(sol.t_events[3]) > 0:
            t_esc = t_now + float(sol.t_events[3][0])
            if verbose:
                print(f"  ESCAPE (alt > {ESCAPE_ALT_KM:.0f} km) at "
                      f"t={t_esc/3600:.2f} hr after {len(burns)} burns")
            return {
                "survived": False,
                "outcome": "escape",
                "survival_time_s": t_esc,
                "n_burns": len(burns),
                "total_dv_ms": total_dv_ms,
                "burns": burns,
                "min_peri_alt_km": float(min_peri_alt) if np.isfinite(min_peri_alt) else np.nan,
            }

        # Crash check (event index 1) — terminal failure.
        if len(sol.t_events[1]) > 0:
            t_crash = t_now + float(sol.t_events[1][0])
            min_peri_alt = min(min_peri_alt, PERIAPSIS_CRASH_ALT)
            if verbose:
                print(f"  CRASH at t={t_crash/3600:.2f} hr "
                      f"after {len(burns)} burns")
            return {
                "survived": False,
                "outcome": "crash",
                "survival_time_s": t_crash,
                "n_burns": len(burns),
                "total_dv_ms": total_dv_ms,
                "burns": burns,
                "min_peri_alt_km": float(min_peri_alt),
            }

        # Control trigger (event index 0)?
        if len(sol.t_events[0]) == 0:
            # No 600-km descending crossing before the horizon. Two cases:
            #  - the orbit stayed safely above the shell the whole time → survived;
            #  - it never came back inbound (drifting) → also "survived to horizon"
            #    in the sense of no crash, but flag it as idle so the caller knows the
            #    controller stopped acting. Escape is already caught above.
            break

        t_ctrl = float(sol.t_events[0][0])
        state_ctrl = sol.y_events[0][0].copy()
        t_now += t_ctrl

        burn = solve_burn(state_ctrl, period_s, n_revs=n_revs, eom=cr3bp_eom,
                          peri_target_km=peri_target_km,
                          apo_target_km=apo_target_km,
                          mode=mode, r_peri_nom=r_peri_nom, r_apo_nom=r_apo_nom)
        state_post = state_ctrl.copy()
        state_post[3:] = state_post[3:] + burn["dv"]
        total_dv_ms += burn["dv_mag_ms"]

        # Coast past the control shell before re-arming the (descending) 600-km event,
        # otherwise it re-triggers immediately at the same point and the loop makes no
        # time progress. Coast under TRUTH dynamics to the next periapsis (inbound leg
        # completes), watching for a crash en route. This sets the burn cadence to
        # one burn per periapsis approach (MacKenzie Strategy 3 fires once per descent).
        crash_ev2 = make_crash_event(PERIAPSIS_CRASH_ALT)
        peri_ev2 = make_periapsis_event(terminal=True)
        escape_ev2 = _make_escape_event()
        coast = propagate(
            truth_eom, state_post, (0.0, horizon_s - t_now),
            events=[peri_ev2, crash_ev2, escape_ev2],
            rtol=rtol_truth, atol=atol_truth,
        )
        # Record this burn before any terminal-outcome return.
        burns.append({
            "t_s": t_now, "alt_km": _altitude_km(state_ctrl),
            "dv_ms": burn["dv_mag_ms"], "converged": burn["converged"],
            "residual_km": burn["residual_km"],
        })
        if verbose:
            print(f"  burn {len(burns):2d} @ t={t_now/3600:7.2f} hr  "
                  f"ΔV={burn['dv_mag_ms']:7.3f} m/s  "
                  f"res={burn['residual_km']:.3f} km  "
                  f"{'ok' if burn['converged'] else 'NO-CONV'}")

        if len(coast.t_events[2]) > 0:  # escape during the post-burn coast
            t_esc = t_now + float(coast.t_events[2][0])
            if verbose:
                print(f"  ESCAPE (post-burn coast) at t={t_esc/3600:.2f} hr")
            return {
                "survived": False, "outcome": "escape",
                "survival_time_s": t_esc, "n_burns": len(burns),
                "total_dv_ms": total_dv_ms, "burns": burns,
                "min_peri_alt_km": float(min_peri_alt) if np.isfinite(min_peri_alt) else np.nan,
            }
        if len(coast.t_events[1]) > 0:  # crash during the inbound coast
            t_crash = t_now + float(coast.t_events[1][0])
            min_peri_alt = min(min_peri_alt, PERIAPSIS_CRASH_ALT)
            if verbose:
                print(f"  CRASH (post-burn coast) at t={t_crash/3600:.2f} hr "
                      f"after {len(burns)} burns")
            return {
                "survived": False, "outcome": "crash",
                "survival_time_s": t_crash, "n_burns": len(burns),
                "total_dv_ms": total_dv_ms, "burns": burns,
                "min_peri_alt_km": float(min_peri_alt),
            }
        if len(coast.t_events[0]) > 0:
            state = coast.y_events[0][0].copy()
            min_peri_alt = min(min_peri_alt, _altitude_km(state))
            t_now += float(coast.t_events[0][0])
        else:
            # No periapsis before horizon — advance to the horizon end and finish.
            state = coast.y[:, -1].copy()
            t_now = horizon_s

    # Reached the horizon with no crash and no escape: survived. "idle" flags that the
    # controller stopped triggering (orbit drifted off the 600-km shell) before the
    # horizon — survived in the no-crash sense but no longer actively held.
    idle = t_now < horizon_s
    return {
        "survived": True,
        "outcome": "idle" if idle else "held",
        "survival_time_s": horizon_s,
        "controller_active_until_s": t_now,
        "n_burns": len(burns),
        "total_dv_ms": total_dv_ms,
        "burns": burns,
        "min_peri_alt_km": float(min_peri_alt) if np.isfinite(min_peri_alt) else np.nan,
    }