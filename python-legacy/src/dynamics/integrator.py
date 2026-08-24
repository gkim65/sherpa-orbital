"""
RK45 integrator wrapper for CR3BP / CR3BP+J2 propagation.

Provides:
  - propagate(): general-purpose IVP integration with optional event detection
  - propagate_to_apoapsis(): propagate until next apoapsis passage
  - propagate_to_periapsis(): propagate until next periapsis passage
  - propagate_n_orbits(): propagate for a fixed number of orbital periods

Event detection uses the radial-velocity sign-change condition:
  - Apoapsis:  d/dt(r_enc) = 0 with ṙ going from + to -
  - Periapsis: d/dt(r_enc) = 0 with ṙ going from - to +

r_enc is the distance from Enceladus, NOT from the barycentre.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Optional
from src.constants import R_ENCELADUS, OMEGA
from src.dynamics.cr3bp import X_ENCELADUS


# ── Integration tolerances ────────────────────────────────────────────────────
RTOL_TRUTH: float = 1e-10   # truth model (CR3BP+J2)
ATOL_TRUTH: float = 1e-12
RTOL_ONBOARD: float = 1e-8  # onboard model (CR3BP only)
ATOL_ONBOARD: float = 1e-10


def _r_enceladus(state: np.ndarray) -> float:
    """Distance from Enceladus centre (km)."""
    x, y, z = state[0], state[1], state[2]
    return np.sqrt((x - X_ENCELADUS)**2 + y**2 + z**2)


def _rdot_enceladus(state: np.ndarray) -> float:
    """Radial velocity relative to Enceladus (km/s). Positive = moving away."""
    x, y, z, vx, vy, vz = state
    dx = x - X_ENCELADUS
    r = np.sqrt(dx**2 + y**2 + z**2)
    return (dx * vx + y * vy + z * vz) / r


def make_apoapsis_event(terminal: bool = True) -> Callable:
    """
    Build a scipy event function that triggers at apoapsis (ṙ = 0, ṙ going -).

    Parameters
    ----------
    terminal : bool
        If True, integration stops at this event.
    """
    def apoapsis(t: float, state: np.ndarray) -> float:
        return _rdot_enceladus(state)

    apoapsis.terminal = terminal
    apoapsis.direction = -1.0   # only triggers when ṙ crosses zero going negative
    return apoapsis


def make_periapsis_event(terminal: bool = True) -> Callable:
    """
    Build a scipy event function that triggers at periapsis (ṙ = 0, ṙ going +).

    Parameters
    ----------
    terminal : bool
        If True, integration stops at this event.
    """
    def periapsis(t: float, state: np.ndarray) -> float:
        return _rdot_enceladus(state)

    periapsis.terminal = terminal
    periapsis.direction = +1.0  # only triggers when ṙ crosses zero going positive
    return periapsis


def make_altitude_event(altitude_km: float, terminal: bool = False) -> Callable:
    """
    Build a scipy event function that triggers when altitude above Enceladus
    equals altitude_km (crossing downward).

    Used in stationkeeping (Strategy 3 fires at 600-km altitude crossing).

    Parameters
    ----------
    altitude_km : float
        Target altitude above Enceladus surface (km).
    terminal : bool
        If True, integration stops at this event.
    """
    target_r = R_ENCELADUS + altitude_km

    def altitude_cross(t: float, state: np.ndarray) -> float:
        return _r_enceladus(state) - target_r

    altitude_cross.terminal = terminal
    altitude_cross.direction = -1.0  # triggers on descending crossing
    return altitude_cross


def make_crash_event(crash_altitude_km: float = 5.0) -> Callable:
    """
    Build a terminal event that fires when periapsis altitude < crash_altitude_km.

    Parameters
    ----------
    crash_altitude_km : float
        Surface impact threshold altitude (km).
    """
    crash_r = R_ENCELADUS + crash_altitude_km

    def crash(t: float, state: np.ndarray) -> float:
        return _r_enceladus(state) - crash_r

    crash.terminal = True
    crash.direction = -1.0
    return crash


def propagate(
    eom: Callable,
    state0: np.ndarray,
    t_span: tuple[float, float],
    events: Optional[list] = None,
    rtol: float = RTOL_TRUTH,
    atol: float = ATOL_TRUTH,
    dense_output: bool = False,
    max_step: float = np.inf,
) -> object:
    """
    Integrate equations of motion from t_span[0] to t_span[1].

    Parameters
    ----------
    eom : Callable
        f(t, state) -> d(state)/dt
    state0 : np.ndarray, shape (6,)
        Initial state [x, y, z, vx, vy, vz] in km / km/s.
    t_span : tuple[float, float]
        (t0, tf) in seconds.
    events : list, optional
        scipy event functions for event detection.
    rtol, atol : float
        Relative and absolute tolerances for RK45.
    dense_output : bool
        If True, return a continuous solution object (for plotting).
    max_step : float
        Maximum allowed step size (s).

    Returns
    -------
    scipy OdeResult
        .t : times (s)
        .y : states (6, N)
        .t_events, .y_events if events provided
        .success : bool
    """
    return solve_ivp(
        eom,
        t_span,
        state0,
        method="RK45",
        rtol=rtol,
        atol=atol,
        events=events,
        dense_output=dense_output,
        max_step=max_step,
    )


def propagate_to_apoapsis(
    eom: Callable,
    state0: np.ndarray,
    t_max: float,
    rtol: float = RTOL_TRUTH,
    atol: float = ATOL_TRUTH,
) -> tuple[np.ndarray, float]:
    """
    Propagate until the next apoapsis (ṙ_enc = 0, decelerating).

    Parameters
    ----------
    eom : Callable
        Equations of motion.
    state0 : np.ndarray, shape (6,)
        Initial state.
    t_max : float
        Maximum integration time (s) as a safety limit.
    rtol, atol : float
        Integration tolerances.

    Returns
    -------
    state_apo : np.ndarray, shape (6,)
        State at apoapsis.
    t_apo : float
        Time of apoapsis (s).

    Raises
    ------
    RuntimeError
        If apoapsis not found within t_max.
    """
    event = make_apoapsis_event(terminal=True)
    sol = propagate(eom, state0, (0.0, t_max), events=[event], rtol=rtol, atol=atol)

    if len(sol.t_events[0]) == 0:
        raise RuntimeError(
            f"Apoapsis not found within t_max={t_max:.1f} s. "
            f"Final r_enc={_r_enceladus(sol.y[:, -1]):.1f} km"
        )

    return sol.y_events[0][0], sol.t_events[0][0]


def propagate_to_periapsis(
    eom: Callable,
    state0: np.ndarray,
    t_max: float,
    rtol: float = RTOL_TRUTH,
    atol: float = ATOL_TRUTH,
) -> tuple[np.ndarray, float]:
    """
    Propagate until the next periapsis (ṙ_enc = 0, accelerating).

    Parameters
    ----------
    eom : Callable
        Equations of motion.
    state0 : np.ndarray, shape (6,)
        Initial state.
    t_max : float
        Maximum integration time (s).
    rtol, atol : float
        Integration tolerances.

    Returns
    -------
    state_peri : np.ndarray, shape (6,)
        State at periapsis.
    t_peri : float
        Time of periapsis (s).
    """
    event = make_periapsis_event(terminal=True)
    sol = propagate(eom, state0, (0.0, t_max), events=[event], rtol=rtol, atol=atol)

    if len(sol.t_events[0]) == 0:
        raise RuntimeError(
            f"Periapsis not found within t_max={t_max:.1f} s."
        )

    return sol.y_events[0][0], sol.t_events[0][0]


def propagate_n_orbits(
    eom: Callable,
    state0: np.ndarray,
    n_orbits: int,
    orbit_period_s: float,
    rtol: float = RTOL_TRUTH,
    atol: float = ATOL_TRUTH,
    dense_output: bool = True,
) -> object:
    """
    Propagate for n_orbits complete orbital periods.

    Parameters
    ----------
    eom : Callable
        Equations of motion.
    state0 : np.ndarray, shape (6,)
        Initial state.
    n_orbits : int
        Number of orbits to propagate.
    orbit_period_s : float
        Estimated orbital period (s). Used to set t_span.
    rtol, atol : float
        Integration tolerances.
    dense_output : bool
        If True, return continuous solution (enables smooth plotting).

    Returns
    -------
    scipy OdeResult
    """
    t_end = n_orbits * orbit_period_s
    crash_event = make_crash_event()
    return propagate(
        eom,
        state0,
        (0.0, t_end),
        events=[crash_event],
        rtol=rtol,
        atol=atol,
        dense_output=dense_output,
    )
