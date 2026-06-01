"""
Utility functions for orbital element conversions and CR3BP unit normalization.

Provides:
  - cartesian_to_keplerian(): 6D Cartesian → Keplerian elements (km, km/s → km, rad)
  - keplerian_to_cartesian(): Keplerian → Cartesian
  - altitude_from_state(): periapsis / apoapsis altitudes from a CR3BP state
  - cr3bp_to_nondim(): physical state → non-dimensional CR3BP state
  - nondim_to_cr3bp(): non-dimensional → physical state
  - ic_nondim_to_physical(): convert literature ICs (non-dim) to km/km/s

The normalisation uses the Saturn-Enceladus system:
  L* = A_ENCELADUS = 238020 km
  T* = 1/OMEGA           s
  V* = L* * OMEGA      km/s

Reference for normalization: Koon et al. (2011), "Dynamical Systems, the Three-Body
Problem and Space Mission Design", Ch. 2.
"""

import numpy as np
from src.constants import (
    GM_SATURN, GM_ENCELADUS, A_ENCELADUS, R_ENCELADUS,
    MU, OMEGA, L_STAR, T_STAR, V_STAR
)
from src.dynamics.cr3bp import X_SATURN, X_ENCELADUS


# ── CR3BP Unit Normalisation ──────────────────────────────────────────────────

def cr3bp_to_nondim(state_phys: np.ndarray) -> np.ndarray:
    """
    Convert a physical CR3BP state to non-dimensional units.

    Non-dimensional: length in units of L_STAR, time in units of T_STAR,
    velocity in units of V_STAR = L_STAR / T_STAR.

    Parameters
    ----------
    state_phys : np.ndarray, shape (6,)
        [x, y, z, vx, vy, vz] in km and km/s.

    Returns
    -------
    np.ndarray, shape (6,)
        Non-dimensional state. In non-dim units, mu_nd = MU, omega_nd = 1.
    """
    state_nd = state_phys.copy()
    state_nd[:3] /= L_STAR
    state_nd[3:] /= V_STAR
    return state_nd


def nondim_to_cr3bp(state_nd: np.ndarray) -> np.ndarray:
    """
    Convert a non-dimensional CR3BP state to physical units (km, km/s).

    Parameters
    ----------
    state_nd : np.ndarray, shape (6,)
        Non-dimensional state [x, y, z, vx, vy, vz].

    Returns
    -------
    np.ndarray, shape (6,)
        Physical state in km and km/s.
    """
    state_phys = state_nd.copy()
    state_phys[:3] *= L_STAR
    state_phys[3:] *= V_STAR
    return state_phys


def ic_nondim_to_physical(x_nd: float, y_nd: float, z_nd: float,
                           vx_nd: float, vy_nd: float, vz_nd: float) -> np.ndarray:
    """
    Convert literature-reported non-dimensional CR3BP initial conditions to km/km/s.

    In standard non-dim CR3BP notation the barycentre is at the origin,
    Saturn is at x = -mu, Enceladus is at x = (1-mu).

    Parameters
    ----------
    x_nd, y_nd, z_nd : float
        Non-dimensional position components.
    vx_nd, vy_nd, vz_nd : float
        Non-dimensional velocity components.

    Returns
    -------
    np.ndarray, shape (6,)
        Physical state in km and km/s.
    """
    return nondim_to_cr3bp(np.array([x_nd, y_nd, z_nd, vx_nd, vy_nd, vz_nd]))


# ── Distance helpers in the CR3BP frame ──────────────────────────────────────

def r_from_enceladus(state: np.ndarray) -> float:
    """
    Distance from spacecraft to Enceladus centre (km).

    Parameters
    ----------
    state : np.ndarray, shape (6,)
        Physical CR3BP state.

    Returns
    -------
    float
        Distance in km.
    """
    x, y, z = state[0], state[1], state[2]
    return float(np.sqrt((x - X_ENCELADUS)**2 + y**2 + z**2))


def altitude_from_enceladus(state: np.ndarray) -> float:
    """
    Altitude above Enceladus surface (km).

    Parameters
    ----------
    state : np.ndarray, shape (6,)
        Physical CR3BP state.

    Returns
    -------
    float
        Altitude in km.
    """
    return r_from_enceladus(state) - R_ENCELADUS


# ── Keplerian elements (two-body, Enceladus-centred) ─────────────────────────

def cartesian_to_keplerian(state: np.ndarray) -> dict:
    """
    Convert Cartesian state (Enceladus-centred inertial) to Keplerian elements.

    For use in two-body approximation diagnostics. Not valid in the full CR3BP.
    State must be expressed relative to Enceladus (subtract X_ENCELADUS from x).

    Parameters
    ----------
    state : np.ndarray, shape (6,)
        [x, y, z, vx, vy, vz] relative to Enceladus, km and km/s.

    Returns
    -------
    dict with keys:
        a  : float  semi-major axis (km)
        e  : float  eccentricity
        i  : float  inclination (rad)
        raan : float  right ascension of ascending node (rad)
        aop  : float  argument of periapsis (rad)
        ta   : float  true anomaly (rad)
        period : float  orbital period (s)
    """
    r_vec = state[:3]
    v_vec = state[3:]
    GM = GM_ENCELADUS

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    # Eccentricity vector
    e_vec = np.cross(v_vec, h_vec) / GM - r_vec / r
    e = np.linalg.norm(e_vec)

    # Semi-major axis from vis-viva
    energy = v**2 / 2.0 - GM / r
    a = -GM / (2.0 * energy)

    # Inclination
    i = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

    # Node vector
    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec)
    n = np.linalg.norm(n_vec)

    # RAAN
    if n > 1e-10:
        raan = np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0))
        if n_vec[1] < 0:
            raan = 2.0 * np.pi - raan
    else:
        raan = 0.0

    # Argument of periapsis
    if n > 1e-10 and e > 1e-10:
        aop = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0))
        if e_vec[2] < 0:
            aop = 2.0 * np.pi - aop
    else:
        aop = 0.0

    # True anomaly
    if e > 1e-10:
        ta = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0))
        if np.dot(r_vec, v_vec) < 0:
            ta = 2.0 * np.pi - ta
    else:
        ta = 0.0

    period = 2.0 * np.pi * np.sqrt(a**3 / GM) if a > 0 else float("inf")

    return dict(a=a, e=e, i=i, raan=raan, aop=aop, ta=ta, period=period)
