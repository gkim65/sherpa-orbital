"""
CR3BP equations of motion in the Saturn-Enceladus rotating frame.

This is the ONBOARD model — no J2, no solar perturbation.
State vector: [x, y, z, vx, vy, vz] in km and km/s.

Frame origin: Saturn-Enceladus barycentre.
x-axis points from Saturn toward Enceladus.
z-axis is normal to the orbital plane (aligned with angular momentum).

Saturn is located at x = -mu * L_STAR  (left of barycentre)
Enceladus is at      x = (1-mu) * L_STAR  (right of barycentre)

Reference: Szebehely (1967), "Theory of Orbits"; Koon et al. (2011) "Dynamical Systems".
Physical-unit formulation: Scheeres (2012), Ch. 3.
"""

import numpy as np
from src.constants import GM_SATURN, GM_ENCELADUS, A_ENCELADUS, MU, OMEGA, L_STAR


# Positions of the primaries in the rotating frame (km)
X_SATURN: float = -MU * L_STAR          # Saturn is at negative x
X_ENCELADUS: float = (1.0 - MU) * L_STAR  # Enceladus is at positive x


def cr3bp_eom(t: float, state: np.ndarray) -> np.ndarray:
    """
    CR3BP equations of motion (onboard model, no J2).

    Parameters
    ----------
    t : float
        Time (s). Not used explicitly; included for scipy solve_ivp compatibility.
    state : np.ndarray, shape (6,)
        [x, y, z, vx, vy, vz] in km and km/s, barycentre frame.

    Returns
    -------
    np.ndarray, shape (6,)
        [vx, vy, vz, ax, ay, az] in km/s and km/s².
    """
    x, y, z, vx, vy, vz = state

    # Distance from Saturn and Enceladus (km)
    r1 = np.sqrt((x - X_SATURN)**2 + y**2 + z**2)
    r2 = np.sqrt((x - X_ENCELADUS)**2 + y**2 + z**2)

    r1_3 = r1**3
    r2_3 = r2**3

    # Gravitational accelerations
    gx = -GM_SATURN * (x - X_SATURN) / r1_3 - GM_ENCELADUS * (x - X_ENCELADUS) / r2_3
    gy = -GM_SATURN * y / r1_3 - GM_ENCELADUS * y / r2_3
    gz = -GM_SATURN * z / r1_3 - GM_ENCELADUS * z / r2_3

    # Coriolis and centrifugal accelerations (physical-unit rotating frame)
    ax = gx + 2.0 * OMEGA * vy + OMEGA**2 * x
    ay = gy - 2.0 * OMEGA * vx + OMEGA**2 * y
    az = gz

    return np.array([vx, vy, vz, ax, ay, az])


def jacobi_constant(state: np.ndarray) -> float:
    """
    Compute the Jacobi constant (energy integral) of the CR3BP.

    C = 2*Omega(x,y,z) - v²

    where Omega is the effective potential:
        Omega = (omega²/2)*(x²+y²) + GM_S/r1 + GM_E/r2

    Parameters
    ----------
    state : np.ndarray, shape (6,)
        [x, y, z, vx, vy, vz] in km and km/s.

    Returns
    -------
    float
        Jacobi constant (km²/s²). Conserved along a CR3BP trajectory.
    """
    x, y, z, vx, vy, vz = state

    r1 = np.sqrt((x - X_SATURN)**2 + y**2 + z**2)
    r2 = np.sqrt((x - X_ENCELADUS)**2 + y**2 + z**2)

    omega_eff = 0.5 * OMEGA**2 * (x**2 + y**2) + GM_SATURN / r1 + GM_ENCELADUS / r2
    v_sq = vx**2 + vy**2 + vz**2

    return 2.0 * omega_eff - v_sq


def libration_points_x() -> tuple[float, float, float]:
    """
    Approximate x-coordinates of L1, L2, L3 collinear libration points (km).

    Uses the quintic polynomial formulation in the rotating frame.
    Only L1 (between primaries) and L2 (beyond Enceladus) are physically
    relevant for halo orbit design.

    Returns
    -------
    tuple[float, float, float]
        (x_L1, x_L2, x_L3) in km, barycentre frame.
    """
    from numpy.polynomial import polynomial as P

    # Non-dimensional search, then scale back
    mu = MU

    # L1: between primaries, solve 5th-order polynomial (Murray & Dermott 1999)
    # gamma = distance from Enceladus to L1 (non-dim)
    # polynomial: gamma^5 - (3-mu)*gamma^4 + (3-2*mu)*gamma^3 - mu*gamma^2 + 2*mu*gamma - mu = 0
    coeffs_L1 = [-mu, 2*mu, -mu, (3 - 2*mu), -(3 - mu), 1.0]
    roots_L1 = np.roots(coeffs_L1[::-1])
    gamma_L1 = float(np.real(roots_L1[np.isreal(roots_L1) & (np.real(roots_L1) > 0)].min()))
    x_L1 = (1.0 - mu - gamma_L1) * L_STAR

    # L2: beyond Enceladus
    # polynomial: gamma^5 + (3-mu)*gamma^4 + (3-2*mu)*gamma^3 - mu*gamma^2 - 2*mu*gamma - mu = 0
    coeffs_L2 = [-mu, -2*mu, -mu, (3 - 2*mu), (3 - mu), 1.0]
    roots_L2 = np.roots(coeffs_L2[::-1])
    gamma_L2 = float(np.real(roots_L2[np.isreal(roots_L2) & (np.real(roots_L2) > 0)].min()))
    x_L2 = (1.0 - mu + gamma_L2) * L_STAR

    # L3: beyond Saturn (negative x side)
    coeffs_L3 = [7*mu/12, mu, -mu/12, 1.0, -(7/12 - 7*mu/12), -1.0]  # approximate
    # Use simpler Newton iteration for L3
    gamma_L3_guess = 1.0 - 7*mu/12
    x_L3 = -(mu + gamma_L3_guess) * L_STAR

    return x_L1, x_L2, x_L3
