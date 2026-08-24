"""
CR3BP + Enceladus J2 perturbation — the TRUTH model.

The J2 term accounts for Enceladus' oblateness as measured by Cassini.
It adds an axisymmetric gravitational perturbation to Enceladus' monopole term.

J2 perturbation acceleration on the spacecraft (in the inertial frame, then
rotated to the rotating CR3BP frame) is:

    a_J2 = (3/2) * J2 * GM_enc * R_enc² / r2⁵
           * [ (5*(z/r2)² - 1)*dx_enc,
               (5*(z/r2)² - 1)*dy_enc,
               (5*(z/r2)² - 3)*z        ]

where (dx_enc, dy_enc, z) = position relative to Enceladus.

Note: The z-component of J2 has a different coefficient (−3 vs −1).
This is the standard zonal harmonic form for an oblate body with
symmetry axis aligned with z (Enceladus spin axis ≈ orbit normal).

Reference: Balmino (1994), eq. 4; Schaub & Junkins (2018) §9.2.
"""

import numpy as np
from src.constants import (
    GM_SATURN, GM_ENCELADUS, J2_ENCELADUS, R_ENCELADUS,
    MU, OMEGA, L_STAR
)
from src.dynamics.cr3bp import X_SATURN, X_ENCELADUS, cr3bp_eom


def j2_acceleration(state: np.ndarray) -> np.ndarray:
    """
    J2 perturbation acceleration from Enceladus' oblateness (km/s²).

    Parameters
    ----------
    state : np.ndarray, shape (6,)
        [x, y, z, vx, vy, vz] in km and km/s, barycentre frame.

    Returns
    -------
    np.ndarray, shape (3,)
        [ax_j2, ay_j2, az_j2] in km/s².
    """
    x, y, z = state[0], state[1], state[2]

    dx = x - X_ENCELADUS
    dy = y
    r2 = np.sqrt(dx**2 + dy**2 + z**2)
    r2_sq = r2**2

    # Common factor: (3/2) * J2 * GM_enc * R_enc² / r2⁵
    factor = 1.5 * J2_ENCELADUS * GM_ENCELADUS * R_ENCELADUS**2 / r2**5

    z_ratio_sq = (z / r2)**2

    ax = factor * (5.0 * z_ratio_sq - 1.0) * dx
    ay = factor * (5.0 * z_ratio_sq - 1.0) * dy
    az = factor * (5.0 * z_ratio_sq - 3.0) * z

    return np.array([ax, ay, az])


def cr3bp_j2_eom(t: float, state: np.ndarray) -> np.ndarray:
    """
    CR3BP + Enceladus J2 equations of motion (truth model).

    Adds the J2 oblateness perturbation to the standard CR3BP dynamics.
    Use rtol=1e-10, atol=1e-12 (tighter tolerances than onboard model).

    Parameters
    ----------
    t : float
        Time (s).
    state : np.ndarray, shape (6,)
        [x, y, z, vx, vy, vz] in km and km/s, barycentre frame.

    Returns
    -------
    np.ndarray, shape (6,)
        [vx, vy, vz, ax, ay, az] in km/s and km/s².
    """
    dstate = cr3bp_eom(t, state).copy()

    # Add J2 perturbation to accelerations
    a_j2 = j2_acceleration(state)
    dstate[3] += a_j2[0]
    dstate[4] += a_j2[1]
    dstate[5] += a_j2[2]

    return dstate
