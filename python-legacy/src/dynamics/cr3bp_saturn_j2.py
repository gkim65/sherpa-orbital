"""
CR3BP + Saturn J2 + Enceladus J2 — the high-fidelity TRUTH model (Phase 1.5 Step A).

External review (2026-06-22) identified Saturn's oblateness (J2) as the dominant
non-point-mass perturbation in the Saturn-Enceladus system — larger than
Enceladus' own J2 across most of the science orbit, because Enceladus J2 falls
off as 1/r⁴ from the (tiny) moon while Saturn J2, though referenced to a far
larger body, acts at the comparatively short Enceladus orbital distance.

This module adds the Saturn J2 zonal-harmonic perturbation on top of the existing
CR3BP + Enceladus J2 truth model. The onboard model (cr3bp.py) is NOT touched —
the truth/onboard split is preserved (CLAUDE.md rule).

Saturn J2 perturbation acceleration on the spacecraft (relative to Saturn):

    a_J2 = (3/2) * J2 * GM_sat * R_sat² / r1⁵
           * [ (5*(z/r1)² - 1)*dx_sat,
               (5*(z/r1)² - 1)*dy_sat,
               (5*(z/r1)² - 3)*z        ]

where (dx_sat, dy_sat, z) = position relative to Saturn.

Symmetry-axis assumption
------------------------
The formula takes the rotating-frame z-axis as Saturn's spin/oblateness axis.
This is exact only if Saturn's pole is parallel to the Saturn-Enceladus orbit
normal. Enceladus orbits very nearly in Saturn's equatorial plane (orbital
inclination to the Saturn equator ≈ 0°), so the misalignment is sub-degree and
its effect on a ~1e-6 km/s² perturbation is negligible for the Step A divergence
study. This approximation is LOCAL to the rotating-frame model; the SPICE
inertial model (Phase 1.5 Step B) will use Saturn's true pole from a PCK kernel
and does not inherit it.

Reference: Balmino (1994), eq. 4; Schaub & Junkins (2018) §9.2.
Saturn J2, R_sat: Jacobson et al. (2006), AJ 132:2520.
"""

import numpy as np
from src.constants import GM_SATURN, J2_SATURN, R_SATURN
from src.dynamics.cr3bp import X_SATURN
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom, j2_acceleration


def saturn_j2_acceleration(state: np.ndarray) -> np.ndarray:
    """
    J2 perturbation acceleration from Saturn's oblateness (km/s²).

    The position is taken relative to Saturn (at X_SATURN in the rotating frame),
    with the rotating-frame z-axis as Saturn's symmetry axis (see module docstring).

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

    dx = x - X_SATURN
    dy = y
    r1 = np.sqrt(dx**2 + dy**2 + z**2)

    # Common factor: (3/2) * J2 * GM_sat * R_sat² / r1⁵
    factor = 1.5 * J2_SATURN * GM_SATURN * R_SATURN**2 / r1**5

    z_ratio_sq = (z / r1)**2

    ax = factor * (5.0 * z_ratio_sq - 1.0) * dx
    ay = factor * (5.0 * z_ratio_sq - 1.0) * dy
    az = factor * (5.0 * z_ratio_sq - 3.0) * z

    return np.array([ax, ay, az])


def cr3bp_saturn_enc_j2_eom(t: float, state: np.ndarray) -> np.ndarray:
    """
    CR3BP + Saturn J2 + Enceladus J2 equations of motion (high-fidelity truth model).

    Builds on cr3bp_j2_eom (which already includes the CR3BP base dynamics and the
    Enceladus J2 term) by adding Saturn's J2 perturbation. Use the truth-model
    tolerances rtol=1e-10, atol=1e-12.

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
    dstate = cr3bp_j2_eom(t, state).copy()

    a_sat_j2 = saturn_j2_acceleration(state)
    dstate[3] += a_sat_j2[0]
    dstate[4] += a_sat_j2[1]
    dstate[5] += a_sat_j2[2]

    return dstate


# Re-export the Enceladus J2 acceleration for convenience / divergence studies.
__all__ = ["saturn_j2_acceleration", "cr3bp_saturn_enc_j2_eom", "j2_acceleration"]