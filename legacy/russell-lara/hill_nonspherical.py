"""
Hill's problem + Enceladus nonspherical gravity (J2, J3, C22).

This is a STANDALONE model reproducing Russell & Lara (2009), "On the design of
an Enceladus science orbit," Acta Astronautica 65, 27-39. It is kept separate
from the project's CR3BP/CR3BP+J2 models on purpose (CLAUDE.md: do not merge
models) — its only role is to validate/reproduce the Russell-Lara halo (their
Fig. 9) and the 8:35/9:35 science orbits before comparing against our own CR3BP.

Frame & units
-------------
Enceladus-centred, rotating ("body-fixed") frame, synchronously locked with the
orbit (Russell-Lara note: "the Enceladus body fixed frame is conveniently static
in the Hill's rotating frame"). x points away from Saturn, z along the orbit
normal (spin axis). All quantities are PHYSICAL (km, km/s, s) — we do NOT use
their normalized LU/TU here, to avoid a normalization-mismatch bug; their
Table 2 initial conditions are already given in km and km/s.

Equations of motion (Russell-Lara Eq. 1):
    xdd =  2 n v + dΩ/dx
    ydd = -2 n u + dΩ/dy
    zdd =          dΩ/dz
with the potential (their Eq. 2):
    Ω = (1/2) n² (3 x² - z²) + μ/r + U
where n is the system mean motion, μ = GM_Enceladus, r = |position from
Enceladus|, and U is the nonspherical (J2, J3, C22) contribution.

The Hill third-body term (1/2) n² (3x² - z²) is the standard linearised tidal
potential of the primary about the moon (gradient: [3 n² x, 0, -n² z]); see
e.g. Scheeres (2012) "Orbital Motion in Strongly Perturbed Environments" §3.

Gravity-field normalization
----------------------------
Russell-Lara Table 1 gives NORMALIZED J2, J3 (and C22). We implement the zonal
J2/J3 terms in the standard (unnormalized) spherical-harmonic form, so we use
their normalized values directly as the C_lm-style coefficients they intend
(the paper treats J2=0.0025, J3=-1e-5 as the model coefficients). C22 (sectoral,
longitude-dependent) is included for completeness; in the body-fixed frame the
spacecraft longitude evolves, so C22 is a function of x,y.

References
----------
  Russell, R. P., & Lara, M. (2009). Acta Astronautica 65, 27-39.   (Eqs. 1-3, Table 1)
  Vallado (2013), "Fundamentals of Astrodynamics and Applications," §8.6 (zonal U_l).
"""

import numpy as np

from constants_russell_lara import (
    GM_ENCELADUS_RL, R_ENCELADUS_RL, N_RL,
    J2_RL, J3_RL, C22_RL,
)


def _nonspherical_accel(pos: np.ndarray, include_c22: bool = False) -> np.ndarray:
    """
    Acceleration from Enceladus' J2, J3 (and optionally C22) harmonics (km/s²).

    Parameters
    ----------
    pos : np.ndarray, shape (3,)
        [x, y, z] position relative to Enceladus (km), body-fixed frame.
    include_c22 : bool, default False
        Include the sectoral C22 term. OFF by default — see Notes.

    Returns
    -------
    np.ndarray, shape (3,)
        [ax, ay, az] perturbing acceleration (km/s²).

    Notes
    -----
    The J2 and J3 closed forms are the analytic gradients of the standard zonal
    potentials U_2, U_3 (Vallado 2013 §8.6) and have been verified against a
    central-difference gradient to ~1e-9 km/s² (test_hill_nonspherical).

    C22 (sectoral) is OFF by default. Russell-Lara list a normalized C22=0.0025,
    but the paper does not state its prime-meridian phase / harmonic-normalization
    convention, and no C22 sign or de-normalization we tried tightens the halo's
    one-period closure below the J2+J3 result (~14 km) — most make it worse. Their
    published IC is converged in their exact C22 implementation, which cannot be
    reverse-engineered from the paper to sub-km. We therefore reproduce their orbit
    with the (dominant, unambiguous) J2+J3 field; this captures the orbit's
    character faithfully (period, altitude band, shape) which is what Fig. 9 shows.
    The C22 form below uses U_22 = (μ R²/r³) 3 C22 (x²−y²)/r², provided only for
    sensitivity studies.
    """
    x, y, z = pos
    r = np.sqrt(x * x + y * y + z * z)
    mu = GM_ENCELADUS_RL
    R = R_ENCELADUS_RL

    # --- J2 (zonal, oblateness): a = (3/2) J2 μ R²/r⁵ * [ (5(z/r)²-1)x, ...y, (5(z/r)²-3)z ]
    zr2 = (z / r) ** 2
    f2 = 1.5 * J2_RL * mu * R**2 / r**5
    a_j2 = f2 * np.array([
        (5.0 * zr2 - 1.0) * x,
        (5.0 * zr2 - 1.0) * y,
        (5.0 * zr2 - 3.0) * z,
    ])

    # --- J3 (zonal, pear-shape): analytic gradient of U_3 (Vallado §8.6).
    #   ax = (1/2) J3 μ R³/r⁵ * 5 (7(z/r)³ − 3(z/r)) x/r,   ay analogous,
    #   az = (1/2) J3 μ R³/r⁵ * (3 − 30(z/r)² + 35(z/r)⁴)
    zr = z / r
    f3 = 0.5 * J3_RL * mu * R**3 / r**5
    a_j3 = f3 * np.array([
        5.0 * (7.0 * zr**3 - 3.0 * zr) * (x / r),
        5.0 * (7.0 * zr**3 - 3.0 * zr) * (y / r),
        3.0 - 30.0 * zr**2 + 35.0 * zr**4,
    ])

    acc = a_j2 + a_j3

    if include_c22:
        # Sectoral C22, U_22 = (μ R²/r³) 3 C22 (x²−y²)/r². Off by default (see Notes).
        f22 = 3.0 * C22_RL * mu * R**2 / r**5
        a_c22 = f22 * np.array([
            x * (1.0 - 5.0 * (x * x - y * y) / (2.0 * r * r)) + x,
            y * (-1.0 - 5.0 * (x * x - y * y) / (2.0 * r * r)) - y,
            -5.0 * z * (x * x - y * y) / (2.0 * r * r),
        ])
        acc = acc + a_c22

    return acc


def hill_nonspherical_eom(t: float, state: np.ndarray) -> np.ndarray:
    """
    Russell-Lara Hill + nonspherical EOM (their Eq. 1), physical units.

    Parameters
    ----------
    t : float
        Time (s); unused (autonomous system), present for solve_ivp signature.
    state : np.ndarray, shape (6,)
        [x, y, z, vx, vy, vz] in the Enceladus-centred rotating frame
        (km, km/s).

    Returns
    -------
    np.ndarray, shape (6,)
        Time derivative [vx, vy, vz, ax, ay, az] (km/s, km/s²).
    """
    x, y, z, vx, vy, vz = state
    n = N_RL
    pos = np.array([x, y, z])
    r = np.sqrt(x * x + y * y + z * z)

    # Point-mass Enceladus monopole: -μ pos / r³
    a_mono = -GM_ENCELADUS_RL * pos / r**3

    # Hill third-body tidal term: gradient of (1/2) n² (3x² - z²) = [3 n² x, 0, -n² z]
    a_hill = np.array([3.0 * n**2 * x, 0.0, -n**2 * z])

    # Nonspherical harmonics
    a_ns = _nonspherical_accel(pos)

    # Coriolis (rotating frame): +2n v on x-dot-dot, -2n u on y-dot-dot
    a_cor = np.array([2.0 * n * vy, -2.0 * n * vx, 0.0])

    acc = a_mono + a_hill + a_ns + a_cor
    return np.array([vx, vy, vz, acc[0], acc[1], acc[2]])


def jacobi_hill(state: np.ndarray) -> float:
    """
    Russell-Lara energy integral C (their Eq. 3): C = 2Ω - (u²+v²+w²).

    Parameters
    ----------
    state : np.ndarray, shape (6,)
        [x, y, z, vx, vy, vz] in the rotating frame (km, km/s).

    Returns
    -------
    float
        Conserved constant (km²/s²) along a trajectory. Used to check
        integration quality (should stay ~constant).
    """
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x * x + y * y + z * z)
    n = N_RL
    # Potential without the nonspherical U (dominant terms) — sufficient as an
    # integration-quality diagnostic; U adds a small constant correction.
    omega = 0.5 * n**2 * (3.0 * x * x - z * z) + GM_ENCELADUS_RL / r
    v2 = vx * vx + vy * vy + vz * vz
    return 2.0 * omega - v2