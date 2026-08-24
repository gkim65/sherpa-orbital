"""
Navigation observation model: Gaussian position/altitude noise.

The Enceladus Orbilander performs autonomous optical navigation using limb
localization and landmark tracking (MacKenzie et al. 2020, §C.1, "Autonomous
Optical Navigation"). The steady-state position error reported there is on
the order of 300 m (notional scenario, Exhibit C-8) degrading to 2-3x worse
(~1 km) under the conservative parameter set (Exhibit C-9); the project's
onboard-belief navigation noise is set as the round, slightly-conservative
figure SIGMA_NAV_POS = 2 km, cited to §C.1 in `src/constants.py`.

This module is deliberately upstream of the dynamics: it maps a true position
(or altitude) to a noisy observation and never calls the CR3BP integrators.
"""

from __future__ import annotations

import numpy as np

from src.constants import SIGMA_NAV_POS


def observe_position(
    true_position: np.ndarray,
    sigma_r: float = SIGMA_NAV_POS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Return a noisy position observation: independent isotropic Gaussian noise
    of standard deviation sigma_r added to each Cartesian component.

    Args:
        true_position: true position vector, km, shape (3,) (or (N, 3)).
        sigma_r: 1-sigma position noise per axis, km (default: SIGMA_NAV_POS,
            MacKenzie 2020 §C.1).
        rng: numpy random Generator for reproducibility. If None, a fresh
            default Generator (unseeded) is created.

    Returns:
        Noisy position observation, km, same shape as true_position.
    """
    if rng is None:
        rng = np.random.default_rng()
    true_position = np.asarray(true_position, dtype=float)
    noise = rng.normal(loc=0.0, scale=sigma_r, size=true_position.shape)
    return true_position + noise


def observe_altitude(
    true_altitude_km: float,
    sigma_r: float = SIGMA_NAV_POS,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Return a noisy scalar altitude observation: true altitude plus zero-mean
    Gaussian noise of standard deviation sigma_r.

    Args:
        true_altitude_km: true altitude above the Enceladus surface, km.
        sigma_r: 1-sigma altitude noise, km (default: SIGMA_NAV_POS,
            MacKenzie 2020 §C.1).
        rng: numpy random Generator for reproducibility. If None, a fresh
            default Generator (unseeded) is created.

    Returns:
        Noisy altitude observation, km.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(loc=0.0, scale=sigma_r)
    return float(true_altitude_km) + float(noise)
