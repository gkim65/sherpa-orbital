"""
Thruster model: deterministic ΔV application and noisy burn execution.

The Enceladus Orbilander uses Aerojet Rocketdyne MR-106E 22-N monopropellant
thrusters for stationkeeping burns (MacKenzie et al. 2020, Exhibit 3-11,
"Number of thrusters (specific impulse, Isp): ... 8x - 22N (220 s)"). The
associated specific impulse and spacecraft wet mass are recorded as cited
constants in `src/constants.py` (ISP_MR106E, M_SPACECRAFT_WET) and are used
here only for documentation / future fuel-mass bookkeeping — this module
itself is ΔV-only (matching `baselines/mpc.py`'s existing ΔV-only convention).

This module is deliberately upstream of the dynamics: it only maps a
commanded ΔV vector to an applied ΔV vector. It never calls the CR3BP
integrators and must not be used to merge the truth/onboard models.

Burn execution errors (thruster throughput variability, valve response,
attitude control coupling) are modeled here as a single scalar efficiency
factor η_eff, per the Phase 3 spec in `docs/todo.md`:
    η_eff ~ Uniform(0.8, 1.0)
    ΔV_applied = η_eff * ΔV_commanded
This is a minimal stand-in for burn execution error, not a full thruster
degradation model (a random-walk degradation model is explicitly deferred
to a later phase per `docs/todo.md` Phase 2).
"""

from __future__ import annotations

import numpy as np

# Uniform burn-efficiency bounds (Phase 3 spec, docs/todo.md: "sample halo IC
# + eta_eff ~ Uniform(0.8, 1.0)").
ETA_EFF_MIN: float = 0.8
ETA_EFF_MAX: float = 1.0


def apply_dv(dv_commanded: np.ndarray) -> np.ndarray:
    """
    Deterministic ΔV application: the applied ΔV exactly equals the commanded ΔV.

    Args:
        dv_commanded: commanded delta-V vector, km/s, shape (3,).

    Returns:
        Applied delta-V vector, km/s, shape (3,) (identical to the input).
    """
    dv_commanded = np.asarray(dv_commanded, dtype=float)
    return dv_commanded.copy()


def sample_eta_eff(
    rng: np.random.Generator | None = None,
    size: int | None = None,
) -> float | np.ndarray:
    """
    Draw one or more burn-efficiency samples from Uniform(ETA_EFF_MIN, ETA_EFF_MAX).

    Args:
        rng: numpy random Generator for reproducibility. If None, a fresh
            default Generator (unseeded) is created.
        size: number of samples to draw. If None, a single scalar float is
            returned; otherwise an array of shape (size,) is returned.

    Returns:
        Dimensionless burn efficiency (efficiency), in [ETA_EFF_MIN, ETA_EFF_MAX].
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(ETA_EFF_MIN, ETA_EFF_MAX, size=size)


def apply_dv_noisy(
    dv_commanded: np.ndarray,
    rng: np.random.Generator | None = None,
    eta_eff: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Noisy-execution ΔV application: applied ΔV = commanded ΔV * eta_eff, with
    eta_eff ~ Uniform(ETA_EFF_MIN, ETA_EFF_MAX) unless explicitly supplied.

    Args:
        dv_commanded: commanded delta-V vector, km/s, shape (3,).
        rng: numpy random Generator for reproducibility. If None, a fresh
            default Generator (unseeded) is created. Ignored if eta_eff is given.
        eta_eff: optional fixed burn efficiency (dimensionless) to use instead
            of drawing a random sample. Useful for deterministic tests.

    Returns:
        Tuple of (applied delta-V vector [km/s, shape (3,)], eta_eff used).
    """
    dv_commanded = np.asarray(dv_commanded, dtype=float)
    if eta_eff is None:
        eta_eff = float(sample_eta_eff(rng=rng))
    return dv_commanded * eta_eff, eta_eff
