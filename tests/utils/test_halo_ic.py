"""
Tests for the halo orbit differential corrector (src/utils/halo_ic.py).

The verified period-3 mission IC (PERIOD3_IC_ND) is used as the primary test fixture.
It is a JPL-catalog fixed point corrected in our unit system — period 36 hr,
periapsis ~31 km, south polar.  MacKenzie et al. 2020 §B.2.3.

Slow tests (full pipeline): run with pytest -m slow.
"""

import numpy as np
import pytest

from src.utils.halo_ic import (
    _propagate_half_period_nd,
    _cr3bp_stm_nd,
    _nd_to_phys,
    differential_corrector,
    characterise_orbit,
    find_halo_ic,
    PERIOD3_IC_ND,
    PERIOD3_PERIOD_S,
)
from src.constants import L_STAR, V_STAR, T_STAR, PERIAPSIS_ALT_MIN, PERIAPSIS_ALT_MAX

# Extract IC components for convenience
_X0 = PERIOD3_IC_ND[0]
_Z0 = PERIOD3_IC_ND[2]
_VY0 = PERIOD3_IC_ND[4]


# ── STM sanity ────────────────────────────────────────────────────────────────

def test_stm_derivative_finite():
    """STM derivative should be all-finite at t=0."""
    aug0 = np.concatenate([PERIOD3_IC_ND, np.eye(6).ravel()])
    daug = _cr3bp_stm_nd(0.0, aug0)
    assert np.all(np.isfinite(daug))


def test_stm_symplecticity():
    """det(Phi) ≈ 1 at the half-period crossing — Hamiltonian flow."""
    _, phi, _ = _propagate_half_period_nd(_X0, _Z0, _VY0, n_crossings=2, t_half_max=8.0)
    det = np.linalg.det(phi)
    assert abs(det - 1.0) < 1e-3, f"det(Phi)={det:.6f}, expected ≈ 1"


# ── n_crossings propagator mechanics ─────────────────────────────────────────

def test_n_crossings_parameter_accepted():
    """
    _propagate_half_period_nd(n_crossings=2) finds the 2nd descending y=0 crossing
    (half of the 36-hr period = 18 hr) and returns correct shapes.
    """
    sf, phi, t_half = _propagate_half_period_nd(_X0, _Z0, _VY0,
                                                 n_crossings=2, t_half_max=8.0)
    # 18 hr / TU ≈ 3.43 nondim
    assert 3.0 < t_half < 4.5, f"Half-period t={t_half:.3f} nondim, expected ~3.43"
    assert phi.shape == (6, 6)
    assert sf.shape == (6,)


# ── Corrector mechanics ───────────────────────────────────────────────────────

def test_corrector_convergence_from_period3_ic():
    """
    Corrector with n_crossings=2 converges in 1 iteration from the JPL-catalog
    IC (which is already a fixed point to 1e-12 in our system).
    """
    result = differential_corrector(
        _X0, _Z0, _VY0,
        n_crossings=2,
        t_half_max_nd=8.0,
        tol=1e-10, max_iter=10, damp=0.5, verbose=False,
    )
    assert result["converged"], \
        f"Corrector did not converge. Residual: {result['residual']:.3e}"
    assert result["residual"] < 1e-10


# ── Verified period-3 mission IC ─────────────────────────────────────────────

def test_period3_ic_closes():
    """
    The hardcoded period-3 IC closes to < 5 km over 36 hr.
    Verifies: periapsis in mission band, period ~36 hr.
    Mission orbit: MacKenzie 2020 §B.2.3 southern L1 halo, south polar.
    """
    ic = _nd_to_phys(PERIOD3_IC_ND)
    info = characterise_orbit(ic, PERIOD3_PERIOD_S, verbose=False)
    assert info["closure_km"] < 5.0, \
        f"Closure = {info['closure_km']:.3f} km"
    assert PERIAPSIS_ALT_MIN <= info["periapsis_alt_km"] <= PERIAPSIS_ALT_MAX, \
        f"Periapsis {info['periapsis_alt_km']:.1f} km outside mission band"
    assert 30.0 <= info["period_hr"] <= 42.0, \
        f"Period {info['period_hr']:.2f} hr outside expected 30-42 hr"


def test_period3_ic_three_periapses():
    """
    Propagating the period-3 IC for 36 hr should produce exactly 3 periapsis
    passes, each in the mission altitude band.
    """
    from src.dynamics.cr3bp import cr3bp_eom, X_ENCELADUS
    from src.dynamics.integrator import propagate
    from src.constants import R_ENCELADUS

    ic = _nd_to_phys(PERIOD3_IC_ND)
    sol = propagate(cr3bp_eom, ic, (0.0, PERIOD3_PERIOD_S), dense_output=True)

    t_dense = np.linspace(0, PERIOD3_PERIOD_S, 50000)
    r = (np.sqrt((sol.sol(t_dense)[0] - X_ENCELADUS)**2
                 + sol.sol(t_dense)[1]**2
                 + sol.sol(t_dense)[2]**2)
         - R_ENCELADUS)
    mins = [i for i in range(1, len(r)-1)
            if r[i] < r[i-1] and r[i] < r[i+1]]

    assert len(mins) == 3, f"Expected 3 periapsis passes, got {len(mins)}"
    for idx in mins:
        assert PERIAPSIS_ALT_MIN <= r[idx] <= PERIAPSIS_ALT_MAX, \
            f"Periapsis at t={t_dense[idx]/3600:.1f} hr: alt={r[idx]:.1f} km out of band"


# ── Full pipeline (slow) ──────────────────────────────────────────────────────

@pytest.mark.slow
def test_find_halo_ic_period3():
    """
    Corrector + characterise starting from the JPL IC (no seed scan needed —
    the IC is already converged to 1e-12).  Verifies the full pipeline works.
    Run with: pytest -m slow tests/utils/test_halo_ic.py
    """
    result = differential_corrector(
        _X0, _Z0, _VY0,
        n_crossings=2,
        t_half_max_nd=8.0,
        tol=1e-10, max_iter=20, damp=0.5, verbose=True,
    )
    assert result["converged"], "Corrector did not converge."
    info = characterise_orbit(result["ic"], result["period_s"], verbose=True)
    assert info["closure_km"] < 5.0
    peri = info["periapsis_alt_km"]
    assert PERIAPSIS_ALT_MIN <= peri <= PERIAPSIS_ALT_MAX, \
        f"Periapsis {peri:.1f} km outside mission band."