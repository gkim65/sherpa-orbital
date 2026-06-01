"""
Tests for src/utils/orbital_elements.py.
"""

import numpy as np
import pytest
from src.constants import MU, R_ENCELADUS, L_STAR, V_STAR, GM_ENCELADUS
from src.utils.orbital_elements import (
    cr3bp_to_nondim,
    nondim_to_cr3bp,
    ic_nondim_to_physical,
    r_from_enceladus,
    altitude_from_enceladus,
    cartesian_to_keplerian,
)
from src.dynamics.cr3bp import X_ENCELADUS

_X_ND = 1.0 - MU + 0.0085
_Z_ND = 0.0006
_VY_ND = -0.0180
IC = nondim_to_cr3bp(np.array([_X_ND, 0.0, _Z_ND, 0.0, _VY_ND, 0.0]))


class TestNormalization:
    def test_roundtrip_physical_to_nondim(self):
        nd = cr3bp_to_nondim(IC)
        back = nondim_to_cr3bp(nd)
        np.testing.assert_allclose(back, IC, rtol=1e-14)

    def test_nondim_position_scale(self):
        """x=L_STAR in physical → x=1.0 in non-dim."""
        state = np.array([L_STAR, 0, 0, 0, 0, 0], dtype=float)
        nd = cr3bp_to_nondim(state)
        assert abs(nd[0] - 1.0) < 1e-14

    def test_nondim_velocity_scale(self):
        """vx=V_STAR in physical → vx=1.0 in non-dim."""
        state = np.array([0, 0, 0, V_STAR, 0, 0], dtype=float)
        nd = cr3bp_to_nondim(state)
        assert abs(nd[3] - 1.0) < 1e-14

    def test_ic_nondim_to_physical_matches_nondim_to_cr3bp(self):
        result1 = ic_nondim_to_physical(_X_ND, 0.0, _Z_ND, 0.0, _VY_ND, 0.0)
        result2 = nondim_to_cr3bp(np.array([_X_ND, 0.0, _Z_ND, 0.0, _VY_ND, 0.0]))
        np.testing.assert_array_equal(result1, result2)

    def test_nondim_preserves_zero(self):
        """Zero state maps to zero state."""
        zero = np.zeros(6)
        np.testing.assert_array_equal(cr3bp_to_nondim(zero), zero)
        np.testing.assert_array_equal(nondim_to_cr3bp(zero), zero)


class TestDistanceFunctions:
    def test_r_from_enceladus_positive(self):
        assert r_from_enceladus(IC) > 0

    def test_r_from_enceladus_at_surface(self):
        """State exactly on Enceladus surface should give r = R_ENCELADUS."""
        state_surface = np.array([X_ENCELADUS + R_ENCELADUS, 0, 0, 0, 0, 0])
        r = r_from_enceladus(state_surface)
        assert abs(r - R_ENCELADUS) < 1e-10

    def test_altitude_positive(self):
        """IC altitude above Enceladus surface should be positive."""
        alt = altitude_from_enceladus(IC)
        assert alt > 0

    def test_altitude_zero_at_surface(self):
        state_surface = np.array([X_ENCELADUS + R_ENCELADUS, 0, 0, 0, 0, 0])
        alt = altitude_from_enceladus(state_surface)
        assert abs(alt) < 1e-10

    def test_altitude_consistent_with_r(self):
        r = r_from_enceladus(IC)
        alt = altitude_from_enceladus(IC)
        assert abs(r - alt - R_ENCELADUS) < 1e-10


class TestKeplerianElements:
    def _elliptical_state(self):
        """
        A genuine two-body elliptical orbit around Enceladus.

        Circular orbit at 500 km altitude: v_circ = sqrt(GM_enc / r).
        The CR3BP halo IC is NOT two-body elliptical (the rotating frame
        velocity makes it hyperbolic relative to Enceladus alone), so we
        use a purpose-built circular orbit here.
        """
        r = R_ENCELADUS + 500.0  # km from Enceladus centre
        v_circ = np.sqrt(GM_ENCELADUS / r)
        # State: position along x-axis, velocity along y-axis (prograde circular)
        return np.array([r, 0.0, 0.0, 0.0, v_circ, 0.0])

    def test_keplerian_returns_dict(self):
        elem = cartesian_to_keplerian(self._elliptical_state())
        for key in ("a", "e", "i", "raan", "aop", "ta", "period"):
            assert key in elem

    def test_semi_major_axis_positive(self):
        elem = cartesian_to_keplerian(self._elliptical_state())
        assert elem["a"] > 0

    def test_circular_orbit_eccentricity_near_zero(self):
        """Circular orbit should have e ≈ 0."""
        elem = cartesian_to_keplerian(self._elliptical_state())
        assert elem["e"] < 1e-10, f"Circular orbit eccentricity = {elem['e']:.2e}"

    def test_eccentricity_bounded(self):
        elem = cartesian_to_keplerian(self._elliptical_state())
        assert 0 <= elem["e"] < 1.0

    def test_inclination_in_range(self):
        elem = cartesian_to_keplerian(self._elliptical_state())
        assert 0 <= elem["i"] <= np.pi

    def test_circular_orbit_period(self):
        """500-km circular orbit period: T = 2π sqrt(r³/GM_enc)."""
        r = R_ENCELADUS + 500.0
        expected_period = 2 * np.pi * np.sqrt(r**3 / GM_ENCELADUS)
        elem = cartesian_to_keplerian(self._elliptical_state())
        assert abs(elem["period"] - expected_period) / expected_period < 1e-10

    def test_period_order_of_magnitude(self):
        """500-km circular orbit period should be a few hours."""
        elem = cartesian_to_keplerian(self._elliptical_state())
        period_h = elem["period"] / 3600.0
        assert 1.0 < period_h < 50.0, f"Period {period_h:.1f} h out of range"

    def test_keplerian_hyperbolic_returns_negative_a(self):
        """cartesian_to_keplerian correctly returns a<0 for hyperbolic orbit."""
        r = R_ENCELADUS + 500.0
        v_escape = np.sqrt(2 * GM_ENCELADUS / r) * 1.5  # 1.5x escape velocity
        state_hyp = np.array([r, 0.0, 0.0, 0.0, v_escape, 0.0])
        elem = cartesian_to_keplerian(state_hyp)
        assert elem["a"] < 0, "Hyperbolic orbit should have a < 0"
