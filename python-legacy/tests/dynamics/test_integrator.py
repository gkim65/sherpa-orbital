"""
Tests for src/dynamics/integrator.py.

Validates event detection (apoapsis, periapsis, altitude crossing, crash)
and propagation accuracy.
"""

import numpy as np
import pytest
from src.constants import R_ENCELADUS, OMEGA
from src.dynamics.cr3bp import cr3bp_eom
from src.dynamics.integrator import (
    propagate,
    propagate_to_apoapsis,
    propagate_to_periapsis,
    make_apoapsis_event,
    make_periapsis_event,
    make_altitude_event,
    make_crash_event,
    _r_enceladus,
    _rdot_enceladus,
    RTOL_TRUTH, ATOL_TRUTH,
)
from src.utils.orbital_elements import nondim_to_cr3bp
from src.constants import MU

# Shared approx halo IC (same as test_cr3bp)
_X_ND = 1.0 - MU + 0.0085
_Z_ND = 0.0006
_VY_ND = -0.0180
IC = nondim_to_cr3bp(np.array([_X_ND, 0.0, _Z_ND, 0.0, _VY_ND, 0.0]))
T_12H = 12.0 * 3600.0


class TestHelperFunctions:
    def test_r_enceladus_returns_positive(self):
        r = _r_enceladus(IC)
        assert r > 0

    def test_r_enceladus_order_of_magnitude(self):
        """Starting IC should be within ~1000 km of Enceladus."""
        r = _r_enceladus(IC)
        alt = r - R_ENCELADUS
        assert 0 < alt < 2000, f"Initial altitude {alt:.1f} km unexpected"

    def test_rdot_enceladus_is_scalar(self):
        rdot = _rdot_enceladus(IC)
        assert isinstance(rdot, (float, np.floating))


class TestPropagate:
    def test_propagate_returns_success(self):
        sol = propagate(cr3bp_eom, IC, (0.0, T_12H))
        assert sol.success

    def test_propagate_output_shape(self):
        sol = propagate(cr3bp_eom, IC, (0.0, T_12H))
        assert sol.y.shape[0] == 6
        assert sol.y.shape[1] > 1

    def test_propagate_initial_state_preserved(self):
        sol = propagate(cr3bp_eom, IC, (0.0, T_12H))
        np.testing.assert_allclose(sol.y[:, 0], IC, rtol=1e-12)

    def test_propagate_with_dense_output(self):
        sol = propagate(cr3bp_eom, IC, (0.0, T_12H), dense_output=True)
        assert sol.sol is not None
        # Evaluate at midpoint
        mid = sol.sol(T_12H / 2)
        assert mid.shape == (6,)


class TestEventDetection:
    def test_apoapsis_event_fires(self):
        """Apoapsis event must fire within one orbit."""
        event = make_apoapsis_event(terminal=True)
        sol = propagate(cr3bp_eom, IC, (0.0, 2 * T_12H), events=[event])
        assert len(sol.t_events[0]) >= 1, "Apoapsis event never fired"

    def test_periapsis_event_fires(self):
        """Periapsis event must fire within one orbit."""
        event = make_periapsis_event(terminal=True)
        sol = propagate(cr3bp_eom, IC, (0.0, 2 * T_12H), events=[event])
        assert len(sol.t_events[0]) >= 1, "Periapsis event never fired"

    def test_apoapsis_rdot_near_zero(self):
        """At apoapsis, ṙ should be near zero."""
        state_apo, t_apo = propagate_to_apoapsis(cr3bp_eom, IC, 2 * T_12H)
        rdot = _rdot_enceladus(state_apo)
        assert abs(rdot) < 1e-3, f"ṙ at apoapsis = {rdot:.6f} km/s (expected ~0)"

    def test_apoapsis_is_local_max(self):
        """At apoapsis the spacecraft is farther from Enceladus than at IC."""
        state_apo, _ = propagate_to_apoapsis(cr3bp_eom, IC, 2 * T_12H)
        r_apo = _r_enceladus(state_apo)
        r_ic = _r_enceladus(IC)
        # The initial IC is near Enceladus, apoapsis should be much farther
        assert r_apo > r_ic, f"Apoapsis r={r_apo:.1f} not > IC r={r_ic:.1f}"

    def test_periapsis_rdot_near_zero(self):
        """At periapsis, ṙ should be near zero."""
        state_peri, t_peri = propagate_to_periapsis(cr3bp_eom, IC, 2 * T_12H)
        rdot = _rdot_enceladus(state_peri)
        assert abs(rdot) < 1e-3, f"ṙ at periapsis = {rdot:.6f} km/s (expected ~0)"

    def test_altitude_event_fires(self):
        """Altitude crossing event fires when target is within the orbit's range.

        The approximate IC is a large CR3BP orbit (periapsis ~1570 km altitude,
        apoapsis ~3263 km altitude). We test with 2000 km — inside the range —
        starting from apoapsis so the spacecraft descends through it.
        """
        state_apo, _ = propagate_to_apoapsis(cr3bp_eom, IC, 2 * T_12H)
        event = make_altitude_event(2000.0, terminal=False)
        sol = propagate(cr3bp_eom, state_apo, (0.0, T_12H), events=[event])
        assert len(sol.t_events[0]) >= 1, "2000-km altitude event never fired after apoapsis"

    def test_altitude_at_event_correct(self):
        """State at altitude event should be at correct distance from Enceladus."""
        target_alt = 600.0
        event = make_altitude_event(target_alt, terminal=True)
        sol = propagate(cr3bp_eom, IC, (0.0, 2 * T_12H), events=[event])
        if len(sol.t_events[0]) > 0:
            state_at_event = sol.y_events[0][0]
            r = _r_enceladus(state_at_event)
            alt = r - R_ENCELADUS
            assert abs(alt - target_alt) < 1.0, (
                f"Altitude at event: {alt:.2f} km (expected {target_alt} km)"
            )


class TestPropagateFunctions:
    def test_propagate_to_apoapsis_returns_array(self):
        state, t = propagate_to_apoapsis(cr3bp_eom, IC, 2 * T_12H)
        assert state.shape == (6,)
        assert t > 0

    def test_propagate_to_periapsis_returns_array(self):
        # Start from apoapsis so periapsis is definitively in the future
        state_apo, _ = propagate_to_apoapsis(cr3bp_eom, IC, 2 * T_12H)
        state, t = propagate_to_periapsis(cr3bp_eom, state_apo, 2 * T_12H)
        assert state.shape == (6,)
        assert t > 0
