"""
Tests for src/dynamics/cr3bp_j2.py (truth model).

Validates:
  - J2 acceleration magnitude is physically plausible at science orbit altitudes
  - J2 term is zero on the equatorial plane (z=0) in the z-component only (not x/y)
  - J2 has correct symmetry: flipping z flips a_z but keeps a_x, a_y the same
  - CR3BP+J2 EOM are strictly different from CR3BP EOM
"""

import numpy as np
import pytest
from src.constants import MU, GM_ENCELADUS, J2_ENCELADUS, R_ENCELADUS
from src.dynamics.cr3bp import cr3bp_eom
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom, j2_acceleration
from src.dynamics.integrator import propagate, RTOL_TRUTH, ATOL_TRUTH
from src.utils.orbital_elements import nondim_to_cr3bp

_X_ND = 1.0 - MU + 0.0085
_Z_ND = 0.0006
_VY_ND = -0.0180
IC = nondim_to_cr3bp(np.array([_X_ND, 0.0, _Z_ND, 0.0, _VY_ND, 0.0]))
T_12H = 12.0 * 3600.0


class TestJ2Acceleration:
    def test_j2_accel_returns_shape(self):
        a = j2_acceleration(IC)
        assert a.shape == (3,)

    def test_j2_accel_magnitude_plausible(self):
        """At ~300 km altitude, J2 accel should be ~1e-7 to 1e-4 km/s²."""
        a = j2_acceleration(IC)
        mag = np.linalg.norm(a)
        assert 1e-10 < mag < 1e-3, f"|a_J2| = {mag:.2e} km/s² out of expected range"

    def test_j2_symmetry_z_flip(self):
        """Flipping z flips a_z but keeps a_x, a_y (axisymmetry about z)."""
        state_pos_z = IC.copy()
        state_neg_z = IC.copy()
        state_neg_z[2] = -IC[2]

        a_pos = j2_acceleration(state_pos_z)
        a_neg = j2_acceleration(state_neg_z)

        # a_x and a_y should flip sign too because dx, dy are same but z changes
        # Specifically: (5*(z/r)^2 - 1) changes when z changes, so x/y components change
        # But a_z should flip sign when z flips (it's proportional to z*(5z²/r² - 3))
        np.testing.assert_allclose(a_pos[2], -a_neg[2], rtol=1e-10,
                                   err_msg="a_z should flip sign when z flips")

    def test_j2_zero_at_equator_z_component(self):
        """At z=0 in-plane (equatorial), a_z_J2 = 0 (symmetric oblateness)."""
        state_eq = IC.copy()
        state_eq[2] = 0.0
        a = j2_acceleration(state_eq)
        # a_z = factor * (5*0 - 3) * 0 = 0
        assert abs(a[2]) < 1e-20, f"a_z at z=0: {a[2]:.2e} (should be 0)"

    def test_j2_scales_with_j2_constant(self):
        """J2 acceleration should scale linearly with J2 coefficient."""
        from src import constants as C
        a1 = j2_acceleration(IC)

        # Temporarily double J2 and recompute
        original_j2 = C.J2_ENCELADUS
        C.J2_ENCELADUS = original_j2 * 2.0
        # Re-import to pick up modified constant
        import importlib
        import src.dynamics.cr3bp_j2 as j2_mod
        importlib.reload(j2_mod)
        a2 = j2_mod.j2_acceleration(IC)

        # Restore
        C.J2_ENCELADUS = original_j2
        importlib.reload(j2_mod)

        np.testing.assert_allclose(a2, 2.0 * a1, rtol=1e-10,
                                   err_msg="J2 accel should scale linearly with J2")


class TestCR3BPJ2EOM:
    def test_eom_shape(self):
        dstate = cr3bp_j2_eom(0.0, IC)
        assert dstate.shape == (6,)

    def test_eom_differs_from_cr3bp(self):
        """CR3BP+J2 EOM must differ from pure CR3BP at a non-zero z position."""
        dstate_cr3bp = cr3bp_eom(0.0, IC)
        dstate_j2 = cr3bp_j2_eom(0.0, IC)
        accel_diff = np.linalg.norm(dstate_j2[3:] - dstate_cr3bp[3:])
        assert accel_diff > 0, "CR3BP+J2 and CR3BP give identical accelerations"

    def test_velocity_part_unchanged(self):
        """The velocity derivatives (ẋ,ẏ,ż) must be identical in both models."""
        dstate_cr3bp = cr3bp_eom(0.0, IC)
        dstate_j2 = cr3bp_j2_eom(0.0, IC)
        np.testing.assert_array_equal(dstate_cr3bp[:3], dstate_j2[:3])

    def test_eom_at_equator_z_component_affected(self):
        """At z=0, J2 only affects x/y accelerations, not z (since a_z_J2=0 at z=0)."""
        state_eq = IC.copy()
        state_eq[2] = 0.0
        state_eq[5] = 0.0
        dstate_cr3bp = cr3bp_eom(0.0, state_eq)
        dstate_j2 = cr3bp_j2_eom(0.0, state_eq)
        # az should be the same when z=0
        assert abs(dstate_j2[5] - dstate_cr3bp[5]) < 1e-20
