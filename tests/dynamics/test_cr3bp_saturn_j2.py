"""
Tests for src/dynamics/cr3bp_saturn_j2.py (high-fidelity truth model, Phase 1.5 Step A).

Validates:
  - Saturn J2 acceleration has the right shape and is physically plausible at
    Enceladus' orbital distance (~1e-6 km/s², per the reviewer-prompted estimate)
  - Saturn J2 is z-axisymmetric: flipping z flips a_z but the in-plane components
    behave consistently with the zonal-harmonic form
  - a_z = 0 on Saturn's equatorial plane (z=0)
  - Saturn J2 scales linearly with the J2 coefficient
  - The combined EOM = CR3BP + Enceladus J2 + Saturn J2 exactly (no double count)
  - Saturn J2 dominates Enceladus J2 across the bulk of the science orbit
    (the central finding of the external review)
"""

import numpy as np
import pytest

from src.constants import MU, GM_SATURN, J2_SATURN, R_SATURN, R_ENCELADUS, L_STAR
from src.dynamics.cr3bp import cr3bp_eom, X_SATURN
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom, j2_acceleration
from src.dynamics.cr3bp_saturn_j2 import (
    saturn_j2_acceleration,
    cr3bp_saturn_enc_j2_eom,
)
from src.utils.orbital_elements import nondim_to_cr3bp

# Same science-orbit-like IC used in the Enceladus J2 tests.
_X_ND = 1.0 - MU + 0.0085
_Z_ND = 0.0006
_VY_ND = -0.0180
IC = nondim_to_cr3bp(np.array([_X_ND, 0.0, _Z_ND, 0.0, _VY_ND, 0.0]))


class TestSaturnJ2Acceleration:
    def test_shape(self):
        a = saturn_j2_acceleration(IC)
        assert a.shape == (3,)

    def test_magnitude_plausible(self):
        """At Enceladus' orbital distance, Saturn J2 accel should be ~1e-6 km/s²."""
        a = saturn_j2_acceleration(IC)
        mag = np.linalg.norm(a)
        assert 1e-8 < mag < 1e-4, f"|a_SatJ2| = {mag:.2e} km/s² out of expected range"

    def test_z_flip_flips_az(self):
        """Flipping z flips a_z (zonal harmonic is odd in z for the z-component)."""
        s_pos = IC.copy()
        s_neg = IC.copy()
        s_neg[2] = -IC[2]
        a_pos = saturn_j2_acceleration(s_pos)
        a_neg = saturn_j2_acceleration(s_neg)
        np.testing.assert_allclose(a_pos[2], -a_neg[2], rtol=1e-10,
                                   err_msg="a_z should flip sign when z flips")

    def test_az_zero_at_equator(self):
        """At z=0 (Saturn equatorial plane), a_z = 0."""
        s_eq = IC.copy()
        s_eq[2] = 0.0
        a = saturn_j2_acceleration(s_eq)
        assert abs(a[2]) < 1e-20, f"a_z at z=0: {a[2]:.2e} (should be 0)"

    def test_scales_linearly_with_j2(self):
        a1 = saturn_j2_acceleration(IC)

        from src import constants as C
        original = C.J2_SATURN
        C.J2_SATURN = original * 2.0
        import importlib
        import src.dynamics.cr3bp_saturn_j2 as sat_mod
        importlib.reload(sat_mod)
        a2 = sat_mod.saturn_j2_acceleration(IC)

        C.J2_SATURN = original
        importlib.reload(sat_mod)

        np.testing.assert_allclose(a2, 2.0 * a1, rtol=1e-10,
                                   err_msg="Saturn J2 accel should scale linearly with J2")


class TestCombinedEOM:
    def test_shape(self):
        dstate = cr3bp_saturn_enc_j2_eom(0.0, IC)
        assert dstate.shape == (6,)

    def test_velocity_part_unchanged(self):
        """The velocity derivatives must match the base CR3BP model exactly."""
        d_base = cr3bp_eom(0.0, IC)
        d_full = cr3bp_saturn_enc_j2_eom(0.0, IC)
        np.testing.assert_array_equal(d_base[:3], d_full[:3])

    def test_is_sum_of_components(self):
        """
        Combined accel = CR3BP base + Enceladus J2 + Saturn J2, with no double
        counting. Verify against the explicit decomposition.
        """
        d_full = cr3bp_saturn_enc_j2_eom(0.0, IC)
        d_base = cr3bp_eom(0.0, IC)
        a_enc = j2_acceleration(IC)
        a_sat = saturn_j2_acceleration(IC)
        expected_accel = d_base[3:] + a_enc + a_sat
        np.testing.assert_allclose(d_full[3:], expected_accel, rtol=1e-12, atol=0.0)

    def test_reduces_to_enc_j2_when_saturn_j2_zero(self):
        """With J2_SATURN=0 the combined EOM must equal the Enceladus-only J2 EOM."""
        from src import constants as C
        original = C.J2_SATURN
        C.J2_SATURN = 0.0
        import importlib
        import src.dynamics.cr3bp_saturn_j2 as sat_mod
        importlib.reload(sat_mod)
        d_full = sat_mod.cr3bp_saturn_enc_j2_eom(0.0, IC)

        C.J2_SATURN = original
        importlib.reload(sat_mod)

        d_enc = cr3bp_j2_eom(0.0, IC)
        np.testing.assert_allclose(d_full, d_enc, rtol=1e-12, atol=0.0)

    def test_differs_from_enc_only(self):
        """With the real Saturn J2, the combined EOM must differ from Enc-only J2."""
        d_full = cr3bp_saturn_enc_j2_eom(0.0, IC)
        d_enc = cr3bp_j2_eom(0.0, IC)
        assert np.linalg.norm(d_full[3:] - d_enc[3:]) > 0.0


class TestSaturnJ2SignPhysical:
    """
    Convention-free regression test for the Saturn J2 SIGN.

    Same physics as the Enceladus J2 sign test: for an oblate body the total
    gravity at fixed radius is stronger at the equator than at the pole. We
    evaluate at Saturn-centred positions (radius ~ Enceladus orbital distance)
    so the geometry matches the real use case.
    """

    def _total_radial_accel(self, at_pole: bool, r_km: float = L_STAR) -> float:
        if at_pole:
            p = np.array([X_SATURN, 0.0, r_km])
            rhat = np.array([0.0, 0.0, 1.0])
        else:
            p = np.array([X_SATURN + r_km, 0.0, 0.0])
            rhat = np.array([1.0, 0.0, 0.0])
        mono = -GM_SATURN / r_km**2 * rhat            # monopole, inward
        a_total = mono + saturn_j2_acceleration(np.array([*p, 0, 0, 0]))
        return float(np.dot(a_total, rhat))           # negative = inward

    def test_equator_more_attractive_than_pole(self):
        a_pole = self._total_radial_accel(at_pole=True)
        a_eq = self._total_radial_accel(at_pole=False)
        assert a_eq < a_pole, (
            f"Oblate Saturn must pull harder at equator: "
            f"a_eq={a_eq:.3e}, a_pole={a_pole:.3e} (Saturn J2 sign error?)"
        )


class TestReviewFinding:
    def test_saturn_j2_dominates_at_apoapsis(self):
        """
        External review's central claim: across the bulk of the science orbit
        (near apoapsis, ~1065 km altitude) Saturn J2 >> Enceladus J2, because the
        Enceladus term collapses as 1/r⁴ from the small moon.
        """
        from src.dynamics.cr3bp import X_ENCELADUS
        # Point at ~apoapsis altitude above Enceladus, off the equator.
        r2 = R_ENCELADUS + 1065.0
        s = np.zeros(6)
        s[0] = X_ENCELADUS + 0.6 * r2  # offset toward +x
        s[2] = 0.6 * r2                # and +z, so off-equator (both J2 terms active)
        a_sat = np.linalg.norm(saturn_j2_acceleration(s))
        a_enc = np.linalg.norm(j2_acceleration(s))
        assert a_sat > 10.0 * a_enc, (
            f"Expected Saturn J2 to dominate near apoapsis: "
            f"|a_sat|={a_sat:.2e}, |a_enc|={a_enc:.2e}"
        )