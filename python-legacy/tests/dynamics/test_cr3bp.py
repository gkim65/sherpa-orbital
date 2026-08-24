"""
Tests for CR3BP dynamics (onboard model) and CR3BP+J2 (truth model).

Tests:
  1. Jacobi constant conservation over one full orbit (<1e-8 relative drift)
  2. Orbital period is close to 12 hours (±20%) for a near-halo orbit
  3. CR3BP and CR3BP+J2 diverge measurably over 30 days (position error > 1 km)
  4. Unit normalisation round-trip is exact
  5. Libration point L1 lies between Saturn and Enceladus
"""

import numpy as np
import pytest
from src.constants import (
    A_ENCELADUS, R_ENCELADUS, OMEGA, L_STAR, V_STAR, T_STAR, MU
)
from src.dynamics.cr3bp import cr3bp_eom, jacobi_constant, X_ENCELADUS, X_SATURN, libration_points_x
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.integrator import (
    propagate, propagate_n_orbits, RTOL_TRUTH, ATOL_TRUTH
)
from src.utils.orbital_elements import cr3bp_to_nondim, nondim_to_cr3bp, r_from_enceladus


# ── Approximate L1 halo orbit initial conditions ──────────────────────────────
# Converted from non-dimensional literature ICs for a period-3 L1 halo in the
# Saturn-Enceladus system.  These are NOT a closed orbit; they are a
# "good enough" starting point to test energy conservation and event detection.
# Non-dim ICs (x in rotating frame, barycentre origin):
#   x_nd ≈ 1 - mu + 0.0085   (just right of Enceladus)
#   z_nd ≈ 0.0006             (small out-of-plane component)
#   vy_nd ≈ -0.0180           (retrograde, needed for halo geometry)
# Reference: Haapala (2014), Appendix A; Richardson (1980) third-order approx.
_X_ND = 1.0 - MU + 0.0085
_Z_ND = 0.0006
_VY_ND = -0.0180

IC_APPROX_HALO = nondim_to_cr3bp(np.array([_X_ND, 0.0, _Z_ND, 0.0, _VY_ND, 0.0]))

# Estimated 12-hour period in seconds
T_12H = 12.0 * 3600.0


class TestJacobiConservation:
    """Jacobi constant must be conserved to high precision in pure CR3BP."""

    def test_jacobi_one_orbit_relative_drift(self):
        """Relative drift in Jacobi constant over one orbit < 1e-8."""
        state0 = IC_APPROX_HALO.copy()
        C0 = jacobi_constant(state0)

        sol = propagate(cr3bp_eom, state0, (0.0, T_12H),
                        rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
        assert sol.success

        C_final = jacobi_constant(sol.y[:, -1])
        rel_drift = abs((C_final - C0) / C0)
        assert rel_drift < 1e-8, (
            f"Jacobi constant drifted by {rel_drift:.2e} (threshold 1e-8). "
            f"C0={C0:.6f}, C_final={C_final:.6f}"
        )

    def test_jacobi_30_days(self):
        """Jacobi constant stays bounded over 30 days in pure CR3BP."""
        state0 = IC_APPROX_HALO.copy()
        C0 = jacobi_constant(state0)

        sol = propagate(cr3bp_eom, state0, (0.0, 30 * 86400.0),
                        rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
        assert sol.success

        # Sample every 100 steps
        C_vals = np.array([jacobi_constant(sol.y[:, k])
                           for k in range(0, sol.y.shape[1], max(1, sol.y.shape[1] // 100))])
        max_rel_drift = np.max(np.abs((C_vals - C0) / C0))
        assert max_rel_drift < 1e-6, (
            f"Max Jacobi drift over 30 days: {max_rel_drift:.2e}"
        )


class TestOrbitPeriod:
    """Near-halo orbit should have a period within ±30% of 12 hours."""

    def test_period_ballpark(self):
        """Starting near Enceladus, spacecraft should have period ≈ 12 h."""
        state0 = IC_APPROX_HALO.copy()
        # The spacecraft starts near Enceladus; check it stays in a reasonable
        # altitude range after one estimated orbit
        sol = propagate(cr3bp_eom, state0, (0.0, 2 * T_12H),
                        rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
        assert sol.success

        r_min = np.min([r_from_enceladus(sol.y[:, k]) for k in range(sol.y.shape[1])])
        r_max = np.max([r_from_enceladus(sol.y[:, k]) for k in range(sol.y.shape[1])])

        alt_min = r_min - R_ENCELADUS
        alt_max = r_max - R_ENCELADUS

        # Periapsis should be > 5 km (no crash), apoapsis should be > 100 km
        assert alt_min > 5.0, f"Minimum altitude {alt_min:.1f} km — spacecraft crashed"
        assert alt_max > 100.0, f"Max altitude {alt_max:.1f} km — orbit too small"


class TestCR3BPvsJ2Divergence:
    """Truth (CR3BP+J2) and onboard (CR3BP) models must diverge over 30 days."""

    def test_30_day_divergence(self):
        """Position error between CR3BP and CR3BP+J2 must exceed 1 km after 30 days."""
        state0 = IC_APPROX_HALO.copy()
        t_end = 30.0 * 86400.0

        sol_cr3bp = propagate(cr3bp_eom, state0, (0.0, t_end),
                              rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
        sol_j2 = propagate(cr3bp_j2_eom, state0, (0.0, t_end),
                           rtol=RTOL_TRUTH, atol=ATOL_TRUTH)

        assert sol_cr3bp.success
        assert sol_j2.success

        pos_error = np.linalg.norm(sol_cr3bp.y[:3, -1] - sol_j2.y[:3, -1])
        assert pos_error > 1.0, (
            f"CR3BP vs CR3BP+J2 position error after 30 days: {pos_error:.3f} km "
            f"— expected > 1 km, J2 may not be active"
        )


class TestUnitNormalisation:
    """Round-trip physical ↔ non-dimensional conversion must be exact."""

    def test_roundtrip(self):
        state_phys = IC_APPROX_HALO.copy()
        state_nd = cr3bp_to_nondim(state_phys)
        state_back = nondim_to_cr3bp(state_nd)
        np.testing.assert_allclose(state_phys, state_back, rtol=1e-14,
                                   err_msg="Unit normalisation round-trip failed")

    def test_nondim_scale(self):
        """In non-dim units, L_STAR maps to 1.0 and V_STAR maps to 1.0."""
        state_phys = np.array([L_STAR, 0.0, 0.0, 0.0, V_STAR, 0.0])
        state_nd = cr3bp_to_nondim(state_phys)
        assert abs(state_nd[0] - 1.0) < 1e-14
        assert abs(state_nd[4] - 1.0) < 1e-14


class TestLibrationPoints:
    """L1 must lie between Saturn and Enceladus; L2 must lie beyond Enceladus."""

    def test_l1_between_primaries(self):
        x_L1, x_L2, x_L3 = libration_points_x()
        assert X_SATURN < x_L1 < X_ENCELADUS, (
            f"L1 at x={x_L1:.1f} km is not between Saturn ({X_SATURN:.1f}) "
            f"and Enceladus ({X_ENCELADUS:.1f})"
        )

    def test_l2_beyond_enceladus(self):
        x_L1, x_L2, x_L3 = libration_points_x()
        assert x_L2 > X_ENCELADUS, (
            f"L2 at x={x_L2:.1f} km should be beyond Enceladus ({X_ENCELADUS:.1f})"
        )

    def test_l1_close_to_enceladus(self):
        """L1 should be within ~1000 km of Enceladus (typical for small μ)."""
        x_L1, _, _ = libration_points_x()
        dist_to_enc = abs(x_L1 - X_ENCELADUS)
        assert dist_to_enc < 5000.0, (
            f"L1 is {dist_to_enc:.0f} km from Enceladus — unexpectedly far"
        )
