"""
Tests for src/constants.py.

Validates that all physical constants are self-consistent and agree with
published values from Cassini mission data and MacKenzie et al. 2020.
"""

import math
import pytest
from src.constants import (
    GM_SATURN, GM_ENCELADUS, A_ENCELADUS, J2_ENCELADUS, R_ENCELADUS,
    THRUST_NOM, MU, OMEGA, L_STAR, T_STAR, V_STAR,
    PERIAPSIS_ALT_MIN, PERIAPSIS_ALT_MAX, PERIAPSIS_CRASH_ALT, SIGMA_NAV_POS,
)


class TestGravitationalParameters:
    def test_gm_saturn_order_of_magnitude(self):
        """GM_Saturn should be ~3.8e7 km³/s² (Jacobson et al. 2006)."""
        assert 3.7e7 < GM_SATURN < 4.0e7

    def test_gm_enceladus_order_of_magnitude(self):
        """GM_Enceladus should be ~7.2 km³/s² (Iess et al. 2014)."""
        assert 7.0 < GM_ENCELADUS < 7.5

    def test_mu_is_small(self):
        """μ = GM_enc / (GM_sat + GM_enc) << 1 for a small moon."""
        assert 1e-7 < MU < 1e-6, f"μ={MU:.2e} out of expected range"

    def test_mu_definition(self):
        """μ must equal GM_enc / (GM_sat + GM_enc) exactly."""
        expected = GM_ENCELADUS / (GM_SATURN + GM_ENCELADUS)
        assert abs(MU - expected) < 1e-20


class TestOrbitalGeometry:
    def test_a_enceladus_km(self):
        """Enceladus semi-major axis ~238,020 km (Cassini)."""
        assert 237_000 < A_ENCELADUS < 239_000

    def test_r_enceladus_km(self):
        """Mean radius of Enceladus is ~252 km."""
        assert 250 < R_ENCELADUS < 255

    def test_j2_enceladus_cassini(self):
        """J2 from Cassini gravity science (Iess et al. 2014): ~5.435e-3."""
        assert 5.0e-3 < J2_ENCELADUS < 6.0e-3


class TestDerivedConstants:
    def test_omega_rad_s(self):
        """OMEGA should equal sqrt(GM_total / a³) in rad/s."""
        expected = math.sqrt((GM_SATURN + GM_ENCELADUS) / A_ENCELADUS**3)
        assert abs(OMEGA - expected) / expected < 1e-12

    def test_omega_period_hours(self):
        """Orbital period 2π/OMEGA should be ≈ 32.9 hours (Enceladus orbit)."""
        period_h = (2 * math.pi / OMEGA) / 3600.0
        assert 32 < period_h < 34, f"Period={period_h:.2f} h (expected ~32.9 h)"

    def test_l_star(self):
        assert L_STAR == A_ENCELADUS

    def test_t_star(self):
        assert abs(T_STAR - 1.0 / OMEGA) < 1e-10

    def test_v_star(self):
        assert abs(V_STAR - A_ENCELADUS * OMEGA) < 1e-10

    def test_v_star_km_s(self):
        """V_STAR should be ~12.6 km/s (orbital speed of Enceladus around Saturn)."""
        assert 12.0 < V_STAR < 13.0, f"V_STAR={V_STAR:.3f} km/s"


class TestMissionParameters:
    def test_thrust_nom(self):
        """MR-106E nominal thrust is 22 N (MacKenzie §3.5)."""
        assert THRUST_NOM == 22.0

    def test_periapsis_bounds(self):
        """Science orbit periapsis range: 19.8–64.3 km (Exhibit 3-14)."""
        assert PERIAPSIS_ALT_MIN == pytest.approx(19.8)
        assert PERIAPSIS_ALT_MAX == pytest.approx(64.3)

    def test_crash_altitude_below_min_periapsis(self):
        """Crash altitude must be below the minimum science periapsis."""
        assert PERIAPSIS_CRASH_ALT < PERIAPSIS_ALT_MIN

    def test_nav_noise(self):
        """Optical nav 1-sigma noise: 2 km (§C.1)."""
        assert SIGMA_NAV_POS == 2.0
