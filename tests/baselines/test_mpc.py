"""
Tests for the deterministic MPC stationkeeping baseline (baselines/mpc.py).

These exercise the burn solver and the event-driven control loop end-to-end without
crashing. They are correctness/smoke checks, not the feasibility study itself (that
lives in scripts/mpc_feasibility.py and is reported in the session log).
"""

import numpy as np
import pytest

from baselines.mpc import (
    predict_apses,
    solve_burn,
    run_mpc,
    _altitude_km,
    CONTROL_ALT_KM,
    PERIAPSIS_ALT_TARGET,
    APOAPSIS_ALT_TARGET,
)
from src.constants import R_ENCELADUS, PERIAPSIS_CRASH_ALT
from src.dynamics.cr3bp import cr3bp_eom, X_ENCELADUS
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.cr3bp_saturn_j2 import cr3bp_saturn_enc_j2_eom
from src.utils.halo_ic import PERIOD3_IC_ND, PERIOD3_PERIOD_S, _nd_to_phys

# Single-revolution period: the period-3 orbit completes 3 periapsis passes per
# full 3-rev period, so one "rev" between control crossings is ~T/3.
IC = _nd_to_phys(PERIOD3_IC_ND)
ONE_REV_S = PERIOD3_PERIOD_S / 3.0


def test_altitude_km_matches_constants():
    """_altitude_km is distance-from-Enceladus minus the mean radius."""
    # Place a point exactly 600 km above the surface on the +x side of Enceladus.
    state = np.array([X_ENCELADUS + R_ENCELADUS + 600.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert _altitude_km(state) == pytest.approx(600.0, abs=1e-6)


def test_predict_apses_returns_finite_onboard():
    """Predicting apses on the onboard model over a few revs gives finite altitudes."""
    peri, apo = predict_apses(IC, n_revs=3, period_s=ONE_REV_S, eom=cr3bp_eom)
    assert np.isfinite(peri)
    assert np.isfinite(apo)
    # Onboard CR3BP holds the period-3 orbit: periapsis low, apoapsis high.
    assert peri < apo


def test_solve_burn_zero_dv_when_already_on_target():
    """
    A burn solved from a state whose apses are already near the target should return
    a small ΔV (the solver does not invent large corrections out of nothing).
    """
    out = solve_burn(IC, period_s=ONE_REV_S, n_revs=3, eom=cr3bp_eom, max_iter=10)
    # The unperturbed period-3 IC's apses are not exactly the band midpoints, but the
    # min-norm solver must keep the correction physically small (< 50 m/s).
    assert out["dv_mag_ms"] < 50.0
    assert out["dv"].shape == (3,)
    assert np.all(np.isfinite(out["dv"]))


def test_solve_burn_reduces_residual():
    """Gauss-Newton must not increase the apse residual from its zero-ΔV value."""
    from baselines.mpc import _apse_residual
    r0 = np.linalg.norm(_apse_residual(IC, np.zeros(3), 3, ONE_REV_S, cr3bp_eom))
    out = solve_burn(IC, period_s=ONE_REV_S, n_revs=3, eom=cr3bp_eom, max_iter=20)
    assert out["residual_km"] <= r0 + 1e-6


@pytest.mark.parametrize("truth_eom", [cr3bp_j2_eom, cr3bp_saturn_enc_j2_eom])
def test_run_mpc_short_rollout_truth_agnostic(truth_eom):
    """
    The control loop runs against BOTH truth models without raising, returns a
    well-formed result dict, and never reports a positive ΔV with zero burns.
    Short horizon keeps the test fast; this is a smoke check, not feasibility.
    """
    horizon = 2.0 * ONE_REV_S  # ~24 hr — long enough to hit ≥1 control crossing
    result = run_mpc(
        IC, truth_eom=truth_eom,
        period_s=ONE_REV_S, horizon_s=horizon,
        n_revs=2,
    )
    assert set(result) >= {
        "survived", "survival_time_s", "n_burns", "total_dv_ms",
        "burns", "min_peri_alt_km",
    }
    assert result["n_burns"] == len(result["burns"])
    assert result["total_dv_ms"] >= 0.0
    if result["n_burns"] == 0:
        assert result["total_dv_ms"] == 0.0
    # Survival time is bounded by the horizon.
    assert result["survival_time_s"] <= horizon + 1e-6


def test_run_mpc_planning_uses_onboard_only(monkeypatch):
    """
    Guard the truth/onboard split: solve_burn must never be handed the truth EOM by
    run_mpc. We spy on solve_burn's `eom` argument during a short rollout.
    """
    import baselines.mpc as mpc

    seen_eoms = []
    real_solve = mpc.solve_burn

    def spy(state0, period_s, **kwargs):
        seen_eoms.append(kwargs.get("eom", cr3bp_eom))
        return real_solve(state0, period_s, **kwargs)

    monkeypatch.setattr(mpc, "solve_burn", spy)
    mpc.run_mpc(IC, truth_eom=cr3bp_saturn_enc_j2_eom,
                period_s=ONE_REV_S, horizon_s=2.0 * ONE_REV_S, n_revs=2)

    # Every planning call used the onboard CR3BP model, never the truth EOM.
    assert all(e is cr3bp_eom for e in seen_eoms)


def test_run_mpc_classifies_escape_under_saturn_j2():
    """
    Regression lock on the feasibility finding: under Saturn J2 the period-3 orbit is
    destabilized faster than the controller can recover, so a multi-day rollout must
    terminate with outcome='escape' (not a false 'survived'), in well under the horizon.
    """
    horizon = 7.0 * 86400.0
    r = run_mpc(IC, truth_eom=cr3bp_saturn_enc_j2_eom,
                period_s=ONE_REV_S, horizon_s=horizon, n_revs=3)
    assert r["outcome"] == "escape"
    assert r["survived"] is False
    assert r["survival_time_s"] < horizon