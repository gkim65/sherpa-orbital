"""
Tests for the science+safety SARSOP-policy rollout harness (baselines/pomdp_rollout.py).

Correctness/smoke checks on the policy loader, the belief filter, the dev binning, and
the rollout loop's bookkeeping + terminal conditions over the (dev, cov) state space.
NOT the feasibility study itself. Short horizons keep them fast.

Guards the Julia↔Python seam: the Python dev edges / band targets / labels must match
the Julia export exactly.
"""

import numpy as np
import pytest

from baselines.pomdp_rollout import (
    SarsopPolicy,
    run_pomdp_rollout,
    ONE_REV_S,
    DEFAULT_POLICY_PATH,
)
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.cr3bp_saturn_j2 import cr3bp_saturn_enc_j2_eom
from src.utils.halo_ic import PERIOD3_IC_ND, _nd_to_phys

IC = _nd_to_phys(PERIOD3_IC_ND)

EXPECTED_FIELDS = {
    "survived", "outcome", "survival_time_s", "n_burns", "n_steps",
    "total_dv_ms", "min_peri_alt_km", "science_cov", "n_bands", "steps",
}


@pytest.fixture(scope="module")
def policy() -> SarsopPolicy:
    if not DEFAULT_POLICY_PATH.exists():
        pytest.skip(f"exported policy not found at {DEFAULT_POLICY_PATH} "
                    "(run pomdp-julia/src/export_policy.jl)")
    return SarsopPolicy.load()


# ── Loader + seam integrity ─────────────────────────────────────────────────────
def test_policy_loads_expected_labels(policy):
    """Actions + dev-observation labels match the science+safety formulation."""
    assert policy.actions == [
        "OBSERVE", "CORRECT", "EXCURSE_LOW", "EXCURSE_MID", "EXCURSE_HIGH"]
    assert policy.observations == ["OK", "DRIFT", "FAR", "LOST", "CRASHED"]
    assert policy.band_names == ["LOW", "MID", "HIGH"]
    assert policy.initial_state == "OK|cov0"
    assert "LOST|cov0" in policy.terminal_states
    assert "CRASHED|cov0" in policy.terminal_states


def test_dev_edges_match_julia(policy):
    """dev bin edges must match pomdp-julia/src/dynamics.jl DEV_EDGES."""
    assert policy.dev_edges == [15.0, 60.0, 200.0]


def test_dev_bin_boundaries(policy):
    """dev_bin uses the half-open [lo, hi) convention of the Julia toy."""
    assert policy.dev_bin(0.0) == "OK"
    assert policy.dev_bin(14.9) == "OK"
    assert policy.dev_bin(15.0) == "DRIFT"
    assert policy.dev_bin(59.9) == "DRIFT"
    assert policy.dev_bin(60.0) == "FAR"
    assert policy.dev_bin(199.9) == "FAR"
    assert policy.dev_bin(200.0) == "LOST"
    assert policy.dev_bin(np.inf) == "LOST"


def test_tables_row_stochastic(policy):
    """T[s,a,:] and O[s,:] rows sum to 1."""
    assert np.allclose(policy.T.sum(axis=2), 1.0, atol=1e-6)
    assert np.allclose(policy.O.sum(axis=1), 1.0, atol=1e-6)


# ── Policy query reproduces the intended science-vs-safety tradeoff ──────────────
def _act(policy, dev, cov):
    label = SarsopPolicy.make_label(dev, cov)
    b = np.zeros(policy.n_states)
    b[policy.state_index[label]] = 1.0
    return policy.action(b)


def test_policy_safety_first_when_drifting(policy):
    """When the orbit is drifting/far, the policy CORRECTs regardless of coverage."""
    for cov in (0, 3, 7):
        assert _act(policy, "DRIFT", cov) == "CORRECT"
        assert _act(policy, "FAR", cov) == "CORRECT"


def test_policy_excurses_for_science_when_safe(policy):
    """When safe (OK) with science remaining, the policy takes an EXCURSE action;
    when all bands are done it stops excursing (holds)."""
    assert _act(policy, "OK", 0).startswith("EXCURSE_")     # nothing sampled yet
    assert _act(policy, "OK", 7) == "CORRECT"               # all 3 bands done → hold


# ── Belief filter ────────────────────────────────────────────────────────────────
def test_belief_update_stays_normalized(policy):
    b = policy.initial_belief()
    assert b.sum() == pytest.approx(1.0)
    b2 = policy.update_belief(b, "CORRECT", "OK")
    assert b2.sum() == pytest.approx(1.0)
    assert np.all(b2 >= 0.0)


# ── Rollout loop ─────────────────────────────────────────────────────────────────
def test_rollout_records_expected_fields(policy):
    rng = np.random.default_rng(0)
    r = run_pomdp_rollout(IC, truth_eom=cr3bp_j2_eom, period_s=ONE_REV_S,
                          horizon_s=2.0 * 86400.0, policy=policy, ref_ic=IC, rng=rng)
    assert EXPECTED_FIELDS.issubset(r.keys())
    assert isinstance(r["survived"], bool)
    assert r["outcome"] in {"held", "idle", "crash", "escape", "max_steps"}
    assert r["total_dv_ms"] >= 0.0
    assert r["n_burns"] <= r["n_steps"]
    assert 0 <= r["n_bands"] <= 3
    assert r["n_bands"] == bin(r["science_cov"]).count("1")


def test_rollout_reproducible_with_seed(policy):
    kw = dict(truth_eom=cr3bp_j2_eom, period_s=ONE_REV_S,
              horizon_s=2.0 * 86400.0, policy=policy, ref_ic=IC)
    r1 = run_pomdp_rollout(IC, rng=np.random.default_rng(42), **kw)
    r2 = run_pomdp_rollout(IC, rng=np.random.default_rng(42), **kw)
    assert r1["outcome"] == r2["outcome"]
    assert r1["total_dv_ms"] == pytest.approx(r2["total_dv_ms"])
    assert r1["science_cov"] == r2["science_cov"]


def test_rollout_gathers_science_on_encj2(policy):
    """On the holdable EncJ2 rung the policy should sample bands (science > 0) within a
    few days — the whole point of the science+safety design."""
    rng = np.random.default_rng(0)
    r = run_pomdp_rollout(IC, truth_eom=cr3bp_j2_eom, period_s=ONE_REV_S,
                          horizon_s=4.0 * 86400.0, policy=policy, ref_ic=IC, rng=rng)
    assert r["n_bands"] >= 1


def test_max_steps_guard(policy):
    rng = np.random.default_rng(2)
    r = run_pomdp_rollout(IC, truth_eom=cr3bp_j2_eom, period_s=ONE_REV_S,
                          horizon_s=30.0 * 86400.0, policy=policy, ref_ic=IC, rng=rng,
                          max_steps=1)
    assert r["n_steps"] <= 1
