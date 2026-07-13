"""
Tests for the high-fidelity SARSOP-policy rollout harness (baselines/pomdp_rollout.py).

These are correctness/smoke checks on the policy loader, the belief filter, the
action→burn mapping, and the rollout loop's bookkeeping + terminal conditions —
NOT the feasibility study itself (that lives in scripts/pomdp_rollout_feasibility.py
and is reported in the session log). Short horizons keep them fast.

They also guard the ONE thing the Julia↔Python seam must get right: the Python bin
edges/labels must match the Julia toy exactly.
"""

import numpy as np
import pytest

from baselines.pomdp_rollout import (
    SarsopPolicy,
    action_to_dv,
    run_pomdp_rollout,
    DEFAULT_POLICY_PATH,
)
from src.dynamics.cr3bp import X_ENCELADUS
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.cr3bp_saturn_j2 import cr3bp_saturn_enc_j2_eom
from src.constants import R_ENCELADUS
from src.utils.halo_ic import PERIOD3_IC_ND, PERIOD3_PERIOD_S, _nd_to_phys

IC = _nd_to_phys(PERIOD3_IC_ND)
ONE_REV_S = PERIOD3_PERIOD_S / 3.0

# Expected metric fields every rollout result must carry (compared vs the MPC baseline).
EXPECTED_FIELDS = {
    "survived", "outcome", "survival_time_s", "n_burns", "n_steps",
    "total_dv_ms", "min_peri_alt_km", "steps",
}


@pytest.fixture(scope="module")
def policy() -> SarsopPolicy:
    """Load the exported SARSOP policy once for the module."""
    if not DEFAULT_POLICY_PATH.exists():
        pytest.skip(f"exported policy not found at {DEFAULT_POLICY_PATH} "
                    "(run pomdp-julia/src/export_policy.jl)")
    return SarsopPolicy.load()


# ── Loader + seam integrity ─────────────────────────────────────────────────────
def test_policy_loads_expected_labels(policy):
    """The exported labels match the toy formulation (5 states, 3 actions)."""
    assert policy.states == ["CRASHED", "LOW", "NOMINAL", "HIGH", "ESCAPED"]
    assert policy.actions == ["NO_BURN", "SMALL_BURN", "LARGE_BURN"]
    assert policy.terminal_states == {"CRASHED", "ESCAPED"}
    assert policy.initial_state == "NOMINAL"


def test_bin_edges_match_julia(policy):
    """The Python bin edges must match pomdp-julia/src/dynamics.jl exactly."""
    assert policy.bin_edges == [5.0, 25.0, 60.0, 120.0]


def test_bin_of_boundaries(policy):
    """bin_of uses the half-open [lo, hi) convention of the Julia toy."""
    assert policy.bin_of(2.0) == "CRASHED"
    assert policy.bin_of(5.0) == "LOW"       # edge belongs to the upper bin
    assert policy.bin_of(24.9) == "LOW"
    assert policy.bin_of(25.0) == "NOMINAL"
    assert policy.bin_of(42.5) == "NOMINAL"
    assert policy.bin_of(60.0) == "HIGH"
    assert policy.bin_of(119.9) == "HIGH"
    assert policy.bin_of(120.0) == "ESCAPED"
    assert policy.bin_of(1000.0) == "ESCAPED"


def test_tables_row_stochastic(policy):
    """T[s,a,:] and O[s',:] rows sum to 1."""
    assert np.allclose(policy.T.sum(axis=2), 1.0, atol=1e-6)
    assert np.allclose(policy.O.sum(axis=1), 1.0, atol=1e-6)


# ── Policy query reproduces the Julia greedy policy ──────────────────────────────
def test_greedy_policy_matches_julia(policy):
    """
    Belief concentrated on each state reproduces the greedy actions reported in
    docs/session-log/2026-07-06-julia-pomdp.md §4: LOW→LARGE_BURN,
    NOMINAL→SMALL_BURN, HIGH→LARGE_BURN.
    """
    def act_from(state_name):
        b = np.zeros(policy.n_states)
        b[policy.state_index[state_name]] = 1.0
        return policy.action(b)

    assert act_from("LOW") == "LARGE_BURN"
    assert act_from("NOMINAL") == "SMALL_BURN"
    assert act_from("HIGH") == "LARGE_BURN"


# ── Belief filter ────────────────────────────────────────────────────────────────
def test_belief_update_stays_normalized(policy):
    """The Bayes filter returns a normalized distribution."""
    b = policy.initial_belief()
    assert b.sum() == pytest.approx(1.0)
    b2 = policy.update_belief(b, "SMALL_BURN", "NOMINAL")
    assert b2.sum() == pytest.approx(1.0)
    assert np.all(b2 >= 0.0)


def test_belief_update_shifts_toward_observation(policy):
    """Observing LOW should raise the LOW-belief relative to the prior."""
    b = policy.initial_belief()  # all mass on NOMINAL
    b_low = policy.update_belief(b, "NO_BURN", "LOW")
    assert b_low[policy.state_index["LOW"]] > b[policy.state_index["LOW"]]


# ── Action → burn mapping ────────────────────────────────────────────────────────
def test_no_burn_is_zero(policy):
    """NO_BURN maps to a zero ΔV vector."""
    dv = action_to_dv("NO_BURN", IC, policy)
    assert np.allclose(dv, 0.0)


def test_burn_is_prograde_and_ordered(policy):
    """Burns point along +velocity, and LARGE_BURN exceeds SMALL_BURN in magnitude."""
    # Use a state with nonzero velocity (the period-3 IC).
    dv_small = action_to_dv("SMALL_BURN", IC, policy)
    dv_large = action_to_dv("LARGE_BURN", IC, policy)
    v = IC[3:]
    # Prograde: positive projection on the velocity direction.
    assert float(dv_small @ v) > 0.0
    assert np.linalg.norm(dv_large) > np.linalg.norm(dv_small)


# ── Rollout loop ─────────────────────────────────────────────────────────────────
def test_rollout_records_expected_fields(policy):
    """A short rollout returns all expected metric fields with sane types."""
    rng = np.random.default_rng(0)
    r = run_pomdp_rollout(IC, truth_eom=cr3bp_j2_eom, period_s=ONE_REV_S,
                          horizon_s=1.5 * 86400.0, policy=policy, rng=rng)
    assert EXPECTED_FIELDS.issubset(r.keys())
    assert isinstance(r["survived"], bool)
    assert r["outcome"] in {"held", "idle", "crash", "escape", "max_steps"}
    assert r["total_dv_ms"] >= 0.0
    assert r["n_burns"] <= r["n_steps"]
    # Each step record carries the diagnostic fields (incl. the toy-bin label).
    for step in r["steps"]:
        assert {"t_s", "peri_alt_km", "true_bin", "obs_bin", "action",
                "dv_ms", "eta_eff"} <= step.keys()
        assert step["true_bin"] in policy.states


def test_rollout_reproducible_with_seed(policy):
    """Same seed → identical rollout outcome and ΔV."""
    r1 = run_pomdp_rollout(IC, truth_eom=cr3bp_j2_eom, period_s=ONE_REV_S,
                           horizon_s=1.5 * 86400.0, policy=policy,
                           rng=np.random.default_rng(42))
    r2 = run_pomdp_rollout(IC, truth_eom=cr3bp_j2_eom, period_s=ONE_REV_S,
                           horizon_s=1.5 * 86400.0, policy=policy,
                           rng=np.random.default_rng(42))
    assert r1["outcome"] == r2["outcome"]
    assert r1["total_dv_ms"] == pytest.approx(r2["total_dv_ms"])
    assert r1["survival_time_s"] == pytest.approx(r2["survival_time_s"])


def test_rollout_escape_is_terminal(policy):
    """
    The period-3 orbit escapes under the truth model; the rollout must classify
    that as a terminal 'escape' (or 'crash'), not survive, over a short horizon.
    The escape event is watched continuously, so loss time < horizon.
    """
    rng = np.random.default_rng(1)
    r = run_pomdp_rollout(IC, truth_eom=cr3bp_saturn_enc_j2_eom, period_s=ONE_REV_S,
                          horizon_s=3.0 * 86400.0, policy=policy, rng=rng)
    assert r["outcome"] in {"escape", "crash"}
    assert r["survived"] is False
    assert r["survival_time_s"] < 3.0 * 86400.0


def test_max_steps_guard(policy):
    """The step cap is honored (defensive guard against a no-progress loop)."""
    rng = np.random.default_rng(2)
    r = run_pomdp_rollout(IC, truth_eom=cr3bp_j2_eom, period_s=ONE_REV_S,
                          horizon_s=30.0 * 86400.0, policy=policy, rng=rng,
                          max_steps=1)
    # With max_steps=1 the loop must stop at or before 1 recorded step.
    assert r["n_steps"] <= 1