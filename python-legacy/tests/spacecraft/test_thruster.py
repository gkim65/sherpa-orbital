"""
Tests for src/spacecraft/thruster.py.

Validates:
  - Deterministic ΔV application is an identity map.
  - Noisy-execution eta_eff samples are bounded in [0.8, 1.0] with the correct
    mean/spread (statistical check over N=10,000 samples).
  - Noisy-execution applied ΔV has the correct mean/bounds relative to commanded.
"""

import numpy as np
import pytest

from src.spacecraft.thruster import (
    apply_dv,
    apply_dv_noisy,
    sample_eta_eff,
    ETA_EFF_MIN,
    ETA_EFF_MAX,
)

N_SAMPLES = 10_000


def test_apply_dv_is_identity():
    """Deterministic ΔV application returns exactly the commanded ΔV."""
    dv = np.array([0.001, -0.002, 0.0005])
    out = apply_dv(dv)
    np.testing.assert_allclose(out, dv)
    # Must return a copy, not the same array object.
    assert out is not dv


def test_sample_eta_eff_bounds():
    """All eta_eff samples lie within [ETA_EFF_MIN, ETA_EFF_MAX]."""
    rng = np.random.default_rng(42)
    samples = sample_eta_eff(rng=rng, size=N_SAMPLES)
    assert samples.shape == (N_SAMPLES,)
    assert np.all(samples >= ETA_EFF_MIN)
    assert np.all(samples <= ETA_EFF_MAX)


def test_sample_eta_eff_mean_and_std():
    """Uniform(0.8, 1.0) has mean 0.9 and std (b-a)/sqrt(12) ~= 0.05774."""
    rng = np.random.default_rng(123)
    samples = sample_eta_eff(rng=rng, size=N_SAMPLES)
    expected_mean = (ETA_EFF_MIN + ETA_EFF_MAX) / 2.0
    expected_std = (ETA_EFF_MAX - ETA_EFF_MIN) / np.sqrt(12.0)
    assert samples.mean() == pytest.approx(expected_mean, abs=0.01)
    assert samples.std() == pytest.approx(expected_std, abs=0.01)


def test_sample_eta_eff_reproducible_with_seed():
    """Same seed -> same samples; different seeds -> different samples."""
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    rng3 = np.random.default_rng(8)
    s1 = sample_eta_eff(rng=rng1, size=100)
    s2 = sample_eta_eff(rng=rng2, size=100)
    s3 = sample_eta_eff(rng=rng3, size=100)
    np.testing.assert_array_equal(s1, s2)
    assert not np.array_equal(s1, s3)


def test_apply_dv_noisy_scales_commanded_dv():
    """Applied ΔV = commanded ΔV * eta_eff for a fixed eta_eff."""
    dv = np.array([0.01, 0.0, -0.005])
    applied, eta = apply_dv_noisy(dv, eta_eff=0.9)
    assert eta == pytest.approx(0.9)
    np.testing.assert_allclose(applied, dv * 0.9)


def test_apply_dv_noisy_bounds_and_magnitude_statistics():
    """
    Over N=10,000 samples, the applied ΔV magnitude ratio (applied/commanded)
    matches Uniform(0.8, 1.0) bounds and mean, and never exceeds the commanded
    magnitude (efficiency <= 1.0).
    """
    dv = np.array([0.02, -0.01, 0.005])
    dv_mag = np.linalg.norm(dv)
    rng = np.random.default_rng(99)

    ratios = np.empty(N_SAMPLES)
    for i in range(N_SAMPLES):
        applied, eta = apply_dv_noisy(dv, rng=rng)
        ratios[i] = np.linalg.norm(applied) / dv_mag
        assert eta >= ETA_EFF_MIN and eta <= ETA_EFF_MAX

    assert ratios.max() <= ETA_EFF_MAX + 1e-12
    assert ratios.min() >= ETA_EFF_MIN - 1e-12
    assert ratios.mean() == pytest.approx(0.9, abs=0.01)


def test_apply_dv_noisy_zero_vector_stays_zero():
    """A zero commanded ΔV always yields a zero applied ΔV regardless of eta_eff."""
    dv = np.zeros(3)
    applied, eta = apply_dv_noisy(dv, eta_eff=0.85)
    np.testing.assert_allclose(applied, np.zeros(3))
