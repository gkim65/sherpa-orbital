"""
Tests for src/spacecraft/nav.py.

Validates:
  - Noisy position/altitude observations are unbiased with the correct
    standard deviation (statistical check over N=10,000 samples).
  - Reproducibility via a seeded numpy Generator.
"""

import numpy as np
import pytest

from src.spacecraft.nav import observe_position, observe_altitude
from src.constants import SIGMA_NAV_POS

N_SAMPLES = 10_000


def test_observe_position_default_sigma_matches_constant():
    """Default sigma_r is SIGMA_NAV_POS (MacKenzie 2020 §C.1)."""
    true_pos = np.array([100.0, 0.0, 0.0])
    rng = np.random.default_rng(1)
    samples = np.array([observe_position(true_pos, rng=rng) for _ in range(N_SAMPLES)])
    errors = samples - true_pos
    assert errors.std(axis=0) == pytest.approx(SIGMA_NAV_POS, abs=0.1)


def test_observe_position_unbiased():
    """Mean noisy position observation matches the true position (zero-mean noise)."""
    true_pos = np.array([50.0, -20.0, 300.0])
    rng = np.random.default_rng(2)
    samples = np.array([observe_position(true_pos, rng=rng) for _ in range(N_SAMPLES)])
    np.testing.assert_allclose(samples.mean(axis=0), true_pos, atol=0.1)


def test_observe_position_custom_sigma():
    """A custom sigma_r overrides the default and is reflected in the sample std."""
    true_pos = np.zeros(3)
    sigma_r = 0.5
    rng = np.random.default_rng(3)
    samples = np.array(
        [observe_position(true_pos, sigma_r=sigma_r, rng=rng) for _ in range(N_SAMPLES)]
    )
    assert samples.std(axis=0) == pytest.approx(sigma_r, abs=0.02)


def test_observe_position_reproducible_with_seed():
    """Same seed -> identical noisy observation; different seed -> different."""
    true_pos = np.array([1.0, 2.0, 3.0])
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    rng3 = np.random.default_rng(43)
    obs1 = observe_position(true_pos, rng=rng1)
    obs2 = observe_position(true_pos, rng=rng2)
    obs3 = observe_position(true_pos, rng=rng3)
    np.testing.assert_array_equal(obs1, obs2)
    assert not np.array_equal(obs1, obs3)


def test_observe_altitude_mean_and_std():
    """Noisy altitude observations are unbiased with std == sigma_r."""
    true_alt = 600.0  # km
    rng = np.random.default_rng(4)
    samples = np.array([observe_altitude(true_alt, rng=rng) for _ in range(N_SAMPLES)])
    assert samples.mean() == pytest.approx(true_alt, abs=0.1)
    assert samples.std() == pytest.approx(SIGMA_NAV_POS, abs=0.1)


def test_observe_altitude_reproducible_with_seed():
    """Same seed -> identical noisy altitude; different seed -> different."""
    rng1 = np.random.default_rng(5)
    rng2 = np.random.default_rng(5)
    rng3 = np.random.default_rng(6)
    obs1 = observe_altitude(31.0, rng=rng1)
    obs2 = observe_altitude(31.0, rng=rng2)
    obs3 = observe_altitude(31.0, rng=rng3)
    assert obs1 == obs2
    assert obs1 != obs3
