"""Deterministic invariants for the GARCH forecasting experiments.

These check the mathematical properties the paper relies on -- MLE consistency,
the analytic mean-reversion of the forecast, the GARCH advantage under a GARCH
DGP, and proxy-robustness of the QLIKE/MSE losses -- without pinning
machine-specific magnitudes. Run: python -m pytest tests/ -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from garch import (  # noqa: E402
    simulate_garch, fit_garch, long_run_variance, half_life,
    garch_filter, ewma_filter, rolling_filter, garch_multistep,
    mse_loss, qlike_loss, mae_loss,
)
import run_all  # noqa: E402


# --------------------------------------------------------------------------- #
# DGP and closed-form identities
# --------------------------------------------------------------------------- #
def test_long_run_variance_identity():
    assert abs(long_run_variance(0.04, 0.09, 0.90) - 4.0) < 1e-12


def test_half_life_identity():
    # (alpha+beta)^h = 1/2 at h = half_life
    p = 0.99
    h = half_life(p)
    assert abs(p ** h - 0.5) < 1e-9
    assert half_life(0.95) < half_life(0.99)          # more persistent -> longer


def test_simulate_reproducible_and_stationary():
    a = simulate_garch(0.04, 0.09, 0.90, 2000, np.random.default_rng(1))[0]
    b = simulate_garch(0.04, 0.09, 0.90, 2000, np.random.default_rng(1))[0]
    assert np.array_equal(a, b)
    # sample variance is near the long-run variance 4.0 (loose band)
    assert 2.5 < a.var() < 6.0


def test_simulate_rejects_nonstationary():
    import pytest
    with pytest.raises(ValueError):
        simulate_garch(0.04, 0.2, 0.85, 100, np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# Estimation: MLE recovers the truth and is consistent in T
# --------------------------------------------------------------------------- #
def test_fit_recovers_parameters():
    rng = np.random.default_rng(3)
    r, _ = simulate_garch(0.04, 0.09, 0.90, 5000, rng)
    fit = fit_garch(r)
    assert abs(fit["persistence"] - 0.99) < 0.03
    assert abs(fit["alpha"] - 0.09) < 0.05
    assert abs(fit["beta"] - 0.90) < 0.06


def test_rmse_decreases_with_sample_size():
    pr = run_all.param_recovery([500, 4000], M=40, seed=0)
    small, big = pr["rows"][0], pr["rows"][1]
    assert big["persistence_rmse"] < small["persistence_rmse"]
    assert big["beta_rmse"] < small["beta_rmse"]


# --------------------------------------------------------------------------- #
# Forecast filters
# --------------------------------------------------------------------------- #
def test_garch_filter_matches_truth_with_true_params():
    """Filtering with the TRUE parameters reproduces the true conditional
    variance path once the seed transient washes out."""
    rng = np.random.default_rng(4)
    r, s2 = simulate_garch(0.04, 0.09, 0.90, 3000, rng)
    h = garch_filter(r, 0.04, 0.09, 0.90, sigma2_0=s2[0])
    # after burn-in the filtered variance equals the generating variance
    assert np.allclose(h[50:], s2[50:], atol=1e-8)


def test_ewma_and_rolling_are_causal_and_positive():
    rng = np.random.default_rng(5)
    r, _ = simulate_garch(0.04, 0.09, 0.90, 500, rng)
    for h in (ewma_filter(r), rolling_filter(r, 22)):
        assert np.all(h > 0)
        assert h.shape == r.shape


def test_multistep_reverts_to_long_run_variance():
    omega, alpha, beta = 0.04, 0.09, 0.90
    lrv = long_run_variance(omega, alpha, beta)
    # start well above the long-run level; forecast must decay toward it
    f = garch_multistep(3 * lrv, omega, alpha, beta, [1, 10, 100, 1000])
    assert f[0] > f[1] > f[2] > f[3]
    assert abs(f[-1] - lrv) < 0.05 * lrv
    # geometric decay factor matches (alpha+beta)^{h-1}
    p = alpha + beta
    assert abs((f[1] - lrv) - p ** 9 * (3 * lrv - lrv)) < 1e-9


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def test_losses_minimized_at_truth():
    rng = np.random.default_rng(6)
    _, s2 = simulate_garch(0.04, 0.09, 0.90, 2000, rng)
    perfect = s2.copy()
    worse = s2 * 1.5
    assert qlike_loss(s2, perfect) < 1e-12
    assert mse_loss(s2, perfect) < 1e-12
    assert qlike_loss(s2, worse) > qlike_loss(s2, perfect)
    assert mse_loss(s2, worse) > mse_loss(s2, perfect)


def test_qlike_nonnegative_and_scale_free():
    rng = np.random.default_rng(7)
    _, s2 = simulate_garch(0.04, 0.09, 0.90, 1000, rng)
    pred = s2 * 0.7
    assert qlike_loss(s2, pred) >= 0.0
    # QLIKE depends only on the ratio target/pred -> invariant to common scaling
    q1 = qlike_loss(s2, pred)
    q2 = qlike_loss(10 * s2, 10 * pred)
    assert abs(q1 - q2) < 1e-12


# --------------------------------------------------------------------------- #
# Experiment-level invariants
# --------------------------------------------------------------------------- #
def test_garch_wins_qlike_under_garch_dgp():
    fc = run_all.forecast_contest([0.99], T_total=1500, T_train=1000, M=15,
                                  seed=10)
    d = fc["dgps"][0]
    assert d["garch_wins_true"]                       # GARCH minimizes QLIKE
    assert d["qlike_gap"] > 0                          # strictly beats best naive


def test_gap_shrinks_toward_random():
    fc = run_all.forecast_contest([0.90, 0.20], T_total=1500, T_train=1000,
                                  M=20, seed=10)
    high, low = fc["dgps"][0], fc["dgps"][1]
    # the absolute QLIKE advantage of GARCH over the best naive shrinks as the
    # DGP loses persistence (less volatility dynamics to exploit)
    assert 0 < low["qlike_gap"] < high["qlike_gap"]


def test_half_life_grows_toward_igarch():
    ph = run_all.persistence_halflife([0.90, 0.99], T=2000, M=20, seed=30)
    lo, hi = ph["rows"][0], ph["rows"][1]
    assert hi["half_life_true"] > lo["half_life_true"]
    # long-run-variance sensitivity blows up as persistence -> 1
    assert hi["lrv_sensitivity"] > lo["lrv_sensitivity"]


def test_proxy_robust_losses_preserve_ordering():
    fc = run_all.forecast_contest([0.99, 0.40], T_total=1500, T_train=1000,
                                  M=20, seed=10)
    px = run_all.proxy_robustness(fc)
    # robust losses concord at least as well as the non-robust MAE
    assert px["concordance"]["qlike"] >= px["concordance"]["mae"]
    assert px["concordance"]["mse"] >= px["concordance"]["mae"]
    # GARCH is the QLIKE winner under both true variance and the noisy proxy
    assert px["garch_top1_true"]["qlike"] == px["n_dgps"]
    assert px["garch_top1_proxy"]["qlike"] == px["n_dgps"]


def test_determinism():
    a = run_all.param_recovery([800], M=10, seed=1)["rows"][0]
    b = run_all.param_recovery([800], M=10, seed=1)["rows"][0]
    assert a["persistence_mean"] == b["persistence_mean"]
    assert a["beta_rmse"] == b["beta_rmse"]
