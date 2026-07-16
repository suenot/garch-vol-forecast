"""GARCH(1,1) simulation, estimation, forecasting, and forecast-loss metrics.

Everything here operates on a synthetic, seeded GARCH(1,1) data-generating
process, so the TRUE conditional variance path is known by construction and can
be used as ground truth. All Sharpe-free; the object of study is the conditional
variance sigma_t^2 and how well several forecasters recover it.

Model (Bollerslev 1986), zero conditional mean:
    r_t   = sigma_t z_t,     z_t ~ iid N(0, 1)
    sigma_t^2 = omega + alpha r_{t-1}^2 + beta sigma_{t-1}^2

Estimation uses the `arch` library (Sheppard) by Gaussian maximum likelihood.
Forecast losses are the two proxy-robust losses of Patton (2011): MSE and
QLIKE. A non-robust loss (MAE) is included as a deliberate contrast.

References (see paper/FORMULAS.md for the exact equations and provenance):
  Engle (1982), ARCH; Bollerslev (1986), GARCH; Patton (2011), robust loss
  functions under an imperfect volatility proxy; Sheppard, the `arch` library.
"""
import warnings

import numpy as np
from arch import arch_model

RISKMETRICS_LAMBDA = 0.94   # RiskMetrics/EWMA decay


# --------------------------------------------------------------------------- #
# Data-generating process (known ground truth)
# --------------------------------------------------------------------------- #
def simulate_garch(omega, alpha, beta, T, rng, burn=1000):
    """Simulate a GARCH(1,1) path of length T with Gaussian innovations.

    Returns (r, sigma2), where r[t] is the return and sigma2[t] is the TRUE
    conditional variance Var(r_t | F_{t-1}) used to generate it -- the ground
    truth every forecaster is scored against. A burn-in of `burn` observations
    is discarded so the path starts from the stationary distribution. Requires
    alpha + beta < 1 (covariance stationarity).
    """
    if not (omega > 0 and alpha >= 0 and beta >= 0):
        raise ValueError("need omega>0, alpha>=0, beta>=0")
    if alpha + beta >= 1.0:
        raise ValueError("need alpha+beta<1 for a stationary DGP")
    n = T + burn
    r = np.empty(n)
    s2 = np.empty(n)
    s2[0] = omega / (1.0 - alpha - beta)          # long-run variance seed
    z = rng.standard_normal(n)
    r[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, n):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    return r[burn:], s2[burn:]


def long_run_variance(omega, alpha, beta):
    """Unconditional variance omega / (1 - alpha - beta)."""
    return omega / (1.0 - alpha - beta)


def half_life(persistence):
    """Volatility half-life ln(0.5) / ln(alpha+beta) in observations."""
    p = float(persistence)
    if p <= 0.0 or p >= 1.0:
        return float("inf")
    return float(np.log(0.5) / np.log(p))


# --------------------------------------------------------------------------- #
# Estimation (arch, Gaussian MLE)
# --------------------------------------------------------------------------- #
def fit_garch(r):
    """Fit a zero-mean GARCH(1,1) by Gaussian MLE with the `arch` library.

    Returns a dict with the estimated (omega, alpha, beta), the persistence
    alpha+beta, the implied long-run variance and half-life, and the
    log-likelihood. The DGP is scaled so returns are O(1); `rescale=False`
    keeps the estimates in the native units of the simulated returns.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(np.asarray(r, dtype=float), mean="Zero", vol="Garch",
                        p=1, q=1, dist="normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)
    p = res.params
    omega = float(p["omega"])
    alpha = float(p["alpha[1]"])
    beta = float(p["beta[1]"])
    persistence = alpha + beta
    return dict(
        omega=omega, alpha=alpha, beta=beta,
        persistence=float(persistence),
        long_run_var=float(long_run_variance(omega, alpha, beta))
        if persistence < 1.0 else float("inf"),
        half_life=half_life(persistence),
        loglik=float(res.loglikelihood),
    )


# --------------------------------------------------------------------------- #
# One-step-ahead conditional-variance forecasters
# --------------------------------------------------------------------------- #
def garch_filter(r, omega, alpha, beta, sigma2_0=None):
    """Filtered one-step-ahead conditional variance under fixed GARCH params.

    Returns h[t] = Var_hat(r_t | F_{t-1}) for every t, computed causally from
    realized past returns: h[0] = sigma2_0 (sample variance if None), then
    h[t] = omega + alpha r_{t-1}^2 + beta h[t-1]. With parameters estimated on
    a training block, this is a genuine out-of-sample one-step forecast at each
    test t (it never uses r_t or any future return).
    """
    r = np.asarray(r, dtype=float)
    n = r.size
    h = np.empty(n)
    h[0] = float(np.var(r)) if sigma2_0 is None else float(sigma2_0)
    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
    return h


def ewma_filter(r, lam=RISKMETRICS_LAMBDA, sigma2_0=None):
    """RiskMetrics EWMA variance: h[t] = (1-lam) r_{t-1}^2 + lam h[t-1]."""
    r = np.asarray(r, dtype=float)
    n = r.size
    h = np.empty(n)
    h[0] = float(np.var(r)) if sigma2_0 is None else float(sigma2_0)
    for t in range(1, n):
        h[t] = (1.0 - lam) * r[t - 1] ** 2 + lam * h[t - 1]
    return h


def rolling_filter(r, window):
    """Rolling-window sample variance forecast: h[t] = mean(r[t-w:t]^2).

    Uses the mean of the last `window` squared returns (zero-mean DGP), a
    causal estimate of next-step variance. h[t] for t < window uses the
    expanding mean of what is available.
    """
    r = np.asarray(r, dtype=float)
    n = r.size
    r2 = r ** 2
    h = np.empty(n)
    csum = np.concatenate([[0.0], np.cumsum(r2)])
    for t in range(n):
        lo = max(0, t - window)
        cnt = t - lo
        if cnt <= 0:
            h[t] = float(np.var(r))          # cold start for t=0
        else:
            h[t] = (csum[t] - csum[lo]) / cnt
    return h


# --------------------------------------------------------------------------- #
# Analytic multi-step GARCH forecast
# --------------------------------------------------------------------------- #
def garch_multistep(sigma2_next, omega, alpha, beta, horizons):
    """Analytic h-step variance forecast from a one-step-ahead value.

    E_T[sigma^2_{T+h}] = lrv + (alpha+beta)^{h-1} (sigma^2_{T+1} - lrv),
    the geometric mean-reversion of the GARCH forecast toward the long-run
    variance lrv = omega/(1-alpha-beta). `horizons` is an iterable of h>=1.
    Returns an array aligned with `horizons`.
    """
    p = alpha + beta
    lrv = omega / (1.0 - p)
    out = np.array([lrv + p ** (h - 1) * (sigma2_next - lrv) for h in horizons],
                   dtype=float)
    return out


# --------------------------------------------------------------------------- #
# Forecast-loss metrics
# --------------------------------------------------------------------------- #
def mse_loss(target, pred):
    """Mean squared error against a variance target (Patton-robust)."""
    target = np.asarray(target, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean((target - pred) ** 2))


def qlike_loss(target, pred):
    """QLIKE loss (Patton-robust): mean(target/pred - log(target/pred) - 1).

    Bregman divergence generated by -log; >= 0, uniquely minimized at
    pred == target. Robust to replacing `target` by any conditionally
    unbiased proxy (e.g. squared returns), which is the Patton (2011) result.
    """
    target = np.asarray(target, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ratio = target / pred
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mae_loss(target, pred):
    """Mean absolute error against a variance target (NON-robust contrast).

    Included to demonstrate Patton (2011): MAE targets the conditional median
    of the proxy, not its mean, so its forecaster ranking is NOT guaranteed to
    survive the switch from true variance to a noisy proxy.
    """
    target = np.asarray(target, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(target - pred)))
