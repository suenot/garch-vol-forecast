# Does GARCH(1,1) Actually Forecast Volatility? A Controlled Study

**Read online (interactive paper):** <https://garch-forecast.marketmaker.cc> · **Blog post:** <https://marketmaker.cc/en/blog/post/garch-volatility-forecasting-crypto>


A reproducible, controlled study of **GARCH(1,1) volatility forecasting on a
synthetic DGP with known ground truth**. On real markets the conditional
variance is never observed, so "GARCH forecasts well" is hard to verify. We
instead simulate a seeded GARCH(1,1) whose true conditional-variance path is
known by construction, and calibrate exactly when the model beats naive
forecasters and whether the evaluation can be trusted when true volatility is
unobservable. GARCH's win here is guaranteed (the DGP is GARCH); the
contribution is the **calibration** and the **proxy-robustness** result, not a
claim that GARCH wins on real crypto.

Accompanies the [marketmaker.cc](https://marketmaker.cc) blog post
`garch-volatility-forecasting-crypto`. Suggested arXiv category: primary
**q-fin.ST**, cross-list **q-fin.RM**, **econ.EM**.

## Headline (from `results/results.json`, seeded, deterministic)

Base "crypto-like" DGP: `omega = 0.04`, `alpha = 0.09`, `beta = 0.90`,
persistence `0.99` (near-IGARCH), long-run variance `4.0`, true half-life
`69.0` observations.

**1. MLE is consistent.** Persistence estimate converges to the true `0.99` and
every RMSE shrinks like `1/sqrt(T)`:

| T | persistence est. | pers. RMSE | alpha RMSE | beta RMSE | omega RMSE |
|---|---|---|---|---|---|
| 500  | 0.977 | 0.028  | 0.031  | 0.040  | 0.078 |
| 1000 | 0.984 | 0.013  | 0.018  | 0.022  | 0.031 |
| 2000 | 0.987 | 0.0072 | 0.014  | 0.014  | 0.017 |
| 5000 | 0.989 | 0.0044 | 0.0083 | 0.0092 | 0.011 |

**2. GARCH wins the one-step forecast contest at every DGP** (scored against
the true conditional variance by QLIKE), beating the best naive competitor by
83–93%. The margin peaks at moderate persistence and shrinks toward the
near-random regime; near-IGARCH the RiskMetrics EWMA is the closest competitor
because it *is* a persistence-one GARCH:

| DGP persistence | GARCH QLIKE | best naive | best-naive QLIKE | QLIKE gap |
|---|---|---|---|---|
| 0.99 | 0.0018 | EWMA          | 0.0119 | 85.0% |
| 0.90 | 0.0016 | EWMA          | 0.0222 | 92.8% |
| 0.70 | 0.0016 | rolling (132) | 0.0148 | 89.2% |
| 0.40 | 0.0012 | rolling (132) | 0.0093 | 87.1% |
| 0.20 | 0.0011 | rolling (132) | 0.0085 | 87.3% |
| 0.05 | 0.0013 | rolling (132) | 0.0079 | 82.9% |

Multi-step forecasts mean-revert toward the long-run variance `4.0` at the
analytic rate `(alpha+beta)^(h-1)` (`0.810` at h=22 under persistence `0.99`);
forecast MSE grows from `0.58` (h=1) to `12.65` (h=22).

**3. The near-IGARCH fragility.** Persistence is estimated *more* tightly toward
the unit root (RMSE `0.032 → 0.005`), but the half-life and long-run variance
derived from it explode:

| true persistence | true half-life | median est. HL | est. HL IQR | median long-run var |
|---|---|---|---|---|
| 0.90  | 6.6   | 6.7   | 2.6   | 3.97 |
| 0.95  | 13.5  | 13.4  | 5.1   | 4.01 |
| 0.97  | 22.8  | 21.8  | 10.1  | 3.97 |
| 0.98  | 34.3  | 32.4  | 14.1  | 3.90 |
| 0.99  | 69.0  | 62.4  | 34.9  | 3.79 |
| 0.995 | 138.3 | 105.5 | 121.4 | 3.56 |

**4. Proxy robustness (Patton 2011).** Re-scoring against the noisy
squared-return proxy instead of the true variance preserves the ranking for the
robust losses: the proxy inflates every QLIKE by a nearly forecaster-independent
offset (`≈ 1.267`) but keeps the order — at the crypto DGP, GARCH `0.0018 →
1.269`, EWMA `0.0119 → 1.279`. GARCH is selected under the proxy at all 6 DGPs.
Seed by seed on 12,000 forecaster-pair comparisons, QLIKE agrees with the
true-variance ordering **96.2%** of the time and MSE **95.2%**, versus only
**81.5%** for the non-robust MAE.

The synthetic GARCH DGP is chosen for **controlled ground truth, not market
realism** — the deliverable is the calibrated method, not a strategy.

## Reproduce everything

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py            # full run -> results/results.json (~1 min)
python scripts/run_all.py --quick    # smoke run -> results/results_quick.json
python scripts/check_paper_numbers.py  # verify every number in the paper vs results.json
python -m pytest tests/ -q           # deterministic invariants
tectonic paper/main.tex              # -> paper/main.pdf
```

## Layout

```
scripts/
  garch.py               # GARCH simulate/fit (arch), forecasters (GARCH/EWMA/rolling),
                         # analytic multi-step forecast, MSE/QLIKE/MAE losses
  run_all.py             # 4 controlled experiments -> results.json
  check_paper_numbers.py # verifies every numeric claim in main.tex against results.json
tests/test_experiments.py  # deterministic invariants
results/results.json       # committed representative run
paper/main.tex             # the paper   |   paper/FORMULAS.md  formulas + provenance
```

## License

Code: MIT. Paper text and figures: CC BY 4.0.

