"""Controlled GARCH(1,1) volatility-forecasting experiments -> results.json.

Four seeded experiments on a synthetic GARCH(1,1) DGP with KNOWN ground truth
(the true conditional variance path is generated, not estimated):

  1. param_recovery   : MLE consistency. Simulate a crypto-like GARCH(1,1)
     (alpha=0.09, beta=0.90, persistence 0.99) and refit over M seeds at
     several sample sizes T. Bias and RMSE of each parameter shrink with T and
     the persistence estimate converges to the true 0.99.
  2. forecast_contest : one-step-ahead forecasters scored against the TRUE
     conditional variance (MSE and QLIKE): correctly-specified GARCH refit,
     EWMA/RiskMetrics (lambda=0.94), and rolling-window sample variance. The
     correctly-specified GARCH wins under a GARCH DGP; the gap shrinks as the
     DGP persistence falls toward near-random. Includes multi-step (h=1,5,10,22)
     error and the analytic mean-reversion of the forecast toward the long-run
     variance.
  3. persistence_halflife : across a grid of true persistence (0.90..0.995),
     the estimated persistence, implied half-life ln(0.5)/ln(alpha+beta), and
     how estimation and long-horizon forecast uncertainty blow up as
     alpha+beta -> 1 (near-IGARCH).
  4. proxy_robustness : re-score experiment 2's forecasters against the noisy
     squared-return proxy instead of the true variance. The Patton (2011)
     robust losses (MSE, QLIKE) preserve the forecaster ranking; a non-robust
     loss (MAE) need not -- confirming the evaluation is trustworthy without
     observing true volatility.

Everything is seeded and deterministic. Run: python scripts/run_all.py [--quick]
"""
import argparse
import json
import os
import sys

import arch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garch import (  # noqa: E402
    simulate_garch, fit_garch, long_run_variance, half_life,
    garch_filter, ewma_filter, rolling_filter, garch_multistep,
    mse_loss, qlike_loss, mae_loss, RISKMETRICS_LAMBDA,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Crypto-like base DGP: high persistence (near-IGARCH), long-run variance 4.0
# (per-observation vol 2.0, i.e. a 2% daily move in percent units).
TRUE_OMEGA = 0.04
TRUE_ALPHA = 0.09
TRUE_BETA = 0.90
TRUE_PERSISTENCE = TRUE_ALPHA + TRUE_BETA          # 0.99
ROLLING_WINDOWS = [22, 66, 132]                    # ~1, 3, 6 trading months
FORECAST_HORIZONS = [1, 5, 10, 22]


# --------------------------------------------------------------------------- #
# 1. Parameter recovery / MLE consistency
# --------------------------------------------------------------------------- #
def param_recovery(sample_sizes, M, seed=0):
    rng = np.random.default_rng(seed)
    truth = dict(omega=TRUE_OMEGA, alpha=TRUE_ALPHA, beta=TRUE_BETA,
                 persistence=TRUE_PERSISTENCE)
    rows = []
    for T in sample_sizes:
        est = {k: [] for k in ("omega", "alpha", "beta", "persistence",
                               "half_life")}
        for _ in range(M):
            r, _ = simulate_garch(TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA, T, rng)
            fit = fit_garch(r)
            for k in est:
                est[k].append(fit[k])
        row = dict(T=int(T), M=int(M))
        for k in ("omega", "alpha", "beta", "persistence"):
            a = np.array(est[k])
            tv = truth[k]
            row[k + "_mean"] = float(a.mean())
            row[k + "_bias"] = float(a.mean() - tv)
            row[k + "_rmse"] = float(np.sqrt(np.mean((a - tv) ** 2)))
            row[k + "_std"] = float(a.std(ddof=1))
        # half-life recovered from each fit (true half-life ~ 68.97 obs)
        hl = np.array(est["half_life"])
        row["half_life_true"] = float(half_life(TRUE_PERSISTENCE))
        row["half_life_median"] = float(np.median(hl))
        rows.append(row)
    return dict(truth=truth, sample_sizes=[int(x) for x in sample_sizes],
                M=int(M), rows=rows)


# --------------------------------------------------------------------------- #
# 2. Forecast contest against the TRUE conditional variance
# --------------------------------------------------------------------------- #
def _forecasters(r, train, sigma2_seed):
    """Build every one-step-ahead variance forecast over the full path.

    Parameters are estimated on r[:train]; forecasts are evaluated later only
    on the test block, so GARCH's parameters never see the test returns.
    Returns (dict name->h array, fitted_params).
    """
    fit = fit_garch(r[:train])
    fc = {}
    fc["garch"] = garch_filter(r, fit["omega"], fit["alpha"], fit["beta"],
                               sigma2_0=sigma2_seed)
    fc["ewma"] = ewma_filter(r, lam=RISKMETRICS_LAMBDA, sigma2_0=sigma2_seed)
    for w in ROLLING_WINDOWS:
        fc["roll_%d" % w] = rolling_filter(r, w)
    return fc, fit


def _score_block(fc, target, test_slice):
    out = {}
    tgt = target[test_slice]
    for name, h in fc.items():
        hs = h[test_slice]
        out[name] = dict(mse=mse_loss(tgt, hs), qlike=qlike_loss(tgt, hs),
                         mae=mae_loss(tgt, hs))
    return out


def _rank(scores, metric):
    items = sorted(scores.items(), key=lambda kv: kv[1][metric])
    return [name for name, _ in items]


def forecast_contest(persistences, T_total, T_train, M, seed=10):
    """For each DGP persistence, average one-step forecast losses over M seeded
    paths, scoring every forecaster against the true conditional variance and
    (for experiment 4) the squared-return proxy. Reports the GARCH advantage
    over the best naive competitor and how it shrinks as persistence falls."""
    rng = np.random.default_rng(seed)
    test_slice = slice(T_train, T_total)
    dgps = []
    # per-seed proxy concordance: for each seed and each forecaster pair, does
    # the ordering under the noisy squared-return proxy match the ordering
    # under the true conditional variance? Aggregated across all seeds, pairs,
    # and DGPs, per loss. Patton (2011): robust losses (MSE, QLIKE) should
    # agree on a higher fraction of noisy draws than the non-robust MAE.
    ps_conc = {m: [0, 0] for m in ("mse", "qlike", "mae")}
    for pers in persistences:
        # keep alpha's share of persistence fixed at the base ratio, and hold
        # the long-run variance at 4.0 so paths are comparable across DGPs
        alpha = TRUE_ALPHA / TRUE_PERSISTENCE * pers
        beta = pers - alpha
        omega = 4.0 * (1.0 - pers)
        acc_true = {}
        acc_proxy = {}
        # accumulate per-seed average losses
        seed_true = {}
        seed_proxy = {}
        for _ in range(M):
            r, s2 = simulate_garch(omega, alpha, beta, T_total, rng)
            proxy = r ** 2
            fc, _ = _forecasters(r, T_train, float(np.var(r[:T_train])))
            st = _score_block(fc, s2, test_slice)
            sp = _score_block(fc, proxy, test_slice)
            names = sorted(fc.keys())
            for name in fc:
                for m in ("mse", "qlike", "mae"):
                    seed_true.setdefault(name, {}).setdefault(m, []).append(st[name][m])
                    seed_proxy.setdefault(name, {}).setdefault(m, []).append(sp[name][m])
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    for m in ("mse", "qlike", "mae"):
                        agree = (np.sign(st[a][m] - st[b][m])
                                 == np.sign(sp[a][m] - sp[b][m]))
                        ps_conc[m][0] += int(agree)
                        ps_conc[m][1] += 1
        for name in seed_true:
            acc_true[name] = {m: float(np.mean(seed_true[name][m]))
                              for m in ("mse", "qlike", "mae")}
            acc_proxy[name] = {m: float(np.mean(seed_proxy[name][m]))
                               for m in ("mse", "qlike", "mae")}
        best_naive = min((n for n in acc_true if n != "garch"),
                         key=lambda n: acc_true[n]["qlike"])
        garch_q = acc_true["garch"]["qlike"]
        naive_q = acc_true[best_naive]["qlike"]
        dgps.append(dict(
            persistence=float(pers),
            alpha=float(alpha), beta=float(beta), omega=float(omega),
            half_life=float(half_life(pers)),
            long_run_var=float(long_run_variance(omega, alpha, beta)),
            loss_true=acc_true, loss_proxy=acc_proxy,
            best_naive=best_naive,
            garch_qlike=float(garch_q), best_naive_qlike=float(naive_q),
            qlike_gap=float(naive_q - garch_q),
            qlike_gap_pct=float((naive_q - garch_q) / naive_q * 100.0),
            garch_wins_true=bool(garch_q == min(v["qlike"] for v in acc_true.values())),
            rank_true_qlike=_rank(acc_true, "qlike"),
        ))
    per_seed_concordance = {m: float(ps_conc[m][0] / ps_conc[m][1])
                            for m in ps_conc}
    return dict(T_total=int(T_total), T_train=int(T_train), M=int(M),
                rolling_windows=list(ROLLING_WINDOWS),
                ewma_lambda=RISKMETRICS_LAMBDA,
                persistences=[float(p) for p in persistences], dgps=dgps,
                per_seed_concordance=per_seed_concordance,
                per_seed_pairs_total=int(ps_conc["qlike"][1]))


def multistep_forecast(T_total, T_train, M, seed=20):
    """Analytic multi-step GARCH forecast error and mean-reversion toward the
    long-run variance, on the crypto-like DGP. At each test origin we forecast
    h steps ahead and compare with the realized TRUE variance sigma^2_{t+h}."""
    rng = np.random.default_rng(seed)
    horizons = FORECAST_HORIZONS
    lrv = long_run_variance(TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA)
    sq_err = {h: [] for h in horizons}          # forecast vs realized true var
    dev = {h: [] for h in horizons}             # |forecast - lrv| (mean-revert)
    for _ in range(M):
        r, s2 = simulate_garch(TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA, T_total, rng)
        fit = fit_garch(r[:T_train])
        h1 = garch_filter(r, fit["omega"], fit["alpha"], fit["beta"],
                          sigma2_0=float(np.var(r[:T_train])))
        # one-step-ahead value at each origin t is h1[t+1] = forecast of t+1;
        # multistep from that origin uses fitted params
        maxh = max(horizons)
        for t in range(T_train, T_total - maxh):
            sig2_next = h1[t + 1]
            fpath = garch_multistep(sig2_next, fit["omega"], fit["alpha"],
                                    fit["beta"], horizons)
            for i, h in enumerate(horizons):
                sq_err[h].append((fpath[i] - s2[t + h]) ** 2)
                dev[h].append(abs(fpath[i] - lrv))
    rows = []
    dev1 = float(np.mean(dev[1]))
    for h in horizons:
        mean_dev = float(np.mean(dev[h]))
        rows.append(dict(
            horizon=int(h),
            mse=float(np.mean(sq_err[h])),
            mean_abs_dev_from_lrv=mean_dev,
            dev_ratio=float(mean_dev / dev1),
            analytic_decay=float(TRUE_PERSISTENCE ** (h - 1)),
        ))
    return dict(T_total=int(T_total), T_train=int(T_train), M=int(M),
                horizons=list(horizons), long_run_var=float(lrv),
                persistence=float(TRUE_PERSISTENCE), rows=rows)


# --------------------------------------------------------------------------- #
# 3. Persistence and half-life across the near-IGARCH grid
# --------------------------------------------------------------------------- #
def persistence_halflife(persistences, T, M, seed=30):
    rng = np.random.default_rng(seed)
    rows = []
    for pers in persistences:
        alpha = TRUE_ALPHA / TRUE_PERSISTENCE * pers
        beta = pers - alpha
        omega = 4.0 * (1.0 - pers)
        lrv_true = long_run_variance(omega, alpha, beta)
        est_p, est_hl, est_lrv = [], [], []
        for _ in range(M):
            r, _ = simulate_garch(omega, alpha, beta, T, rng)
            fit = fit_garch(r)
            est_p.append(fit["persistence"])
            est_hl.append(min(fit["half_life"], 1e6))
            est_lrv.append(min(fit["long_run_var"], 1e9))
        est_p = np.array(est_p)
        est_hl = np.array(est_hl)
        est_lrv = np.array(est_lrv)
        rows.append(dict(
            persistence_true=float(pers),
            alpha=float(alpha), beta=float(beta), omega=float(omega),
            half_life_true=float(half_life(pers)),
            long_run_var_true=float(lrv_true),
            persistence_mean=float(est_p.mean()),
            persistence_rmse=float(np.sqrt(np.mean((est_p - pers) ** 2))),
            half_life_median=float(np.median(est_hl)),
            half_life_iqr=float(np.percentile(est_hl, 75)
                                - np.percentile(est_hl, 25)),
            long_run_var_median=float(np.median(est_lrv)),
            # analytic sensitivity of long-run variance to persistence:
            # d(lrv)/d(p) = lrv / (1 - p); blows up as p -> 1
            lrv_sensitivity=float(lrv_true / (1.0 - pers)),
        ))
    return dict(persistences=[float(p) for p in persistences], T=int(T),
                M=int(M), rows=rows)


# --------------------------------------------------------------------------- #
# 4. Proxy robustness (Patton 2011)
# --------------------------------------------------------------------------- #
def proxy_robustness(contest):
    """From the forecast contest, compare forecaster orderings scored against
    the TRUE conditional variance versus the noisy squared-return proxy, for
    each loss. Patton (2011): the robust losses (MSE, QLIKE) preserve the
    ordering; the non-robust MAE need not. We report three stable summaries:

      * pairwise concordance -- across all DGPs and all forecaster pairs, the
        fraction whose true-vs-proxy ordering agrees (a Kendall-style measure);
      * top-1 agreement -- fraction of DGPs whose loss-minimizing forecaster is
        the same under true variance and under the proxy;
      * the headline crypto-like DGP full ordering under each loss.
    """
    def rank(losses, metric):
        return [n for n, _ in sorted(losses.items(), key=lambda kv: kv[1][metric])]

    metrics = ("mse", "qlike", "mae")
    per_dgp = []
    concord = {m: [0, 0] for m in metrics}      # [matches, total_pairs]
    top1 = {m: 0 for m in metrics}
    garch_top1_true = {m: 0 for m in metrics}
    garch_top1_proxy = {m: 0 for m in metrics}
    for d in contest["dgps"]:
        lt, lp = d["loss_true"], d["loss_proxy"]
        names = sorted(lt.keys())
        entry = dict(persistence=d["persistence"])
        for m in metrics:
            rt, rp = rank(lt, m), rank(lp, m)
            entry[m + "_rank_true"] = rt
            entry[m + "_rank_proxy"] = rp
            entry[m + "_rank_match"] = bool(rt == rp)
            entry[m + "_top_true"] = rt[0]
            entry[m + "_top_proxy"] = rp[0]
            top1[m] += int(rt[0] == rp[0])
            garch_top1_true[m] += int(rt[0] == "garch")
            garch_top1_proxy[m] += int(rp[0] == "garch")
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    st = np.sign(lt[a][m] - lt[b][m])
                    sp = np.sign(lp[a][m] - lp[b][m])
                    concord[m][0] += int(st == sp)
                    concord[m][1] += 1
        per_dgp.append(entry)
    n = len(contest["dgps"])
    headline = max(contest["dgps"], key=lambda d: d["persistence"])
    head_entry = [e for e in per_dgp
                  if e["persistence"] == headline["persistence"]][0]
    return dict(
        n_dgps=int(n), headline_persistence=float(headline["persistence"]),
        robust_losses=["mse", "qlike"], nonrobust_loss="mae",
        pairwise_total=int(concord["qlike"][1]),
        concordance={m: float(concord[m][0] / concord[m][1]) for m in metrics},
        per_seed_concordance=contest.get("per_seed_concordance"),
        per_seed_pairs_total=contest.get("per_seed_pairs_total"),
        concordant_pairs={m: int(concord[m][0]) for m in metrics},
        top1_agree_count={m: int(top1[m]) for m in metrics},
        top1_agree_rate={m: float(top1[m] / n) for m in metrics},
        garch_top1_true={m: int(garch_top1_true[m]) for m in metrics},
        garch_top1_proxy={m: int(garch_top1_proxy[m]) for m in metrics},
        headline_qlike_rank_true=head_entry["qlike_rank_true"],
        headline_qlike_rank_proxy=head_entry["qlike_rank_proxy"],
        headline_qlike_match=head_entry["qlike_rank_match"],
        per_dgp=per_dgp,
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        cfg = dict(
            pr_sizes=[500, 1000, 2000], pr_M=25,
            fc_T=1500, fc_train=1000, fc_M=20,
            ms_T=1500, ms_train=1000, ms_M=15,
            ph_T=2000, ph_M=25,
        )
    else:
        cfg = dict(
            pr_sizes=[500, 1000, 2000, 5000], pr_M=300,
            fc_T=3000, fc_train=2000, fc_M=200,
            ms_T=3000, ms_train=2000, ms_M=120,
            ph_T=3000, ph_M=300,
        )
    contest_persistences = [0.99, 0.90, 0.70, 0.40, 0.20, 0.05]
    grid_persistences = [0.90, 0.95, 0.97, 0.98, 0.99, 0.995]

    print("[1/4] parameter recovery (MLE consistency) ...", flush=True)
    pr = param_recovery(cfg["pr_sizes"], cfg["pr_M"], seed=0)

    print("[2/4] forecast contest vs true conditional variance ...", flush=True)
    fc = forecast_contest(contest_persistences, cfg["fc_T"], cfg["fc_train"],
                          cfg["fc_M"], seed=10)
    print("[2b] multi-step forecast / mean-reversion ...", flush=True)
    ms = multistep_forecast(cfg["ms_T"], cfg["ms_train"], cfg["ms_M"], seed=20)

    print("[3/4] persistence and half-life grid ...", flush=True)
    ph = persistence_halflife(grid_persistences, cfg["ph_T"], cfg["ph_M"],
                              seed=30)

    print("[4/4] proxy robustness (Patton 2011) ...", flush=True)
    px = proxy_robustness(fc)

    results = dict(
        meta=dict(
            seed=0, quick=bool(args.quick),
            python=sys.version.split()[0], numpy=np.__version__,
            arch=arch.__version__, scipy=__import__("scipy").__version__,
            true_dgp=dict(omega=TRUE_OMEGA, alpha=TRUE_ALPHA, beta=TRUE_BETA,
                          persistence=TRUE_PERSISTENCE),
            rolling_windows=list(ROLLING_WINDOWS),
            ewma_lambda=RISKMETRICS_LAMBDA,
            forecast_horizons=list(FORECAST_HORIZONS),
            contest_persistences=list(contest_persistences),
            grid_persistences=list(grid_persistences),
            config=cfg,
        ),
        param_recovery=pr,
        forecast_contest=fc,
        multistep=ms,
        persistence_halflife=ph,
        proxy_robustness=px,
    )

    out = os.path.join(ROOT, "results",
                       "results_quick.json" if args.quick else "results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out)

    # headline
    print("\n=== headline ===")
    last = pr["rows"][-1]
    print(f"param recovery @T={last['T']}: persistence "
          f"{last['persistence_mean']:.4f} (true {TRUE_PERSISTENCE}), "
          f"alpha rmse {last['alpha_rmse']:.4f}, beta rmse {last['beta_rmse']:.4f}")
    for d in fc["dgps"]:
        lt = d["loss_true"]
        print(f"  contest p={d['persistence']:.2f}: GARCH qlike "
              f"{lt['garch']['qlike']:.4f} vs best naive {d['best_naive']} "
              f"{d['best_naive_qlike']:.4f} (gap {d['qlike_gap_pct']:.1f}%) "
              f"win={d['garch_wins_true']}")
    print("  multistep mse by h:",
          {r["horizon"]: round(r["mse"], 3) for r in ms["rows"]})
    for r in ph["rows"]:
        print(f"  p*={r['persistence_true']:.3f}: est {r['persistence_mean']:.4f}, "
              f"HL true {r['half_life_true']:.1f} med {r['half_life_median']:.1f}, "
              f"p-rmse {r['persistence_rmse']:.4f}")
    print("  proxy top-1 agree:", px["top1_agree_count"],
          "of", px["n_dgps"], "DGPs (avg loss)")
    print("  per-seed proxy concordance:",
          {k: round(v, 4) for k, v in px["per_seed_concordance"].items()})


if __name__ == "__main__":
    main()
