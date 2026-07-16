#!/usr/bin/env python3
"""Verify every headline numeric claim in paper/main.tex against results.json.

Forward check (REQUIRED): each claim's literal must appear in the manuscript
body (word-boundary matched) and agree with the value computed from
results/results.json within rounding tolerance (half a unit in the literal's
last quoted decimal). Plus two supporting passes: internal-consistency of
results.json (annualization/identity relations) and code-constant checks (DGP
parameters quoted in the paper are the ones actually in the source).

Exit code 0 iff everything passes. Run: python3 scripts/check_paper_numbers.py
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEX = os.path.join(ROOT, "paper", "main.tex")
RESULTS = os.path.join(ROOT, "results", "results.json")

with open(RESULTS) as f:
    R = json.load(f)

META = R["meta"]
DGP = META["true_dgp"]
PR = {row["T"]: row for row in R["param_recovery"]["rows"]}
FC = {round(d["persistence"], 3): d for d in R["forecast_contest"]["dgps"]}
MS = {row["horizon"]: row for row in R["multistep"]["rows"]}
PH = {round(row["persistence_true"], 3): row
      for row in R["persistence_halflife"]["rows"]}
PX = R["proxy_robustness"]
CFG = META["config"]
LRV = R["multistep"]["long_run_var"]                 # 4.0


# --------------------------------------------------------------------------- #
# internal consistency of results.json (identities the paper leans on)
# --------------------------------------------------------------------------- #
def _consistency():
    e = []
    if abs(DGP["omega"] / (1 - DGP["persistence"]) - LRV) > 1e-6:
        e.append("base long-run variance != omega/(1-persistence)")
    if abs(DGP["alpha"] + DGP["beta"] - DGP["persistence"]) > 1e-12:
        e.append("alpha+beta != persistence")
    for p, row in PH.items():
        hl = math.log(0.5) / math.log(p)
        if abs(row["half_life_true"] - hl) > 1e-6:
            e.append(f"half_life_true[{p}] != ln0.5/ln(p)")
        if abs(row["lrv_sensitivity"] - row["long_run_var_true"] / (1 - p)) > 1e-6:
            e.append(f"lrv_sensitivity[{p}] != lrv/(1-p)")
    for h, row in MS.items():
        if abs(row["analytic_decay"] - DGP["persistence"] ** (h - 1)) > 1e-9:
            e.append(f"analytic_decay[h={h}] != persistence^(h-1)")
    for p, d in FC.items():
        gp = (d["best_naive_qlike"] - d["garch_qlike"]) / d["best_naive_qlike"] * 100
        if abs(d["qlike_gap_pct"] - gp) > 1e-6:
            e.append(f"qlike_gap_pct[{p}] inconsistent")
        if not d["garch_wins_true"]:
            e.append(f"GARCH does not win QLIKE at persistence {p}")
    if PX["per_seed_pairs_total"] != 12000:
        e.append("per-seed pair total changed")
    # config sanity
    for k, v in dict(pr_M=300, fc_M=200, ms_M=120, ph_M=300,
                     fc_T=3000, fc_train=2000).items():
        if CFG[k] != v:
            e.append(f"config {k} != {v}")
    return e


# --------------------------------------------------------------------------- #
# code constants quoted in the paper (verified by literal presence in source)
# --------------------------------------------------------------------------- #
CODE_CONSTANTS = [
    ("TRUE_OMEGA = 0.04", "scripts/run_all.py", "DGP omega"),
    ("TRUE_ALPHA = 0.09", "scripts/run_all.py", "DGP alpha"),
    ("TRUE_BETA = 0.90", "scripts/run_all.py", "DGP beta"),
    ("RISKMETRICS_LAMBDA = 0.94", "scripts/garch.py", "EWMA decay"),
    ("ROLLING_WINDOWS = [22, 66, 132]", "scripts/run_all.py", "rolling windows"),
]


def _code_constants():
    e = []
    for needle, rel, what in CODE_CONSTANTS:
        with open(os.path.join(ROOT, rel)) as fh:
            if needle not in fh.read():
                e.append(f"code constant '{needle}' ({what}) not in {rel}")
    return e


# --------------------------------------------------------------------------- #
# the claims: (label, tex literal, value)
# --------------------------------------------------------------------------- #
def _f(x):
    return float(x)


CLAIMS = []


def add(label, lit, value):
    CLAIMS.append((label, lit, value))


# --- DGP / configuration -------------------------------------------------- #
add("DGP alpha", "0.09", DGP["alpha"])
add("DGP beta", "0.90", DGP["beta"])
add("DGP persistence", "0.99", DGP["persistence"])
add("DGP omega", "0.04", DGP["omega"])
add("long-run variance", "4.0", LRV)
add("long-run volatility (derived sqrt)", "2.0", math.sqrt(LRV))
add("true half-life base (derived)", "69.0",
    math.log(0.5) / math.log(DGP["persistence"]))
add("sample size 500", "500", 500)
add("sample size 1000", "1000", 1000)
add("sample size 2000", "2000", 2000)
add("sample size 5000", "5000", 5000)
add("recovery seeds M", "300", CFG["pr_M"])
add("contest seeds M", "200", CFG["fc_M"])
add("multistep seeds M", "120", CFG["ms_M"])
add("contest path length", "3000", CFG["fc_T"])
add("contest train length", "2000", CFG["fc_train"])
add("contest test length (derived)", "1000", CFG["fc_T"] - CFG["fc_train"])
add("rolling window 22", "22", META["rolling_windows"][0])
add("rolling window 66", "66", META["rolling_windows"][1])
add("rolling window 132", "132", META["rolling_windows"][2])
add("EWMA lambda", "0.94", META["ewma_lambda"])
add("horizon 10", "10", 10)
add("horizon 22", "22", 22)
add("grid persistence 0.95", "0.95", 0.95)
add("grid persistence 0.97", "0.97", 0.97)
add("grid persistence 0.98", "0.98", 0.98)
add("grid persistence 0.995", "0.995", 0.995)
add("contest persistence 0.70", "0.70", 0.70)
add("contest persistence 0.40", "0.40", 0.40)
add("contest persistence 0.20", "0.20", 0.20)
add("contest persistence 0.05", "0.05", 0.05)

# --- Experiment 1: parameter recovery ------------------------------------- #
add("pers mean T=500", "0.977", PR[500]["persistence_mean"])
add("pers mean T=1000", "0.984", PR[1000]["persistence_mean"])
add("pers mean T=2000", "0.987", PR[2000]["persistence_mean"])
add("pers mean T=5000", "0.989", PR[5000]["persistence_mean"])
add("pers rmse T=500", "0.028", PR[500]["persistence_rmse"])
add("pers rmse T=1000", "0.013", PR[1000]["persistence_rmse"])
add("pers rmse T=2000", "0.0072", PR[2000]["persistence_rmse"])
add("pers rmse T=5000 (table)", "0.0044", PR[5000]["persistence_rmse"])
add("pers rmse T=5000 (abstract)", "0.004", PR[5000]["persistence_rmse"])
add("alpha rmse T=500", "0.031", PR[500]["alpha_rmse"])
add("alpha rmse T=1000", "0.018", PR[1000]["alpha_rmse"])
add("alpha rmse T=5000", "0.0083", PR[5000]["alpha_rmse"])
add("beta rmse T=500", "0.040", PR[500]["beta_rmse"])
add("beta rmse T=1000", "0.022", PR[1000]["beta_rmse"])
add("beta rmse T=5000", "0.0092", PR[5000]["beta_rmse"])
add("omega rmse T=500", "0.078", PR[500]["omega_rmse"])
add("omega rmse T=1000", "0.031", PR[1000]["omega_rmse"])
add("omega rmse T=2000", "0.017", PR[2000]["omega_rmse"])
add("omega rmse T=5000", "0.011", PR[5000]["omega_rmse"])
add("HL median T=500", "41.3", PR[500]["half_life_median"])
add("HL median T=1000", "50.7", PR[1000]["half_life_median"])
add("HL median T=2000", "56.8", PR[2000]["half_life_median"])
add("HL median T=5000", "63.0", PR[5000]["half_life_median"])
add("pers bias T=500", "0.013", -PR[500]["persistence_bias"])
add("alpha mean T=5000", "0.089", PR[5000]["alpha_mean"])
add("beta mean T=5000", "0.900", PR[5000]["beta_mean"])
add("omega bias T=500", "0.036", PR[500]["omega_bias"])
add("omega bias T=5000", "0.003", PR[5000]["omega_bias"])

# --- Experiment 2: forecast contest --------------------------------------- #
add("garch qlike p=0.99", "0.0018", FC[0.99]["garch_qlike"])
add("garch qlike p=0.90", "0.0016", FC[0.90]["garch_qlike"])
add("garch qlike p=0.70", "0.0016", FC[0.70]["garch_qlike"])
add("garch qlike p=0.40", "0.0012", FC[0.40]["garch_qlike"])
add("garch qlike p=0.20", "0.0011", FC[0.20]["garch_qlike"])
add("garch qlike p=0.05", "0.0013", FC[0.05]["garch_qlike"])
add("naive qlike p=0.99", "0.0119", FC[0.99]["best_naive_qlike"])
add("naive qlike p=0.90", "0.0222", FC[0.90]["best_naive_qlike"])
add("naive qlike p=0.70", "0.0148", FC[0.70]["best_naive_qlike"])
add("naive qlike p=0.40", "0.0093", FC[0.40]["best_naive_qlike"])
add("naive qlike p=0.20", "0.0085", FC[0.20]["best_naive_qlike"])
add("naive qlike p=0.05", "0.0079", FC[0.05]["best_naive_qlike"])
add("gap pct p=0.99", "85.0", FC[0.99]["qlike_gap_pct"])
add("gap pct p=0.90", "92.8", FC[0.90]["qlike_gap_pct"])
add("gap pct p=0.70", "89.2", FC[0.70]["qlike_gap_pct"])
add("gap pct p=0.40", "87.1", FC[0.40]["qlike_gap_pct"])
add("gap pct p=0.20", "87.3", FC[0.20]["qlike_gap_pct"])
add("gap pct p=0.05", "82.9", FC[0.05]["qlike_gap_pct"])
add("garch mse p=0.99", "0.219", FC[0.99]["loss_true"]["garch"]["mse"])
add("garch mse p=0.90", "0.070", FC[0.90]["loss_true"]["garch"]["mse"])
add("garch mse p=0.70", "0.060", FC[0.70]["loss_true"]["garch"]["mse"])
add("garch mse p=0.40", "0.043", FC[0.40]["loss_true"]["garch"]["mse"])
add("garch mse p=0.20", "0.037", FC[0.20]["loss_true"]["garch"]["mse"])
add("garch mse p=0.05", "0.042", FC[0.05]["loss_true"]["garch"]["mse"])
add("ewma mse p=0.99", "1.13", FC[0.99]["loss_true"]["ewma"]["mse"])
add("abs gap p=0.90", "0.0206", FC[0.90]["qlike_gap"])
add("abs gap p=0.05", "0.0065", FC[0.05]["qlike_gap"])
add("gap true garch-ewma p=0.99", "0.0101",
    FC[0.99]["best_naive_qlike"] - FC[0.99]["garch_qlike"])
add("gap range low (derived round)", "83",
    round(min(d["qlike_gap_pct"] for d in FC.values())))
add("gap range high (derived round)", "93",
    round(max(d["qlike_gap_pct"] for d in FC.values())))

# --- multistep ------------------------------------------------------------ #
add("multistep mse h=1", "0.58", MS[1]["mse"])
add("multistep mse h=5", "3.22", MS[5]["mse"])
add("multistep mse h=10", "6.40", MS[10]["mse"])
add("multistep mse h=22", "12.65", MS[22]["mse"])
add("analytic decay h=5", "0.961", MS[5]["analytic_decay"])
add("analytic decay h=10", "0.914", MS[10]["analytic_decay"])
add("analytic decay h=22", "0.810", MS[22]["analytic_decay"])
add("dev ratio h=5", "0.947", MS[5]["dev_ratio"])
add("dev ratio h=10", "0.887", MS[10]["dev_ratio"])
add("dev ratio h=22", "0.767", MS[22]["dev_ratio"])
add("decay/ratio h=1", "1.000", MS[1]["analytic_decay"])

# --- Experiment 3: persistence / half-life -------------------------------- #
for p, lit in [(0.90, "0.896"), (0.95, "0.947"), (0.97, "0.968"),
               (0.98, "0.978"), (0.99, "0.988"), (0.995, "0.993")]:
    add(f"est pers {p}", lit, PH[p]["persistence_mean"])
for p, lit in [(0.90, "0.032"), (0.95, "0.015"), (0.97, "0.011"),
               (0.98, "0.0080"), (0.99, "0.0061"), (0.995, "0.0053")]:
    add(f"pers rmse {p}", lit, PH[p]["persistence_rmse"])
add("pers rmse 0.995 (abstract round)", "0.005", PH[0.995]["persistence_rmse"])
for p, lit in [(0.90, "6.6"), (0.95, "13.5"), (0.97, "22.8"),
               (0.98, "34.3"), (0.99, "69.0"), (0.995, "138.3")]:
    add(f"HL true {p}", lit, PH[p]["half_life_true"])
for p, lit in [(0.90, "6.7"), (0.95, "13.4"), (0.97, "21.8"),
               (0.98, "32.4"), (0.99, "62.4"), (0.995, "105.5")]:
    add(f"HL median {p}", lit, PH[p]["half_life_median"])
for p, lit in [(0.90, "2.6"), (0.95, "5.1"), (0.97, "10.1"),
               (0.98, "14.1"), (0.99, "34.9"), (0.995, "121.4")]:
    add(f"HL IQR {p}", lit, PH[p]["half_life_iqr"])
for p, lit in [(0.90, "3.97"), (0.95, "4.01"), (0.98, "3.90"),
               (0.99, "3.79"), (0.995, "3.56")]:
    add(f"lrv median {p}", lit, PH[p]["long_run_var_median"])
add("lrv sensitivity 0.90", "40", PH[0.90]["lrv_sensitivity"])
add("lrv sensitivity 0.995", "800", PH[0.995]["lrv_sensitivity"])

# --- Experiment 4: proxy robustness --------------------------------------- #
add("per-seed concordance qlike (pct)", "96.2",
    PX["per_seed_concordance"]["qlike"] * 100)
add("per-seed concordance mse (pct)", "95.2",
    PX["per_seed_concordance"]["mse"] * 100)
add("per-seed concordance mae (pct)", "81.5",
    PX["per_seed_concordance"]["mae"] * 100)
add("per-seed concordance qlike (abstract round)", "96",
    round(PX["per_seed_concordance"]["qlike"] * 100))
add("per-seed concordance mae (abstract round)", "82",
    round(PX["per_seed_concordance"]["mae"] * 100))
add("per-seed pairs total", "12000", PX["per_seed_pairs_total"])
add("n_dgps", "6", PX["n_dgps"])
_pt = FC[0.99]["loss_true"]
_pp = FC[0.99]["loss_proxy"]
add("proxy qlike garch", "1.269", _pp["garch"]["qlike"])
add("proxy qlike ewma", "1.279", _pp["ewma"]["qlike"])
add("proxy qlike roll22", "1.297", _pp["roll_22"]["qlike"])
add("proxy qlike roll66", "1.342", _pp["roll_66"]["qlike"])
add("proxy qlike roll132", "1.407", _pp["roll_132"]["qlike"])
add("true qlike roll22", "0.0290", _pt["roll_22"]["qlike"])
add("true qlike roll66", "0.0738", _pt["roll_66"]["qlike"])
add("true qlike roll132", "0.1365", _pt["roll_132"]["qlike"])
add("proxy offset (derived)", "1.267",
    _pp["garch"]["qlike"] - _pt["garch"]["qlike"])
add("gap proxy garch-ewma (derived)", "0.0108",
    _pp["ewma"]["qlike"] - _pp["garch"]["qlike"])

# --- software versions ---------------------------------------------------- #
add("python version", "3.14.6", META["python"])
add("numpy version", "2.5.1", META["numpy"])
add("arch version", "8.0.0", META["arch"])


# --------------------------------------------------------------------------- #
def _parse_literal(lit):
    if re.fullmatch(r"-?\d+\.\d+", lit):
        dec = len(lit.split(".")[1])
        return float(lit), 0.5 * 10.0 ** -dec + 1e-9
    if re.fullmatch(r"-?\d+", lit):
        return float(lit), 1e-9
    return None


def _body(tex):
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", tex, re.S)
    body = m.group(1)
    body = re.sub(
        r"\\(?:eqref|ref|label|pageref|cite[tp]?\*?|href|url|"
        r"bibliographystyle|bibliography)\s*(?:\[[^\]]*\])*\{[^}]*\}", " ", body)
    return body


def main():
    failures = []
    failures += [f"[consistency] {e}" for e in _consistency()]
    failures += [f"[code-const] {e}" for e in _code_constants()]

    with open(TEX) as f:
        body = _body(f.read())

    literals = sorted({c[1] for c in CLAIMS}, key=len, reverse=True)
    counts = {}
    text = body
    for lit in literals:
        esc = re.escape(lit)
        if lit[0].isdigit() or lit[0] == "-":
            pat = re.compile(r"(?<![\d.])" + esc + r"(?!\d)")
        else:
            pat = re.compile(esc)
        counts[lit] = len(pat.findall(text))
        text = pat.sub(" ", text)

    for label, lit, value in CLAIMS:
        if counts[lit] < 1:
            failures.append(f"[presence] {label}: literal '{lit}' not found")
        parsed = _parse_literal(lit)
        if parsed is None:
            if lit != str(value):
                failures.append(f"[value] {label}: '{lit}' vs {value!r}")
        else:
            num, tol = parsed
            if abs(num - float(value)) > tol:
                failures.append(f"[value] {label}: literal '{lit}' vs computed "
                                f"{float(value):.6g} (tol {tol:g})")

    if failures:
        print(f"check_paper_numbers: FAIL ({len(failures)} problem(s))")
        for f_ in failures:
            print("  -", f_)
        return 1
    print(f"check_paper_numbers: OK -- {len(CLAIMS)} numeric claims in main.tex "
          f"verified against results.json (plus consistency and code-constant "
          f"checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
