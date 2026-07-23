"""
Commodity Volatility & Macro Model
==================================

Two questions, one script:

  1. VOLATILITY  - How does the risk of WTI crude oil futures evolve over time?
                   Fitted with a GARCH(1,1), which lets today's variance depend on
                   yesterday's shock and yesterday's variance (volatility clustering).

  2. DRIVERS     - What actually moves crude returns? WTI daily returns are regressed
                   on a set of macro factors (dollar, equities, rates, gold, risk
                   appetite) using OLS with Newey-West (HAC) standard errors, which
                   stay honest in the presence of autocorrelation and heteroskedasticity.

Data: Yahoo Finance via yfinance (free, no API key).

Usage:  python commodity_vol_macro.py
Output: charts/volatility.png, charts/macro_betas.png, results/*.txt
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from arch import arch_model

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

START = "2010-01-01"
TARGET = "CL=F"          # WTI crude oil front-month futures

FACTORS = {
    "DX-Y.NYB": "US Dollar Index",     # dollar strength -> oil priced in USD
    "^GSPC":    "S&P 500",             # global growth / risk appetite
    "^VIX":     "VIX",                 # market fear
    "GC=F":     "Gold",                # inflation hedge / real asset comovement
    "^TNX":     "10Y Treasury Yield",  # real rates & growth expectations
    "HG=F":     "Copper",              # industrial demand proxy
}

CHART_DIR = "charts"
RESULT_DIR = "results"

# Muted, print-friendly palette
INK = "#1a1a1a"
ACCENT = "#0b5d8a"
ACCENT_2 = "#b5451c"
GRID = "#d9d9d9"


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------

def download_prices() -> pd.DataFrame:
    """Download adjusted closes for the target and all macro factors."""
    tickers = [TARGET] + list(FACTORS)
    raw = yf.download(tickers, start=START, progress=False, auto_adjust=True)

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.dropna(how="all")

    # Forward-fill across mismatched trading calendars (holidays differ by market),
    # then require every series present so the regression sample is balanced.
    prices = prices.ffill(limit=3).dropna()
    return prices


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, in percent."""
    return 100.0 * np.log(prices / prices.shift(1)).dropna()


# ----------------------------------------------------------------------------
# 2. GARCH(1,1) volatility model
# ----------------------------------------------------------------------------

def fit_garch(returns: pd.Series):
    """
    Fit GARCH(1,1) with Student-t errors.

        r_t     = mu + e_t,        e_t = sigma_t * z_t
        sigma^2 = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2

    Student-t rather than normal because commodity returns have fat tails;
    assuming normality understates the frequency of large moves.
    """
    model = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    return model.fit(disp="off")


def annualized_vol(res) -> pd.Series:
    """Conditional volatility, expressed as an annualized percentage."""
    return res.conditional_volatility * np.sqrt(252)


def vol_persistence(res) -> float:
    """alpha + beta. Approaching 1.0 means shocks to volatility decay slowly."""
    p = res.params
    return float(p["alpha[1]"] + p["beta[1]"])


def half_life(persistence: float) -> float:
    """Trading days for a volatility shock to decay halfway back to its long-run level."""
    return float(np.log(0.5) / np.log(persistence))


# ----------------------------------------------------------------------------
# 3. Macro regression
# ----------------------------------------------------------------------------

def macro_regression(returns: pd.DataFrame):
    """
    OLS of crude returns on contemporaneous macro factor returns,
    with Newey-West (HAC) standard errors, 5 lags.

    Interpretation note: this is a decomposition of comovement, not a causal
    claim and not a forecast - every regressor is same-day.
    """
    y = returns[TARGET]
    X = returns[list(FACTORS)].rename(columns=FACTORS)
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})


def fit_markov_regimes(returns: pd.Series):
    """
    Two-state Markov-switching model with regime-dependent variance.

        r_t | S_t = s  ~  N(mu_s, sigma^2_s),   S_t in {calm, crisis}

    The regimes are estimated from the data rather than imposed by a threshold.
    That matters: a median split forces exactly half the sample into each state
    and puts the boundary in an arbitrary place. The Markov model infers both
    how many days are genuinely stressed and how sticky each state is.

    Returns (fitted result, boolean crisis mask, crisis regime index).
    """
    mod = sm.tsa.MarkovRegression(
        returns, k_regimes=2, trend="c", switching_variance=True
    )
    res = mod.fit(em_iter=25, search_reps=10)

    # Identify which estimated state is the high-variance one - label order
    # is not guaranteed across runs.
    variances = [res.params[f"sigma2[{i}]"] for i in (0, 1)]
    crisis_idx = int(np.argmax(variances))

    smoothed = res.smoothed_marginal_probabilities[crisis_idx]
    return res, smoothed > 0.5, crisis_idx


def regime_regressions(returns: pd.DataFrame, crisis: pd.Series):
    """
    Re-estimate the macro betas separately within each Markov regime.

    This is the "what only appears to drive price" test: a full-sample beta
    averages across states where the relationship genuinely differs, and that
    average can look like no relationship at all.
    """
    out = {}
    for label, mask in [("Calm regime", ~crisis), ("Crisis regime", crisis)]:
        sub = returns[mask.values]
        X = sm.add_constant(sub[list(FACTORS)].rename(columns=FACTORS))
        out[label] = sm.OLS(sub[TARGET], X).fit(
            cov_type="HAC", cov_kwds={"maxlags": 5}
        )
    return out


# ----------------------------------------------------------------------------
# 4. Charts
# ----------------------------------------------------------------------------

def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)


def plot_volatility(prices: pd.Series, ann_vol: pd.Series, path: str):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.3], "hspace": 0.12},
    )

    ax1.plot(prices.index, prices.values, color=INK, linewidth=0.9)
    ax1.set_ylabel("WTI front-month ($/bbl)", fontsize=9.5, color=INK)
    ax1.set_title(
        "WTI Crude Oil: Price and GARCH(1,1) Conditional Volatility",
        fontsize=13, color=INK, loc="left", pad=12,
    )
    _style(ax1)

    ax2.fill_between(ann_vol.index, ann_vol.values, color=ACCENT, alpha=0.16)
    ax2.plot(ann_vol.index, ann_vol.values, color=ACCENT, linewidth=1.0)
    ax2.axhline(ann_vol.mean(), color=ACCENT_2, linestyle="--", linewidth=1.0,
                label=f"Sample mean ({ann_vol.mean():.0f}%)")
    ax2.set_ylabel("Annualized volatility (%)", fontsize=9.5, color=INK)
    ax2.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax2)

    # Label the episodes a reader will recognize
    events = {
        "2014-11-27": "OPEC holds output",
        "2020-04-20": "Negative WTI",
        "2022-03-08": "Ukraine invasion",
    }
    for date, label in events.items():
        ts = pd.Timestamp(date)
        if ann_vol.index.min() <= ts <= ann_vol.index.max():
            nearest = ann_vol.index[ann_vol.index.get_indexer([ts], method="nearest")[0]]
            ax2.annotate(
                label, xy=(nearest, ann_vol.loc[nearest]),
                xytext=(0, 26), textcoords="offset points",
                ha="center", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.7),
            )

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_betas(fit, path: str):
    params = fit.params.drop("const")
    err = 1.96 * fit.bse.drop("const")
    order = params.abs().sort_values().index
    params, err = params[order], err[order]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [ACCENT if v > 0 else ACCENT_2 for v in params]
    ax.barh(params.index, params.values, xerr=err.values, color=colors,
            alpha=0.85, height=0.6,
            error_kw=dict(ecolor=INK, elinewidth=0.9, capsize=3))
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_xlabel("Beta: % move in WTI per 1% move in factor", fontsize=9.5, color=INK)
    ax.set_title(
        f"What Moves Crude? OLS betas with 95% HAC bands  (adj. R² = {fit.rsquared_adj:.2f})",
        fontsize=12.5, color=INK, loc="left", pad=12,
    )
    _style(ax)
    ax.grid(axis="y", visible=False)

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_regimes(prices: pd.Series, crisis: pd.Series, path: str):
    """Price series with Markov-identified crisis periods shaded."""
    fig, ax = plt.subplots(figsize=(11, 4.5))

    px = prices.loc[crisis.index]
    ax.plot(px.index, px.values, color=INK, linewidth=0.9, zorder=3)

    # Shade contiguous runs of crisis days
    flag = crisis.astype(int).values
    edges = np.diff(np.concatenate([[0], flag, [0]]))
    for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1):
        ax.axvspan(crisis.index[start], crisis.index[end],
                   color=ACCENT_2, alpha=0.16, linewidth=0, zorder=1)

    share = 100 * crisis.mean()
    ax.set_ylabel("WTI front-month ($/bbl)", fontsize=9.5, color=INK)
    ax.set_title(
        f"Markov-Switching Crisis Regime  (shaded: {share:.0f}% of trading days)",
        fontsize=12.5, color=INK, loc="left", pad=12,
    )
    _style(ax)

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_regime_betas(regimes: dict, path: str):
    """Grouped comparison of macro betas across the two estimated regimes."""
    calm = regimes["Calm regime"]
    crisis = regimes["Crisis regime"]

    names = list(FACTORS.values())
    order = sorted(names, key=lambda n: abs(calm.params[n]))
    y = np.arange(len(order))
    h = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.barh(y + h / 2, [calm.params[n] for n in order], height=h,
            xerr=[1.96 * calm.bse[n] for n in order],
            color=ACCENT, alpha=0.9, label="Calm regime",
            error_kw=dict(ecolor=INK, elinewidth=0.8, capsize=2.5))
    ax.barh(y - h / 2, [crisis.params[n] for n in order], height=h,
            xerr=[1.96 * crisis.bse[n] for n in order],
            color=ACCENT_2, alpha=0.9, label="Crisis regime",
            error_kw=dict(ecolor=INK, elinewidth=0.8, capsize=2.5))

    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Beta: % move in WTI per 1% move in factor", fontsize=9.5, color=INK)
    ax.set_title(
        "The Same Factors, Two Different Markets",
        fontsize=12.5, color=INK, loc="left", pad=12,
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    ax.grid(axis="y", visible=False)

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. Run
# ----------------------------------------------------------------------------

def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("Downloading data...")
    prices = download_prices()
    returns = to_log_returns(prices)
    print(f"  {len(returns):,} trading days, "
          f"{returns.index.min():%Y-%m-%d} to {returns.index.max():%Y-%m-%d}\n")

    print("Fitting GARCH(1,1)...")
    garch = fit_garch(returns[TARGET])
    ann_vol = annualized_vol(garch)
    persist = vol_persistence(garch)

    print(f"  Persistence (alpha+beta): {persist:.4f}")
    print(f"  Shock half-life:          {half_life(persist):.1f} trading days")
    print(f"  Mean annualized vol:      {ann_vol.mean():.1f}%")
    print(f"  Current annualized vol:   {ann_vol.iloc[-1]:.1f}%\n")

    print("Running macro regression...")
    fit = macro_regression(returns)
    print(f"  Adjusted R-squared: {fit.rsquared_adj:.3f}\n")

    print("Estimating Markov-switching regimes...")
    ms, crisis, crisis_idx = fit_markov_regimes(returns[TARGET])
    calm_idx = 1 - crisis_idx
    daily_calm = np.sqrt(ms.params[f"sigma2[{calm_idx}]"])
    daily_crisis = np.sqrt(ms.params[f"sigma2[{crisis_idx}]"])
    dur = ms.expected_durations

    print(f"  Crisis days:         {int(crisis.sum()):,} ({100 * crisis.mean():.1f}% of sample)")
    print(f"  Calm vol (ann.):     {daily_calm * np.sqrt(252):.1f}%")
    print(f"  Crisis vol (ann.):   {daily_crisis * np.sqrt(252):.1f}%")
    print(f"  Expected duration:   {dur[calm_idx]:.0f} days calm / {dur[crisis_idx]:.0f} days crisis\n")

    print("Estimating regime-conditional betas...")
    regimes = regime_regressions(returns, crisis)
    for label, sub in regimes.items():
        print(f"  {label:14s} adj. R² = {sub.rsquared_adj:.3f}")
    print()

    plot_volatility(prices[TARGET], ann_vol, f"{CHART_DIR}/volatility.png")
    plot_betas(fit, f"{CHART_DIR}/macro_betas.png")
    plot_regimes(prices[TARGET], crisis, f"{CHART_DIR}/regimes.png")
    plot_regime_betas(regimes, f"{CHART_DIR}/regime_betas.png")

    with open(f"{RESULT_DIR}/garch_summary.txt", "w") as f:
        f.write(str(garch.summary()))
    with open(f"{RESULT_DIR}/markov_summary.txt", "w") as f:
        f.write(str(ms.summary()))
    with open(f"{RESULT_DIR}/macro_regression.txt", "w") as f:
        f.write(str(fit.summary()))
        for label, sub in regimes.items():
            f.write(f"\n\n{'=' * 78}\n{label}\n{'=' * 78}\n{sub.summary()}")

    print("Wrote charts/ and results/.")
    return garch, ms, fit, regimes, ann_vol, crisis


if __name__ == "__main__":
    main()
