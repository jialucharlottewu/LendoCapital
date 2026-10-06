"""
Week 3 risk metrics.

compute_risk_metrics() describes how a strategy behaves, not just what it returned:
volatility, Sharpe, Sortino, max drawdown, Calmar, skewness, kurtosis and hit rate,
all on one stated, annualized basis.

Annualization follows the data's actual frequency (daily -> 252, weekly -> 52,
monthly -> 12), inferred from the dates unless you pass periods_per_year yourself.
"""
import numpy as np
import pandas as pd

PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4, "annual": 1}


def infer_periods_per_year(index):
    """
    Work out how many return periods there are per year from the dates themselves,
    using the median gap between consecutive dates:
      about 1 day   -> daily     (252)
      about 7 days  -> weekly    (52)
      about 30 days -> monthly   (12)
      about 91 days -> quarterly (4)
    Raises an error if the dates are missing or the spacing is unclear, so a wrong
    annualization factor can never slip through silently.
    """
    idx = pd.DatetimeIndex(index)
    if len(idx) < 3:
        raise ValueError("need at least 3 dated observations to infer the frequency")
    gap = pd.Series(idx.sort_values()).diff().dt.days.median()
    if gap <= 4:
        return PERIODS_PER_YEAR["daily"]
    if 5 <= gap <= 10:
        return PERIODS_PER_YEAR["weekly"]
    if 25 <= gap <= 35:
        return PERIODS_PER_YEAR["monthly"]
    if 85 <= gap <= 95:
        return PERIODS_PER_YEAR["quarterly"]
    raise ValueError(f"unclear return frequency (median gap {gap} days); pass periods_per_year")


def max_drawdown(returns):
    """Largest fall from a previous peak in cumulative growth (a negative number)."""
    growth = (1 + returns).cumprod()
    return (growth / growth.cummax() - 1).min()


def _metrics_one(r, n, rf_annual):
    r = r.dropna()
    rf = (1 + rf_annual) ** (1 / n) - 1                 # risk-free rate per period
    excess = r - rf
    years = len(r) / n

    ann_mean = r.mean() * n                               # arithmetic, used in Sharpe
    cagr = (1 + r).prod() ** (1 / years) - 1              # compound, used in Calmar
    vol = r.std() * np.sqrt(n)
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean()) * np.sqrt(n)
    mdd = max_drawdown(r)

    return {
        "ann_return_cagr": cagr,
        "ann_return_mean": ann_mean,
        "ann_volatility": vol,
        "downside_deviation": downside,
        "sharpe": excess.mean() * n / vol if vol > 0 else np.nan,
        "sortino": excess.mean() * n / downside if downside > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "skewness": r.skew(),
        "excess_kurtosis": r.kurt(),
        "hit_rate": (r > 0).mean(),
        "best_period": r.max(),
        "worst_period": r.min(),
        "n_periods": len(r),
        "periods_per_year": n,
        "start": r.index.min(),
        "end": r.index.max(),
    }


def compute_risk_metrics(returns, periods_per_year=None, rf_annual=0.0):
    """
    Risk metrics for one return series or every column of a DataFrame.

    returns          : simple periodic returns (0.02 = +2%), indexed by date
    periods_per_year : 252 daily, 12 monthly, ... ; None -> inferred from the dates
    rf_annual        : annual risk-free rate (default 0). For a long-short, zero-cost
                       portfolio, 0 is the right choice.

    Every return and risk number is ANNUALIZED with the same factor, so rows are
    comparable. Definitions:
      ann_return_cagr     compound annual growth rate
      ann_return_mean     mean periodic return x periods_per_year (arithmetic)
      ann_volatility      std of periodic returns x sqrt(periods_per_year)
      downside_deviation  sqrt(mean(min(r - rf, 0)^2)) x sqrt(periods_per_year)
      sharpe              annualized mean excess return / ann_volatility
      sortino             annualized mean excess return / downside_deviation
      max_drawdown        worst fall from a previous peak of cumulative growth
      calmar              ann_return_cagr / |max_drawdown|
      skewness            sample skewness of periodic returns (not annualized)
      excess_kurtosis     sample kurtosis minus 3 (0 = normal tails; not annualized)
      hit_rate            share of periods with a positive return
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame(returns.name or "returns")
    n = periods_per_year or infer_periods_per_year(returns.index)
    rows = {c: _metrics_one(returns[c], n, rf_annual) for c in returns.columns}
    out = pd.DataFrame(rows).T
    dates = ["start", "end"]
    out[dates] = out[dates].apply(pd.to_datetime)
    num = [c for c in out.columns if c not in dates]
    out[num] = out[num].astype(float)
    out[["n_periods", "periods_per_year"]] = out[["n_periods", "periods_per_year"]].astype(int)
    return out
