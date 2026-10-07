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


# ---------------------------------------------------------------------------
# Benchmark-relative measures and diversification (Task 2)
# ---------------------------------------------------------------------------
def align_to_benchmark(returns, benchmark):
    """
    Put portfolio and benchmark returns on the SAME dates and the SAME frequency.
    Refuses to continue if their frequencies differ (e.g. daily portfolio vs monthly
    benchmark) or if they share fewer than 12 dates.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame(returns.name or "portfolio")
    n_p = infer_periods_per_year(returns.index)
    n_b = infer_periods_per_year(benchmark.index)
    if n_p != n_b:
        raise ValueError(f"frequency mismatch: portfolio {n_p}/yr vs benchmark {n_b}/yr")
    both = returns.join(benchmark.rename("__bench__"), how="inner")
    if len(both) < 12:
        raise ValueError(f"only {len(both)} shared dates with the benchmark")
    return both.drop(columns="__bench__"), both["__bench__"], n_p


def benchmark_relative_metrics(returns, benchmark, periods_per_year=None):
    """
    Beta, alpha, correlation, R-squared, tracking error and information ratio of each
    portfolio against ONE benchmark, on the SAME dates and frequency.

      beta              cov(portfolio, benchmark) / var(benchmark)   (OLS slope)
      alpha_ann         (mean portfolio - beta x mean benchmark) x periods_per_year
      correlation, r2   how closely the portfolio moves with the benchmark (r2 = corr^2)
      tracking_error    std(portfolio - benchmark) x sqrt(periods_per_year)
      active_return_ann mean(portfolio - benchmark) x periods_per_year
      information_ratio active_return_ann / tracking_error

    Tracking error and information ratio compare a portfolio WITH the benchmark, so they
    are meaningful for long-only portfolios; for a zero-cost long-short, read beta,
    alpha and correlation instead.
    """
    port, bench, n = align_to_benchmark(returns, benchmark)
    n = periods_per_year or n
    rows = {}
    for c in port.columns:
        p = port[c].dropna()
        b = bench.loc[p.index]
        beta = np.cov(p, b, ddof=1)[0, 1] / b.var(ddof=1)
        active = p - b
        te = active.std() * np.sqrt(n)
        corr = p.corr(b)
        rows[c] = {
            "beta": beta,
            "alpha_ann": (p.mean() - beta * b.mean()) * n,
            "correlation": corr,
            "r2": corr ** 2,
            "tracking_error": te,
            "active_return_ann": active.mean() * n,
            "information_ratio": active.mean() * n / te if te > 0 else np.nan,
            "n_periods": len(p),
            "periods_per_year": n,
        }
    out = pd.DataFrame(rows).T
    return out.astype({"n_periods": int, "periods_per_year": int})


def diversification_curve(returns, sizes=None, n_draws=500, periods_per_year=None, seed=0):
    """
    How portfolio volatility falls as more stocks are added.

    For each portfolio size k, draws `n_draws` random sets of k stocks (equal weight)
    and records the annualized volatility of each. Returns, per k: the average, 10th
    and 90th percentile volatility, plus the THEORETICAL average for equal-weight
    portfolios of k stocks:
        var(k) = avg_var / k + (1 - 1/k) x avg_cov
    As k grows, var(k) -> avg_cov: the floor set by how much the stocks move together,
    which no number of names removes.

    Only stocks with a complete history over the window are used.
    """
    r = returns.dropna(axis=1)
    n = periods_per_year or infer_periods_per_year(r.index)
    N = r.shape[1]
    sizes = sizes or list(range(1, N + 1))
    rng = np.random.default_rng(seed)
    cov = r.cov().values
    avg_var = np.mean(np.diag(cov))
    avg_cov = (cov.sum() - np.trace(cov)) / (N * (N - 1))
    rows = []
    for k in sizes:
        vols = []
        for _ in range(n_draws if k < N else 1):
            idx = rng.choice(N, size=k, replace=False)
            w = np.full(k, 1 / k)
            vols.append(np.sqrt(w @ cov[np.ix_(idx, idx)] @ w * n))
        theory = np.sqrt((avg_var / k + (1 - 1 / k) * avg_cov) * n)
        rows.append({"n_stocks": k, "avg_vol": np.mean(vols), "p10_vol": np.percentile(vols, 10),
                     "p90_vol": np.percentile(vols, 90), "theory_vol": theory})
    curve = pd.DataFrame(rows).set_index("n_stocks")

    # Where adding names stops helping: the size at which the theoretical curve has
    # captured 90% of the possible fall from one stock down to the floor.
    floor = np.sqrt(avg_cov * n)
    one = np.sqrt(avg_var * n)
    k_grid = np.arange(1, N + 1)
    theory_all = np.sqrt((avg_var / k_grid + (1 - 1 / k_grid) * avg_cov) * n)
    k90 = int(k_grid[np.argmax((one - theory_all) / (one - floor) >= 0.9)])

    corr = r.corr().values
    curve.attrs.update({
        "floor_vol": floor,                                   # volatility no stock count removes
        "avg_stock_vol": one,                                 # typical single-stock volatility
        "avg_pairwise_corr": (corr.sum() - N) / (N * (N - 1)),
        "n_for_90pct": k90,                                   # names needed for 90% of the reduction
        "n_universe": N,
    })
    return curve
