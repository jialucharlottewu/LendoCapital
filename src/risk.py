"""
Week 3 risk metrics.

compute_risk_metrics() describes how a strategy behaves, not just what it returned:
volatility, Sharpe, Sortino, max drawdown, Calmar, skewness, kurtosis and hit rate,
all on one stated, annualized basis.

Annualization follows the data's actual frequency (daily -> 252, weekly -> 52,
monthly -> 12), inferred from the dates unless you pass periods_per_year yourself.
"""
import inspect

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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


# ---------------------------------------------------------------------------
# Allocation methods and backtest (Task 3)
# ---------------------------------------------------------------------------
def equal_weights(cov):
    """1/N in every stock. Uses no estimates at all (cov is taken only for its tickers)."""
    n = len(cov)
    return pd.Series(1.0 / n, index=cov.index)


def inverse_vol_weights(cov):
    """Weight proportional to 1 / volatility: calmer stocks get more money. Ignores correlation."""
    inv = 1.0 / np.sqrt(np.diag(cov))
    return pd.Series(inv / inv.sum(), index=cov.index)


def risk_contributions(w, cov):
    """
    Share of portfolio variance coming from each stock:  w_i x (cov w)_i / (w' cov w).
    The shares sum to 1.
    """
    w = np.asarray(w, dtype=float)
    c = np.asarray(cov, dtype=float)
    marginal = c @ w
    return w * marginal / (w @ marginal)


def risk_parity_weights(cov, tol=1e-10, max_iter=10_000):
    """
    Equal risk contribution: every stock adds the same share of portfolio variance.
    Unlike inverse volatility, it uses the full covariance, so a stock that moves with
    many others gets less weight than its volatility alone would give it.

    Solved by cyclical coordinate descent (Griveau-Billion, Richard and Roncalli, 2013):
    for each stock in turn, solve  c_ii w_i^2 + (sum_j!=i c_ij w_j) w_i - 1/N = 0  for
    w_i > 0, repeat until weights stop changing, then rescale to sum to 1.
    """
    c = np.asarray(cov, dtype=float)
    n = len(c)
    b = 1.0 / n
    w = 1.0 / np.sqrt(np.diag(c))
    w = w / w.sum()                                  # start from inverse volatility
    for _ in range(max_iter):
        w_old = w.copy()
        for i in range(n):
            rest = c[i] @ w - c[i, i] * w[i]
            w[i] = (-rest + np.sqrt(rest ** 2 + 4 * c[i, i] * b)) / (2 * c[i, i])
        if np.max(np.abs(w - w_old)) < tol:
            break
    w = w / w.sum()
    rc = risk_contributions(w, c)
    assert np.allclose(rc, 1.0 / n, atol=1e-6), "risk parity did not converge"
    return pd.Series(w, index=cov.index)


ALLOCATORS = {
    "Equal weight": equal_weights,
    "Inverse volatility": inverse_vol_weights,
    "Risk parity": risk_parity_weights,
}


def concentration_stats(w, cov=None):
    """
    How concentrated one set of weights is:
      max_weight        largest single position
      top10_weight      share of the book in the 10 largest positions
      effective_n       1 / sum(w^2): number of equal positions with the same concentration
      effective_n_risk  same measure on risk contributions (needs cov): how many stocks
                        really drive the portfolio's risk
    """
    w = pd.Series(w).sort_values(ascending=False)
    out = {
        "n_stocks": int((w > 0).sum()),
        "max_weight": w.iloc[0],
        "min_weight": w.iloc[-1],
        "top10_weight": w.iloc[:10].sum(),
        "effective_n": 1.0 / (w ** 2).sum(),
    }
    if cov is not None:
        rc = risk_contributions(w.values, cov.loc[w.index, w.index].values)
        out["effective_n_risk"] = 1.0 / (rc ** 2).sum()
    return out


def backtest_allocation(daily_returns, monthly_returns, method, rebalance_dates, lookback=252):
    """
    Monthly-rebalanced backtest of one allocation method. Every method gets the SAME
    universe and the SAME schedule, so results differ only because of the weights.

    At each rebalance date t (a month-end):
      1. eligible stocks = those with a full `lookback` days of daily returns up to t
      2. covariance from those daily returns, using data up to and including t only
      3. weights = method(covariance)
      4. hold the weights to the next month-end; portfolio return = sum(w x stock return)

    Turnover: within the month the weights drift with prices. At the next rebalance,
    turnover = 0.5 x sum|new weight - drifted weight| (one-way: share of the book traded).
    The first build is not counted.

    method          : a name in ALLOCATORS, or a function of cov (or of mu and cov, in which
                      case expected returns = trailing mean daily return x 252 are passed too)
    daily_returns   : wide daily returns (dates x tickers)
    monthly_returns : wide month-end to month-end returns, indexed by the month-end it ends on
    Returns (returns, weights, turnover): portfolio returns indexed by the month they are
    earned in, a weights table (one row per rebalance date), and turnover per rebalance.
    """
    allocator = ALLOCATORS[method] if isinstance(method, str) else method
    needs_mu = "mu" in inspect.signature(allocator).parameters
    month_ends = monthly_returns.index
    rets, weights, turnover = {}, {}, {}
    drifted = None
    for t in rebalance_dates:
        nxt = month_ends[month_ends > t]
        if len(nxt) == 0:
            break
        nxt = nxt[0]
        window = daily_returns.loc[:t].tail(lookback)
        eligible = window.columns[window.notna().all()]
        eligible = eligible[monthly_returns.loc[nxt, eligible].notna()]
        cov = window[eligible].cov() * 252
        if needs_mu:                                   # e.g. mean-variance: also needs expected returns
            w = allocator(mu=window[eligible].mean() * 252, cov=cov)
        else:
            w = allocator(cov)
        assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()

        if drifted is not None:
            all_names = w.index.union(drifted.index)
            turnover[t] = 0.5 * (w.reindex(all_names, fill_value=0)
                                 - drifted.reindex(all_names, fill_value=0)).abs().sum()

        r = monthly_returns.loc[nxt, w.index]
        port = (w * r).sum()
        rets[nxt] = port
        weights[t] = w
        drifted = w * (1 + r) / (1 + port)

    returns = pd.Series(rets, name=method if isinstance(method, str) else "portfolio")
    returns.index.name = "return_date"
    weights = pd.DataFrame(weights).T.fillna(0.0)
    weights.index.name = "rebalance_date"
    return returns, weights, pd.Series(turnover, name="turnover")


# ---------------------------------------------------------------------------
# Mean-variance optimization (Task 4)
# ---------------------------------------------------------------------------
SECTORS = {
    "AAPL": "Tech", "ADBE": "Tech", "AMD": "Tech", "ASML": "Tech", "AVGO": "Tech", "CRM": "Tech",
    "CRWD": "Tech", "CSCO": "Tech", "INTC": "Tech", "MSFT": "Tech", "MU": "Tech", "NVDA": "Tech",
    "ORCL": "Tech", "PLTR": "Tech", "QCOM": "Tech", "SMCI": "Tech", "STX": "Tech", "TSM": "Tech",
    "TXN": "Tech", "WDC": "Tech",
    "GOOGL": "Communication", "META": "Communication", "NFLX": "Communication",
    "AMZN": "Consumer discretionary", "HD": "Consumer discretionary", "MCD": "Consumer discretionary",
    "NKE": "Consumer discretionary", "TSLA": "Consumer discretionary",
    "COST": "Consumer staples", "KO": "Consumer staples", "PEP": "Consumer staples",
    "PG": "Consumer staples", "WMT": "Consumer staples",
    "ABT": "Healthcare", "JNJ": "Healthcare", "MRK": "Healthcare", "PFE": "Healthcare",
    "TMO": "Healthcare", "UNH": "Healthcare",
    "BAC": "Financials", "JPM": "Financials", "MA": "Financials", "V": "Financials", "WFC": "Financials",
    "CAT": "Industrials", "GE": "Industrials", "HON": "Industrials", "RKLB": "Industrials",
    "CVX": "Energy", "XOM": "Energy",
}


def mean_variance_weights(mu, cov, objective="max_sharpe", target_return=None, max_weight=1.0,
                          sectors=None, sector_cap=None, rf=0.0):
    """
    Long-only mean-variance optimizer, solved with scipy.optimize.minimize (SLSQP).

    mu, cov       : annualized expected returns (Series) and covariance (DataFrame), same tickers
    objective     : "min_variance"  - lowest variance (ignores mu)
                    "max_sharpe"    - highest (w'mu - rf) / sqrt(w' cov w)
                    "target_return" - lowest variance with w'mu = target_return (frontier points)
                    "max_return"    - highest w'mu (the top end of the frontier)
    Constraints   : weights sum to 1; 0 <= w <= max_weight (long-only, position cap);
                    optional sector caps: sectors = {ticker: sector}, sector_cap = a number
                    (same cap for every sector) or {sector: cap}.
    If no stock's expected return beats rf, no portfolio has a positive Sharpe ratio and
    "max_sharpe" falls back to the minimum-variance portfolio.
    """
    tickers = list(cov.index)
    m = np.asarray(pd.Series(mu).reindex(tickers), dtype=float)
    c = np.asarray(cov, dtype=float)
    n = len(tickers)
    if max_weight * n < 1 - 1e-12:
        raise ValueError(f"max_weight {max_weight} too small for {n} stocks")

    # sector membership masks and caps
    groups = []
    if sectors is not None and sector_cap is not None:
        for sec in sorted({sectors[t] for t in tickers}):
            cap = sector_cap[sec] if isinstance(sector_cap, dict) else sector_cap
            groups.append((np.array([sectors[t] == sec for t in tickers], dtype=float), cap))

    if objective == "max_sharpe":
        ex = m - rf
        if (ex <= 0).all():
            # no stock is expected to beat rf, so no portfolio has a positive Sharpe ratio
            return mean_variance_weights(mu, cov, "min_variance", max_weight=max_weight,
                                         sectors=sectors, sector_cap=sector_cap)
        # Max Sharpe solved in its convex form: minimize y' cov y subject to y'(mu - rf) = 1,
        # y >= 0, then w = y / sum(y). Caps scale with sum(y): y_i <= max_weight x sum(y).
        cons = [{"type": "eq", "fun": lambda y: y @ ex - 1, "jac": lambda y: ex}]
        if max_weight < 1:
            A = np.eye(n) - max_weight * np.ones((n, n))
            cons.append({"type": "ineq", "fun": lambda y: -(A @ y), "jac": lambda y: -A})
        for k, cap in groups:
            g = k - cap * np.ones(n)
            cons.append({"type": "ineq", "fun": lambda y, g=g: -(g @ y), "jac": lambda y, g=g: -g})
        y0 = np.full(n, 1.0 / max(np.full(n, 1.0 / n) @ ex, 1e-3))
        res = minimize(lambda y: y @ c @ y, y0, jac=lambda y: 2 * c @ y, method="SLSQP",
                       bounds=[(0.0, None)] * n, constraints=cons,
                       options={"ftol": 1e-14, "maxiter": 2000})
        if not res.success:
            raise RuntimeError(f"optimizer failed ({objective}): {res.message}")
        w = np.clip(res.x, 0, None)
        w = w / w.sum()
        w[w < 1e-6] = 0.0
        return pd.Series(w / w.sum(), index=tickers)

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}]
    if objective == "target_return":
        cons.append({"type": "eq", "fun": lambda w: w @ m - target_return, "jac": lambda w: m})
    for k, cap in groups:
        cons.append({"type": "ineq", "fun": lambda w, k=k, cp=cap: cp - k @ w, "jac": lambda w, k=k: -k})

    if objective in ("min_variance", "target_return"):
        f, jac = (lambda w: w @ c @ w), (lambda w: 2 * c @ w)
    elif objective == "max_return":
        f, jac = (lambda w: -(w @ m)), (lambda w: -m)
    else:
        raise ValueError(f"unknown objective {objective}")

    res = minimize(f, np.full(n, 1.0 / n), jac=jac, method="SLSQP", bounds=[(0.0, max_weight)] * n,
                   constraints=cons, options={"ftol": 1e-12, "maxiter": 2000})
    if not res.success:
        raise RuntimeError(f"optimizer failed ({objective}): {res.message}")
    w = np.clip(res.x, 0, None)
    w[w < 1e-6] = 0.0
    return pd.Series(w / w.sum(), index=tickers)


def efficient_frontier(mu, cov, n_points=30, **constraints):
    """
    Efficient frontier under the given constraints (max_weight, sectors, sector_cap):
    the lowest-volatility portfolio for each target return, from the minimum-variance
    portfolio's return up to the highest return the constraints allow.
    Returns a table of target points (expected return, volatility, Sharpe) and, in
    .attrs["weights"], the weights at each point.
    """
    mu = pd.Series(mu).reindex(cov.index)
    lo = mean_variance_weights(mu, cov, "min_variance", **constraints) @ mu
    hi = mean_variance_weights(mu, cov, "max_return", **constraints) @ mu
    rows, weights = [], {}
    for target in np.linspace(lo, hi - 1e-6 * abs(hi), n_points):
        try:
            w = mean_variance_weights(mu, cov, "target_return", target_return=target, **constraints)
        except RuntimeError:
            continue
        ret, vol = w @ mu, np.sqrt(w @ cov.values @ w)
        rows.append({"exp_return": ret, "volatility": vol, "sharpe": ret / vol, "n_stocks": int((w > 0).sum())})
        weights[round(ret, 6)] = w
    out = pd.DataFrame(rows)
    out.attrs["weights"] = pd.DataFrame(weights).T
    return out


def bootstrap_weights(daily_returns, allocator, n_boot=200, periods_per_year=252, seed=0):
    """
    Sensitivity of an allocation to estimation error. Re-draws the history (days sampled
    with replacement), re-estimates expected returns and covariance, and recomputes the
    weights each time. Every redrawn history is statistically just as plausible as the
    real one, so the spread of the weights shows how much of the allocation is noise.
    Returns a table: one row per redraw, one column per stock.
    """
    r = daily_returns.dropna()
    needs_mu = "mu" in inspect.signature(allocator).parameters
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        sample = r.iloc[rng.integers(0, len(r), len(r))]
        cov = sample.cov() * periods_per_year
        w = allocator(mu=sample.mean() * periods_per_year, cov=cov) if needs_mu else allocator(cov)
        out.append(w)
    return pd.DataFrame(out).reset_index(drop=True)
