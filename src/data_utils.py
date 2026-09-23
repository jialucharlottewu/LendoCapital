import pandas as pd
import yfinance as yf
import numpy as np


def load_ohlcv(tickers, start, end, benchmark="SPY"):
    all_data = []

    for ticker in tickers:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if raw.empty:
            print(f"Warning: no data returned for {ticker}")
            continue

        # Flatten MultiIndex columns if present (newer yfinance versions do this by default)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.reset_index()
        raw["ticker"] = ticker
        raw = raw.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })
        all_data.append(raw[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]])

    df = pd.concat(all_data, ignore_index=True)

    bench_raw = yf.download(benchmark, start=start, end=end, progress=False)
    if isinstance(bench_raw.columns, pd.MultiIndex):
        bench_raw.columns = bench_raw.columns.get_level_values(0)
    benchmark_days = set(bench_raw.reset_index()["Date"])

    for ticker in df["ticker"].unique():
        ticker_days = set(df.loc[df["ticker"] == ticker, "date"])
        mismatch = benchmark_days.symmetric_difference(ticker_days)
        if mismatch:
            print(f"Warning: {ticker} calendar differs from {benchmark} on {len(mismatch)} day(s)")

    return df

def check_missing_prices(df):
    """Return rows where any OHLC price is null."""
    cols = ["open", "high", "low", "close", "adj_close"]
    missing = df[df[cols].isnull().any(axis=1)]
    return missing


def check_missing_volumes(df):
    """Return rows where volume is null or zero."""
    return df[df["volume"].isnull() | (df["volume"] == 0)]


def check_duplicates(df):
    """Return duplicate (date, ticker) rows."""
    return df[df.duplicated(subset=["date", "ticker"], keep=False)]


def check_invalid_dates(df, start, end):
    """Return rows with dates outside the expected requested range."""
    return df[(df["date"] < pd.to_datetime(start)) | (df["date"] > pd.to_datetime(end))]


def check_zero_prices(df):
    """Return rows where any price is exactly zero (likely bad data)."""
    cols = ["open", "high", "low", "close", "adj_close"]
    return df[(df[cols] == 0).any(axis=1)]


def check_extreme_returns(df, threshold=0.5):
    """
    Flag single-day returns larger than `threshold` (e.g. 50%) in magnitude.
    Large jumps can indicate splits, data errors, or real extreme events.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["daily_return"] = df.groupby("ticker")["adj_close"].pct_change()
    return df[df["daily_return"].abs() > threshold]


def check_volume_spikes(df, z_thresh=5):
    """
    Flag days where log(volume) is more than z_thresh standard deviations
    above a ticker's own mean log(volume). Using log(volume) instead of raw
    volume accounts for the natural right-skew of trading volume (huge
    outlier days), so the z-score threshold reflects genuinely unusual
    activity rather than an artifact of the skewed distribution.
    """
    df = df.copy()
    log_vol = np.log(df["volume"].replace(0, np.nan))
    stats = log_vol.groupby(df["ticker"]).transform(lambda x: (x - x.mean()) / x.std())
    return df[stats.abs() > z_thresh]

def compute_returns(df, method="simple"):
    """
    Add a 'return' column computed from adj_close, per ticker.

    method : "simple" or "log"
        simple: (P_t / P_t-1) - 1
        log:    ln(P_t / P_t-1)
    """
    if method not in ("simple", "log"):
        raise ValueError("method must be 'simple' or 'log'")

    df = df.sort_values(["ticker", "date"]).copy()

    if method == "simple":
        df["return"] = df.groupby("ticker")["adj_close"].pct_change()
    else:
        df["return"] = df.groupby("ticker")["adj_close"].transform(
            lambda x: np.log(x / x.shift(1))
        )

    return df


def aggregate_returns(df, freq="W", method="simple"):
    """
    Aggregate daily returns into weekly ('W') or monthly ('ME') returns,
    compounding correctly rather than summing simple returns.

    Requires df to already have a 'return' column from compute_returns(),
    computed with the SAME method passed here.
    """
    if method not in ("simple", "log"):
        raise ValueError("method must be 'simple' or 'log'")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    if method == "simple":
        # Compounding: (1 + r1) * (1 + r2) * ... - 1, NOT summed
        agg = df.groupby("ticker").resample(freq)["return"].apply(
            lambda r: (1 + r).prod() - 1
        )
    else:
        # Log returns ARE additive across time, so summing is correct here
        agg = df.groupby("ticker").resample(freq)["return"].sum()

    return agg.reset_index().rename(columns={"return": "period_return"})

def rolling_stats(df, window=21):
    """
    Add rolling mean, volatility, skewness, and kurtosis of daily returns,
    computed per ticker over a trailing `window`-day period (default 21
    trading days ≈ 1 calendar month).
    Requires df to already have a 'return' column (from compute_returns()).
    """
    df = df.sort_values(["ticker", "date"]).copy()
    grp = df.groupby("ticker")["return"]
    df["roll_mean"] = grp.transform(lambda x: x.rolling(window).mean())
    df["roll_vol"] = grp.transform(lambda x: x.rolling(window).std())
    df["roll_skew"] = grp.transform(lambda x: x.rolling(window).skew())
    df["roll_kurt"] = grp.transform(lambda x: x.rolling(window).kurt())
    return df


def build_equal_weight_portfolio(df):
    """Daily equal-weight portfolio return: simple average of all tickers' returns each day."""
    port = df.groupby("date")["return"].mean().reset_index()
    return port.rename(columns={"return": "portfolio_return"})


def build_value_weight_portfolio(df, weights):
    """
    Daily value-weight portfolio return, using a fixed weights dict {ticker: weight}.
    Weights should already be normalized to sum to 1.
    NOTE: this uses a single, current-day market-cap snapshot as a static weight
    applied across the whole period — a simplification worth stating explicitly,
    since real value-weighting uses weights that drift over time as prices change.
    """
    df = df.copy()
    df["weight"] = df["ticker"].map(weights)
    df = df.dropna(subset=["weight"])
    df["weighted_return"] = df["return"] * df["weight"]
    port = df.groupby("date")["weighted_return"].sum().reset_index()
    return port.rename(columns={"weighted_return": "portfolio_return"})


def performance_summary(portfolio_returns, periods_per_year=252, risk_free_rate=0.0):
    """
    Compute annualized return, annualized volatility, Sharpe ratio, and max drawdown
    from a Series of periodic (daily) portfolio returns.

    Annualization convention: periods_per_year=252 assumes DAILY returns
    (252 trading days/year). If using monthly returns instead, pass periods_per_year=12.
    """
    r = portfolio_returns.dropna()

    ann_return = r.mean() * periods_per_year
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = (ann_return - risk_free_rate) / ann_vol

    cumulative = (1 + r).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1
    max_drawdown = drawdown.min()

    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }

def build_value_weight_portfolio_historic(df, shares_hist_dict):
    """
    True historical value-weight portfolio: market cap computed as
    HISTORICAL shares outstanding (from get_shares_full) x HISTORICAL
    RAW close price (not adj_close - shares data isn't split-rescaled,
    so pairing it with raw close keeps both values on the same,
    real-at-the-time basis). Weights use the PREVIOUS day's market cap
    to avoid look-ahead bias.

    shares_hist_dict: {ticker: pd.Series of shares outstanding, indexed by date}
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    frames = []
    for t, shares_series in shares_hist_dict.items():
        if shares_series is None or shares_series.empty:
            continue
        shares_df = shares_series.rename("shares_outstanding").reset_index()
        shares_df.columns = ["date", "shares_outstanding"]
        shares_df["date"] = pd.to_datetime(shares_df["date"])
        if shares_df["date"].dt.tz is not None:
            shares_df["date"] = shares_df["date"].dt.tz_localize(None)
        shares_df["ticker"] = t
        frames.append(shares_df)

    shares_all = pd.concat(frames, ignore_index=True)

    merged = pd.merge_asof(
        df.sort_values("date"),
        shares_all.sort_values("date"),
        on="date",
        by="ticker",
        direction="backward"
    )

    merged["market_cap"] = merged["close"] * merged["shares_outstanding"]
    merged = merged.dropna(subset=["market_cap"])

    merged = merged.sort_values(["ticker", "date"])
    merged["market_cap_lag"] = merged.groupby("ticker")["market_cap"].shift(1)

    daily_total = merged.groupby("date")["market_cap_lag"].transform("sum")
    merged["weight"] = merged["market_cap_lag"] / daily_total

    merged["weighted_return"] = merged["return"] * merged["weight"]
    port = merged.groupby("date")["weighted_return"].sum().reset_index()
    return port.rename(columns={"weighted_return": "portfolio_return"})