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