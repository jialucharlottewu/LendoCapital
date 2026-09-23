import pandas as pd
import yfinance as yf


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