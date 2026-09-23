import pandas as pd
import yfinance as yf


def load_ohlcv(tickers, start, end, benchmark="SPY"):
    """
    Load daily OHLCV data for one or more tickers using yfinance.

    Parameters
    ----------
    tickers : list of str
        e.g. ["AAPL", "MSFT"]
    start, end : str
        Date strings, e.g. "2023-01-01"
    benchmark : str
        Ticker used to define the expected trading calendar.
        Any ticker whose trading days disagree with this gets flagged.

    Returns
    -------
    pd.DataFrame with columns:
        date, ticker, open, high, low, close, adj_close, volume
    """
    all_data = []

    for ticker in tickers:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if raw.empty:
            print(f"Warning: no data returned for {ticker}")
            continue

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

    # --- Calendar check against benchmark ---
    bench_raw = yf.download(benchmark, start=start, end=end, progress=False)
    benchmark_days = set(bench_raw.reset_index()["Date"])

    for ticker in df["ticker"].unique():
        ticker_days = set(df.loc[df["ticker"] == ticker, "date"])
        mismatch = benchmark_days.symmetric_difference(ticker_days)
        if mismatch:
            print(f"Warning: {ticker} calendar differs from {benchmark} on {len(mismatch)} day(s)")

    return df