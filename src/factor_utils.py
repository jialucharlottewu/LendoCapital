"""
Week 2 factor utilities: value and momentum signals, ranking, and quintile portfolios.
"""
import numpy as np
import pandas as pd

from data_utils import split_factor_after

BANKS = {"JPM", "BAC", "WFC"}
FOREIGN = {"TSM", "ASML"}


# ---------------------------------------------------------------------------
# Generic helpers (reused by value and momentum)
# ---------------------------------------------------------------------------
def get_ranking_dates(prices, month=None, start=None, end=None):
    """
    Last trading day of each month in `prices`.
    month=6 keeps only June (annual value sorts); month=None keeps every month (momentum).
    """
    d = pd.Series(sorted(pd.to_datetime(prices["date"].unique())))
    last = d.groupby(d.dt.to_period("M")).max()
    if month is not None:
        last = last[last.dt.month == month]
    if start is not None:
        last = last[last >= pd.Timestamp(start)]
    if end is not None:
        last = last[last <= pd.Timestamp(end)]
    return list(last)


def winsorize(s, lower=0.05, upper=0.95):
    """Clip values outside the lower/upper percentiles of THIS cross-section (one date)."""
    lo, hi = s.quantile([lower, upper])
    return s.clip(lo, hi)


def rank_cross_section(s):
    """Percentile rank within one date: near 0 = lowest score, 1 = highest score."""
    return s.rank(pct=True)


# ---------------------------------------------------------------------------
# Value (Task 2)
# ---------------------------------------------------------------------------
def fiscal_year(fy_end):
    """
    The fiscal year a year-end belongs to. Some firms use 52/53-week years that end on
    the weekend closest to Dec 31, so a year-end can land in the first days of January
    (e.g. JNJ's FY2022 ended 2023-01-01). Year-ends in the first two weeks of January
    count as the previous year; everything else uses the calendar year of the year-end.
    """
    return (pd.to_datetime(fy_end) - pd.Timedelta(days=14)).dt.year


def compute_bm(t, tickers, be_hist, shares_hist, splits_hist, prices,
               shares_split_adjusted=False, max_shares_age_days=548):
    """
    Book-to-market for ranking date t (last trading day of June, year Y), following
    Fama & French (1992):
      book equity : fiscal year ending in calendar year Y-1 (see fiscal_year() for
                    52/53-week years ending in early January), filed on or before t
      market cap  : actual traded price x shares outstanding, on the last trading
                    day of December Y-1

    Every stock gets a row. Stocks that can't be scored are kept and FLAGGED in
    `excluded_reason` rather than silently dropped.

    shares_split_adjusted : set True only if yfinance share counts turn out to be
                            split-adjusted (then 'close' is used as-is).
    max_shares_age_days   : share counts older than this are flagged as stale. Share
                            counts move slowly (buybacks of ~1-3%/yr), so 18 months is
                            allowed; any split in between is applied to the old count.
    """
    t = pd.Timestamp(t)
    Y = t.year
    dec = prices[(prices["date"].dt.year == Y - 1) & (prices["date"].dt.month == 12)]
    me_date = dec["date"].max()
    close_on_me = dec[dec["date"] == me_date].set_index("ticker")["close"]

    rows = []
    for tk in tickers:
        r = {"rebalance_date": t, "ticker": tk, "me_date": me_date,
             "fy_end": pd.NaT, "be_filed": pd.NaT, "book_equity": np.nan,
             "shares": np.nan, "shares_date": pd.NaT, "price": np.nan}

        # Book equity: FY ending in Y-1, and already filed by t
        be = be_hist.get(tk)
        if be is not None and not be.empty:
            cand = be[(fiscal_year(be["fy_end"]) == Y - 1) & (be["be_filed"] <= t)]
            if not cand.empty:
                last = cand.sort_values("fy_end").iloc[-1]
                r.update(fy_end=last["fy_end"], be_filed=last["be_filed"],
                         book_equity=last["book_equity"])

        # Price actually traded on me_date (undo yfinance's backward split adjustment)
        adj_px = close_on_me.get(tk, np.nan)
        factor = 1.0 if shares_split_adjusted else split_factor_after(splits_hist.get(tk), me_date)
        r["price"] = adj_px * factor

        # Shares outstanding as of me_date (latest value on or before it)
        sh = shares_hist.get(tk)
        if sh is not None:
            sh = sh[sh.index <= me_date]
            if not sh.empty:
                n, sh_date = float(sh.iloc[-1]), sh.index[-1]
                # If a split happened between the last share observation and me_date,
                # the old ACTUAL count is out of date by the split ratio: scale it forward.
                if not shares_split_adjusted:
                    spl = splits_hist.get(tk)
                    if spl is not None and len(spl):
                        d = spl.index.normalize()
                        between = spl[(d > sh_date.normalize()) & (d <= me_date)]
                        n *= float(between.prod()) if len(between) else 1.0
                r.update(shares=n, shares_date=sh_date)

        rows.append(r)

    df = pd.DataFrame(rows)
    df["shares_age_days"] = (df["me_date"] - df["shares_date"]).dt.days
    df["market_cap"] = df["price"] * df["shares"]
    df["bm"] = df["book_equity"] / df["market_cap"]

    # Flags: later lines override earlier ones, so the most basic reason wins
    reason = pd.Series(None, index=df.index, dtype=object)
    stale = (df["me_date"] - df["shares_date"]).dt.days > max_shares_age_days
    reason[df["market_cap"].isna()] = "missing price/shares"
    reason[stale] = "stale shares data (>18 months old)"
    reason[df["book_equity"].isna()] = "no FY t-1 report"
    reason[df["book_equity"] <= 0] = "negative book equity"
    reason[df["ticker"].isin(FOREIGN)] = "foreign currency"
    reason[df["ticker"].isin(BANKS)] = "bank"
    df["excluded_reason"] = reason
    return df


def value_signal(t, tickers, be_hist, shares_hist, splits_hist, prices,
                 lower=0.05, upper=0.95, shares_split_adjusted=False):
    """
    Value scoring function for ranking date t:
    B/M -> flag unusable stocks -> winsorize (within this date) -> percentile rank.
    rank_pct = 1 is the cheapest stock (highest B/M) = long side.
    """
    df = compute_bm(t, tickers, be_hist, shares_hist, splits_hist, prices,
                    shares_split_adjusted=shares_split_adjusted)
    ok = df["excluded_reason"].isna()
    df["bm_w"] = np.nan
    df["rank_pct"] = np.nan
    if ok.sum() > 0:
        df.loc[ok, "bm_w"] = winsorize(df.loc[ok, "bm"], lower, upper)
        df.loc[ok, "rank_pct"] = rank_cross_section(df.loc[ok, "bm_w"])
    return df


def detect_shares_split_adjusted(shares_hist, splits_hist, window_days=365, min_obs=3):
    """
    Work out from the data whether yfinance share counts are already split-adjusted.

    For every real stock split (ratio >= 1.5 or <= 0.67; small "splits" in yfinance are
    usually spin-offs, where the share count doesn't change) inside a ticker's share
    history, compare the MEDIAN share count over the year before vs the year after
    (medians over a year are robust to yfinance's noisy values near split dates):
      - jumps by about the split ratio (e.g. x10) -> counts are ACTUAL historical numbers
      - roughly unchanged                        -> counts are already SPLIT-ADJUSTED

    Returns (is_adjusted, evidence_table). is_adjusted is None if there is no usable
    evidence, so the caller can stop instead of guessing.
    """
    rows = []
    for tk, splits in splits_hist.items():
        sh = shares_hist.get(tk)
        if splits is None or len(splits) == 0 or sh is None or sh.empty:
            continue
        for split_date, ratio in splits.items():
            if 0.67 < ratio < 1.5:
                continue
            w = pd.Timedelta(days=window_days)
            before = sh[(sh.index < split_date) & (sh.index >= split_date - w)]
            after = sh[(sh.index > split_date + pd.Timedelta(days=5)) & (sh.index <= split_date + w)]
            if len(before) < min_obs or len(after) < min_obs:
                continue
            observed = after.median() / before.median()
            looks_adjusted = abs(np.log(observed)) < abs(np.log(observed / ratio))
            rows.append({"ticker": tk, "split_date": split_date, "split_ratio": ratio,
                         "median_before": before.median(), "median_after": after.median(),
                         "observed_ratio": round(observed, 2), "looks_adjusted": looks_adjusted})

    evidence = pd.DataFrame(rows)
    if evidence.empty:
        return None, evidence
    share_adjusted = evidence["looks_adjusted"].mean()
    if 0.2 < share_adjusted < 0.8:
        print("Warning: mixed evidence on share-count adjustment; inspect the evidence table.")
    return bool(share_adjusted >= 0.5), evidence


# ---------------------------------------------------------------------------
# Momentum (Task 3)
# ---------------------------------------------------------------------------
# Dates a ticker started trading as the company itself. Earlier prices belong to a
# different security (RKLB traded as the SPAC Vector Acquisition until its merger on
# 2021-08-25, near its $10 cash value), so they are dropped before computing returns.
LISTING_DATES = {
    "RKLB": "2021-08-25",
}


def month_end_prices(prices, col="adj_close", end=None, listing_dates=None):
    """
    Wide table of prices on each market month-end (last trading day of each month):
    rows = month-end dates, columns = tickers.

    Uses adj_close (split- and dividend-adjusted), since momentum is a total return.
    A ticker with no price on a month-end (not yet listed, halted) is NaN there.
    end: drop month-ends after this date (use it to leave out a partial current month).
    listing_dates: {ticker: date}; prices before that date are treated as missing
                   (default LISTING_DATES).
    """
    listing_dates = LISTING_DATES if listing_dates is None else listing_dates
    month_ends = get_ranking_dates(prices, end=end)
    wide = prices.pivot(index="date", columns="ticker", values=col)
    for tk, d in listing_dates.items():
        if tk in wide.columns:
            wide.loc[wide.index < pd.Timestamp(d), tk] = np.nan
    return wide.loc[wide.index.isin(month_ends)].sort_index()


def momentum_signal(t, me_prices, lookback=12, skip=1, lower=0.05, upper=0.95):
    """
    Momentum scoring function for ranking date t (a month-end).

    12-1 momentum = cumulative return from the end of month t-12 to the end of month t-1:
        mom = price(end of t-1) / price(end of t-12) - 1
    i.e. the 11 monthly returns t-11 ... t-1. Month t itself is SKIPPED (short-term reversal).

    Every ticker gets a row; tickers without both prices are flagged, not ranked.
    Then: winsorize (within this date) -> percentile rank. rank_pct = 1 is the biggest
    past winner = long side.
    """
    dates = list(me_prices.index)
    t = pd.Timestamp(t)
    if t not in dates:
        raise ValueError(f"{t.date()} is not a month-end in the price table")
    i = dates.index(t)
    if i < lookback:
        raise ValueError(f"need {lookback} month-ends before {t.date()}")
    start, end = dates[i - lookback], dates[i - skip]

    # Timing guard: end must be month t-1 and start month t-12, never month t itself
    t_m = t.to_period("M")
    assert end.to_period("M") == t_m - skip, "look-back must end at month t-1"
    assert start.to_period("M") == t_m - lookback, "look-back must start at month t-12"
    assert end < t, "signal must not use month t"

    df = pd.DataFrame({
        "rebalance_date": t,
        "ticker": me_prices.columns,
        "start_date": start,
        "end_date": end,
        "p_start": me_prices.loc[start].values,
        "p_end": me_prices.loc[end].values,
    })
    df["mom"] = df["p_end"] / df["p_start"] - 1

    reason = pd.Series(None, index=df.index, dtype=object)
    reason[df["p_end"].isna()] = "missing recent price"
    reason[df["p_start"].isna()] = "less than 12 months of history"
    df["excluded_reason"] = reason

    ok = df["excluded_reason"].isna()
    df["mom_w"] = np.nan
    df["rank_pct"] = np.nan
    if ok.sum() > 0:
        df.loc[ok, "mom_w"] = winsorize(df.loc[ok, "mom"], lower, upper)
        df.loc[ok, "rank_pct"] = rank_cross_section(df.loc[ok, "mom_w"])
    return df


# ---------------------------------------------------------------------------
# Quintile portfolios (Task 4) - one function for both factors
# ---------------------------------------------------------------------------
def forward_returns(me_prices):
    """
    Each stock's return over the month AFTER each month-end:
        fwd.loc[t, ticker] = price(end of t+1) / price(end of t) - 1
    This is what a portfolio formed at the close of month t earns. The last
    month-end has no following month, so its row is NaN.
    """
    return me_prices.shift(-1) / me_prices - 1


def carry_forward_signal(signals, month_ends):
    """
    Hold each ranking until the next ranking date. Value is ranked once a year (June),
    so the June ranking is used at every month-end until the next June. Adds a `date`
    column (the month-end the ranking is used at) next to `rebalance_date` (when it was
    formed). Only uses rankings formed on or before each date, so there is no look-ahead.
    """
    formed = sorted(pd.to_datetime(signals["rebalance_date"].unique()))
    rows = []
    for d in month_ends:
        past = [f for f in formed if f <= d]
        if past:
            rows.append({"date": d, "rebalance_date": past[-1]})
    mapping = pd.DataFrame(rows)
    return mapping.merge(signals, on="rebalance_date", how="left")


def form_quintile_portfolios(signals, fwd, score_col="rank_pct", tiebreak_col=None,
                             date_col="date", n_groups=5, long_high=True):
    """
    Sort stocks into quintiles on `score_col` at each date, and measure each quintile's
    EQUAL-WEIGHT return over the FOLLOWING month.

    signals     : one row per (date, ticker) with a score; unranked stocks have NaN score
    fwd         : forward_returns(me_prices) - next-month return for each month-end
    tiebreak_col: raw score used to order stocks tied after winsorizing
    long_high   : True  -> long-short = Q5 - Q1 (high score is the long side)
                  False -> long-short = Q1 - Q5
                  Fixed by the Task 1 design, never chosen from results.

    Returns (returns, holdings):
      returns  : one row per return month: Q1..Q5, long_short, and stocks per quintile
      holdings : every stock's quintile and next-month return at each date
    """
    month_ends = list(fwd.index)
    ret_rows, hold = [], []
    for d, g in signals.groupby(date_col):
        d = pd.Timestamp(d)
        i = month_ends.index(d)
        if i + 1 >= len(month_ends):
            continue                                  # no following month to earn a return in
        ret_date = month_ends[i + 1]
        # Lag guard: the score formed at the close of month t earns the return of month t+1
        assert ret_date.to_period("M") == d.to_period("M") + 1

        g = g[g[score_col].notna()].copy()
        g["fwd_ret"] = g["ticker"].map(fwd.loc[d])
        g = g[g["fwd_ret"].notna()]
        if len(g) < n_groups:
            continue
        keys = [score_col] + ([tiebreak_col] if tiebreak_col else [])
        g = g.sort_values(keys + ["ticker"]).reset_index(drop=True)
        g["quintile"] = pd.qcut(np.arange(len(g)), n_groups, labels=range(1, n_groups + 1)).astype(int)
        g["return_date"] = ret_date
        hold.append(g[[date_col, "return_date", "ticker", score_col, "quintile", "fwd_ret"]])

        q = g.groupby("quintile")["fwd_ret"].mean()
        row = {"formation_date": d, "return_date": ret_date}
        row.update({f"Q{k}": q[k] for k in range(1, n_groups + 1)})
        row["long_short"] = (q[n_groups] - q[1]) if long_high else (q[1] - q[n_groups])
        row.update({f"n_Q{k}": int((g["quintile"] == k).sum()) for k in range(1, n_groups + 1)})
        ret_rows.append(row)

    returns = pd.DataFrame(ret_rows).set_index("return_date")
    holdings = pd.concat(hold, ignore_index=True)
    return returns, holdings


# ---------------------------------------------------------------------------
# Performance summary (Task 5)
# ---------------------------------------------------------------------------
def performance_table(returns, cols=("Q1", "Q2", "Q3", "Q4", "Q5", "long_short"), periods_per_year=12):
    """
    One row per portfolio: annualized return, volatility, Sharpe ratio, max drawdown
    (from Week 1's performance_summary), plus hit rate (share of positive months),
    average monthly return, t-statistic of the mean, and number of months.
    Risk-free rate = 0: fine for the long-short (a zero-cost portfolio), but it flatters
    the long-only quintiles' Sharpe ratios.
    """
    from data_utils import performance_summary
    rows = {}
    for c in cols:
        r = returns[c].dropna()
        p = performance_summary(r, periods_per_year=periods_per_year)
        p["hit_rate"] = (r > 0).mean()
        p["avg_monthly"] = r.mean()
        p["t_stat"] = r.mean() / r.std() * np.sqrt(len(r))
        p["n_months"] = len(r)
        rows[c] = p
    return pd.DataFrame(rows).T


def drawdown(returns):
    """Drawdown series: how far cumulative growth is below its previous peak."""
    growth = (1 + returns).cumprod()
    return growth / growth.cummax() - 1
