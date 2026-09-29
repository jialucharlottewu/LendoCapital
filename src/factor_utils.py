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
def compute_bm(t, tickers, be_hist, shares_hist, splits_hist, prices,
               shares_split_adjusted=False, max_shares_age_days=548):
    """
    Book-to-market for ranking date t (last trading day of June, year Y), following
    Fama & French (1992):
      book equity : fiscal year ending in calendar year Y-1, filed on or before t
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
            cand = be[(be["fy_end"].dt.year == Y - 1) & (be["be_filed"] <= t)]
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
