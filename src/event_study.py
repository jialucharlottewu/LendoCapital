"""
Week 4 event study tools.

Task 1 (this part): the study design. Every window is fixed here, in TRADING days
relative to day 0, before any event data is loaded, and checked so the estimation
window can never overlap the event window.

Day 0 = the first trading session in which the market can react to the announcement
(the announcement day if released before the close, the next session if after it).
Day -1 is the session before day 0, day +1 the session after, and so on.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventStudyDesign:
    """
    Window boundaries in trading days relative to day 0 (both ends included).

    est_start, est_end   : estimation window, used only to fit the normal-return model
    event_start, event_end: event window, where abnormal returns are measured
    main_car             : the pre-registered test window (the hypothesis is about this CAR)
    drift_car            : secondary window, after the announcement reaction
    """
    est_start: int = -250
    est_end: int = -30
    event_start: int = -10
    event_end: int = 10
    main_car: tuple = (-1, 1)
    drift_car: tuple = (2, 10)

    @property
    def est_length(self):
        return self.est_end - self.est_start + 1

    @property
    def event_length(self):
        return self.event_end - self.event_start + 1

    @property
    def gap(self):
        """Trading days strictly between the end of estimation and the start of the event window."""
        return self.event_start - self.est_end - 1

    def validate(self, min_gap=10, min_est_length=200):
        """Stop if any window is misplaced. Returns self so it can be chained."""
        assert self.est_start < self.est_end, "estimation window is empty"
        assert self.event_start <= 0 <= self.event_end, "event window must contain day 0"
        assert self.est_end < self.event_start, "estimation window overlaps the event window"
        assert self.gap >= min_gap, f"gap of {self.gap} days is below the minimum of {min_gap}"
        assert self.est_length >= min_est_length, f"estimation window of {self.est_length} days is too short"
        for name, (a, b) in [("main_car", self.main_car), ("drift_car", self.drift_car)]:
            assert self.event_start <= a <= b <= self.event_end, f"{name} {a, b} lies outside the event window"
        return self

    def table(self):
        """The design as a table: every window with its first and last trading day."""
        rows = [
            ("Estimation window", self.est_start, self.est_end),
            ("Gap (not used)", self.est_end + 1, self.event_start - 1),
            ("Event window", self.event_start, self.event_end),
            ("Main test: CAR", *self.main_car),
            ("Secondary test: drift CAR", *self.drift_car),
        ]
        out = pd.DataFrame(rows, columns=["window", "first day", "last day"]).set_index("window")
        out["trading days"] = out["last day"] - out["first day"] + 1
        return out


def window_dates(trading_days, day0, design):
    """
    Map the design onto the real trading calendar for one event.

    trading_days : sorted DatetimeIndex of every trading session available
    day0         : the event's day 0 (must itself be a trading day)
    Returns the calendar dates of each window's first and last day, or raises if the
    calendar does not reach far enough back or forward (such an event is dropped, never
    truncated).
    """
    days = pd.DatetimeIndex(trading_days).sort_values()
    day0 = pd.Timestamp(day0)
    if day0 not in days:
        raise ValueError(f"{day0.date()} is not a trading day; align it first")
    i = days.get_loc(day0)
    lo, hi = i + design.est_start, i + design.event_end
    if lo < 0 or hi >= len(days):
        raise ValueError(f"not enough trading days around {day0.date()} for the design")
    pick = lambda k: days[i + k]
    out = {
        "est_first": pick(design.est_start), "est_last": pick(design.est_end),
        "event_first": pick(design.event_start), "day0": day0, "event_last": pick(design.event_end),
    }
    assert out["est_last"] < out["event_first"], "estimation window reaches into the event window"
    return out


# ---------------------------------------------------------------------------
# Task 2: event data preparation
# ---------------------------------------------------------------------------
MARKET_OPEN = pd.Timedelta(hours=9, minutes=30)
MARKET_CLOSE = pd.Timedelta(hours=16)


def fetch_earnings_dates(tickers, limit=40, pause=1.0):
    """
    Earnings announcement dates, times and EPS surprises from Yahoo Finance (yfinance).

    Returns (table, failed): one row per ticker and announcement with
      announce_ts   announcement time in New York time (Yahoo shows 12 AM when the time is unknown)
      eps_estimate  analysts' consensus EPS estimate
      eps_reported  reported EPS (empty for announcements that have not happened yet)
      surprise_pct  Yahoo's surprise in %
    `limit` rows per ticker reach back about limit / 4 years.
    """
    import time
    import yfinance as yf

    rows, failed = [], []
    for tk in tickers:
        try:
            df = yf.Ticker(tk).get_earnings_dates(limit=limit)
        except Exception as e:
            print(f"{tk}: {type(e).__name__}: {e}")
            df = None
        if df is None or df.empty:
            failed.append(tk)
            continue
        df = df.reset_index()
        ts = pd.to_datetime(df.iloc[:, 0])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
        col = lambda name: pd.to_numeric(df[name], errors="coerce") if name in df else np.nan
        rows.append(pd.DataFrame({
            "ticker": tk, "announce_ts": ts,
            "eps_estimate": col("EPS Estimate"), "eps_reported": col("Reported EPS"),
            "surprise_pct": col("Surprise(%)"),
        }))
        time.sleep(pause)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out, failed


def classify_timing(ts):
    """When in the day the announcement came out, from its New York time stamp."""
    t = pd.Timestamp(ts)
    tod = t - t.normalize()
    if tod == pd.Timedelta(0):
        return "unknown"            # Yahoo's 12 AM = time not given
    if tod < MARKET_OPEN:
        return "before open"
    if tod < MARKET_CLOSE:
        return "during market"
    return "after close"


def align_day0(ts, calendar):
    """
    Day 0 = the first trading session in which the market can react:
      before open / during market / unknown -> the announcement date (or the next trading
                                               day if that date is a weekend or holiday)
      after close                           -> the next trading day after the announcement date
    An unknown time stays on the announcement date; the main test window CAR(-1, +1)
    covers the next session either way.
    """
    cal = pd.DatetimeIndex(calendar)
    d = pd.Timestamp(ts).normalize()
    i = cal.searchsorted(d)                       # first trading day on or after d
    if classify_timing(ts) == "after close" and i < len(cal) and cal[i] == d:
        i += 1
    return cal[i] if i < len(cal) else pd.NaT


def build_event_table(raw, calendar, start, end):
    """
    Clean event table: one row per (ticker, day 0), earnings announcements only.

    Steps, each counted in the returned log:
      1. drop announcements without reported EPS (not happened yet, or no data)
      2. classify the time of day and align to day 0 on the trading calendar
      3. keep day 0 between start and end
      4. drop duplicates (the same ticker and day 0 listed twice)
    Surprise = reported EPS - estimate; its sign is the grouping used for H1.
    """
    log = {"raw rows": len(raw)}
    ev = raw.dropna(subset=["eps_reported"]).copy()
    log["dropped: no reported EPS (future or missing)"] = len(raw) - len(ev)

    ev["announce_ts"] = pd.to_datetime(ev["announce_ts"])
    ev["timing"] = ev["announce_ts"].map(classify_timing)
    ev["day0"] = ev["announce_ts"].map(lambda t: align_day0(t, calendar))
    n = len(ev)
    ev = ev[(ev["day0"] >= pd.Timestamp(start)) & (ev["day0"] <= pd.Timestamp(end))]
    log["outside the sample period"] = n - len(ev)

    n = len(ev)
    ev = ev.sort_values(["ticker", "day0", "eps_estimate"]).drop_duplicates(["ticker", "day0"], keep="first")
    log["dropped: duplicate ticker and day 0"] = n - len(ev)

    ev["event_type"] = "earnings"
    ev["surprise"] = ev["eps_reported"] - ev["eps_estimate"]
    ev["surprise_sign"] = np.select([ev["surprise"] > 0, ev["surprise"] < 0, ev["surprise"] == 0],
                                    ["positive", "negative", "zero"], default="no estimate")
    ev["event_id"] = ev["ticker"] + "_" + ev["day0"].dt.strftime("%Y-%m-%d")
    cols = ["event_id", "ticker", "event_type", "announce_ts", "timing", "day0",
            "eps_estimate", "eps_reported", "surprise", "surprise_pct", "surprise_sign"]
    ev = ev[cols].sort_values(["day0", "ticker"]).reset_index(drop=True)
    log["events kept"] = len(ev)
    return ev, log


def build_window_table(events, stock_returns, market_returns, design):
    """
    Per-event window table: the calendar dates of each event's estimation and event
    windows, built on the trading calendar shared by the stocks and the market.

    An event is DROPPED (never truncated) when:
      - the calendar does not reach design.est_start days before day 0 or design.event_end after
      - the stock or the market has any missing return inside the estimation window
        (e.g. a stock that was not yet listed)
      - the stock or the market has any missing return inside the event window
    Returns (windows, dropped): kept events with their window dates, and dropped
    events with the reason.
    """
    cal = stock_returns.index.intersection(market_returns.index).sort_values()
    keep, drop = [], []
    for ev in events.itertuples(index=False):
        base = {"event_id": ev.event_id, "ticker": ev.ticker, "day0": ev.day0}
        if ev.day0 not in cal or ev.ticker not in stock_returns:
            drop.append({**base, "reason": "day 0 or ticker not in the return data"})
            continue
        i = cal.get_loc(ev.day0)
        if i + design.est_start < 0:
            drop.append({**base, "reason": "not enough trading days before day 0"})
            continue
        if i + design.event_end >= len(cal):
            drop.append({**base, "reason": "not enough trading days after day 0"})
            continue
        est = cal[i + design.est_start: i + design.est_end + 1]
        win = cal[i + design.event_start: i + design.event_end + 1]
        r = stock_returns[ev.ticker]
        if r.loc[est].isna().any() or market_returns.loc[est].isna().any():
            drop.append({**base, "reason": "missing returns in the estimation window (short history)"})
            continue
        if r.loc[win].isna().any() or market_returns.loc[win].isna().any():
            drop.append({**base, "reason": "missing returns in the event window"})
            continue
        keep.append({**base, "est_first": est[0], "est_last": est[-1], "event_first": win[0],
                     "event_last": win[-1], "n_est": len(est), "n_event": len(win)})
    windows = events.merge(pd.DataFrame(keep), on=["event_id", "ticker", "day0"], how="inner")
    dropped = pd.DataFrame(drop, columns=["event_id", "ticker", "day0", "reason"])
    return windows, dropped


def check_no_leakage(windows, calendar, design):
    """
    Checks every event's windows against the design on the trading calendar.
    Raises on the first failure; returns a one-row summary when all pass.
      - estimation window has exactly design.est_length trading days, none missing
      - estimation window ends strictly before the event window starts
      - the gap between them is exactly design.gap trading days
      - the event window runs exactly from design.event_start to design.event_end around day 0
    """
    cal = pd.DatetimeIndex(calendar).sort_values()
    pos = {d: k for k, d in enumerate(cal)}
    for w in windows.itertuples(index=False):
        i = pos[w.day0]
        assert w.n_est == design.est_length, f"{w.event_id}: estimation window has {w.n_est} days"
        assert w.est_last < w.event_first, f"{w.event_id}: estimation window reaches the event window"
        assert pos[w.event_first] - pos[w.est_last] - 1 == design.gap, f"{w.event_id}: gap is wrong"
        assert pos[w.est_first] == i + design.est_start and pos[w.est_last] == i + design.est_end, \
            f"{w.event_id}: estimation window misplaced"
        assert pos[w.event_first] == i + design.event_start and pos[w.event_last] == i + design.event_end, \
            f"{w.event_id}: event window misplaced"
    return pd.DataFrame({"events checked": [len(windows)], "estimation days each": [design.est_length],
                         "gap days each": [design.gap], "all checks passed": [True]})


def fetch_8k_earnings_filings(tickers, since="2021-01-01", pause=0.15):
    """
    Dates of each company's SEC 8-K filings under Item 2.02 ("Results of Operations"),
    the form companies file when they release earnings. Used only to cross-check the
    Yahoo announcement dates. Needs SEC_USER_AGENT in .env (see data_utils.py).
    """
    import time
    import requests
    from data_utils import _sec_headers, get_cik_map, CIK_OVERRIDES

    cik_map = get_cik_map()
    rows = []
    for tk in tickers:
        ciks = CIK_OVERRIDES.get(tk, [cik_map.get(tk.replace(".", "-"))])
        for cik in [c for c in ciks if c]:
            r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=_sec_headers())
            time.sleep(pause)
            if r.status_code != 200:
                print(f"{tk}: HTTP {r.status_code}")
                continue
            js = r.json()
            blocks = [js["filings"]["recent"]]
            for f in js["filings"].get("files", []):          # older filings live in extra files
                if f.get("filingTo", "9999") >= since:
                    rf = requests.get(f"https://data.sec.gov/submissions/{f['name']}", headers=_sec_headers())
                    time.sleep(pause)
                    if rf.status_code == 200:
                        blocks.append(rf.json())
            for b in blocks:
                df = pd.DataFrame({k: b[k] for k in ["form", "filingDate", "items"]})
                df = df[df["form"].isin(["8-K", "8-K/A"]) & df["items"].fillna("").str.contains("2.02", regex=False)]
                df = df[df["filingDate"] >= since]
                rows += [{"ticker": tk, "filing_date": pd.Timestamp(d)} for d in df["filingDate"]]
    return pd.DataFrame(rows, columns=["ticker", "filing_date"]).drop_duplicates().sort_values(["ticker", "filing_date"])


def cross_check_dates(events, filings, before=1, after=3, search=30):
    """
    Compare each announcement date with the nearest 8-K Item 2.02 filing date.
      match     filing between `before` days before and `after` days after the announcement
                (after-close releases are often filed that evening or the next business day)
      mismatch  a filing exists within `search` days but outside that range
      no 8-K    no Item 2.02 filing within `search` days
    """
    out = events.copy()
    ann = out["announce_ts"].dt.normalize()
    gaps = []
    for tk, d in zip(out["ticker"], ann):
        f = filings.loc[filings["ticker"] == tk, "filing_date"]
        diff = (f - d).dt.days
        diff = diff[diff.abs() <= search]
        gaps.append(diff.iloc[diff.abs().argmin()] if len(diff) else np.nan)
    out["sec_gap_days"] = gaps
    out["date_check"] = np.where(out["sec_gap_days"].isna(), "no 8-K",
                                 np.where(out["sec_gap_days"].between(-before, after), "match", "mismatch"))
    return out


# ---------------------------------------------------------------------------
# Task 3: normal returns, abnormal returns (AR) and cumulative abnormal returns (CAR)
# ---------------------------------------------------------------------------
NORMAL_MODELS = ("market_model", "market_adjusted", "mean_adjusted")


def fit_market_model(stock, market):
    """
    Ordinary least squares fit of  stock return = alpha + beta x market return + error
    on one estimation window. Returns alpha, beta, the residual standard deviation
    (n - 2 degrees of freedom; the per-day noise used for t-tests in Task 4), R-squared
    and the number of days used.
    """
    y = np.asarray(stock, dtype=float)
    x = np.asarray(market, dtype=float)
    beta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    alpha = y.mean() - beta * x.mean()
    resid = y - alpha - beta * x
    return {"alpha": alpha, "beta": beta, "sigma": np.sqrt((resid ** 2).sum() / (len(y) - 2)),
            "r2": 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum(), "n_est": len(y)}


def normal_return(params, market, model="market_model"):
    """The return expected without the event, under the chosen normal-return model."""
    if model == "market_model":
        return params["alpha"] + params["beta"] * market
    if model == "market_adjusted":
        return market                                   # beta fixed at 1, alpha at 0
    if model == "mean_adjusted":
        return params["mean"] + 0 * market              # the stock's own estimation-window mean
    raise ValueError(f"model must be one of {NORMAL_MODELS}")


def compute_abnormal_returns(windows, stock_returns, market_returns, design, model="market_model"):
    """
    Daily abnormal returns for every event and every day of its event window.

    For each event: fit the model on the estimation window ONLY (est_first..est_last),
    then for each event-window day: AR = actual return - normal return.

    Returns (ar, params):
      ar     : one row per event and event day, with rel_day (-10..+10), date, the stock
               and market return, the normal return and the abnormal return
      params : one row per event with alpha, beta, sigma, R-squared and estimation-window size
    """
    cal = stock_returns.index
    rows, params = [], []
    for w in windows.itertuples(index=False):
        r, m = stock_returns[w.ticker], market_returns
        est = cal[(cal >= w.est_first) & (cal <= w.est_last)]
        p = fit_market_model(r.loc[est], m.loc[est])
        p["mean"] = r.loc[est].mean()
        params.append({"event_id": w.event_id, **p})

        i0 = cal.get_loc(w.day0)
        days = cal[i0 + design.event_start: i0 + design.event_end + 1]
        nr = normal_return(p, m.loc[days].values, model)
        rows.append(pd.DataFrame({
            "event_id": w.event_id, "ticker": w.ticker,
            "rel_day": np.arange(design.event_start, design.event_end + 1), "date": days,
            "ret": r.loc[days].values, "mkt": m.loc[days].values, "normal": nr,
            "ar": r.loc[days].values - nr,
        }))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(params)


def car(ar, first, last):
    """CAR(first, last) for every event: the sum of its daily ARs from rel_day first to last, inclusive."""
    sel = ar[(ar["rel_day"] >= first) & (ar["rel_day"] <= last)]
    out = sel.groupby("event_id").agg(car=("ar", "sum"), days=("ar", "size"))
    assert (out["days"] == last - first + 1).all(), "some event is missing days in the CAR window"
    return out["car"].rename(f"CAR({first:+d},{last:+d})")


def event_car_table(ar, windows, design):
    """Event-level table: each event's description, its day-0 AR and the CARs fixed in the design."""
    cars = pd.concat([car(ar, *design.main_car), car(ar, *design.drift_car),
                      car(ar, design.event_start, -2), car(ar, design.event_start, design.event_end)], axis=1)
    ar0 = ar[ar["rel_day"] == 0].set_index("event_id")["ar"].rename("AR(0)")
    keep = ["event_id", "ticker", "day0", "timing", "surprise", "surprise_sign"]
    return windows[keep].set_index("event_id").join(ar0).join(cars).reset_index()


def aar_table(ar, by=None):
    """
    Aggregated table by event day: average abnormal return (AAR) across events, the
    number of events, and the cumulative average abnormal return (CAAR) from the
    first event-window day. `by` = a column of `ar` to split on (e.g. surprise_sign).
    """
    keys = ([by] if by else []) + ["rel_day"]
    out = ar.groupby(keys)["ar"].agg(AAR="mean", n="size").reset_index()
    out["CAAR"] = out.groupby(by)["AAR"].cumsum() if by else out["AAR"].cumsum()
    return out
