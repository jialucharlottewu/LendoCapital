"""
Week 5 backtest engine.

Turns target weights (decided on signal dates) into daily holdings, trades and returns,
under one explicit execution assumption, and records everything needed to audit a run.

INPUT CONTRACT
  prices          wide table (trading days x tickers) of ADJUSTED closes (splits and
                  dividends), so price changes are total returns. Missing = no price that day.
  target_weights  wide table (signal dates x tickers): the portfolio a strategy wants,
                  computed ONLY from data up to and including the signal date's close.
                  Weights are fractions of the portfolio's value; 1 - sum(weights) is cash.
  config          execution lag, costs, cash rate and edge-case rules (BacktestConfig).

EXECUTION ASSUMPTION
  A signal formed at the close of day t is traded at the close of day t + exec_lag
  (default 1: the next trading day's close). Trading at the same close that produced the
  signal (exec_lag = 0) uses a price nobody could have traded on after seeing the signal,
  so it is refused unless allow_same_close=True is passed on purpose (used only to
  reconcile with earlier research that made that assumption).

EDGE CASES (policies, all recorded in the diagnostics)
  new entrant      a ticker can only be bought once it has a price on the execution day;
                   whether it has enough history to be in the signal is the strategy's job
  missing price    a held name with no price on a day is carried at its last price (0%
                   return) for up to max_stale_days; on an execution day it cannot be
                   traded, so it is frozen at its current weight and the other targets are
                   scaled to fit
  delisting        a held name with no price for more than max_stale_days, or past its last
                   price ever, is sold into cash at its last price
  untradable target weight aimed at a name with no price on the execution day goes to cash
  cash             1 - sum(weights); earns cash_rate (annual, default 0)
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    exec_lag: int = 1                 # trading days from signal close to execution close
    allow_same_close: bool = False    # must be True to permit exec_lag = 0
    cost_bps: float = 0.0             # flat cost per unit traded (Task 3 adds richer models)
    cost_fn: object = None            # optional f(trades, date, context) -> cost as a fraction of NAV
    cash_rate: float = 0.0            # annual return on cash
    max_stale_days: int = 5           # missing-price days tolerated before a forced exit
    initial_nav: float = 1.0


@dataclass
class BacktestResult:
    daily: pd.DataFrame               # per day: nav, gross/net return, cost, cash weight, names held
    rebalances: pd.DataFrame          # per rebalance: dates, turnover, cost, names, cash, diagnostics
    trades: pd.DataFrame              # per rebalance and ticker: weight before, target, trade
    holdings: pd.DataFrame            # per rebalance and ticker: weight and value after trading
    events: pd.DataFrame = field(default_factory=pd.DataFrame)   # forced exits and frozen names


def execution_dates(calendar, signal_dates, exec_lag):
    """Map each signal date to its execution date, exec_lag trading days later (NaT if past the data)."""
    cal = pd.DatetimeIndex(calendar)
    out = {}
    for s in pd.DatetimeIndex(signal_dates):
        if s not in cal:
            raise ValueError(f"signal date {s.date()} is not a trading day")
        i = cal.get_loc(s) + exec_lag
        out[s] = cal[i] if i < len(cal) else pd.NaT
    return pd.Series(out, name="exec_date")


def run_backtest(prices, target_weights, config=None):
    """
    Simulate the portfolio day by day.

    Each day: every position moves with its own return (weights drift), cash earns the
    cash rate, and the gross return is recorded. On an execution day, after that day's
    return, the portfolio is traded from its drifted weights to the targets at the close,
    costs are charged, and the trades, holdings and diagnostics are recorded.
    """
    cfg = config or BacktestConfig()
    if cfg.exec_lag < 0:
        raise ValueError("exec_lag cannot be negative")
    if cfg.exec_lag == 0 and not cfg.allow_same_close:
        raise ValueError("exec_lag=0 trades on the same close that formed the signal (look-ahead). "
                         "Pass allow_same_close=True only to reproduce research that assumed it.")

    prices = prices.sort_index()
    cal = prices.index
    tw = target_weights.reindex(columns=prices.columns).fillna(0.0).sort_index()
    ex = execution_dates(cal, tw.index, cfg.exec_lag)
    if (ex.dropna().index >= ex.dropna().values).any() and cfg.exec_lag > 0:
        raise AssertionError("an execution date is not after its signal date")
    exec_map = {d: s for s, d in ex.dropna().items()}

    has_px = prices.notna()
    last_px = prices.ffill()
    miss = (~has_px).astype(int).cumsum()
    stale = miss - miss.where(has_px).ffill().fillna(0)                 # consecutive missing days
    last_valid = prices.apply(lambda s: s.last_valid_index())
    rf_day = (1 + cfg.cash_rate) ** (1 / 252) - 1
    rets = last_px.pct_change(fill_method=None).fillna(0.0)             # stale days -> 0% return

    start = min(exec_map) if exec_map else None
    if start is None:
        raise ValueError("no execution date falls inside the price data")

    w = pd.Series(0.0, index=prices.columns)     # weights at the previous close
    cash = 1.0
    nav = cfg.initial_nav
    daily, rebal, trades, holds, events = [], [], [], [], []

    for d in cal[cal >= start]:
        # 1) the day's return on the positions held overnight
        held = w[w != 0].index
        r = rets.loc[d, held]
        gross = float((w[held] * r).sum() + cash * rf_day)
        w.loc[held] = w[held] * (1 + r) / (1 + gross)
        cash = cash * (1 + rf_day) / (1 + gross)
        nav_before_cost = nav * (1 + gross)

        # 2) forced exits: held names with no price for too long, or past their last price
        for tk in w[w != 0].index:
            gone = d > last_valid[tk]
            too_stale = stale.loc[d, tk] > cfg.max_stale_days
            if gone or too_stale:
                events.append({"date": d, "ticker": tk, "event": "delisted" if gone else "stale price",
                               "weight moved to cash": w[tk]})
                cash += w[tk]
                w[tk] = 0.0

        cost = 0.0
        # 3) rebalance at the close on execution days
        if d in exec_map:
            s = exec_map[d]
            target = tw.loc[s].copy()
            tradable = has_px.loc[d]
            frozen = w[(w != 0) & ~tradable]                       # held, but no price today
            untradable = target[(target != 0) & ~tradable & ~target.index.isin(frozen.index)]
            target[untradable.index] = 0.0
            target[frozen.index] = 0.0
            room = 1.0 - frozen.sum()
            if target.sum() > room and target.sum() > 0:           # make room for frozen names
                target *= room / target.sum()
            target[frozen.index] = frozen
            for tk, x in frozen.items():
                events.append({"date": d, "ticker": tk, "event": "frozen (no price on execution day)",
                               "weight moved to cash": 0.0})

            trade = target - w
            traded = float(trade.abs().sum())
            if cfg.cost_fn is not None:
                cost = float(cfg.cost_fn(trade, d, {"prices": prices, "weights_before": w.copy()}))
            else:
                cost = cfg.cost_bps / 1e4 * traded

            before = w.copy()
            w = target
            cash = 1.0 - float(w.sum())
            names = w[w != 0].index.union(before[before != 0].index)
            trades.append(pd.DataFrame({"signal_date": s, "exec_date": d, "ticker": names,
                                        "weight_before": before[names].values, "target_weight": w[names].values,
                                        "trade": trade[names].values}))
            nav_after = nav_before_cost * (1 - cost)
            hold = w[w != 0]
            holds.append(pd.DataFrame({"exec_date": d, "ticker": hold.index, "weight": hold.values,
                                       "value": hold.values * nav_after}))
            rebal.append({"signal_date": s, "exec_date": d, "names_held": int((w != 0).sum()),
                          "cash_weight": cash, "traded (sum |trade|)": traded,
                          "cost (fraction of NAV)": cost,
                          "untradable target weight": float(untradable.sum()),
                          "frozen weight": float(frozen.sum()), "nav_after": nav_after})

        nav_new = nav_before_cost * (1 - cost)
        daily.append({"date": d, "gross_return": gross, "cost": cost, "net_return": nav_new / nav - 1,
                      "nav": nav_new, "cash_weight": cash, "names_held": int((w != 0).sum())})
        nav = nav_new

    skipped = ex[ex.isna()]
    res = BacktestResult(
        daily=pd.DataFrame(daily).set_index("date"),
        rebalances=pd.DataFrame(rebal),
        trades=pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(),
        holdings=pd.concat(holds, ignore_index=True) if holds else pd.DataFrame(),
        events=pd.DataFrame(events, columns=["date", "ticker", "event", "weight moved to cash"]),
    )
    res.daily.attrs["signals_past_end"] = list(skipped.index)
    return res


# ---------------------------------------------------------------------------
# Strategies used for verification
# ---------------------------------------------------------------------------
def month_end_dates(calendar, start=None, end=None):
    """Last trading day of each month in the calendar."""
    cal = pd.Series(pd.DatetimeIndex(calendar), index=pd.DatetimeIndex(calendar))
    me = cal.groupby([cal.index.year, cal.index.month]).max().values
    me = pd.DatetimeIndex(me)
    if start is not None:
        me = me[me >= pd.Timestamp(start)]
    if end is not None:
        me = me[me <= pd.Timestamp(end)]
    return me


def equal_weight_targets(prices, signal_dates, min_history=252, max_missing=5):
    """
    1/N across every stock that, at the signal date's close,
      - has a price that day (so delisted names drop out), and
      - has at least min_history - max_missing daily returns in its last min_history
        trading days (so new listings join only after about a year of history, and a
        brief data gap does not throw a stock out for a whole year).
    Uses data up to the signal date only.
    """
    rets = prices.pct_change(fill_method=None)
    rows = {}
    for s in signal_dates:
        window = rets.loc[:s].tail(min_history)
        ok = (window.notna().sum() >= min_history - max_missing) & prices.loc[s].notna()
        eligible = ok[ok].index
        rows[s] = pd.Series(1.0 / len(eligible), index=eligible)
    return pd.DataFrame(rows).T.fillna(0.0)
