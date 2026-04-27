"""
Rolling formation/trading backtest, following Tadi & Witzany (2025).

Each cycle = 3-week formation + 1-week trading. Window steps forward 1 week.

Within one cycle:
  Formation period
    1. For each alt vs BTC: OLS through origin -> beta, spread = BTC - beta * alt.
    2. ADF or KSS test for stationarity. Compute Kendall tau between BTC and alt.
    3. From tests-passing alts, pick the two with the largest tau -> (sym1, sym2).
    4. Fit Normal / Student-t / Cauchy marginals on the two formation spreads, AIC-select.
    5. PIT to (u, v); fit Gaussian / Student-t / Clayton / Gumbel / Frank copulas, AIC-select.
  Trading period
    6. Use the *frozen* beta to recompute spreads on trading prices.
    7. Apply *frozen* marginal CDFs to get (u_t, v_t).
    8. Apply *frozen* copula to get conditional probabilities h(u|v) and h(v|u).
    9. Per Tables 3 and 4, generate open/close signals; size each leg at fixed
       USDT notional; force-close at end of week. Track PnL.

Outputs equity-by-cycle, per-cycle log, trade log, and summary metrics.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau

from spread_adf_kss import (
    KSS_CRITICAL_10PCT,
    adf_on_series,
    discover_csv_paths,
    kss_auxiliary_t_stat,
    load_close_series,
    merge_on_timestamp,
    ols_btc_on_alt_through_origin,
)
from copula_marginals import DEFAULT_DISTRIBUTIONS, _cdf_from_fit, _fit_distribution
from copula_fitting import fit_copulas_and_select


HOURS_PER_WEEK = 168
FORMATION_HOURS = 3 * HOURS_PER_WEEK   # 504
TRADING_HOURS = 1 * HOURS_PER_WEEK     # 168
ADF_PASS_MAX_PVALUE = 0.10              # paper: 10% sig level for ADF
INITIAL_CAPITAL = 20_000.0
TAKER_FEE = 0.0004                      # Binance USDT-margined taker
EPS = 1e-6


# ---------- formation helpers ----------

def _hours_to_rows(timeframe: str, hours: int) -> int:
    if timeframe == "1h":
        return hours
    if timeframe == "5m":
        return hours * 12
    if timeframe == "1m":
        return hours * 60
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _best_marginal(spread: np.ndarray) -> dict[str, Any]:
    """Try Normal/Student-t/Cauchy, return the AIC-min fit dict (compatible with _cdf_from_fit)."""
    best: dict[str, Any] | None = None
    for dist in DEFAULT_DISTRIBUTIONS:
        try:
            fit = _fit_distribution(spread, dist)
        except Exception:
            continue
        if best is None or fit["aic"] < best["aic"]:
            best = fit
    if best is None:
        raise RuntimeError("All marginal fits failed.")
    return best


def _select_pair_in_formation(
    formation_panel: pd.DataFrame, gate: str
) -> list[dict[str, Any]] | None:
    """
    Return [{sym, beta, spread, tau}, {...}] for the top-2 alts under the chosen gate,
    or None if fewer than 2 pass.
    """
    btc = formation_panel["BTC_USDT"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for sym in formation_panel.columns:
        if sym == "BTC_USDT":
            continue
        alt = formation_panel[sym].to_numpy(dtype=float)
        if not np.all(np.isfinite(alt)) or float(np.dot(alt, alt)) <= 0.0:
            continue
        try:
            spread, beta, _ = ols_btc_on_alt_through_origin(btc, alt)
        except Exception:
            continue
        adf = adf_on_series(spread)
        kss = kss_auxiliary_t_stat(spread)

        kss_pass = bool(np.isfinite(kss["kss_t_stat"]) and kss["kss_t_stat"] < KSS_CRITICAL_10PCT)
        adf_pass = bool(adf["adf_pvalue"] < ADF_PASS_MAX_PVALUE)
        if gate == "kss" and not kss_pass:
            continue
        if gate == "adf" and not adf_pass:
            continue

        tau_res = kendalltau(btc, alt, nan_policy="omit")
        tau = float(tau_res.correlation) if np.isfinite(tau_res.correlation) else float("nan")

        rows.append({"sym": sym, "beta": float(beta), "spread": spread, "tau": tau})

    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r["tau"], reverse=True)
    return rows[:2]


def _fit_pair_models(top2: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit marginals on each formation spread, then a copula on (u, v)."""
    s1 = top2[0]["spread"]
    s2 = top2[1]["spread"]
    m1 = _best_marginal(s1)
    m2 = _best_marginal(s2)

    u = np.clip(_cdf_from_fit(s1, m1), EPS, 1.0 - EPS)
    v = np.clip(_cdf_from_fit(s2, m2), EPS, 1.0 - EPS)
    _, best_copula = fit_copulas_and_select(pd.DataFrame({"u": u, "v": v}))

    return {
        "sym1": top2[0]["sym"],
        "sym2": top2[1]["sym"],
        "beta1": top2[0]["beta"],
        "beta2": top2[1]["beta"],
        "tau1": top2[0]["tau"],
        "tau2": top2[1]["tau"],
        "marginal1": m1,
        "marginal2": m2,
        "copula": best_copula,
    }


# ---------- copula conditionals (consistent with copula_fitting.py keys) ----------

def _h_conditionals(
    u: np.ndarray, v: np.ndarray, copula_info: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute h(u|v) and h(v|u) for the AIC-selected copula. u, v in (eps, 1-eps)."""
    name = copula_info["copula"]
    p = copula_info["params"]

    if name == "gaussian":
        rho = float(p["rho"])
        x = stats.norm.ppf(u)
        y = stats.norm.ppf(v)
        d = np.sqrt(1.0 - rho * rho)
        return stats.norm.cdf((x - rho * y) / d), stats.norm.cdf((y - rho * x) / d)

    if name == "student_t":
        rho, df = float(p["rho"]), float(p["df"])
        x = stats.t.ppf(u, df=df)
        y = stats.t.ppf(v, df=df)
        sx = np.sqrt((df + y * y) * (1.0 - rho * rho) / (df + 1.0))
        sy = np.sqrt((df + x * x) * (1.0 - rho * rho) / (df + 1.0))
        h12 = stats.t.cdf((x - rho * y) / sx, df=df + 1.0)
        h21 = stats.t.cdf((y - rho * x) / sy, df=df + 1.0)
        return h12, h21

    if name == "clayton":
        theta = float(p["theta"])
        a = np.power(u, -theta) + np.power(v, -theta) - 1.0
        h12 = np.power(a, -1.0 / theta - 1.0) * np.power(v, -theta - 1.0)
        h21 = np.power(a, -1.0 / theta - 1.0) * np.power(u, -theta - 1.0)
        return h12, h21

    if name == "gumbel":
        theta = float(p["theta"])
        lu = -np.log(u)
        lv = -np.log(v)
        a = np.power(lu, theta) + np.power(lv, theta)
        c = np.exp(-np.power(a, 1.0 / theta))
        h12 = c * np.power(a, 1.0 / theta - 1.0) * np.power(lv, theta - 1.0) / v
        h21 = c * np.power(a, 1.0 / theta - 1.0) * np.power(lu, theta - 1.0) / u
        return h12, h21

    if name == "frank":
        theta = float(p["theta"])
        if abs(theta) < 1e-8:
            return u.copy(), v.copy()
        e0 = np.exp(-theta)
        eu = np.exp(-theta * u)
        ev = np.exp(-theta * v)
        denom = (e0 - 1.0) + (eu - 1.0) * (ev - 1.0)
        h12 = ev * (eu - 1.0) / denom
        h21 = eu * (ev - 1.0) / denom
        return h12, h21

    raise ValueError(f"Unsupported copula in h_conditionals: {name}")


# ---------- trading-period simulator ----------

@dataclass
class Trade:
    cycle: int
    pair: tuple[str, str]
    side: int          # +1 = long S1 / short S2, -1 = short S1 / long S2
    open_idx: int
    close_idx: int
    open_p1: float
    open_p2: float
    close_p1: float
    close_p2: float
    q1: float
    q2: float
    gross_pnl: float
    fee_paid: float
    net_pnl: float
    forced_close: bool


def _simulate_trading_week(
    trading_panel: pd.DataFrame,
    fit: dict[str, Any],
    alpha_open: float,
    alpha_close: float,
    notional_per_leg: float,
    fee: float,
    cycle: int,
    allow_flip: bool = False,
) -> tuple[list[Trade], float]:
    """
    Apply Tables 3 + 4 over one trading week.
    Sizing: Q_i = notional_per_leg / P_i^open (fixed for the whole week).
    BTC cancels because long S1 + short S2 = (+BTC -beta1*P1) + (-BTC + beta2*P2)
      net coin trades: -beta1 * P1, +beta2 * P2.
    Equivalent dollar-neutral version: short Q1 of coin1 + long Q2 of coin2.

    allow_flip:
        False (paper Table 3, default): only red close-zone or week-end can close
            a position; an opposite-green signal while in a position is IGNORED.
            This is paper-faithful but has the gap that (u,v) can jump corner-to-corner
            without crossing the red zone, leaving you holding the wrong side.
        True (improved): green-corner signals always set the target position.
            If currently in +1 and the bar enters short-corner, close +1 and open -1
            on the same bar. Closes still happen on red or week-end.
    """
    sym1, sym2 = fit["sym1"], fit["sym2"]
    beta1, beta2 = fit["beta1"], fit["beta2"]

    btc = trading_panel["BTC_USDT"].to_numpy(dtype=float)
    p1 = trading_panel[sym1].to_numpy(dtype=float)
    p2 = trading_panel[sym2].to_numpy(dtype=float)

    # Trading-period spreads use frozen formation beta.
    s1 = btc - beta1 * p1
    s2 = btc - beta2 * p2

    u = np.clip(_cdf_from_fit(s1, fit["marginal1"]), EPS, 1.0 - EPS)
    v = np.clip(_cdf_from_fit(s2, fit["marginal2"]), EPS, 1.0 - EPS)
    h12, h21 = _h_conditionals(u, v, fit["copula"])

    # Fixed quantities for the week.
    q1 = notional_per_leg / float(p1[0])
    q2 = notional_per_leg / float(p2[0])

    position = 0  # +1, -1, or 0
    open_idx = -1
    open_p1 = open_p2 = 0.0
    trades: list[Trade] = []
    week_pnl = 0.0

    n = len(trading_panel)
    for t in range(n):
        is_last = (t == n - 1)

        sig_open_long = (h12[t] < alpha_open) and (h21[t] > 1.0 - alpha_open)
        sig_open_short = (h12[t] > 1.0 - alpha_open) and (h21[t] < alpha_open)
        sig_flat = (abs(h12[t] - 0.5) < alpha_close) and (abs(h21[t] - 0.5) < alpha_close)

        # In flip mode, an opposite-green signal also forces close-then-reopen.
        flip_close = allow_flip and (
            (position == +1 and sig_open_short) or
            (position == -1 and sig_open_long)
        )
        should_close = (position != 0) and (sig_flat or is_last or flip_close)

        if should_close:
            if position == +1:
                gross = q1 * (open_p1 - p1[t]) + q2 * (p2[t] - open_p2)
            else:
                gross = q1 * (p1[t] - open_p1) + q2 * (open_p2 - p2[t])
            close_fee = fee * (q1 * float(p1[t]) + q2 * float(p2[t]))
            open_fee = fee * (q1 * open_p1 + q2 * open_p2)
            net = gross - close_fee - open_fee
            week_pnl += net
            trades.append(
                Trade(
                    cycle=cycle,
                    pair=(sym1, sym2),
                    side=position,
                    open_idx=open_idx,
                    close_idx=t,
                    open_p1=float(open_p1),
                    open_p2=float(open_p2),
                    close_p1=float(p1[t]),
                    close_p2=float(p2[t]),
                    q1=float(q1),
                    q2=float(q2),
                    gross_pnl=float(gross),
                    fee_paid=float(open_fee + close_fee),
                    net_pnl=float(net),
                    forced_close=bool(is_last and not sig_flat and not flip_close),
                )
            )
            position = 0
            open_idx = -1

        # Open path (also fires after a flip-close on the same bar).
        if position == 0 and not is_last:
            if sig_open_long:
                position = +1
            elif sig_open_short:
                position = -1
            if position != 0:
                open_idx = t
                open_p1 = float(p1[t])
                open_p2 = float(p2[t])

    return trades, week_pnl


def _simulate_trading_week_level(
    trading_panel: pd.DataFrame,
    fit: dict[str, Any],
    cmi_open: float,
    cmi_close: float,
    notional_per_leg: float,
    fee: float,
    cycle: int,
) -> tuple[list[Trade], float]:
    """
    Level-based signals (Xie & Wu 2013, paper Eq. 5).

    Cumulative mispricing index, reset to 0 at start of trading week:
        CMI^{1|2}_t = sum_{s=0..t} (h^{1|2}_s - 0.5)
        CMI^{2|1}_t = sum_{s=0..t} (h^{2|1}_s - 0.5)

    Open long S1 / short S2:  CMI^{1|2}_t < -t_o  AND  CMI^{2|1}_t > +t_o
        (asset 1 has been persistently underpriced -> long S1; asset 2 overpriced -> short S2)
    Open short S1 / long S2:  CMI^{1|2}_t > +t_o  AND  CMI^{2|1}_t < -t_o

    Close long S1 / short S2 when both CMIs revert past +/- t_c:
        CMI^{2|1}_t < +t_c  AND  CMI^{1|2}_t > -t_c
    Close short S1 / long S2 when:
        CMI^{1|2}_t < +t_c  AND  CMI^{2|1}_t > -t_c

    Position rule and sizing identical to return-based simulator.
    """
    sym1, sym2 = fit["sym1"], fit["sym2"]
    beta1, beta2 = fit["beta1"], fit["beta2"]

    btc = trading_panel["BTC_USDT"].to_numpy(dtype=float)
    p1 = trading_panel[sym1].to_numpy(dtype=float)
    p2 = trading_panel[sym2].to_numpy(dtype=float)

    s1 = btc - beta1 * p1
    s2 = btc - beta2 * p2

    u = np.clip(_cdf_from_fit(s1, fit["marginal1"]), EPS, 1.0 - EPS)
    v = np.clip(_cdf_from_fit(s2, fit["marginal2"]), EPS, 1.0 - EPS)
    h12, h21 = _h_conditionals(u, v, fit["copula"])

    # Cumulative mispricing indices (start at 0 at t=0).
    cmi12 = np.cumsum(h12 - 0.5)
    cmi21 = np.cumsum(h21 - 0.5)

    q1 = notional_per_leg / float(p1[0])
    q2 = notional_per_leg / float(p2[0])

    position = 0
    open_idx = -1
    open_p1 = open_p2 = 0.0
    trades: list[Trade] = []
    week_pnl = 0.0

    n = len(trading_panel)
    for t in range(n):
        is_last = (t == n - 1)

        sig_open_long = (cmi12[t] < -cmi_open) and (cmi21[t] > cmi_open)
        sig_open_short = (cmi12[t] > cmi_open) and (cmi21[t] < -cmi_open)
        # Close logic depends on which side we are on
        sig_close_long_pos = (cmi21[t] < cmi_close) and (cmi12[t] > -cmi_close)
        sig_close_short_pos = (cmi12[t] < cmi_close) and (cmi21[t] > -cmi_close)
        sig_flat = (
            (position == +1 and sig_close_long_pos)
            or (position == -1 and sig_close_short_pos)
        )

        if position != 0 and (sig_flat or is_last):
            if position == +1:
                gross = q1 * (open_p1 - p1[t]) + q2 * (p2[t] - open_p2)
            else:
                gross = q1 * (p1[t] - open_p1) + q2 * (open_p2 - p2[t])
            close_fee = fee * (q1 * float(p1[t]) + q2 * float(p2[t]))
            open_fee = fee * (q1 * open_p1 + q2 * open_p2)
            net = gross - close_fee - open_fee
            week_pnl += net
            trades.append(
                Trade(
                    cycle=cycle,
                    pair=(sym1, sym2),
                    side=position,
                    open_idx=open_idx,
                    close_idx=t,
                    open_p1=float(open_p1),
                    open_p2=float(open_p2),
                    close_p1=float(p1[t]),
                    close_p2=float(p2[t]),
                    q1=float(q1),
                    q2=float(q2),
                    gross_pnl=float(gross),
                    fee_paid=float(open_fee + close_fee),
                    net_pnl=float(net),
                    forced_close=bool(is_last and not sig_flat),
                )
            )
            position = 0
            open_idx = -1

        if position == 0 and not is_last:
            if sig_open_long:
                position = +1
            elif sig_open_short:
                position = -1
            if position != 0:
                open_idx = t
                open_p1 = float(p1[t])
                open_p2 = float(p2[t])

    return trades, week_pnl


# ---------- top-level orchestration ----------

def _safe_panel(timeframe: str) -> pd.DataFrame:
    paths = discover_csv_paths(timeframe)
    series_map = {sym: load_close_series(p) for sym, p in paths.items()}
    panel = merge_on_timestamp(series_map)
    return panel


def run_backtest(
    timeframe: str = "1h",
    formation_hours: int = FORMATION_HOURS,
    trading_hours: int = TRADING_HOURS,
    alpha_open: float = 0.10,
    alpha_close: float = 0.10,
    notional_per_leg: float = INITIAL_CAPITAL,
    fee: float = TAKER_FEE,
    gate: str = "kss",
    signal_mode: str = "return",
    cmi_open: float = 1.0,
    cmi_close: float = 0.0,
    allow_flip: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the full rolling backtest.

    Args:
        timeframe: data folder under data/.
        formation_hours, trading_hours: window lengths in HOURS (auto-converted to rows).
        alpha_open, alpha_close: trading rule thresholds (paper: 0.10 baseline).
            Used only when signal_mode == "return".
        notional_per_leg: USDT capital per leg per trading week (paper: 20,000).
        fee: per-leg per-trade fee (paper: 0.0004 taker).
        gate: 'kss' (paper Table 6 main row) or 'adf'.
        signal_mode: 'return' (paper Eq. 4 + Table 3) or 'level' (paper Eq. 5, CMI-based).
        cmi_open, cmi_close: level-based thresholds t_o and t_c (paper defaults: 1.0, 0.0).
            Used only when signal_mode == "level".
    """
    if signal_mode not in ("return", "level"):
        raise ValueError(f"signal_mode must be 'return' or 'level', got {signal_mode!r}")
    panel = _safe_panel(timeframe)
    form_rows = _hours_to_rows(timeframe, formation_hours)
    trade_rows = _hours_to_rows(timeframe, trading_hours)
    cycle_rows = form_rows + trade_rows

    n = len(panel)
    if verbose:
        print(f"Panel: {n} rows, {len(panel.columns)} cols, "
              f"formation={form_rows}, trading={trade_rows}, gate={gate}")

    cycle_log: list[dict[str, Any]] = []
    trade_log: list[Trade] = []
    equity_per_cycle: list[float] = []
    cumulative = 0.0

    cycle_idx = 0
    start = 0
    while start + cycle_rows <= n:
        formation = panel.iloc[start : start + form_rows]
        trading = panel.iloc[start + form_rows : start + cycle_rows]

        rec: dict[str, Any] = {
            "cycle": cycle_idx,
            "formation_start": int(formation.index[0]),
            "trading_start": int(trading.index[0]),
            "trading_end": int(trading.index[-1]),
        }

        top2 = None
        try:
            top2 = _select_pair_in_formation(formation, gate=gate)
        except Exception as exc:  # noqa: BLE001
            rec.update({"skipped": True, "reason": f"select error: {exc}", "pnl": 0.0})

        if top2 is None and "reason" not in rec:
            rec.update({"skipped": True, "reason": f"<2 pairs pass {gate}", "pnl": 0.0})

        if top2 is not None:
            try:
                fit = _fit_pair_models(top2)
                if signal_mode == "return":
                    trades, week_pnl = _simulate_trading_week(
                        trading, fit, alpha_open, alpha_close,
                        notional_per_leg, fee, cycle=cycle_idx,
                        allow_flip=allow_flip,
                    )
                else:
                    trades, week_pnl = _simulate_trading_week_level(
                        trading, fit, cmi_open, cmi_close,
                        notional_per_leg, fee, cycle=cycle_idx,
                    )
                cumulative += week_pnl
                trade_log.extend(trades)
                rec.update({
                    "skipped": False,
                    "sym1": fit["sym1"], "sym2": fit["sym2"],
                    "beta1": fit["beta1"], "beta2": fit["beta2"],
                    "tau1": fit["tau1"], "tau2": fit["tau2"],
                    "marginal1": fit["marginal1"]["distribution"],
                    "marginal2": fit["marginal2"]["distribution"],
                    "copula": fit["copula"]["copula"],
                    "n_trades": len(trades),
                    "n_forced": sum(1 for t in trades if t.forced_close),
                    "pnl": float(week_pnl),
                })
            except Exception as exc:  # noqa: BLE001
                rec.update({"skipped": True, "reason": f"fit/sim error: {exc}", "pnl": 0.0})

        equity_per_cycle.append(cumulative)
        cycle_log.append(rec)

        if verbose and (cycle_idx % 10 == 0 or rec.get("skipped")):
            tag = "SKIP" if rec.get("skipped") else f"{rec.get('sym1')}/{rec.get('sym2')}"
            print(f"  cycle {cycle_idx:3d}  {tag:>22s}  pnl={rec['pnl']:+.2f}  cum={cumulative:+.2f}")

        cycle_idx += 1
        start += trade_rows

    metrics = _compute_metrics(cycle_log, equity_per_cycle, notional_per_leg)
    cycle_df = pd.DataFrame(cycle_log)
    trade_df = pd.DataFrame([t.__dict__ for t in trade_log])

    return {
        "metrics": metrics,
        "cycle_log": cycle_df,
        "trade_log": trade_df,
        "equity_per_cycle": equity_per_cycle,
        "config": {
            "timeframe": timeframe,
            "formation_hours": formation_hours,
            "trading_hours": trading_hours,
            "signal_mode": signal_mode,
            "alpha_open": alpha_open,
            "alpha_close": alpha_close,
            "cmi_open": cmi_open,
            "cmi_close": cmi_close,
            "allow_flip": allow_flip,
            "notional_per_leg": notional_per_leg,
            "fee": fee,
            "gate": gate,
        },
    }


def _compute_metrics(
    cycle_log: list[dict[str, Any]],
    equity_per_cycle: list[float],
    notional: float,
) -> dict[str, Any]:
    pnl = np.array([c["pnl"] for c in cycle_log], dtype=float)
    n_weeks = int(len(pnl))
    weekly_ret = pnl / notional  # rough: weekly PnL relative to per-leg capital
    eq = np.array(equity_per_cycle, dtype=float)

    if n_weeks > 0 and np.std(weekly_ret) > 0:
        sharpe = float(np.mean(weekly_ret) / np.std(weekly_ret) * np.sqrt(52))
    else:
        sharpe = float("nan")

    total_pnl = float(eq[-1]) if n_weeks > 0 else 0.0
    total_return = total_pnl / notional
    if n_weeks > 0:
        annual_return = (1.0 + total_return) ** (52.0 / n_weeks) - 1.0
    else:
        annual_return = float("nan")
    annual_vol = float(np.std(weekly_ret) * np.sqrt(52)) if n_weeks > 0 else float("nan")

    if n_weeks > 0:
        running_max = np.maximum.accumulate(eq)
        dd = (eq - running_max) / notional
        max_dd = float(np.min(dd))
    else:
        max_dd = 0.0
    romad = (total_return / abs(max_dd)) if max_dd < 0 else float("nan")

    n_skipped = int(sum(1 for c in cycle_log if c.get("skipped")))
    n_trades = int(sum(c.get("n_trades", 0) for c in cycle_log))

    return {
        "n_cycles": n_weeks,
        "n_skipped": n_skipped,
        "n_trades": n_trades,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "romad": romad,
    }


def save_results(result: dict[str, Any], out_subdir: str = "rolling_backtest") -> str:
    tf = result["config"]["timeframe"]
    out_dir = os.path.join("result", f"result_{tf}", out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": result["metrics"], "config": result["config"]},
            f,
            indent=2,
            default=float,
        )

    result["cycle_log"].to_csv(
        os.path.join(out_dir, "cycle_log.csv"), index=False, encoding="utf-8-sig",
    )
    if not result["trade_log"].empty:
        result["trade_log"].to_csv(
            os.path.join(out_dir, "trade_log.csv"), index=False, encoding="utf-8-sig",
        )
    pd.DataFrame({"equity": result["equity_per_cycle"]}).to_csv(
        os.path.join(out_dir, "equity_per_cycle.csv"), index=True, encoding="utf-8-sig",
    )
    return out_dir


def main():
    result = run_backtest(timeframe="1h", alpha_open=0.10, alpha_close=0.10, gate="kss")
    out_dir = save_results(result)
    print("\n=== Summary ===")
    for k, v in result["metrics"].items():
        if isinstance(v, float):
            print(f"  {k:>16s}: {v:.4f}")
        else:
            print(f"  {k:>16s}: {v}")
    print(f"\nWritten to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
