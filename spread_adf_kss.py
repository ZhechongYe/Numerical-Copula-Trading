from __future__ import annotations

import glob
import json
import os
import sys
import re
import time
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import kendalltau
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller

TIMEFRAME = "1h"
DATA_DIR = "data"
RESULT_ROOT = "result"

# KSS
KSS_CRITICAL_10PCT = -1.92

# ADF
ADF_PASS_MAX_PVALUE = 0.05


def _parse_symbol_from_filename(path: str) -> str | None:
    base = os.path.basename(path)
    m = re.match(r"^(.+)_USDT_[^/]+\.csv$", base, re.IGNORECASE)
    if not m:
        return None
    return f"{m.group(1).upper()}_USDT"


def discover_csv_paths(timeframe: str) -> dict[str, str]:
    """Map symbol_id (e.g. ETH_USDT) to CSV path."""
    folder = os.path.join(DATA_DIR, timeframe)
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Data folder not found: {folder} (run fetch_bar_data.py to download OHLCV first)"
        )
    paths = glob.glob(os.path.join(folder, "*_USDT_*.csv"))
    out: dict[str, str] = {}
    for p in paths:
        sid = _parse_symbol_from_filename(p)
        if sid:
            out[sid] = p
    if "BTC_USDT" not in out:
        raise FileNotFoundError(f"No BTC_USDT CSV under {folder}")
    return out


def load_close_series(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path, usecols=["timestamp", "close"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        return pd.Series(dtype=float, name=os.path.basename(csv_path))
    s = df.set_index("timestamp")["close"].astype(float)
    s.name = os.path.basename(csv_path)
    return s


def merge_on_timestamp(series_by_symbol: dict[str, pd.Series]) -> pd.DataFrame:
    """Inner-join all symbols on timestamp index."""
    if not series_by_symbol:
        raise ValueError("No price series to merge")
    base = pd.concat(series_by_symbol, axis=1, join="inner")
    base.columns = list(series_by_symbol.keys())
    base = base.dropna()
    if base.empty:
        raise ValueError("Merged panel is empty; check timestamp overlap across CSVs")
    return base


def ols_btc_on_alt_through_origin(btc: np.ndarray, alt: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Slide specification: BTC_t = beta^i * P_t^i + residual (no intercept).

    Spread S_t^i = BTC_t - beta^i * P_t^i equals the OLS residual.
    Returns (spread, beta_hat, r_squared_uncentered from statsmodels).
    """
    alt = np.asarray(alt, dtype=float).ravel()
    btc = np.asarray(btc, dtype=float).ravel()
    denom = float(np.dot(alt, alt))
    if denom <= 0.0 or not np.isfinite(denom):
        raise ValueError("Alt price sum of squares is zero or invalid; cannot fit through origin.")
    x = alt.reshape(-1, 1)
    res = OLS(btc, x).fit()
    beta = float(res.params[0])
    spread = np.asarray(btc - beta * alt, dtype=float)
    return spread, beta, float(res.rsquared)


def adf_on_series(x: np.ndarray, maxlag: int | None = None) -> dict[str, Any]:
    """Augmented Dickey-Fuller test; default autolag='AIC'."""
    kw: dict[str, Any] = {"autolag": "AIC"}
    if maxlag is not None:
        kw["maxlag"] = maxlag
    stat, pvalue, usedlag, nobs, crit, icbest = adfuller(x, **kw)
    return {
        "adf_statistic": float(stat),
        "adf_pvalue": float(pvalue),
        "adf_used_lags": int(usedlag),
        "adf_nobs": int(nobs),
        "adf_critical_values": {str(k): float(v) for k, v in crit.items()},
        "adf_icbest": float(icbest) if icbest is not None and not np.isnan(icbest) else None,
    }


def kss_auxiliary_t_stat(spread: np.ndarray) -> dict[str, Any]:
    """
    KSS Taylor auxiliary regression (no drift, common slide specification):

        ΔS_t = δ * (S_{t-1})^3 + ε_t

    Returns the t-statistic for δ (left tail: more negative favors stationarity).
    """
    s = np.asarray(spread, dtype=float).ravel()
    ds = np.diff(s)
    s_lag = s[:-1]
    x = (s_lag**3).reshape(-1, 1)
    y = ds
    mask = np.isfinite(x.ravel()) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(y) < 30:
        return {"kss_t_stat": float("nan"), "kss_delta_hat": float("nan"), "kss_nobs": int(len(y))}
    res = OLS(y, x).fit()
    delta_hat = float(res.params[0])
    t_stat = float(res.tvalues[0])
    return {
        "kss_t_stat": t_stat,
        "kss_delta_hat": delta_hat,
        "kss_nobs": int(res.nobs),
        "kss_rsquared": float(res.rsquared),
    }


def _nls_ssr(params: np.ndarray, ds: np.ndarray, s_lag: np.ndarray, z2: np.ndarray) -> float:
    """Nonlinear SSR for optimizer demo: ΔS ≈ δ * S_{t-1}^3 * exp(-θ * z^2); z is standardized lagged spread."""
    delta, theta = float(params[0]), float(params[1])
    lin = delta * (s_lag**3)
    arg = np.clip(-theta * z2, -50.0, 50.0)
    pred = lin * np.exp(arg)
    r = ds - pred
    return float(np.dot(r, r))


def compare_nelder_mead_vs_bfgs(ds: np.ndarray, s_lag: np.ndarray) -> dict[str, Any]:
    """Compare Nelder-Mead vs BFGS: wall time, iteration/function counts, final SSR."""
    max_n = 8000
    if len(ds) > max_n:
        ds = ds[-max_n:]
        s_lag = s_lag[-max_n:]
    scale = float(np.std(s_lag) + 1e-12)
    z = s_lag / scale
    z2 = z * z
    # Initial δ from no-intercept OLS on lag^3 only; θ starts small
    x0_cube = (s_lag**3).reshape(-1, 1)
    ols = OLS(ds, x0_cube).fit()
    delta0 = float(ols.params[0]) if np.isfinite(ols.params[0]) else 0.0
    x0 = np.array([delta0, 0.05], dtype=float)

    def run(method: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        sol = optimize.minimize(
            lambda p: _nls_ssr(p, ds, s_lag, z2),
            x0,
            method=method,
            options={"maxiter": 3000},
        )
        elapsed = time.perf_counter() - t0
        nit = getattr(sol, "nit", None)
        nfev = getattr(sol, "nfev", None)
        return {
            "method": method,
            "success": bool(sol.success),
            "message": str(sol.message),
            "elapsed_sec": float(elapsed),
            "nit": int(nit) if nit is not None else None,
            "nfev": int(nfev) if nfev is not None else None,
            "final_ssr": float(sol.fun),
            "x": [float(sol.x[0]), float(sol.x[1])],
        }

    out: dict[str, Any] = {}
    for m in ("Nelder-Mead", "BFGS"):
        try:
            out[m.replace("-", "_").lower()] = run(m)
        except Exception as exc:  # noqa: BLE001
            out[m.replace("-", "_").lower()] = {"method": m, "error": str(exc)}
    return out


@dataclass
class PairResult:
    alt_symbol: str
    n_obs: int
    beta: float
    r_squared: float
    kendall_tau: float
    adf_statistic: float
    adf_pvalue: float
    adf_used_lags: int
    adf_pass_5pct: bool
    kss_t_stat: float
    kss_pass_10pct: bool
    kss_critical_10pct: float


def run_analysis(timeframe: str) -> tuple[list[PairResult], dict[str, Any]]:
    paths = discover_csv_paths(timeframe)
    btc_path = paths.pop("BTC_USDT")

    btc_s = load_close_series(btc_path)
    if len(btc_s) == 0:
        raise ValueError(f"BTC series is empty; check file: {btc_path}")
    series_map: dict[str, pd.Series] = {"BTC_USDT": btc_s}
    skipped: list[str] = []
    for sym, p in paths.items():
        s = load_close_series(p)
        if len(s) == 0:
            skipped.append(sym)
            continue
        series_map[sym] = s
    if skipped:
        print("Warning: empty CSV(s), skipped:", ", ".join(skipped))

    panel = merge_on_timestamp(series_map)
    btc = panel["BTC_USDT"].values.astype(float)

    meta_optimizer: dict[str, Any] = {}

    rows: list[PairResult] = []
    for col in panel.columns:
        if col == "BTC_USDT":
            continue
        alt = panel[col].values.astype(float)
        spread, beta, r2 = ols_btc_on_alt_through_origin(btc, alt)
        adf_info = adf_on_series(spread)
        kss_info = kss_auxiliary_t_stat(spread)

        ds = np.diff(spread)
        s_lag = spread[:-1]
        m = np.isfinite(ds) & np.isfinite(s_lag)
        meta_optimizer[col] = compare_nelder_mead_vs_bfgs(ds[m], s_lag[m])

        kss_t = float(kss_info.get("kss_t_stat", float("nan")))
        kss_pass = bool(np.isfinite(kss_t) and kss_t < KSS_CRITICAL_10PCT)
        adf_p = float(adf_info["adf_pvalue"])
        adf_pass = bool(adf_p < ADF_PASS_MAX_PVALUE)

        tau_res = kendalltau(btc, alt, nan_policy="omit")
        kendall_tau = float(tau_res.correlation) if np.isfinite(tau_res.correlation) else float("nan")

        rows.append(
            PairResult(
                alt_symbol=col,
                n_obs=int(len(spread)),
                beta=float(beta),
                r_squared=float(r2),
                kendall_tau=kendall_tau,
                adf_statistic=float(adf_info["adf_statistic"]),
                adf_pvalue=adf_p,
                adf_used_lags=int(adf_info["adf_used_lags"]),
                adf_pass_5pct=adf_pass,
                kss_t_stat=kss_t,
                kss_pass_10pct=kss_pass,
                kss_critical_10pct=KSS_CRITICAL_10PCT,
            )
        )

    rows.sort(key=lambda r: r.alt_symbol)

    eligible = [r for r in rows if r.kss_pass_10pct]
    eligible_sorted = sorted(eligible, key=lambda r: r.kendall_tau, reverse=True)
    top2 = [
        {
            "rank": i + 1,
            "alt_symbol": r.alt_symbol,
            "kendall_tau": r.kendall_tau,
            "adf_pvalue": r.adf_pvalue,
            "kss_t_stat": r.kss_t_stat,
            "beta": r.beta,
        }
        for i, r in enumerate(eligible_sorted[:2])
    ]

    meta = {
        "timeframe": timeframe,
        "regression": "BTC_t = beta^i * alt_t + residual (through origin; no intercept)",
        "spread": "S_t^i = BTC_t - beta^i * alt_t (same as OLS residual)",
        "adf_rule": f"adf_pass_5pct: ADF p-value < {ADF_PASS_MAX_PVALUE} (linear stationarity / linear cointegration narrative)",
        "kss_auxiliary": "Delta_spread_t = delta * spread_{t-1}^3 + eps (no intercept; slide form)",
        "kss_rule": (
            f"kss_pass_10pct: KSS auxiliary t-stat < {KSS_CRITICAL_10PCT} "
            "(10% left tail; nonlinear stationarity narrative)"
        ),
        "kendall_tau": "Kendall tau between aligned BTC and alt closes (for ranking KSS-passing pairs)",
        "copula_selection": (
            "Copula pool gate: kss_pass_10pct only. Among those pairs, take the two largest kendall_tau."
        ),
        "copula_top2": top2,
        "n_pairs_passing_kss_gate": len(eligible),
        "optimizer_objective": (
            "min SSR(Delta_spread - delta * lag^3 * exp(-theta * z^2)); compare Nelder-Mead vs BFGS"
        ),
        "optimizer_runs": meta_optimizer,
    }
    return rows, meta


def save_outputs(timeframe: str, rows: list[PairResult], meta: dict[str, Any]) -> str:
    out_dir = os.path.join(RESULT_ROOT, f"result_{timeframe}", "ADF_KSS")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame([asdict(r) for r in rows])
    csv_path = os.path.join(out_dir, "spread_adf_kss_summary.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    json_path = os.path.join(out_dir, "spread_adf_kss_full.json")
    payload = {
        "meta": meta,
        "pairs": [asdict(r) for r in rows],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    top2_path = os.path.join(out_dir, "copula_top2_by_kendall.csv")
    top2_rows = meta.get("copula_top2") or []
    if top2_rows:
        pd.DataFrame(top2_rows).to_csv(top2_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(
            columns=["rank", "alt_symbol", "kendall_tau", "adf_pvalue", "kss_t_stat", "beta"]
        ).to_csv(top2_path, index=False, encoding="utf-8-sig")

    return out_dir


def run_full_pipeline(timeframe: str) -> tuple[str, list[PairResult], dict[str, Any]]:
    """Run full analysis and write CSV/JSON under result/result_{timeframe}/ADF_KSS/."""
    rows, meta = run_analysis(timeframe)
    out_dir = save_outputs(timeframe, rows, meta)
    return out_dir, rows, meta


def result_adf_kss_dir(timeframe: str) -> str:
    return os.path.join(RESULT_ROOT, f"result_{timeframe}", "ADF_KSS")


def read_copula_top2(timeframe: str) -> pd.DataFrame:
    path = os.path.join(result_adf_kss_dir(timeframe), "copula_top2_by_kendall.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}; run run_full_pipeline('{timeframe}') first.")
    df = pd.read_csv(path)
    return df


def kss_table_for_alts(timeframe: str, alt_symbols: list[str]) -> pd.DataFrame:
    """
    Compact KSS-only table for chosen alts vs BTC (through-origin spread, same as full pipeline).
    """
    paths = discover_csv_paths(timeframe)
    if "BTC_USDT" not in paths:
        raise FileNotFoundError("BTC_USDT CSV required")
    btc_s = load_close_series(paths["BTC_USDT"])
    if len(btc_s) == 0:
        raise ValueError("BTC series is empty")
    series_map: dict[str, pd.Series] = {"BTC_USDT": btc_s}
    for sym in alt_symbols:
        sym = sym.strip().upper()
        if sym == "BTC_USDT":
            continue
        if sym not in paths:
            raise KeyError(f"No CSV for {sym} under {timeframe}")
        s = load_close_series(paths[sym])
        if len(s) == 0:
            raise ValueError(f"Empty series: {sym}")
        series_map[sym] = s
    panel = merge_on_timestamp(series_map)
    btc_a = panel["BTC_USDT"].values.astype(float)
    rows_out: list[dict[str, Any]] = []
    for sym in alt_symbols:
        sym = sym.strip().upper()
        if sym == "BTC_USDT" or sym not in panel.columns:
            continue
        alt_a = panel[sym].values.astype(float)
        spread, beta, _ = ols_btc_on_alt_through_origin(btc_a, alt_a)
        kss = kss_auxiliary_t_stat(spread)
        t = float(kss.get("kss_t_stat", float("nan")))
        rows_out.append(
            {
                "alt_symbol": sym,
                "n_obs": int(len(spread)),
                "beta": float(beta),
                "kss_t_stat": t,
                "kss_pass_10pct": bool(np.isfinite(t) and t < KSS_CRITICAL_10PCT),
                "kss_critical_10pct": float(KSS_CRITICAL_10PCT),
            }
        )
    return pd.DataFrame(rows_out)


def main():
    tf = (sys.argv[1] if len(sys.argv) > 1 else TIMEFRAME).strip()
    if tf not in ("1h", "1m"):
        raise ValueError(
            'Timeframe must be "1h" or "1m" (set TIMEFRAME at top or pass argv, e.g. python spread_adf_kss.py 1m)'
        )
    out_dir, rows, _ = run_full_pipeline(tf)
    print(f"Done: {len(rows)} alt(s) vs BTC; results written to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
