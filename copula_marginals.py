from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from spread_adf_kss import (
    discover_csv_paths,
    load_close_series,
    merge_on_timestamp,
    ols_btc_on_alt_through_origin,
)


DEFAULT_DISTRIBUTIONS = ("normal", "student_t", "cauchy")


def _fit_distribution(x: np.ndarray, distribution: str) -> dict[str, Any]:
    """Fit one marginal distribution and return parameters, log-likelihood, and AIC."""
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("Cannot fit marginal distribution on an empty series.")

    if distribution == "normal":
        loc, scale = stats.norm.fit(x)
        params = {"loc": float(loc), "scale": float(scale)}
        logpdf = stats.norm.logpdf(x, loc=loc, scale=scale)
        n_params = 2
    elif distribution == "student_t":
        df, loc, scale = stats.t.fit(x)
        params = {"df": float(df), "loc": float(loc), "scale": float(scale)}
        logpdf = stats.t.logpdf(x, df=df, loc=loc, scale=scale)
        n_params = 3
    elif distribution == "cauchy":
        loc, scale = stats.cauchy.fit(x)
        params = {"loc": float(loc), "scale": float(scale)}
        logpdf = stats.cauchy.logpdf(x, loc=loc, scale=scale)
        n_params = 2
    else:
        raise ValueError(f"Unsupported marginal distribution: {distribution}")

    if not np.all(np.isfinite(logpdf)):
        raise ValueError(f"Non-finite log-likelihood values for {distribution}.")

    log_likelihood = float(np.sum(logpdf))
    aic = float(2 * n_params - 2 * log_likelihood)
    return {
        "distribution": distribution,
        "params": params,
        "n_params": int(n_params),
        "log_likelihood": log_likelihood,
        "aic": aic,
    }


def _cdf_from_fit(x: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    """Apply the fitted marginal CDF."""
    distribution = fit["distribution"]
    params = fit["params"]
    x = np.asarray(x, dtype=float).ravel()

    if distribution == "normal":
        return stats.norm.cdf(x, loc=params["loc"], scale=params["scale"])
    if distribution == "student_t":
        return stats.t.cdf(x, df=params["df"], loc=params["loc"], scale=params["scale"])
    if distribution == "cauchy":
        return stats.cauchy.cdf(x, loc=params["loc"], scale=params["scale"])
    raise ValueError(f"Unsupported marginal distribution: {distribution}")


def build_btc_reference_spreads(
    timeframe: str,
    alt_symbols: list[str] | tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build BTC-reference spreads with the same through-origin OLS convention as spread_adf_kss.py.

    Returns:
        spread_df: timestamp plus one spread column per alt symbol.
        spread_meta: beta, r-squared, and sample size for each alt symbol.
    """
    symbols = [s.strip().upper() for s in alt_symbols]
    if len(symbols) == 0:
        raise ValueError("alt_symbols must contain at least one symbol.")
    if len(set(symbols)) != len(symbols):
        raise ValueError("alt_symbols contains duplicates.")

    paths = discover_csv_paths(timeframe)
    required = ["BTC_USDT", *symbols]
    missing = [sym for sym in required if sym not in paths]
    if missing:
        raise KeyError(f"Missing CSV data for: {', '.join(missing)}")

    series_map = {"BTC_USDT": load_close_series(paths["BTC_USDT"])}
    for sym in symbols:
        series_map[sym] = load_close_series(paths[sym])

    panel = merge_on_timestamp(series_map)
    btc = panel["BTC_USDT"].values.astype(float)

    spread_data: dict[str, np.ndarray] = {}
    meta_rows: list[dict[str, Any]] = []
    for sym in symbols:
        alt = panel[sym].values.astype(float)
        spread, beta, r_squared = ols_btc_on_alt_through_origin(btc, alt)
        spread_col = f"{sym}_spread"
        spread_data[spread_col] = spread
        meta_rows.append(
            {
                "alt_symbol": sym,
                "spread_column": spread_col,
                "beta": float(beta),
                "r_squared": float(r_squared),
                "n_obs": int(len(spread)),
            }
        )

    spread_df = pd.DataFrame(spread_data, index=panel.index)
    spread_df = spread_df.reset_index().rename(columns={"index": "timestamp"})
    spread_meta = pd.DataFrame(meta_rows)
    return spread_df, spread_meta


def fit_marginals(
    spread_df: pd.DataFrame,
    spread_columns: list[str] | tuple[str, ...],
    distributions: list[str] | tuple[str, ...] = DEFAULT_DISTRIBUTIONS,
) -> pd.DataFrame:
    """Fit candidate marginals for each spread column and mark the minimum-AIC fit."""
    rows: list[dict[str, Any]] = []
    for col in spread_columns:
        if col not in spread_df.columns:
            raise KeyError(f"Missing spread column: {col}")
        x = spread_df[col].to_numpy(dtype=float)
        for dist_name in distributions:
            fit = _fit_distribution(x, dist_name)
            rows.append(
                {
                    "spread_column": col,
                    "distribution": fit["distribution"],
                    "params": fit["params"],
                    "n_params": fit["n_params"],
                    "log_likelihood": fit["log_likelihood"],
                    "aic": fit["aic"],
                }
            )

    fit_summary = pd.DataFrame(rows)
    fit_summary["best"] = False
    best_idx = fit_summary.groupby("spread_column")["aic"].idxmin()
    fit_summary.loc[best_idx, "best"] = True
    return fit_summary.sort_values(["spread_column", "aic"]).reset_index(drop=True)


def fit_marginals_and_transform(
    timeframe: str = "1h",
    alt_symbols: list[str] | tuple[str, str] = ("XRP_USDT", "BCH_USDT"),
    clip_eps: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit marginals on two BTC-reference spreads and transform them to (u, v) in (0, 1)^2.

    Returns:
        uv_df: timestamp, two spread columns, u, and v.
        fit_summary: candidate marginal fits with AIC and best flag.
        spread_meta: beta and sample metadata for the BTC-reference spreads.
    """
    symbols = [s.strip().upper() for s in alt_symbols]
    if len(symbols) != 2:
        raise ValueError("fit_marginals_and_transform expects exactly two alt symbols.")
    if not (0.0 < clip_eps < 0.5):
        raise ValueError("clip_eps must be between 0 and 0.5.")

    spread_df, spread_meta = build_btc_reference_spreads(timeframe, symbols)
    spread_columns = [f"{sym}_spread" for sym in symbols]
    fit_summary = fit_marginals(spread_df, spread_columns)

    uv_df = spread_df[["timestamp", *spread_columns]].copy()
    uv_names = ["u", "v"]
    for spread_col, uv_name in zip(spread_columns, uv_names):
        best_row = fit_summary.loc[
            (fit_summary["spread_column"] == spread_col) & fit_summary["best"]
        ].iloc[0]
        fit = {
            "distribution": best_row["distribution"],
            "params": best_row["params"],
        }
        transformed = _cdf_from_fit(uv_df[spread_col].to_numpy(dtype=float), fit)
        uv_df[uv_name] = np.clip(transformed, clip_eps, 1.0 - clip_eps)

    ordered_cols = ["timestamp", *spread_columns, "u", "v"]
    return uv_df[ordered_cols], fit_summary, spread_meta
