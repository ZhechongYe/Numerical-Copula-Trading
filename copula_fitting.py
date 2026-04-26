from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from scipy.stats import kendalltau


COPULA_FAMILIES = ("gaussian", "student_t", "clayton", "gumbel", "frank")
EPS = 1e-10


def _prepare_uv(df: pd.DataFrame, u_col: str, v_col: str) -> tuple[np.ndarray, np.ndarray]:
    if u_col not in df.columns or v_col not in df.columns:
        raise KeyError(f"Input DataFrame must contain {u_col!r} and {v_col!r}.")

    u = df[u_col].to_numpy(dtype=float)
    v = df[v_col].to_numpy(dtype=float)
    mask = np.isfinite(u) & np.isfinite(v)
    u = u[mask]
    v = v[mask]

    if len(u) == 0:
        raise ValueError("No finite (u, v) observations available for copula fitting.")
    if np.any((u <= 0.0) | (u >= 1.0) | (v <= 0.0) | (v >= 1.0)):
        raise ValueError("Copula inputs u and v must be strictly inside (0, 1).")

    return np.clip(u, EPS, 1.0 - EPS), np.clip(v, EPS, 1.0 - EPS)


def _aic(log_likelihood: float, n_params: int) -> float:
    return float(2 * n_params - 2 * log_likelihood)


def _initial_rho(u: np.ndarray, v: np.ndarray) -> float:
    tau = kendalltau(u, v, nan_policy="omit").correlation
    if not np.isfinite(tau):
        return 0.0
    return float(np.clip(np.sin(0.5 * np.pi * tau), -0.8, 0.8))


def _gaussian_logpdf(u: np.ndarray, v: np.ndarray, rho: float) -> np.ndarray:
    z1 = stats.norm.ppf(u)
    z2 = stats.norm.ppf(v)
    one_minus = 1.0 - rho * rho
    return -0.5 * np.log(one_minus) - (
        rho * rho * (z1 * z1 + z2 * z2) - 2.0 * rho * z1 * z2
    ) / (2.0 * one_minus)


def _student_t_logpdf(u: np.ndarray, v: np.ndarray, rho: float, df: float) -> np.ndarray:
    x = stats.t.ppf(u, df=df)
    y = stats.t.ppf(v, df=df)
    one_minus = 1.0 - rho * rho
    q = (x * x - 2.0 * rho * x * y + y * y) / one_minus

    log_joint = (
        special.gammaln((df + 2.0) / 2.0)
        - special.gammaln(df / 2.0)
        - np.log(df * np.pi)
        - 0.5 * np.log(one_minus)
        - ((df + 2.0) / 2.0) * np.log1p(q / df)
    )
    log_marginals = stats.t.logpdf(x, df=df) + stats.t.logpdf(y, df=df)
    return log_joint - log_marginals


def _clayton_logpdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    a = np.power(u, -theta) + np.power(v, -theta) - 1.0
    return (
        np.log1p(theta)
        + (-theta - 1.0) * (np.log(u) + np.log(v))
        + (-2.0 - 1.0 / theta) * np.log(a)
    )


def _gumbel_logpdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    x = -np.log(u)
    y = -np.log(v)
    a = np.power(x, theta) + np.power(y, theta)
    a_pow = np.power(a, 1.0 / theta)
    c_log = -a_pow
    return (
        c_log
        + (theta - 1.0) * (np.log(x) + np.log(y))
        - np.log(u)
        - np.log(v)
        + (1.0 / theta - 2.0) * np.log(a)
        + np.log(a_pow + theta - 1.0)
    )


def _frank_logpdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    emt = np.exp(-theta)
    emtu = np.exp(-theta * u)
    emtv = np.exp(-theta * v)
    numerator_const = -theta * (emt - 1.0)
    denominator_base = emt - 1.0 + (emtu - 1.0) * (emtv - 1.0)
    return (
        np.log(np.maximum(numerator_const, EPS))
        - theta * (u + v)
        - 2.0 * np.log(np.maximum(np.abs(denominator_base), EPS))
    )


def _neg_log_likelihood(params: np.ndarray, copula: str, u: np.ndarray, v: np.ndarray) -> float:
    try:
        if copula == "gaussian":
            logpdf = _gaussian_logpdf(u, v, float(params[0]))
        elif copula == "student_t":
            logpdf = _student_t_logpdf(u, v, float(params[0]), float(params[1]))
        elif copula == "clayton":
            logpdf = _clayton_logpdf(u, v, float(params[0]))
        elif copula == "gumbel":
            logpdf = _gumbel_logpdf(u, v, float(params[0]))
        elif copula == "frank":
            logpdf = _frank_logpdf(u, v, float(params[0]))
        else:
            raise ValueError(f"Unsupported copula: {copula}")
    except FloatingPointError:
        return 1e100

    if not np.all(np.isfinite(logpdf)):
        return 1e100
    return float(-np.sum(logpdf))


def _fit_with_starts(
    copula: str,
    starts: list[list[float]],
    bounds: list[tuple[float, float]],
    u: np.ndarray,
    v: np.ndarray,
) -> dict[str, Any]:
    best = None
    for start in starts:
        sol = optimize.minimize(
            _neg_log_likelihood,
            np.asarray(start, dtype=float),
            args=(copula, u, v),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000},
        )
        if best is None or float(sol.fun) < float(best.fun):
            best = sol

    if best is None:
        raise RuntimeError(f"No optimizer run completed for {copula}.")
    log_likelihood = float(-best.fun)
    return {
        "x": [float(x) for x in best.x],
        "log_likelihood": log_likelihood,
        "success": bool(best.success),
        "message": str(best.message),
    }


def _fit_one_copula(copula: str, u: np.ndarray, v: np.ndarray) -> dict[str, Any]:
    rho0 = _initial_rho(u, v)

    if copula == "gaussian":
        fit = _fit_with_starts(copula, [[rho0], [0.0]], [(-0.99, 0.99)], u, v)
        params = {"rho": fit["x"][0]}
        n_params = 1
    elif copula == "student_t":
        starts = [[rho0, 4.0], [rho0, 10.0], [0.0, 30.0]]
        fit = _fit_with_starts(copula, starts, [(-0.99, 0.99), (2.01, 100.0)], u, v)
        params = {"rho": fit["x"][0], "df": fit["x"][1]}
        n_params = 2
    elif copula == "clayton":
        tau = kendalltau(u, v, nan_policy="omit").correlation
        theta0 = 2.0 * tau / (1.0 - tau) if np.isfinite(tau) and tau > 0 else 0.5
        theta0 = float(np.clip(theta0, 0.01, 20.0))
        fit = _fit_with_starts(copula, [[theta0], [0.5], [2.0]], [(0.001, 50.0)], u, v)
        params = {"theta": fit["x"][0]}
        n_params = 1
    elif copula == "gumbel":
        tau = kendalltau(u, v, nan_policy="omit").correlation
        theta0 = 1.0 / (1.0 - tau) if np.isfinite(tau) and tau > 0 else 1.1
        theta0 = float(np.clip(theta0, 1.001, 20.0))
        fit = _fit_with_starts(copula, [[theta0], [1.1], [2.0]], [(1.001, 50.0)], u, v)
        params = {"theta": fit["x"][0]}
        n_params = 1
    elif copula == "frank":
        negative_fit = _fit_with_starts(copula, [[-10.0], [-2.0]], [(-50.0, -1e-4)], u, v)
        positive_fit = _fit_with_starts(copula, [[2.0], [10.0]], [(1e-4, 50.0)], u, v)
        fit = (
            negative_fit
            if negative_fit["log_likelihood"] >= positive_fit["log_likelihood"]
            else positive_fit
        )
        theta = fit["x"][0]
        params = {"theta": theta}
        n_params = 1
    else:
        raise ValueError(f"Unsupported copula: {copula}")

    return {
        "copula": copula,
        "params": params,
        "n_params": n_params,
        "log_likelihood": fit["log_likelihood"],
        "aic": _aic(fit["log_likelihood"], n_params),
        "success": fit["success"],
        "message": fit["message"],
    }


def fit_copulas_and_select(
    uv_df: pd.DataFrame,
    u_col: str = "u",
    v_col: str = "v",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fit Gaussian, Student-t, Clayton, Gumbel, and Frank copulas by MLE and select by AIC.
    """
    u, v = _prepare_uv(uv_df, u_col, v_col)
    rows = [_fit_one_copula(copula, u, v) for copula in COPULA_FAMILIES]

    copula_summary = pd.DataFrame(rows)
    copula_summary["best"] = False
    best_idx = copula_summary["aic"].idxmin()
    copula_summary.loc[best_idx, "best"] = True
    copula_summary = copula_summary.sort_values("aic").reset_index(drop=True)

    best_copula = copula_summary.loc[copula_summary["best"]].iloc[0].to_dict()
    return copula_summary, best_copula
