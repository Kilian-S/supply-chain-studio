"""Seasonal naive and ETS, the two baselines of the demand forecast.

Both forecast the mean of every series from a forecast origin T (the last observed month, 1-based) for
any number of months ahead, and both can forecast from earlier origins inside the fitting window, which
gives the past errors the uncertainty is built from.
"""

from __future__ import annotations

import warnings
from multiprocessing import Pool

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from scgame.ml.demand_series import first_sale


def seasonal_naive(y: np.ndarray, T: int, horizon: int) -> np.ndarray:
    """Every future year repeats months T-11..T. A series on sale for less than a year repeats the mean of
    its months on sale instead (its last year would include months before its launch)."""
    fs = first_sale(y, T)
    on_sale = T - fs
    h = np.arange(horizon)
    rep = y[:, T - 12 + (h % 12)]
    mean = np.array([y[i, fs[i]:T].mean() if fs[i] < T else 0.0 for i in range(len(y))])
    return np.where((on_sale >= 12)[:, None], rep, mean[:, None])


# ETS: Holt-Winters with additive damped trend and additive season on log(1 + y), from the first sale on.
# The trend is kept or dropped per series by AIC. Fewer than 24 months on sale falls back to seasonal naive.
ETS_MIN_MONTHS = 24


def _fit_one(args: tuple[np.ndarray, int]) -> dict | None:
    y, start = args
    x = np.log1p(y[start:])
    if len(x) < ETS_MIN_MONTHS:
        return None
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for trend in ("add", None):
            try:
                kw = {"trend": trend, "damped_trend": trend is not None, "seasonal": "add", "seasonal_periods": 12, "initialization_method": "estimated"}
                r = ExponentialSmoothing(x, **kw).fit()
            except Exception:  # noqa: BLE001, S112 (a failed fit falls back to seasonal naive)
                continue
            if not np.isfinite(r.aic):
                continue
            if best is None or r.aic < best[0]:
                phi = r.params.get("damping_trend", np.nan)
                best = (r.aic, {"start": start, "level": r.level.copy(), "trend": np.asarray(r.trend if trend else np.zeros(len(x))).copy(),
                                    "season": r.season.copy(), "phi": float(phi) if trend and np.isfinite(phi) else 0.0,
                                    "sigma2": float(np.mean(r.resid ** 2)), "has_trend": trend is not None})
    return None if best is None else best[1]


def fit_ets(y: np.ndarray, T: int, workers: int = 4) -> list[dict | None]:
    """One fit per series on months 1..T (states kept for every month, so forecasts can start earlier)."""
    fs = first_sale(y, T)
    jobs = [(y[i, :T], int(fs[i])) for i in range(len(y))]
    with Pool(workers) as pool:
        return pool.map(_fit_one, jobs, chunksize=32)


def ets_forecast(fits: list[dict | None], y: np.ndarray, origin: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean forecasts from `origin` (at most the fitted window's end) with the fitted parameters and states. Returns the
    forecasts and a mask of series that used ETS (the rest fall back to seasonal naive)."""
    out = seasonal_naive(y, origin, horizon)
    used = np.zeros(len(y), dtype=bool)
    h = np.arange(1, horizon + 1)
    for i, f in enumerate(fits):
        if f is None:
            continue
        t = origin - 1 - f["start"]  # state index of the origin month
        if t < 12:
            continue
        damp = np.cumsum(f["phi"] ** h) if f["has_trend"] else np.zeros(horizon)
        season = f["season"][t + h - 12 * np.ceil(h / 12).astype(int)]
        mu = f["level"][t] + damp * f["trend"][t] + season
        out[i] = np.maximum(np.expm1(mu + f["sigma2"] / 2), 0.0)  # mean of a log-normal, back on the unit scale
        used[i] = True
    return out, used
