"""LightGBM and the neural network (multilayer perceptron), the two global models of the demand forecast.

Both read one year of a series (scaled by its mean) plus its traits and predict the next year; five years
come from applying them five times, feeding each forecast year back in (agreed design A1 to C7 in
`design/demand-forecast/PROPOSAL.md`, section 4a).
"""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from scgame.ml.demand_series import (
    LEVELS,
    MONTHS_ON_SALE_CAP,
    Windows,
    level_weights,
    windows,
)
from scgame.ml.forecast_baselines import seasonal_naive

LAG_COLS = [f"lag_{k}" for k in range(1, 13)]  # lag_12 is the origin month
CATEGORICAL = ["level", "region", "category"]


def window_frame(win: Windows, static: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(win.lags, columns=LAG_COLS)
    f = pd.concat([f, win.dyn.reset_index(drop=True), static.iloc[win.series].reset_index(drop=True)], axis=1)
    return f


def training_windows(y: np.ndarray, bulk: np.ndarray, fit_end: int) -> Windows:
    """Every origin with a year before and a year after it inside months 1..fit_end (A8)."""
    return windows(y[:, :fit_end], bulk[:, :fit_end], list(range(12, fit_end - 12 + 1)), with_target=True)


# ---------------------------------------------------------------- recursion shared by both models


def recursive_forecast(predict_year, y: np.ndarray, bulk: np.ndarray, T: int, years: int) -> tuple[np.ndarray, np.ndarray]:
    """Forecast `years` years from origin T, one year at a time. `predict_year(win)` returns scaled
    forecasts (series, 12). Series with fewer than 3 months on sale at T use seasonal naive throughout.
    Returns forecasts (series, 12 * years) and the fallback mask."""
    hist = np.zeros((y.shape[0], T + 12 * years))
    hist[:, :T] = y[:, :T]  # nothing after the origin is visible
    b = np.zeros_like(hist)
    b[:, :T] = bulk[:, :T]
    base = windows(hist, b, [T], with_target=False)
    on_sale0 = base.dyn["months_on_sale"].to_numpy()
    es0 = base.dyn["empty_share"].to_numpy()
    bs0 = base.dyn["bulk_share"].to_numpy()
    fallback = ~base.ok
    for k in range(years):
        Tk = T + 12 * k
        win = windows(hist, b, [Tk], with_target=False, months_on_sale=np.minimum(on_sale0 + 12 * k, MONTHS_ON_SALE_CAP), empty_share=es0)
        win.dyn["bulk_share"] = bs0
        pred = np.maximum(predict_year(win), 0.0) * win.scale[:, None]
        pred[~np.isfinite(pred)] = 0.0
        hist[:, Tk:Tk + 12] = pred
    out = hist[:, T:]
    if fallback.any():
        out[fallback] = seasonal_naive(y, T, 12 * years)[fallback]
    return out, fallback


# ---------------------------------------------------------------- LightGBM


def lgb_rows(win: Windows, static: pd.DataFrame) -> pd.DataFrame:
    """One row per window and month ahead (B1)."""
    f = window_frame(win, static)
    n = len(f)
    rows = f.loc[np.repeat(np.arange(n), 12)].reset_index(drop=True)
    h = np.tile(np.arange(1, 13), n)
    rows["months_ahead"] = h
    rows["target_month"] = (rows["origin_month"].to_numpy() - 1 + h) % 12 + 1
    return rows


LGB_GRID = [
    {"num_leaves": nl, "min_data_in_leaf": md, "bagging_fraction": fr, "feature_fraction": fr, "objective": obj}
    for nl in (15, 63) for md in (50, 500) for fr in (0.8, 1.0) for obj in ("regression", "tweedie")
]
LGB_TREES = (250, 500, 1000, 1500)


@dataclass
class LgbModel:
    booster: lgb.Booster
    trees: int
    static: pd.DataFrame
    setting: dict = field(default_factory=dict)

    def predict_year(self, win: Windows) -> np.ndarray:
        X = lgb_rows(win, self.static)
        p = self.booster.predict(X, num_iteration=self.trees)
        return p.reshape(-1, 12)


def fit_lgb(y: np.ndarray, bulk: np.ndarray, meta: pd.DataFrame, static: pd.DataFrame, fit_end: int, setting: dict,
            threads: int = 4) -> lgb.Booster:
    win = training_windows(y, bulk, fit_end)
    keep = win.ok
    X = lgb_rows(win, static)
    target = win.target.ravel()
    w = np.repeat(level_weights(meta, win.series, win.scale, win.origin), 12)
    m = np.repeat(keep, 12)
    w = w[m] / w[m].mean()
    params = dict(setting, learning_rate=0.05, bagging_freq=1 if setting["bagging_fraction"] < 1 else 0, num_threads=threads,
                  verbose=-1, seed=7, deterministic=True, force_col_wise=True)
    if setting["objective"] == "tweedie":
        params["tweedie_variance_power"] = 1.5
    ds = lgb.Dataset(X[m], target[m], weight=w, categorical_feature=CATEGORICAL, free_raw_data=False)
    return lgb.train(params, ds, num_boost_round=max(LGB_TREES))


def lgb_shap(model: LgbModel, win: Windows) -> pd.DataFrame:
    """Mean absolute SHAP value per input over the given windows (B5)."""
    X = lgb_rows(win, model.static)
    contrib = model.booster.predict(X, num_iteration=model.trees, pred_contrib=True)
    return pd.Series(np.abs(contrib[:, :-1]).mean(0), index=X.columns).sort_values(ascending=False)


# ---------------------------------------------------------------- neural network


class MlpInputs:
    """Standardised inputs (C3): numbers as they are, missing values as 0 plus a flag, categories one-hot."""

    def __init__(self):
        self.scaler = StandardScaler()

    def raw(self, win: Windows, static: pd.DataFrame) -> np.ndarray:
        f = window_frame(win, static)
        parts = [f[LAG_COLS].to_numpy()]
        g = f["growth"].to_numpy()
        parts += [np.nan_to_num(g)[:, None], np.isnan(g)[:, None].astype(float)]
        parts += [f[["months_on_sale", "empty_share", "bulk_share"]].to_numpy()]
        parts.append(np.eye(12)[f["origin_month"].to_numpy().astype(int) - 1])
        parts.append(np.eye(len(LEVELS))[f["level"].to_numpy().astype(int)])
        for col, n in (("region", 8), ("category", 8)):
            v = f[col].to_numpy()
            oh = np.zeros((len(f), n + 1))
            idx = np.where(np.isnan(v), n, np.nan_to_num(v - 1 if col == "region" else v)).astype(int)
            oh[np.arange(len(f)), idx] = 1
            parts.append(oh)
        for cols in (["region_population", "region_income", "region_urban"], ["log_price", "log_volume"]):
            v = f[cols].to_numpy()
            parts += [np.nan_to_num(v), np.isnan(v[:, :1]).astype(float)]
        return np.hstack(parts).astype(float)

    def fit_transform(self, win, static):
        return self.scaler.fit_transform(self.raw(win, static))

    def transform(self, win, static):
        return self.scaler.transform(self.raw(win, static))


# Loss is a setting like LightGBM's (B2): squared error, or Poisson deviance, which also predicts the mean
# but is far less pulled by the rare bulk months (scaled targets reach 207, the 99th percentile is 10).
MLP_GRID = [{"width": wd, "alpha": a, "loss": loss} for wd in (32, 64) for a in (1e-4, 1e-3, 1e-2) for loss in ("squared_error", "poisson")]
MLP_LR = 1e-3
MLP_SEEDS = 5
# Epochs are a setting chosen on the validation set, like LightGBM's number of trees. Early stopping on the
# latest training origin (agreed choice C5) stopped after 1 to 5 epochs: one held-out origin's loss is
# dominated by a few heavily weighted windows and bulk months, so it is too noisy to stop on.
MLP_EPOCHS = (25, 50, 100, 200)


def repeat_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal weight per level puts over half the weight on 1% of windows (the few Total and Region ones),
    which makes mini-batch training erratic. Repeating a window round(w) times and weighting each copy by
    w / copies gives the same objective with weights near 1."""
    copies = np.maximum(1, np.round(w)).astype(int)
    idx = np.repeat(np.arange(len(w)), copies)
    return idx, (w / copies)[idx]


def _train_seed(args) -> list[MLPRegressor]:
    X, Y, w, setting, seed, epochs = args
    net = MLPRegressor(hidden_layer_sizes=(setting["width"], setting["width"]), alpha=setting["alpha"], loss=setting["loss"],
                       learning_rate_init=MLP_LR, batch_size=256, random_state=seed, shuffle=True)
    snaps = []
    for epoch in range(1, max(epochs) + 1):
        net.partial_fit(X, Y, sample_weight=w)
        if epoch in epochs:
            snaps.append(copy.deepcopy(net))
    return snaps


@dataclass
class MlpModel:
    nets: list
    inputs: MlpInputs
    static: pd.DataFrame
    setting: dict = field(default_factory=dict)

    def predict_year(self, win: Windows) -> np.ndarray:
        X = self.inputs.transform(win, self.static)
        return np.mean([n.predict(X) for n in self.nets], axis=0)


def fit_mlp(y: np.ndarray, bulk: np.ndarray, meta: pd.DataFrame, static: pd.DataFrame, fit_end: int, setting: dict,
            workers: int = 4, epochs: tuple[int, ...] = MLP_EPOCHS) -> dict[int, MlpModel]:
    """Five networks from different random starts (C6), with snapshots after each epoch count in
    `epochs`. Returns one averaged model per epoch count."""
    win = training_windows(y, bulk, fit_end)
    tr = np.nonzero(win.ok)[0]
    inputs = MlpInputs()
    X = inputs.fit_transform(win, static)
    w = level_weights(meta, win.series, win.scale, win.origin)
    w = w / w[tr].mean()
    idx, wr = repeat_rows(w[tr])
    rows = tr[idx]
    jobs = [(X[rows], win.target[rows], wr, setting, seed, tuple(epochs)) for seed in range(MLP_SEEDS)]
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(_train_seed, jobs))
    return {e: MlpModel([run[i] for run in runs], inputs, static, dict(setting, epochs=e)) for i, e in enumerate(epochs)}


def one_year_forecasts(model, y: np.ndarray, bulk: np.ndarray, origins: list[int]) -> np.ndarray:
    """One-year-ahead forecasts from each origin (series, origin, 12), reading months up to the origin only. Series that
    fall back at an origin get seasonal naive there."""
    out = np.zeros((y.shape[0], len(origins), 12))
    for j, T in enumerate(origins):
        win = windows(y[:, :T], bulk[:, :T], [T], with_target=False)
        pred = np.maximum(model.predict_year(win), 0.0) * win.scale[:, None]
        sn = seasonal_naive(y, T, 12)
        out[:, j] = np.where(win.ok[:, None], pred, sn)
    return out
