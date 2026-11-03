"""Demand forecast (model B): validation, test, ground truth and the capacity check.

Fit on years 1 to 3 (months 1 to 36); every setting is chosen on years 4 and 5 (months 37 to 60, forecast
from month 36). The chosen settings are refitted on years 1 to 5 and years 6 to 10 are forecast once from
month 60 (upper levels five years ahead, SKU levels two). Years 6 to 10 are read only by the test step.
The ground truth (200 replications, `scgame.world.replicate`) checks the forecasts and is never used to
choose. Design and decisions in `design/demand-forecast/PROPOSAL.md`, the build in
`design/demand-forecast/HANDOVER.md`. Years 6 to 10 are also drawn a second time (another seed, same rules)
and the same forecasts are scored on them once.

Usage: uv run python -m scgame.ml.demand_forecast <world dir> <out dir>
Needs world v1 years 1 to 5 (`scgame.world.generator`, seed 7). Years 6 to 10 (`scgame.world.future`) and
the 200 ground-truth replications (`scgame.world.replicate`, about 13 minutes) are built first if missing.
The whole run takes about 50 minutes on 4 cores.
"""

from __future__ import annotations

import os

# One thread per process for the linear algebra, as in the committed run: the network trains five copies in
# parallel processes, and fixed thread counts keep floating-point sums, and so the outputs, reproducible.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
# The imports below must follow the thread settings above, hence E402 (RUF100: newer ruff versions do not
# enable E402 by default and would call the directive unused).
# ruff: noqa: E402, RUF100

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scgame.ml import demand_series as ds
from scgame.ml import forecast_baselines as fb
from scgame.ml import forecast_models as fm
from scgame.ml import forecast_uncertainty as fu
from scgame.ml.demand_series import LEVELS, UPPER

VAL_FIT, TEST_FIT = 36, 60
VAL_YEARS, TEST_YEARS, SKU_YEARS = 2, 5, 2
N_PATHS = 1000
# Width factors, one for the upper levels and one for the SKU levels (one factor for all levels left the
# upper bands 2 to 3 times too narrow on validation, while wider SKU paths exploded).
WIDTHS_UPPER = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
WIDTHS_LOWER = (0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
UPPER_RECS = ("Bottom-up", "MinT")
# A second draw of years 6 to 10 under the same rules (scored once, after the first test years had been seen).
SECOND_FUTURE = "future2"
SECOND_FUTURE_STREAM = 16  # random stream [seed, 16]; the first future uses [seed, 6]
SECOND_REPLICATION_STREAM = 17  # replications [seed, 17, r]; the first future's use [seed, 7, r]
METHODS = ["Seasonal naive", "ETS", "LightGBM", "Neural network"]
GAME_CAPACITY_SHARE = 50_000 / (365 / 12) / 5 / 1_450  # yearly cost of a unit of monthly capacity, share of unit price
RHO = np.round(np.geomspace(0.1, 10, 41), 4)  # capacity cost per unit of monthly capacity per year / lost margin per unit


# ---------------------------------------------------------------- forecasts per method


class Fitted:
    """A method fitted on months 1..fit_end with one setting: mean forecasts from any origin."""

    def __init__(self, name: str, ser: ds.Series, fit_end: int, model=None, ets=None):
        self.name, self.ser, self.fit_end, self.model, self.ets = name, ser, fit_end, model, ets
        self.fallback = 0

    def forecast(self, years: int) -> np.ndarray:
        y, b, T = self.ser.y, self.ser.bulk, self.fit_end
        if self.name == "Seasonal naive":
            return fb.seasonal_naive(y, T, 12 * years)
        if self.name == "ETS":
            out, used = fb.ets_forecast(self.ets, y, T, 12 * years)
            self.fallback = int((~used).sum())
            return out
        out, fallback = fm.recursive_forecast(self.model.predict_year, y, b, T, years)
        self.fallback = int(fallback.sum())
        return out

    def year_from(self, origin: int) -> np.ndarray:
        """Mean forecasts of the twelve months after `origin` (series, 12), reading months up to it only.
        The origin may lie after `fit_end`: the model keeps the parameters it learned on months 1..fit_end."""
        y = self.ser.y
        if self.name == "Seasonal naive":
            return fb.seasonal_naive(y, origin, 12)
        if self.name == "ETS":
            fits = self.ets if origin == self.fit_end else fb.ets_extend(self.ets, y, origin)
            return fb.ets_forecast(fits, y, origin, 12)[0]
        return fm.one_year_forecasts(self.model, y, self.ser.bulk, [origin])[:, 0]


def level_rows(ser: ds.Series) -> dict[str, np.ndarray]:
    lev = ser.meta["level"].to_numpy()
    return {lv: np.nonzero(lev == lv)[0] for lv in LEVELS}


def point_score(ser: ds.Series, fc: np.ndarray, fit_end: int, years: int) -> float:
    """Mean over levels and forecast years of the WAPE of the mean forecast (used to pick hyperparameters)."""
    rows = level_rows(ser)
    cells = []
    for idx in rows.values():
        for j in range(years):
            m = slice(fit_end + 12 * j, fit_end + 12 * (j + 1))
            cells.append(fu.wape(ser.y[idx, m], fc[idx, 12 * j:12 * (j + 1)]))
    return float(np.nanmean(cells))


def tune(name: str, ser: ds.Series, static: pd.DataFrame, fit_end: int, log) -> tuple[Fitted, list[dict]]:
    """Hyperparameters chosen by point_score on the validation years."""
    trials = []
    if name == "Seasonal naive":
        f = Fitted(name, ser, fit_end)
        trials.append({"setting": {}, "score": point_score(ser, f.forecast(VAL_YEARS), fit_end, VAL_YEARS)})
        return f, trials
    if name == "ETS":
        f = Fitted(name, ser, fit_end, ets=fb.fit_ets(ser.y, fit_end))
        trials.append({"setting": {}, "score": point_score(ser, f.forecast(VAL_YEARS), fit_end, VAL_YEARS)})
        return f, trials
    best = None
    if name == "LightGBM":
        for setting in fm.LGB_GRID:
            booster = fm.fit_lgb(ser.y, ser.bulk, ser.meta, static, fit_end, setting)
            for trees in fm.LGB_TREES:
                f = Fitted(name, ser, fit_end, model=fm.LgbModel(booster, trees, static, dict(setting, trees=trees)))
                sc = point_score(ser, f.forecast(VAL_YEARS), fit_end, VAL_YEARS)
                trials.append({"setting": dict(setting, trees=trees), "score": sc})
                if best is None or sc < best[0]:
                    best = (sc, f)
            log(f"  LightGBM {setting} best so far {best[0]:.4f}")
    else:
        for setting in fm.MLP_GRID:
            for epochs, model in fm.fit_mlp(ser.y, ser.bulk, ser.meta, static, fit_end, setting, epochs=fm.MLP_EPOCHS).items():
                f = Fitted(name, ser, fit_end, model=model)
                sc = point_score(ser, f.forecast(VAL_YEARS), fit_end, VAL_YEARS)
                trials.append({"setting": dict(setting, epochs=epochs), "score": sc})
                if best is None or sc < best[0]:
                    best = (sc, f)
            log(f"  Neural network {setting} best so far {best[0]:.4f}")
    return best[1], trials


def refit(f: Fitted, ser: ds.Series, static: pd.DataFrame, fit_end: int) -> Fitted:
    """The same method and setting on a longer window."""
    if f.name == "Seasonal naive":
        return Fitted(f.name, ser, fit_end)
    if f.name == "ETS":
        return Fitted(f.name, ser, fit_end, ets=fb.fit_ets(ser.y, fit_end))
    s = dict(f.model.setting)
    if f.name == "LightGBM":
        trees = s.pop("trees")
        booster = fm.fit_lgb(ser.y, ser.bulk, ser.meta, static, fit_end, s)
        return Fitted(f.name, ser, fit_end, model=fm.LgbModel(booster, trees, static, dict(s, trees=trees)))
    epochs = s.pop("epochs")
    return Fitted(f.name, ser, fit_end, model=fm.fit_mlp(ser.y, ser.bulk, ser.meta, static, fit_end, s, epochs=(epochs,))[epochs])


# Out-of-sample error samples: the method is refitted every REFIT_EVERY months from FIRST_REFIT on, and each
# refit forecasts one year from its own month and the following ones, up to the end of the fitting window.
FIRST_REFIT = 24  # the learned models need a year in and a year out before their first training window
REFIT_EVERY = 3


def rolling_errors(f: Fitted, static: pd.DataFrame, log=None) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
    """Origins, months observed per origin, actual and forecast (series, origin, 12) from rolling-origin
    refits on months 1..origin. Months after the fitting window are left as NaN in `actual`."""
    ser, F = f.ser, f.fit_end
    origins = list(range(FIRST_REFIT, F))
    fc = np.full((len(ser.y), len(origins), 12), np.nan)
    for T_r in range(FIRST_REFIT, F, REFIT_EVERY):
        fr = refit(f, ser, static, T_r)
        for o in range(T_r, min(T_r + REFIT_EVERY, F)):
            fc[:, origins.index(o)] = fr.year_from(o)
        if log:
            log(f"    {f.name} refit at month {T_r}")
    observed = [min(12, F - o) for o in origins]
    actual = np.full_like(fc, np.nan)
    for j, (o, m) in enumerate(zip(origins, observed)):
        actual[:, j, :m] = ser.y[:, o:o + m]
    return origins, observed, actual, fc


# ---------------------------------------------------------------- paths, reconciliation, scores


class Probabilistic:
    """Mean forecasts, error blocks and the coherent sample paths of one fitted method."""

    def __init__(self, f: Fitted, years: int, seed: int, static: pd.DataFrame, log=None):
        ser, T = f.ser, f.fit_end
        self.f, self.ser, self.T, self.years = f, ser, T, years
        self.mean = f.forecast(years)
        origins, observed, actual, past = rolling_errors(f, static, log)
        self.origins, self.observed = origins, observed
        self.blocks = fu.error_blocks(np.nan_to_num(actual), np.nan_to_num(past), origins, observed)
        up = ser.upper_idx
        e = (actual[up] - past[up]).transpose(1, 2, 0).reshape(-1, len(up))  # (origin x month, series)
        e_units = e[np.isfinite(e).all(axis=1)]
        bottom = np.searchsorted(up, ser.rc_idx)
        self.P = {"Bottom-up": fu.bottom_up_matrix(ser.S_upper, bottom), "MinT": fu.mint_matrix(ser.S_upper, fu.shrunk_covariance(e_units))}
        self.draws = fu.draw_origins(self.blocks, years, N_PATHS, np.random.default_rng(seed))

    def upper_paths(self, k: float, rec: str) -> np.ndarray:
        up = self.ser.upper_idx
        return fu.reconcile(self.P[rec], fu.sample_paths(self.mean[up], fu.sub_blocks(self.blocks, up), self.draws, k, self.T))

    def lower_paths(self, k: float, upper: np.ndarray) -> dict[str, np.ndarray]:
        """Region × SKU paths scaled to the reconciled Region × category paths (middle-out), and SKU paths as
        their sums, so every level adds up."""
        ser = self.ser
        Hs = 12 * min(self.years, SKU_YEARS)
        draws = {key: d[:, :Hs // 12] for key, d in self.draws.items()}
        pairs = fu.sample_paths(self.mean[ser.pair_idx, :Hs], fu.sub_blocks(self.blocks, ser.pair_idx), draws, k, self.T)
        rc = upper[:, np.searchsorted(ser.upper_idx, ser.rc_idx), :Hs]
        pairs = fu.middle_out(pairs, ser.pair_parent, rc, len(ser.rc_idx))
        return {"pairs": pairs, "skus": fu.sum_to(pairs, ser.pair_sku, len(ser.sku_idx))}

    def paths(self, k_upper: float, k_lower: float, rec: str) -> dict[str, np.ndarray]:
        upper = self.upper_paths(k_upper, rec)
        return {"upper": upper, **self.lower_paths(k_lower, upper)}

    def quantile_table(self, paths: dict) -> dict[str, np.ndarray]:
        """(3, series, months) per level present in `paths`."""
        out = {}
        if "upper" in paths:
            up_lev = self.ser.meta["level"].to_numpy()[self.ser.upper_idx]
            q_up = fu.quantiles(paths["upper"])
            out.update({lv: q_up[:, up_lev == lv] for lv in UPPER})
        if "skus" in paths:
            out["SKU"] = fu.quantiles(paths["skus"])
            out["Region × SKU"] = fu.quantiles(paths["pairs"])
        return out


def actual_by_level(ser: ds.Series) -> dict[str, np.ndarray]:
    rows = level_rows(ser)
    return {lv: ser.y[idx] for lv, idx in rows.items()}


def score_table(ser: ds.Series, q: dict[str, np.ndarray], T: int, years: dict[str, int]) -> dict:
    """Scores per level and forecast year."""
    act = actual_by_level(ser)
    out = {}
    for lv in [lv for lv in LEVELS if lv in q]:
        out[lv] = []
        for j in range(years[lv]):
            a = act[lv][:, T + 12 * j:T + 12 * (j + 1)]
            out[lv].append(fu.scores(a, q[lv][:, :, 12 * j:12 * (j + 1)]))
    return out


def mean_wql(table: dict) -> float:
    return float(np.nanmean([c["wql"] for cells in table.values() for c in cells]))


def years_by_level(upper_years: int) -> dict[str, int]:
    return {lv: (upper_years if lv in UPPER else min(upper_years, SKU_YEARS)) for lv in LEVELS}


# ---------------------------------------------------------------- ground truth


def truth_by_series(ser: ds.Series, w: ds.World, reps: np.ndarray, rep_sku: np.ndarray) -> np.ndarray:
    """Replications aggregated to the series of `ser`: (replication, series, months 37..120)."""
    row = {s: i for i, s in enumerate(rep_sku)}
    cols = np.array([row[s] for s in w.skus["sku_id"]])
    U = reps[:, :, cols, :].astype(np.float32)  # (rep, region, sku row, month)
    cat = w.skus["category_id"].to_numpy()
    out = np.zeros((U.shape[0], len(ser.meta), U.shape[3]), dtype=np.float32)
    for i, m in enumerate(ser.meta.itertuples()):
        r = slice(None) if m.region < 0 else m.region - 1
        if m.sku_id > 0:
            srow = int(np.nonzero(w.skus["sku_id"].to_numpy() == m.sku_id)[0][0])
            v = U[:, r, srow, :]
        elif m.category >= 0:
            v = U[:, r][..., cat == m.category, :].sum(axis=-2)
        else:
            v = U[:, r].sum(axis=-2)
        out[:, i] = v if v.ndim == 2 else v.sum(axis=1)
    return out


# ---------------------------------------------------------------- capacity check


def capacity_plan(demand_paths: np.ndarray, rho: float) -> float:
    """Capacity minimising 4 c K + m E[sum over 48 months of (D - K)+] over pooled path-months, c / m = rho:
    the (1 - rho / 12) quantile of the pooled monthly demand."""
    share = rho / 12
    if share >= 1:
        return 0.0
    return float(np.quantile(demand_paths.ravel(), 1 - share))


def capacity_cost(K: float, demand: np.ndarray, c: float, m: float) -> np.ndarray:
    """Cost over years 7 to 10 for each row of `demand` (row, 48 months)."""
    return 4 * c * K + m * np.maximum(demand - K, 0).sum(axis=-1)


# ---------------------------------------------------------------- the run


def truth_table(ser: ds.Series, tr: np.ndarray, T: int, yrs: dict[str, int], q: dict | None) -> dict:
    """Per level and forecast year, the ground-truth mean as a forecast and, given our quantiles `q`, the
    share of replications inside our P10 to P90 and the true band width over ours."""
    tq = np.quantile(tr, fu.QUANTILES, axis=0)  # (3, series, months)
    tmean = tr.mean(axis=0)
    tab = {}
    for lv, idx in level_rows(ser).items():
        tab[lv] = []
        for j in range(yrs[lv]):
            m = slice(T + 12 * j - 36, T + 12 * (j + 1) - 36)  # truth months start at 37
            a = ser.y[idx, T + 12 * j:T + 12 * (j + 1)]
            cell = {"wape_truth_mean": fu.wape(a, tmean[idx, m]), "wql_truth": fu.scores(a, tq[:, idx, m])["wql"]}
            if q is not None:
                ours = q[lv][:, :, 12 * j:12 * (j + 1)]
                inside = (tr[:, idx, m] >= ours[0][None]) & (tr[:, idx, m] <= ours[2][None])
                cell["truth_inside_our_p10_p90"] = float(inside.mean())
                cell["true_p10_p90_width_vs_ours"] = float((tq[2, idx, m] - tq[0, idx, m]).sum() / max((ours[2] - ours[0]).sum(), 1e-9))
            tab[lv].append(cell)
    return tab


def capacity_check(ser_t: ds.Series, w: ds.World, truth_t: np.ndarray, test: dict, chosen: str) -> dict:
    """Capacity per region for years 7 to 10 (forecast months 13 to 60 from month 60) under each plan, costed
    on the realised years and on replications 100 to 199; the ground-truth plan uses replications 0 to 99."""
    reg_rows_t = np.nonzero(ser_t.meta["level"].to_numpy() == "Region")[0]
    reg_up = np.searchsorted(ser_t.upper_idx, reg_rows_t)
    months = slice(TEST_FIT + 12, TEST_FIT + 60)  # months 73..120
    fmonths = slice(12, 60)
    actual = ser_t.y[reg_rows_t, months]  # (8, 48)
    tr_reg = truth_t[:, reg_rows_t, 36:84]  # truth months 73..120
    plan_reps, eval_reps = tr_reg[:100], tr_reg[100:]
    sk = w.skus
    price = sk["price"].to_numpy()
    margin_unit = (sk["price"] * sk["margin"]).to_numpy()
    hist = w.units[:, :, :TEST_FIT].sum(axis=2)  # (region, sku) units in years 1 to 5
    p_r = (hist * price).sum(1) / hist.sum(1)
    m_r = (hist * margin_unit).sum(1) / hist.sum(1)
    rho_game = GAME_CAPACITY_SHARE * p_r / m_r
    plans = {n: test[n]["paths"]["upper"][:, reg_up, fmonths] for n in METHODS}

    def capacity(plan: str, r: int, rho: float) -> float:
        if plan == "Our forecast":
            return capacity_plan(plans[chosen][:, r], rho)
        if plan == "Our P50 only":
            return float(np.quantile(plans[chosen][:, r], 0.5, axis=0).max())
        if plan == "Last year's peak":
            return float(ser_t.y[reg_rows_t[r], TEST_FIT - 12:TEST_FIT].max())
        if plan == "Ground truth":
            return capacity_plan(plan_reps[:, r], rho)
        return capacity_plan(plans[plan][:, r], rho)

    curves = {}
    for plan in ["Our forecast", "Our P50 only", "Seasonal naive", "Last year's peak", "Ground truth"] + [n for n in METHODS if n not in (chosen, "Seasonal naive")]:
        realised, expected = [], []
        for rho in RHO:
            rr, ee = 0.0, 0.0
            for r in range(len(reg_rows_t)):
                K, c = capacity(plan, r, rho), rho * m_r[r]
                rr += float(capacity_cost(K, actual[r][None], c, m_r[r])[0])
                ee += float(capacity_cost(K, eval_reps[:, r], c, m_r[r]).mean())
            realised.append(rr)
            expected.append(ee)
        curves[plan] = {"realised": realised, "expected": expected}
    per_region = []  # at each region's game ratio
    for r in range(len(reg_rows_t)):
        rho = float(rho_game[r])
        c = rho * m_r[r]
        row = {"region": int(ser_t.meta.iloc[reg_rows_t[r]]["region"]), "rho_game": rho, "price": float(p_r[r]), "margin": float(m_r[r])}
        for plan in ["Our forecast", "Our P50 only", "Seasonal naive", "Last year's peak", "Ground truth"]:
            K = capacity(plan, r, rho)
            row[plan] = {"capacity": K, "realised": float(capacity_cost(K, actual[r][None], c, m_r[r])[0]),
                         "expected": float(capacity_cost(K, eval_reps[:, r], c, m_r[r]).mean())}
        per_region.append(row)
    return {"rho": RHO.tolist(), "rho_game": rho_game.tolist(), "game_capacity_share": GAME_CAPACITY_SHARE, "curves": curves, "per_region": per_region}


def run(world_dir: str | Path, out_dir: str | Path, log=print) -> dict:
    t_start = time.time()
    world, out = Path(world_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w = ds.load(world)
    rep_file = np.load(world / "truth_replications.npz")
    reps, rep_sku = rep_file["units"], rep_file["sku_id"]
    years5 = years_by_level(TEST_YEARS)
    years2 = years_by_level(VAL_YEARS)

    # The validation series see months 1..60 only; the test months are blanked so nothing can read them.
    w_val = ds.World(w.skus, w.regions, w.units.copy(), w.bulk.copy())
    w_val.units[:, :, TEST_FIT:] = 0
    w_val.bulk[:, :, TEST_FIT:] = 0
    ser_v = ds.build_series(w_val, VAL_FIT)
    st_v = ds.static_features(ser_v, w_val)

    # ---- validation: hyperparameters, then width, reconciliation and method
    log("validation")
    val = {}
    for i, name in enumerate(METHODS):
        f, trials = tune(name, ser_v, st_v, VAL_FIT, log)
        pr = Probabilistic(f, VAL_YEARS, 100 + i, st_v, log)
        # Upper width and reconciliation on the upper levels, then the SKU width on the SKU levels.
        grid_u = []
        for k in WIDTHS_UPPER:
            for rec in UPPER_RECS:
                tab = score_table(ser_v, pr.quantile_table({"upper": pr.upper_paths(k, rec)}), VAL_FIT, years2)
                grid_u.append({"width": k, "reconciliation": rec, "score": mean_wql(tab)})
        bu = min(grid_u, key=lambda g: g["score"])
        upper = pr.upper_paths(bu["width"], bu["reconciliation"])
        grid_l = []
        for k in WIDTHS_LOWER:
            tab = score_table(ser_v, pr.quantile_table(pr.lower_paths(k, upper)), VAL_FIT, years2)
            grid_l.append({"width": k, "score": mean_wql(tab)})
        bl = min(grid_l, key=lambda g: g["score"])
        paths = pr.paths(bu["width"], bl["width"], bu["reconciliation"])
        q_best = pr.quantile_table(paths)
        tab = score_table(ser_v, q_best, VAL_FIT, years2)
        best = {"width_upper": bu["width"], "width_lower": bl["width"], "reconciliation": bu["reconciliation"], "score": mean_wql(tab), "table": tab}
        val[name] = {"fitted": f, "trials": trials, "grid_upper": grid_u, "grid_lower": grid_l, "best": best, "fallback": f.fallback,
                     "q": q_best, "setting": (f.model.setting if f.model is not None else {}),
                     "error_origins": {"origins": pr.origins, "observed": pr.observed, "level_errors": len(pr.blocks.level)}}
        log(f"  {name}: score {best['score']:.4f} widths {bu['width']}/{bl['width']} {bu['reconciliation']} fallback {f.fallback}")
    chosen = min(METHODS, key=lambda n: val[n]["best"]["score"])
    log(f"chosen: {chosen}")

    # ---- test: refit on years 1 to 5, forecast years 6 to 10 once
    log("test")
    ser_t = ds.build_series(w, TEST_FIT)
    st_t = ds.static_features(ser_t, w)
    test = {}
    for i, name in enumerate(METHODS):
        v = val[name]
        f = refit(v["fitted"], ser_t, st_t, TEST_FIT)
        pr = Probabilistic(f, TEST_YEARS, 200 + i, st_t, log)
        b = v["best"]
        paths = pr.paths(b["width_upper"], b["width_lower"], b["reconciliation"])
        q = pr.quantile_table(paths)
        test[name] = {"fitted": f, "pr": pr, "paths": paths, "q": q, "table": score_table(ser_t, q, TEST_FIT, years5), "fallback": f.fallback,
                      "error_origins": {"origins": pr.origins, "observed": pr.observed, "level_errors": len(pr.blocks.level)}}
        log(f"  {name}: test score {mean_wql(test[name]['table']):.4f}")

    # ---- ground truth and capacity, first future
    log("ground truth")
    truth_t = truth_by_series(ser_t, w, reps, rep_sku)  # months 37..120
    truth_v = truth_by_series(ser_v, w, reps, rep_sku)
    truth = {"test": truth_table(ser_t, truth_t, TEST_FIT, years5, test[chosen]["q"]),
             "validation": truth_table(ser_v, truth_v, VAL_FIT, years2, None)}
    log("capacity")
    capacity = capacity_check(ser_t, w, truth_t, test, chosen)

    # ---- second future: years 6 to 10 drawn again under the same rules with another seed, scored once.
    # Years 1 to 5 are the same, so the fits, forecasts and bands above are the ones scored.
    log("second future")
    w2 = ds.load(world, world / SECOND_FUTURE)
    ser_t2 = ds.build_series(w2, TEST_FIT)
    assert ser_t2.meta.equals(ser_t.meta) and np.array_equal(ser_t2.y[:, :TEST_FIT], ser_t.y[:, :TEST_FIT])
    rep2 = np.load(world / SECOND_FUTURE / "truth_replications.npz")
    truth_t2 = truth_by_series(ser_t2, w2, rep2["units"], rep2["sku_id"])
    second = {
        "test": {n: {"table": score_table(ser_t2, test[n]["q"], TEST_FIT, years5)} for n in METHODS},
        "truth": truth_table(ser_t2, truth_t2, TEST_FIT, years5, test[chosen]["q"]),
        "capacity": capacity_check(ser_t2, w2, truth_t2, test, chosen),
        "world": json.loads((world / SECOND_FUTURE / "world_future.json").read_text()),
    }
    for n in METHODS:
        second["test"][n]["score"] = mean_wql(second["test"][n]["table"])
        log(f"  {n}: second-future score {second['test'][n]['score']:.4f}")

    # ---- explanations for LightGBM (test fit)
    shap = None
    lgb_fit = test["LightGBM"]["fitted"]
    win = ds.windows(ser_t.y[:, :TEST_FIT], ser_t.bulk[:, :TEST_FIT], [TEST_FIT], with_target=False)
    sv = fm.lgb_shap(lgb_fit.model, win)
    shap = {k: float(v) for k, v in sv.items()}

    # ---- outputs
    def clean(tab):
        return {lv: [{k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in c.items()} for c in cells] for lv, cells in tab.items()}

    overview = {
        "model": {"id": "demand-forecast", "title": "Demand five years out"},
        "facts": {
            "series_validation": {lv: len(i) for lv, i in level_rows(ser_v).items()},
            "series_test": {lv: len(i) for lv, i in level_rows(ser_t).items()},
            "fit_months": {"validation": VAL_FIT, "test": TEST_FIT}, "paths": N_PATHS, "replications": int(reps.shape[0]),
            "training_windows_validation": len(fm.training_windows(ser_v.y, ser_v.bulk, VAL_FIT).series),
        },
        "validation": {n: {"setting": val[n]["setting"], "fallback": val[n]["fallback"],
                            "trials": [{"setting": t["setting"], "score": t["score"]} for t in val[n]["trials"]],
                            "grid_upper": val[n]["grid_upper"], "grid_lower": val[n]["grid_lower"],
                            "best": {k: (clean(v) if k == "table" else v) for k, v in val[n]["best"].items()},
                            "error_origins": val[n]["error_origins"]}
                    for n in METHODS},
        "chosen": chosen,
        "test": {n: {"table": clean(test[n]["table"]), "score": mean_wql(test[n]["table"]), "fallback": test[n]["fallback"],
                     "error_origins": test[n]["error_origins"]} for n in METHODS},
        "truth": {k: clean(v) for k, v in truth.items()},
        "capacity": capacity,
        "second_future": {"test": {n: {"table": clean(v["table"]), "score": v["score"]} for n, v in second["test"].items()},
                          "truth": clean(second["truth"]), "capacity": second["capacity"], "world": second["world"]},
        "error_samples": {"first_refit": FIRST_REFIT, "refit_every": REFIT_EVERY, "min_level_months": fu.MIN_LEVEL_MONTHS},
        "shap": shap,
        "runtime_s": round(time.time() - t_start),
    }
    (out / "overview.json").write_text(json.dumps(overview, separators=(",", ":"), default=float))
    (out / "series.json").write_text(json.dumps(series_payload(ser_v, ser_t, val, test, truth_v, truth_t, chosen, ser_t2, truth_t2), separators=(",", ":")))
    log(f"done in {overview['runtime_s']} s")
    return overview


def median_series(ser: ds.Series, level: str, upto: int) -> int:
    idx = level_rows(ser)[level]
    vol = ser.y[idx, :upto].sum(1)
    return int(idx[np.argsort(vol)[len(vol) // 2]])


def series_payload(ser_v, ser_t, val, test, truth_v, truth_t, chosen, ser_t2, truth_t2) -> dict:
    """Histories, forecasts and truth bands for the presets (median-volume series per level) and for every
    region, rounded to whole units."""

    def fan(pr_q, lv, pos):
        return [np.round(pr_q[lv][i, pos]).astype(int).tolist() for i in range(3)]

    out = {"presets": [], "regions": []}
    for lv in LEVELS:
        iv = median_series(ser_v, lv, 36)
        key = ser_v.meta.iloc[iv]
        it = int(np.nonzero((ser_t.meta[["level", "region", "category", "sku_id"]] == key[["level", "region", "category", "sku_id"]].values).all(1))[0][0])
        pos_v = int(np.nonzero(level_rows(ser_v)[lv] == iv)[0][0])
        pos_t = int(np.nonzero(level_rows(ser_t)[lv] == it)[0][0])
        out["presets"].append({
            "level": lv, "region": int(key["region"]), "category": int(key["category"]), "sku_id": int(key["sku_id"]),
            "actual": np.round(ser_t.y[it]).astype(int).tolist(),
            "validation": {n: fan(val[n]["q"], lv, pos_v) for n in METHODS},
            "test": fan(test[chosen]["q"], lv, pos_t),
            "truth_test": [np.round(np.quantile(truth_t[:, it, 24:], q, axis=0)).astype(int).tolist() for q in (0.1, 0.5, 0.9)],
            "actual_second_future": np.round(ser_t2.y[it, TEST_FIT:]).astype(int).tolist(),
            "truth_second_future": [np.round(np.quantile(truth_t2[:, it, 24:], q, axis=0)).astype(int).tolist() for q in (0.1, 0.5, 0.9)],
        })
    reg = level_rows(ser_t)["Region"]
    tot = level_rows(ser_t)["Total"]
    for lv, idx in (("Total", tot), ("Region", reg)):
        for p, i in enumerate(idx):
            out["regions"].append({"level": lv, "region": int(ser_t.meta.iloc[i]["region"]), "actual": np.round(ser_t.y[i]).astype(int).tolist(),
                                       "test": fan(test[chosen]["q"], lv, p),
                                   "truth_test": [np.round(np.quantile(truth_t[:, i, 24:], q, axis=0)).astype(int).tolist() for q in (0.1, 0.5, 0.9)],
                                   "actual_second_future": np.round(ser_t2.y[i, TEST_FIT:]).astype(int).tolist(),
                                   "truth_second_future": [np.round(np.quantile(truth_t2[:, i, 24:], q, axis=0)).astype(int).tolist() for q in (0.1, 0.5, 0.9)]})
    return out


def ensure_world(world_dir: str | Path, log=print) -> None:
    """Build years 6 to 10 and the ground-truth replications if they are missing."""
    from scgame.world.future import generate_future
    from scgame.world.replicate import replicate

    world = Path(world_dir)
    if not (world / "bulk_buyers.parquet").exists():
        raise FileNotFoundError(f"{world} lacks bulk_buyers.parquet; run `python -m scgame.world.generator {world} 7` first")
    if not (world / "skus_future.parquet").exists():
        log("building years 6 to 10")
        generate_future(world)
    if not (world / "truth_replications.npz").exists():
        log("building 200 ground-truth replications (about 13 minutes)")
        replicate(world, 200, 4)
    second = world / SECOND_FUTURE
    if not (second / "skus_future.parquet").exists():
        log("building the second future")
        generate_future(world, stream=SECOND_FUTURE_STREAM, out_dir=second)
    if not (second / "truth_replications.npz").exists():
        log("building 200 ground-truth replications of the second future (about 13 minutes)")
        replicate(world, 200, 4, future_dir=second, stream=SECOND_REPLICATION_STREAM)


def main(world_dir: str, out_dir: str) -> None:
    warnings.filterwarnings("ignore")
    def log(s: str) -> None:
        print(s, flush=True)
    ensure_world(world_dir, log)
    run(world_dir, out_dir, log=log)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../data/world_v1", sys.argv[2] if len(sys.argv) > 2 else "../data/models/demand-forecast")
