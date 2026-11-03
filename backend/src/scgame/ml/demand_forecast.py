"""Demand forecast (model B): validation, test, ground truth and the capacity check.

Fit on years 1 to 3 (months 1 to 36); every setting is chosen on years 4 and 5 (months 37 to 60, forecast
from month 36). The chosen settings are refitted on years 1 to 5 and years 6 to 10 are forecast once from
month 60 (upper levels five years ahead, SKU levels two). Years 6 to 10 are read only by the test step.
The ground truth (200 replications, `scgame.world.replicate`) checks the forecasts and is never used to
choose. Design and decisions in `design/demand-forecast/PROPOSAL.md`, the build in
`design/demand-forecast/HANDOVER.md`.

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

    def past_year(self, origins: list[int]) -> np.ndarray:
        """One-year-ahead forecasts at origins inside the fitting window (series, origin, 12)."""
        y = self.ser.y
        if self.name == "Seasonal naive":
            return np.stack([fb.seasonal_naive(y, T, 12) for T in origins], axis=1)
        if self.name == "ETS":
            return np.stack([fb.ets_forecast(self.ets, y, T, 12)[0] for T in origins], axis=1)
        return fm.in_sample_year(self.model, y, self.ser.bulk, origins)


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
            for epochs, model in fm.fit_mlp(ser.y, ser.bulk, ser.meta, static, fit_end, setting).items():
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
    return Fitted(f.name, ser, fit_end, model=fm.fit_mlp(ser.y, ser.bulk, ser.meta, static, fit_end, s)[epochs])


# ---------------------------------------------------------------- paths, reconciliation, scores


class Probabilistic:
    """Mean forecasts, error blocks and the coherent sample paths of one fitted method."""

    def __init__(self, f: Fitted, years: int, seed: int):
        ser, T = f.ser, f.fit_end
        self.f, self.ser, self.T, self.years = f, ser, T, years
        self.mean = f.forecast(years)
        origins = list(range(12, T - 12 + 1))
        past = f.past_year(origins)
        actual = np.stack([ser.y[:, o:o + 12] for o in origins], axis=1)
        self.blocks = fu.error_blocks(actual, past, origins)
        up = ser.upper_idx
        e_units = (actual[up] - past[up]).transpose(1, 2, 0).reshape(-1, len(up))
        bottom = np.searchsorted(up, ser.rc_idx)
        self.P = {"Bottom-up": fu.bottom_up_matrix(ser.S_upper, bottom), "MinT": fu.mint_matrix(ser.S_upper, fu.shrunk_covariance(e_units))}
        self.draws = fu.draw_origins(len(origins), years, N_PATHS, np.random.default_rng(seed))

    def upper_paths(self, k: float, rec: str) -> np.ndarray:
        up = self.ser.upper_idx
        return fu.reconcile(self.P[rec], fu.sample_paths(self.mean[up], _sub(self.blocks, up), self.draws, k, self.T))

    def lower_paths(self, k: float, upper: np.ndarray) -> dict[str, np.ndarray]:
        """Region × SKU paths scaled to the reconciled Region × category paths (middle-out), and SKU paths as
        their sums, so every level adds up."""
        ser = self.ser
        Hs = 12 * min(self.years, SKU_YEARS)
        pairs = fu.sample_paths(self.mean[ser.pair_idx, :Hs], _sub(self.blocks, ser.pair_idx), self.draws[:, :Hs // 12], k, self.T)
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


def _sub(b: fu.ErrorBlocks, idx: np.ndarray) -> fu.ErrorBlocks:
    return fu.ErrorBlocks(b.level[:, idx], b.noise[:, idx])


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
        pr = Probabilistic(f, VAL_YEARS, seed=100 + i)
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
                     "q": q_best, "setting": (f.model.setting if f.model is not None else {})}
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
        pr = Probabilistic(f, TEST_YEARS, seed=200 + i)
        b = v["best"]
        paths = pr.paths(b["width_upper"], b["width_lower"], b["reconciliation"])
        q = pr.quantile_table(paths)
        test[name] = {"fitted": f, "pr": pr, "paths": paths, "q": q, "table": score_table(ser_t, q, TEST_FIT, years5), "fallback": f.fallback}
        log(f"  {name}: test score {mean_wql(test[name]['table']):.4f}")

    # ---- ground truth
    log("ground truth")
    truth_t = truth_by_series(ser_t, w, reps, rep_sku)  # months 37..120
    truth_v = truth_by_series(ser_v, w, reps, rep_sku)
    truth = {}
    for label, ser, tr, T, yrs, qsrc in (("test", ser_t, truth_t, TEST_FIT, years5, test[chosen]["q"]),
                                          ("validation", ser_v, truth_v, VAL_FIT, years2, None)):
        rows = level_rows(ser)
        tq = np.quantile(tr, fu.QUANTILES, axis=0)  # (3, series, months)
        tmean = tr.mean(axis=0)
        tab = {}
        for lv, idx in rows.items():
            tab[lv] = []
            for j in range(yrs[lv]):
                m = slice(T + 12 * j - 36, T + 12 * (j + 1) - 36)  # truth months start at 37
                a = ser.y[idx, T + 12 * j:T + 12 * (j + 1)]
                cell = {"wape_truth_mean": fu.wape(a, tmean[idx, m]), "wql_truth": fu.scores(a, tq[:, idx, m])["wql"]}
                if qsrc is not None:
                    ours = qsrc[lv][:, :, 12 * j:12 * (j + 1)]
                    inside = (tr[:, idx, m] >= ours[0][None]) & (tr[:, idx, m] <= ours[2][None])
                    cell["truth_inside_our_p10_p90"] = float(inside.mean())
                    cell["true_p10_p90_width_vs_ours"] = float((tq[2, idx, m] - tq[0, idx, m]).sum() / max((ours[2] - ours[0]).sum(), 1e-9))
                tab[lv].append(cell)
        truth[label] = tab

    # ---- capacity check on years 7 to 10 (forecast months 13 to 60 from month 60)
    log("capacity")
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
    plans = {f"{n}": test[n]["paths"]["upper"][:, reg_up, fmonths] for n in METHODS}
    curves = {}
    per_region = []
    for plan in ["Our forecast", "Our P50 only", "Seasonal naive", "Last year's peak", "Ground truth"] + [n for n in METHODS if n not in (chosen, "Seasonal naive")]:
        realised, expected = [], []
        for rho_scale in RHO:
            rr, ee = 0.0, 0.0
            for r in range(len(reg_rows_t)):
                rho = rho_scale
                c = rho * m_r[r]
                if plan == "Our forecast":
                    K = capacity_plan(plans[chosen][:, r], rho)
                elif plan == "Our P50 only":
                    K = float(np.quantile(plans[chosen][:, r], 0.5, axis=0).max())
                elif plan == "Last year's peak":
                    K = float(ser_t.y[reg_rows_t[r], TEST_FIT - 12:TEST_FIT].max())
                elif plan == "Ground truth":
                    K = capacity_plan(plan_reps[:, r], rho)
                else:
                    K = capacity_plan(plans[plan][:, r], rho)
                rr += float(capacity_cost(K, actual[r][None], c, m_r[r])[0])
                ee += float(capacity_cost(K, eval_reps[:, r], c, m_r[r]).mean())
            realised.append(rr)
            expected.append(ee)
        curves[plan] = {"realised": realised, "expected": expected}
    # Per region at each region's game ratio.
    for r in range(len(reg_rows_t)):
        rho = float(rho_game[r])
        c = rho * m_r[r]
        row = {"region": int(ser_t.meta.iloc[reg_rows_t[r]]["region"]), "rho_game": rho, "price": float(p_r[r]), "margin": float(m_r[r])}
        for plan in ["Our forecast", "Our P50 only", "Seasonal naive", "Last year's peak", "Ground truth"]:
            if plan == "Our forecast":
                K = capacity_plan(plans[chosen][:, r], rho)
            elif plan == "Our P50 only":
                K = float(np.quantile(plans[chosen][:, r], 0.5, axis=0).max())
            elif plan == "Last year's peak":
                K = float(ser_t.y[reg_rows_t[r], TEST_FIT - 12:TEST_FIT].max())
            elif plan == "Ground truth":
                K = capacity_plan(plan_reps[:, r], rho)
            else:
                K = capacity_plan(plans["Seasonal naive"][:, r], rho)
            row[plan] = {"capacity": K, "realised": float(capacity_cost(K, actual[r][None], c, m_r[r])[0]),
                             "expected": float(capacity_cost(K, eval_reps[:, r], c, m_r[r]).mean())}
        per_region.append(row)

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
                            "best": {k: (clean(v) if k == "table" else v) for k, v in val[n]["best"].items()}}
                    for n in METHODS},
        "chosen": chosen,
        "test": {n: {"table": clean(test[n]["table"]), "score": mean_wql(test[n]["table"]), "fallback": test[n]["fallback"]} for n in METHODS},
        "truth": {k: clean(v) for k, v in truth.items()},
        "capacity": {"rho": RHO.tolist(), "rho_game": rho_game.tolist(), "game_capacity_share": GAME_CAPACITY_SHARE, "curves": curves, "per_region": per_region},
        "shap": shap,
        "runtime_s": round(time.time() - t_start),
    }
    (out / "overview.json").write_text(json.dumps(overview, separators=(",", ":"), default=float))
    (out / "series.json").write_text(json.dumps(series_payload(ser_v, ser_t, val, test, truth_v, truth_t, chosen), separators=(",", ":")))
    log(f"done in {overview['runtime_s']} s")
    return overview


def median_series(ser: ds.Series, level: str, upto: int) -> int:
    idx = level_rows(ser)[level]
    vol = ser.y[idx, :upto].sum(1)
    return int(idx[np.argsort(vol)[len(vol) // 2]])


def series_payload(ser_v, ser_t, val, test, truth_v, truth_t, chosen) -> dict:
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
        })
    reg = level_rows(ser_t)["Region"]
    tot = level_rows(ser_t)["Total"]
    for lv, idx in (("Total", tot), ("Region", reg)):
        for p, i in enumerate(idx):
            out["regions"].append({"level": lv, "region": int(ser_t.meta.iloc[i]["region"]), "actual": np.round(ser_t.y[i]).astype(int).tolist(),
                                       "test": fan(test[chosen]["q"], lv, p),
                                       "truth_test": [np.round(np.quantile(truth_t[:, i, 24:], q, axis=0)).astype(int).tolist() for q in (0.1, 0.5, 0.9)]})
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


def main(world_dir: str, out_dir: str) -> None:
    warnings.filterwarnings("ignore")
    log = lambda s: print(s, flush=True)
    ensure_world(world_dir, log)
    run(world_dir, out_dir, log=log)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../data/world_v1", sys.argv[2] if len(sys.argv) > 2 else "../data/models/demand-forecast")
