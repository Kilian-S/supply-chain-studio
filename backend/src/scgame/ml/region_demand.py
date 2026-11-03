"""New-region demand (model C): yearly units per category of a region before its first order.

Agreed design in `design/new-region-demand/PROPOSAL.md` (sections 4a and 13); build notes in `HANDOVER.md`.

- Example: one region × category. Label: log of the region's units per 1,000 inhabitants over the
  network's (index, network = 1), pooled over the fitting years. Rows weighted by their share of the
  region's units, so each region weighs the same.
- Methods: population share (index 1), nearest analogues (kNN), log-linear ridge regression, LightGBM.
- Units for a target year: index × population × the network's units per 1,000 inhabitants in the last
  observed year (last year repeated). With bulk "network average", only consumer units are modelled and
  bulk units are added at the network's bulk units per inhabitant.
- Validation: leave one region out, fitted on years 1 to 3, scored on the held-out region's year 4. Every
  setting and the method are chosen there.
- Uncertainty: the chosen setting's leave-one-region-out log errors, a region part (t with n − 1 degrees
  of freedom × √(1 + 1/n)) and a cell part (normal), 1,000 draws; months from the network's monthly shape
  of the last year, with count noise whose dispersion comes from the same folds.
- Test: refit on years 1 to 4 of the 8 regions, predict the 30 simulated new regions' year 5, scored once.
- Ground truth (`region_sim`): 200 replications, used to check only.

Usage: uv run python -m scgame.ml.region_demand <world dir> <out dir>
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from scgame.ml import forecast_uncertainty as fu
from scgame.ml import region_sim as rs
from scgame.ml.demand_forecast import (
    GAME_CAPACITY_SHARE,
    RHO,
    capacity_plan,
)
from scgame.world.generator import CATEGORIES

METHODS = ["Population share", "Nearest analogues", "Log-linear regression", "Gradient boosting"]
CATS = [c.name for c in CATEGORIES]
N_CAT = len(CATS)
N_DRAWS = 1_000
BULKS = ["in label", "network average"]
RIDGE_ALPHAS = [float(a) for a in np.round(np.geomspace(0.01, 100, 6), 4)]
RIDGE_SETS = ["income, urban", "income, urban, island", "income, urban, island, category × income"]
VAL_FIT, VAL_LEVEL, VAL_TARGET = [0, 1, 2], 2, 3  # year indices (0 = year 1)
TEST_FIT, TEST_LEVEL = [0, 1, 2, 3], 3


# ---------------------------------------------------------------- data


@dataclass
class Panel:
    regions: pd.DataFrame  # the 8 existing regions
    monthly: np.ndarray  # (8 regions, 8 categories, 60 months, 2 kinds: consumer, bulk)
    price: np.ndarray  # (8 categories) units-weighted price, year 4
    margin: np.ndarray  # (8 categories) units-weighted margin per unit, year 4

    @property
    def yearly(self) -> np.ndarray:  # (8, 8, 5 years, 2)
        m = self.monthly
        return m.reshape(m.shape[0], m.shape[1], 5, 12, 2).sum(axis=3)


def month_index(day: np.ndarray) -> np.ndarray:
    """Month 0..59 of day 1..1825 (twelve equal parts of each 365-day year)."""
    doy = (day - 1) % 365
    return (day - 1) // 365 * 12 + np.minimum((doy / 365 * 12).astype(int), 11)


def load_panel(world_dir: str | Path) -> Panel:
    world = Path(world_dir)
    reg = pd.read_parquet(world / "regions.parquet")
    sk = pd.read_parquet(world / "skus.parquet")
    cat_of = dict(zip(sk["sku_id"], sk["category"].map({c: i for i, c in enumerate(CATS)})))
    lines = pd.read_parquet(world / "order_lines.parquet")
    r = lines["region_id"].to_numpy() - 1
    c = lines["sku_id"].map(cat_of).to_numpy()
    m = month_index(lines["day"].to_numpy())
    k = (lines["quantity"].to_numpy() >= rs.BULK_QTY).astype(int)
    cube = np.zeros((len(reg), N_CAT, 60, 2))
    np.add.at(cube, (r, c, m, k), lines["quantity"].to_numpy())
    y4 = lines[(lines["day"] > 3 * 365) & (lines["day"] <= 4 * 365)].merge(sk[["sku_id", "category", "price", "margin"]], on="sku_id")
    y4 = y4.assign(rev=y4.quantity * y4.price, mar=y4.quantity * y4.price * y4.margin)
    g = y4.groupby("category")[["quantity", "rev", "mar"]].sum().reindex(CATS)
    return Panel(reg, cube, (g.rev / g.quantity).to_numpy(), (g.mar / g.quantity).to_numpy())


def features(reg: pd.DataFrame) -> pd.DataFrame:
    """Region features a shop can look up before entering. Not the generator's online adoption or customer
    share, not customer counts."""
    return pd.DataFrame({
        "log_income": np.log(reg["income_index"].to_numpy()),
        "urban": reg["urban_share"].to_numpy(dtype=float),
        "island": (reg["landmass"] != "continent").to_numpy(dtype=float),
        "log_pop": np.log(reg["population_m"].to_numpy()),
    })


# ---------------------------------------------------------------- methods


def settings() -> list[dict]:
    out = [{"method": "Population share", "bulk": "in label"}]
    for bulk in BULKS:
        for k in (1, 2, 3):
            for weighting in ("plain", "distance"):
                out.append({"method": "Nearest analogues", "bulk": bulk, "k": k, "weighting": weighting})
    for bulk in BULKS:
        for fs in RIDGE_SETS:
            for pop in (False, True):
                for a in RIDGE_ALPHAS:
                    out.append({"method": "Log-linear regression", "bulk": bulk, "features": fs, "log_population": pop, "alpha": a})
    for bulk in BULKS:
        for leaves in (2, 4):
            for rows in (4, 8):
                for trees in (50, 100, 200):
                    out.append({"method": "Gradient boosting", "bulk": bulk, "leaves": leaves, "rows_per_leaf": rows, "trees": trees})
    return out


def _z(train: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise columns on the training regions only; a constant column becomes 0."""
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    ok = sd > 1e-9
    sd = np.where(ok, sd, 1.0)
    return np.where(ok, (train - mu) / sd, 0.0), np.where(ok, (target - mu) / sd, 0.0)


def _ridge_design(z: np.ndarray, s: dict) -> np.ndarray:
    """Rows (region, category), region-major. z columns: log income, urban, island, log population."""
    cols = [0, 1] + ([2] if "island" in s["features"] else []) + ([3] if s["log_population"] else [])
    base = np.repeat(z[:, cols], N_CAT, axis=0)
    if "category" not in s["features"]:
        return base
    onehot = np.tile(np.eye(N_CAT), (len(z), 1))
    return np.hstack([base, onehot * np.repeat(z[:, [0]], N_CAT, axis=0)])


def fit_predict_log_index(s: dict, f_train: pd.DataFrame, label: np.ndarray, weight: np.ndarray, f_target: pd.DataFrame) -> np.ndarray:
    """Predicted log index (target regions, categories) from the training regions' label and weights (n, 8)."""
    n_t = len(f_target)
    if s["method"] == "Population share":
        return np.zeros((n_t, N_CAT))
    if s["method"] == "Nearest analogues":
        zt, zg = _z(f_train[["log_income", "urban", "island"]].to_numpy(), f_target[["log_income", "urban", "island"]].to_numpy())
        out = np.zeros((n_t, N_CAT))
        idx = np.exp(label)
        k = min(s["k"], len(zt))
        for j in range(n_t):
            d = np.sqrt(((zt - zg[j]) ** 2).sum(axis=1))
            near = np.argsort(d, kind="stable")[:k]
            if s["weighting"] == "distance":
                w = 1 / np.maximum(d[near], 1e-6)
            else:
                w = np.ones(k)
            out[j] = np.log((idx[near] * w[:, None]).sum(axis=0) / w.sum())
        return out
    if s["method"] == "Log-linear regression":
        zt, zg = _z(f_train.to_numpy(), f_target.to_numpy())
        model = Ridge(alpha=s["alpha"], fit_intercept=True)
        model.fit(_ridge_design(zt, s), label.ravel(), sample_weight=weight.ravel())
        return model.predict(_ridge_design(zg, s)).reshape(n_t, N_CAT)
    if s["method"] == "Gradient boosting":
        def rows(f: pd.DataFrame) -> np.ndarray:
            x = np.repeat(f.to_numpy(), N_CAT, axis=0)
            return np.hstack([x, np.tile(np.arange(N_CAT), len(f))[:, None]])
        model = lgb.LGBMRegressor(n_estimators=s["trees"], learning_rate=0.05, num_leaves=s["leaves"], min_child_samples=s["rows_per_leaf"],
                                  min_data_per_group=s["rows_per_leaf"], cat_smooth=0.0, cat_l2=0.0, min_child_weight=0.0, min_data_in_bin=1, max_bin=63, subsample=1.0, colsample_bytree=1.0,
                                  deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1, random_state=0)
        model.fit(rows(f_train), label.ravel(), sample_weight=weight.ravel(), categorical_feature=[4])
        return model.predict(rows(f_target)).reshape(n_t, N_CAT)
    raise ValueError(s["method"])


def predict_units(s: dict, yearly: np.ndarray, f_all: pd.DataFrame, pop_all: np.ndarray, train: np.ndarray, fit_years: list[int],
                  level: np.ndarray, f_target: pd.DataFrame, pop_target: np.ndarray) -> np.ndarray:
    """Yearly units (target regions, categories). `yearly` (regions, 8, years, 2) is read only for the training
    regions; `level` (training regions, 8, 2) holds their units in the year whose network rate scales the index."""
    U = yearly[train][:, :, fit_years, :].sum(axis=2)  # (n, 8, 2)
    y = U.sum(axis=-1) if s["bulk"] == "in label" else U[..., 0]
    pop_k = pop_all[train] * 1000
    rate = y / pop_k[:, None]
    net = y.sum(axis=0) / pop_k.sum()
    label = np.log(np.maximum(rate / net, 1e-9))
    weight = y / y.sum(axis=1, keepdims=True)
    p = fit_predict_log_index(s, f_all.iloc[train].reset_index(drop=True), label, weight, f_target.reset_index(drop=True))
    pop_t = pop_target[:, None] * 1000
    if s["bulk"] == "in label":
        return np.exp(p) * pop_t * level.sum(axis=-1).sum(axis=0) / pop_k.sum()
    cons = level[..., 0].sum(axis=0) / pop_k.sum()
    bulk = level[..., 1].sum(axis=0) / pop_k.sum()
    return pop_t * (np.exp(p) * cons + bulk)


# ---------------------------------------------------------------- scores


def region_scores(actual: np.ndarray, point: np.ndarray) -> dict:
    """Per region (rows), WAPE over its categories, WAPE of its total, bias of its total."""
    tot = actual.sum(axis=1)
    return {
        "wape_region_category": np.abs(point - actual).sum(axis=1) / tot,
        "wape_region": np.abs(point.sum(axis=1) - tot) / tot,
        "bias_region": (point.sum(axis=1) - tot) / tot,
    }


def selection_score(actual: np.ndarray, point: np.ndarray) -> float:
    """Mean over regions (equal weight) of the mean of the two levels' WAPE."""
    r = region_scores(actual, point)
    return float(np.mean((r["wape_region_category"] + r["wape_region"]) / 2))


def loo_predict(panel: Panel, s: dict, held_out: int, fit_years: list[int] = VAL_FIT, level_year: int = VAL_LEVEL) -> np.ndarray:
    """The held-out region's yearly units per category, from the other regions only."""
    n = len(panel.regions)
    train = np.array([i for i in range(n) if i != held_out])
    f = features(panel.regions)
    yearly = panel.yearly
    return predict_units(s, yearly, f, panel.regions["population_m"].to_numpy(), train, fit_years, yearly[train][:, :, level_year, :],
                         f.iloc[[held_out]], panel.regions["population_m"].to_numpy()[[held_out]])[0]


# ---------------------------------------------------------------- uncertainty


@dataclass
class ErrorModel:
    sigma_region: float
    sigma_cell: float
    phi: float  # month count-noise dispersion: var(log month ratio) = phi / expected units
    n: int  # held-out regions behind sigma_region

    def as_dict(self) -> dict:
        return {"sigma_region": self.sigma_region, "sigma_cell": self.sigma_cell, "month_dispersion": self.phi, "regions": self.n}


def error_model(actual: np.ndarray, point: np.ndarray, actual_months: np.ndarray, shape: np.ndarray) -> ErrorModel:
    """From held-out regions' yearly units (n, 8), predictions (n, 8), monthly totals (n, 12) and the network's
    monthly shape of the level year (8, 12) per region (n, 8, 12)."""
    e = np.log((actual + 1) / (point + 1))
    w = actual / actual.sum(axis=1, keepdims=True)
    a = (w * e).sum(axis=1)
    cell = e - a[:, None]
    sig_r = float(np.sqrt(np.mean(a ** 2)))
    sig_c = float(np.sqrt(np.mean((w * cell ** 2).sum(axis=1))))
    # Month noise around the year's total spread by the predicted category mix and the network's shape.
    mix = point / point.sum(axis=1, keepdims=True)
    s = np.einsum("nc,ncm->nm", mix, shape)
    expected = actual_months.sum(axis=1, keepdims=True) * s
    r = np.log((actual_months + 1) / (expected + 1))
    phi = float(np.mean(r ** 2 * expected))
    return ErrorModel(sig_r, sig_c, phi, len(actual))


def draws(point: np.ndarray, shape: np.ndarray, em: ErrorModel, rng: np.random.Generator, n: int = N_DRAWS) -> tuple[np.ndarray, np.ndarray]:
    """For one region, yearly units per category (n, 8) and monthly totals (n, 12)."""
    a = em.sigma_region * np.sqrt(1 + 1 / em.n) * rng.standard_t(em.n - 1, size=n)
    # Cell errors average to zero over the region (weighted by its predicted mix), as the fitted cell parts
    # do by construction, so the region total carries the region error only; rescaled so that their
    # weighted variance is sigma_cell^2.
    mix = point / point.sum()
    u = rng.normal(0, 1, size=(n, N_CAT))
    eps = (u - (u @ mix)[:, None]) * em.sigma_cell / np.sqrt(max(1 - (mix ** 2).sum(), 1e-9))
    yearly = point[None] * np.exp(a[:, None] + eps)
    months = np.einsum("dc,cm->dm", yearly, shape)
    expected = point @ shape
    months = months * np.exp(rng.normal(0, 1, size=(n, 12)) * np.sqrt(em.phi / np.maximum(expected, 1.0)))
    return yearly, months


def interval_scores(actual: np.ndarray, yearly_draws: list[np.ndarray]) -> dict:
    """Coverage of P10 to P90 and weighted quantile loss, cells and region totals, over regions."""
    qc = np.stack([np.quantile(d, fu.QUANTILES, axis=0) for d in yearly_draws], axis=1)  # (3, n, 8)
    qt = np.stack([np.quantile(d.sum(axis=1), fu.QUANTILES) for d in yearly_draws], axis=1)  # (3, n)
    tot = actual.sum(axis=1)
    return {
        "coverage_cells": float(np.mean((actual >= qc[0]) & (actual <= qc[2]))),
        "coverage_region": float(np.mean((tot >= qt[0]) & (tot <= qt[2]))),
        "wql_cells": fu.scores(actual, qc)["wql"],
        "wql_region": fu.scores(tot, qt)["wql"],
    }


def network_shape(monthly: np.ndarray, train: np.ndarray, year: int) -> np.ndarray:
    """(8, 12) each category's monthly share of the year in the training regions."""
    m = monthly[train][:, :, 12 * year:12 * (year + 1), :].sum(axis=(0, 3))
    return m / m.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- the run


def validate(panel: Panel, log=print) -> dict:
    n = len(panel.regions)
    yearly = panel.yearly
    actual = yearly[:, :, VAL_TARGET, :].sum(axis=-1)
    trials = []
    preds = {}
    for i, s in enumerate(settings()):
        point = np.stack([loo_predict(panel, s, h) for h in range(n)])
        sc = selection_score(actual, point)
        trials.append({"setting": s, "score": sc})
        preds[i] = point
    out = {}
    for m in METHODS:
        idx = [i for i, t in enumerate(trials) if t["setting"]["method"] == m]
        best = min(idx, key=lambda i: trials[i]["score"])  # first in grid order on ties
        out[m] = {"trials": [trials[i] for i in idx], "setting": trials[best]["setting"], "score": trials[best]["score"], "point": preds[best]}
        log(f"validation {m}: {trials[best]['score']:.4f} {trials[best]['setting']}")
    return out


def run(world_dir: str | Path, out_dir: str | Path, log=print, validation_only: bool = False) -> dict:
    """The whole model. `validation_only` stops after the validation and the learning curve and writes
    `validation.json` instead, so the test is scored once, after validation has been reviewed."""
    t0 = time.time()
    out = Path(out_dir)
    panel = load_panel(world_dir)
    reg = panel.regions
    n = len(reg)
    yearly = panel.yearly
    pop = reg["population_m"].to_numpy()
    f = features(reg)
    rng = np.random.default_rng([7, 34])

    # Validation.
    val = validate(panel, log)
    chosen = min(METHODS, key=lambda m: val[m]["score"])
    log(f"chosen: {chosen}")
    actual_v = yearly[:, :, VAL_TARGET, :].sum(axis=-1)
    months_v = panel.monthly[:, :, 12 * VAL_TARGET:12 * (VAL_TARGET + 1), :].sum(axis=(1, 3))  # (8, 12)
    shapes_v = np.stack([network_shape(panel.monthly, np.array([j for j in range(n) if j != h]), VAL_LEVEL) for h in range(n)])
    validation = {}
    errors = {}
    for m in METHODS:
        v = val[m]
        point = v["point"]
        em = error_model(actual_v, point, months_v, shapes_v)
        errors[m] = em
        # Validation coverage: each held-out region's band from the other 7 regions' errors.
        dv = []
        for h in range(n):
            o = np.array([j for j in range(n) if j != h])
            em_h = error_model(actual_v[o], point[o], months_v[o], shapes_v[o])
            dv.append(draws(point[h], shapes_v[h], em_h, rng)[0])
        rsc = region_scores(actual_v, point)
        validation[m] = {
            "setting": v["setting"], "score": v["score"], "trials": v["trials"], "errors": em.as_dict(),
            "intervals": interval_scores(actual_v, dv),
            "per_region": [{"region": reg["name"].iloc[h], **{k: float(rsc[k][h]) for k in rsc},
                            "actual": actual_v[h].tolist(), "p50": point[h].tolist(),
                            "p10": np.quantile(dv[h], 0.1, axis=0).tolist(), "p90": np.quantile(dv[h], 0.9, axis=0).tolist()} for h in range(n)],
            "wape_region_category": float(rsc["wape_region_category"].mean()), "wape_region": float(rsc["wape_region"].mean()),
        }
    # Realised-network diagnostic on validation (the target year's network level instead of last year's).
    for m in METHODS:
        s = val[m]["setting"]
        pt = np.stack([predict_units(s, yearly, f, pop, np.array([j for j in range(n) if j != h]), VAL_FIT,
                                     yearly[[j for j in range(n) if j != h]][:, :, VAL_TARGET, :], f.iloc[[h]], pop[[h]])[0] for h in range(n)])
        validation[m]["realised_network_score"] = selection_score(actual_v, pt)

    # Learning curve at each method's chosen setting.
    curve = {}
    for m in METHODS:
        s = val[m]["setting"]
        rows = []
        for k in range(2, n):
            per_subset = []
            for sub in itertools.combinations(range(n), k):
                train = np.array(sub)
                rest = np.array([j for j in range(n) if j not in sub])
                pt = predict_units(s, yearly, f, pop, train, VAL_FIT, yearly[train][:, :, VAL_LEVEL, :], f.iloc[rest], pop[rest])
                per_subset.append(selection_score(actual_v[rest], pt))
            rows.append({"training_regions": k, "subsets": len(per_subset), "mean": float(np.mean(per_subset)),
                         "p10": float(np.quantile(per_subset, 0.1)), "p90": float(np.quantile(per_subset, 0.9))})
        curve[m] = rows  # at 7 training regions this is leave one region out
    log("learning curve done")
    if validation_only:
        summary = {"chosen": chosen, "validation": {m: {k: v for k, v in validation[m].items() if k != "trials"} for m in METHODS},
                   "learning_curve": curve}
        (out / "validation.json").write_text(json.dumps(summary, indent=1, default=_json) + "\n")
        return summary
    sim_path = out / "simulation.npz"
    if not sim_path.exists():
        rs.simulate(world_dir, out, 200, 4, log)
    sim = np.load(sim_path)

    test_reg = pd.read_csv(out / "test_regions.csv")
    f_test = features(test_reg)
    pop_test = test_reg["population_m"].to_numpy()

    # Test, scored once: refit on years 1 to 4 of the 8 regions, predict the 30 new regions' year 5.
    all_idx = np.arange(n)
    actual_t = sim["test_realised"].sum(axis=(2, 3)).astype(float)  # (30, 8)
    months_t = sim["test_realised"].sum(axis=(1, 3)).astype(float)  # (30, 12)
    shape_t = network_shape(panel.monthly, all_idx, TEST_LEVEL)
    inside = test_reg["inside_range"].to_numpy(bool)
    test = {}
    test_draws = {}
    for m in METHODS:
        s = val[m]["setting"]
        point = predict_units(s, yearly, f, pop, all_idx, TEST_FIT, yearly[:, :, TEST_LEVEL, :], f_test, pop_test)
        dd = [draws(point[j], shape_t, errors[m], rng) for j in range(len(point))]
        test_draws[m] = (point, dd)
        rsc = region_scores(actual_t, point)
        block = {"score": selection_score(actual_t, point),
                 "wape_region_category": float(rsc["wape_region_category"].mean()), "wape_region": float(rsc["wape_region"].mean()),
                 "bias_region": float(rsc["bias_region"].mean()),
                 **interval_scores(actual_t, [d[0] for d in dd])}
        for name, mask in (("inside_range", inside), ("outside_range", ~inside)):
            block[name] = {"score": selection_score(actual_t[mask], point[mask]),
                           **interval_scores(actual_t[mask], [d[0] for d, k in zip(dd, mask) if k])}
        island = test_reg["landmass"].to_numpy() != "continent"
        for name, mask in (("islands", island), ("continent", ~island)):
            block[name] = {"score": selection_score(actual_t[mask], point[mask])}
        realised_net = predict_units(s, yearly, f, pop, all_idx, TEST_FIT, yearly[:, :, 4, :], f_test, pop_test)
        block["realised_network_score"] = selection_score(actual_t, realised_net)
        block["realised_network_bias_region"] = float(region_scores(actual_t, realised_net)["bias_region"].mean())
        block["per_region"] = [{"region": test_reg["name"].iloc[j], **{k: float(rsc[k][j]) for k in rsc}, "p50": point[j].tolist(),
                                "p10": np.quantile(dd[j][0], 0.1, axis=0).tolist(), "p90": np.quantile(dd[j][0], 0.9, axis=0).tolist()}
                               for j in range(len(point))]
        test[m] = block
        log(f"test {m}: {block['score']:.4f}")

    # Paired per-region differences of the selection score against the chosen method (negative favours it).
    def per_region_score(point: np.ndarray) -> np.ndarray:
        r = region_scores(actual_t, point)
        return (r["wape_region_category"] + r["wape_region"]) / 2
    base = per_region_score(test_draws[chosen][0])
    for m in METHODS:
        if m == chosen:
            continue
        d = base - per_region_score(test_draws[m][0])
        test[m]["paired_vs_chosen"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                                       "chosen_better_regions": int((d < 0).sum()), "regions": len(d)}

    # Ground truth, check only.
    tr_t = sim["test_reps"].sum(axis=-1).astype(float)  # (200, 30, 8, 12)
    tr_v = sim["existing_reps"].sum(axis=-1).astype(float)  # (200, 8, 8, 12)
    truth_mean_t = tr_t.sum(axis=-1).mean(axis=0)
    truth_mean_v = tr_v.sum(axis=-1).mean(axis=0)
    truth = {
        "validation_noise_floor": selection_score(actual_v, truth_mean_v),
        "test_noise_floor": selection_score(actual_t, truth_mean_t),
        "test_noise_floor_inside": selection_score(actual_t[inside], truth_mean_t[inside]),
        "test_noise_floor_outside": selection_score(actual_t[~inside], truth_mean_t[~inside]),
        "calibration": {},
        "error_vs_truth_mean": {},
    }
    yearly_reps = tr_t.sum(axis=-1)  # (200, 30, 8)
    for m in METHODS:
        point, dd = test_draws[m]
        qc = np.stack([np.quantile(d[0], [0.1, 0.9], axis=0) for d in dd], axis=1)  # (2, 30, 8)
        qt = np.stack([np.quantile(d[0].sum(axis=1), [0.1, 0.9]) for d in dd], axis=1)  # (2, 30)
        tot = yearly_reps.sum(axis=-1)
        width_true = np.quantile(tot, 0.9, axis=0) - np.quantile(tot, 0.1, axis=0)
        truth["calibration"][m] = {
            "cells_inside_our_p10_p90": float(np.mean((yearly_reps >= qc[0][None]) & (yearly_reps <= qc[1][None]))),
            "region_inside_our_p10_p90": float(np.mean((tot >= qt[0][None]) & (tot <= qt[1][None]))),
            "true_width_over_ours_region": float(width_true.sum() / (qt[1] - qt[0]).sum()),
        }
        truth["error_vs_truth_mean"][m] = selection_score(truth_mean_t, point)
    truth["elasticity"] = elasticities(panel, val["Log-linear regression"]["setting"])

    # Business check: the new region's FC capacity for year 5.
    capacity = capacity_check(panel, test_reg, test_draws, sim, chosen)

    overview = {
        "methods": METHODS, "categories": CATS, "chosen": chosen,
        "regions": reg[["id", "name", "landmass", "population_m", "income_index", "urban_share"]].to_dict("records"),
        "training": training_tables(panel),
        "test_regions": json.loads(test_reg.to_json(orient="records")),
        "validation": validation, "learning_curve": curve, "test": test, "truth": truth, "capacity": capacity,
        "settings_tried": len(settings()), "runtime_s": round(time.time() - t0, 1),
    }
    (out / "overview.json").write_text(json.dumps(overview, indent=1, default=_json) + "\n")
    regions_payload = {
        "validation_months": {"actual": months_v.tolist()},
        "test": [{"region": test_reg["name"].iloc[j], "actual": actual_t[j].tolist(), "truth_mean": truth_mean_t[j].tolist(),
                  "months_actual": months_t[j].tolist(),
                  "months": {m: np.quantile(test_draws[m][1][j][1], fu.QUANTILES, axis=0).tolist() for m in METHODS}}
                 for j in range(len(test_reg))],
    }
    (out / "regions.json").write_text(json.dumps(regions_payload, default=_json) + "\n")
    log(f"done in {time.time() - t0:.0f} s")
    return overview


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def training_tables(panel: Panel) -> dict:
    """Units per 1,000 inhabitants per year by region and category, training set (years 1 to 3)."""
    U = panel.yearly[:, :, VAL_FIT, :].sum(axis=2) / len(VAL_FIT)
    pop_k = panel.regions["population_m"].to_numpy()[:, None] * 1000
    return {"per_1000_all": (U.sum(axis=-1) / pop_k).tolist(), "per_1000_consumer": (U[..., 0] / pop_k).tolist(),
            "bulk_share": (U[..., 1].sum(axis=1) / U.sum(axis=(1, 2))).tolist()}


def elasticities(panel: Panel, s: dict) -> dict:
    """Learned income elasticity per category of the log-linear regression (its chosen setting, refitted on
    years 1 to 4) against the generator's true consumer elasticity, 0.5 (customer weight) + the category's
    income elasticity. Check only."""
    n = len(panel.regions)
    f = features(panel.regions)
    U = panel.yearly[:, :, TEST_FIT, :].sum(axis=2)
    y = U.sum(axis=-1) if s["bulk"] == "in label" else U[..., 0]
    pop_k = panel.regions["population_m"].to_numpy() * 1000
    label = np.log((y / pop_k[:, None]) / (y.sum(axis=0) / pop_k.sum()))
    weight = y / y.sum(axis=1, keepdims=True)
    mu, sd = f.to_numpy().mean(axis=0), f.to_numpy().std(axis=0)
    z = (f.to_numpy() - mu) / sd
    model = Ridge(alpha=s["alpha"], fit_intercept=True).fit(_ridge_design(z, s), label.ravel(), sample_weight=weight.ravel())
    w = model.coef_
    learned = np.full(N_CAT, w[0] / sd[0])
    if "category" in s["features"]:
        learned = learned + w[-N_CAT:] / sd[0]
    urban_mean = float(panel.regions["urban_share"].mean())
    return {"learned_income": learned.tolist(), "true_income": [0.5 + c.income_elasticity for c in CATEGORIES],
            "learned_urban": float(w[1] / sd[1]), "true_urban_at_mean": 0.5 / (0.4 + 0.5 * urban_mean), "regions": n}


def capacity_check(panel: Panel, test_reg: pd.DataFrame, test_draws: dict, sim, chosen: str) -> dict:
    """One FC per new region, capacity K (units per month) for year 5 decided at the end of year 4. Cost c K for
    the year plus margin m per unit of monthly demand above K, c / m = rho. The best K under a set of draws is
    the (1 - rho / 12) quantile of their months (B's `capacity_plan`; the same optimum holds for one year).
    Plans are costed on the realised year 5 and on replications 100 to 199; the ground-truth plan uses
    replications 0 to 99. Price and margin per unit are the network's of year 4 in the predicted category mix."""
    reps = sim["test_reps"].sum(axis=(2, 4)).astype(float)  # (200, 30, 12)
    plan_reps, eval_reps = reps[:100], reps[100:]
    realised = sim["test_realised"].sum(axis=(1, 3)).astype(float)  # (30, 12)
    point_c = test_draws[chosen][0]
    mix = point_c / point_c.sum(axis=1, keepdims=True)
    p_r = mix @ panel.price
    m_r = mix @ panel.margin
    rho_game = GAME_CAPACITY_SHARE * p_r / m_r
    plans = ["Our forecast", "Our P50 only"] + [m for m in METHODS if m != chosen] + ["Ground truth"]

    def cap(plan: str, j: int, rho: float) -> float:
        if plan == "Our forecast":
            return capacity_plan(test_draws[chosen][1][j][1], rho)
        if plan == "Our P50 only":
            return float(np.quantile(test_draws[chosen][1][j][1], 0.5, axis=0).max())
        if plan == "Ground truth":
            return capacity_plan(plan_reps[:, j], rho)
        return capacity_plan(test_draws[plan][1][j][1], rho)

    def cost(K: float, demand: np.ndarray, c: float, m: float) -> np.ndarray:
        return c * K + m * np.maximum(demand - K, 0).sum(axis=-1)

    curves = {}
    for plan in plans:
        rr, ee = [], []
        for rho in RHO:
            a, b = 0.0, 0.0
            for j in range(len(test_reg)):
                K = cap(plan, j, rho)
                c = rho * m_r[j]
                a += float(cost(K, realised[j], c, m_r[j]))
                b += float(cost(K, eval_reps[:, j], c, m_r[j]).mean())
            rr.append(a)
            ee.append(b)
        curves[plan] = {"realised": rr, "expected": ee}
    per_region = []
    for j in range(len(test_reg)):
        rho = float(rho_game[j])
        c = rho * m_r[j]
        row = {"region": test_reg["name"].iloc[j], "rho_game": rho, "price": float(p_r[j]), "margin": float(m_r[j])}
        for plan in plans:
            K = cap(plan, j, rho)
            row[plan] = {"capacity": K, "realised": float(cost(K, realised[j], c, m_r[j])),
                         "expected": float(cost(K, eval_reps[:, j], c, m_r[j]).mean())}
        per_region.append(row)
    at_game = {plan: {"realised": sum(r[plan]["realised"] for r in per_region), "expected": sum(r[plan]["expected"] for r in per_region)}
               for plan in plans}
    return {"rho": RHO.tolist(), "rho_game": rho_game.tolist(), "game_capacity_share": GAME_CAPACITY_SHARE, "plans": plans,
            "curves": curves, "per_region": per_region, "at_game_ratio": at_game}


def main(world_dir: str, out_dir: str, validation_only: bool = False) -> None:
    run(world_dir, out_dir, validation_only=validation_only)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--validation-only" in sys.argv[3:])
