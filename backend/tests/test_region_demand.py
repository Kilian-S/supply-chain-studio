"""New-region demand: no leakage from the held-out region or the target year, exact baselines, the error
draws, the one-year capacity optimum, the simulated regions, and the committed outputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scgame.ml import region_demand as rd
from scgame.ml import region_sim as rs
from scgame.ml.demand_forecast import capacity_plan

ROOT = Path(__file__).resolve().parents[2]
WORLD = ROOT / "data" / "world_v1"
OUT = ROOT / "data" / "models" / "new-region-demand"
needs_world = pytest.mark.skipif(not (WORLD / "bulk_buyers.parquet").exists(), reason="world v1 not generated")


def toy_panel(seed: int = 0, exact: bool = False) -> rd.Panel:
    """Eight regions whose index is exactly income^0.8 × adoption (exact) or that plus noise."""
    rng = np.random.default_rng(seed)
    reg = pd.DataFrame({"id": range(1, 9), "name": [f"R{i}" for i in range(8)],
                        "landmass": ["continent"] * 5 + ["a", "b", "c"],
                        "population_m": rng.uniform(0.8, 6, 8), "income_index": rng.uniform(0.8, 1.25, 8), "urban_share": rng.uniform(0.4, 0.8, 8)})
    idx = reg["income_index"].to_numpy() ** 0.8 * (0.4 + 0.5 * reg["urban_share"].to_numpy())
    base = np.tile(rng.uniform(5, 20, (1, rd.N_CAT, 12)), (1, 1, 5))  # the same network every year
    cons = reg["population_m"].to_numpy()[:, None, None] * 100 * idx[:, None, None] * base
    if not exact:
        cons = cons * rng.lognormal(0, 0.05, cons.shape)
    monthly = np.stack([cons, np.zeros_like(cons) if exact else 0.1 * cons], axis=-1)
    return rd.Panel(reg, monthly, np.full(rd.N_CAT, 20.0), np.full(rd.N_CAT, 5.0))


def one_per_method() -> list[dict]:
    seen, out = set(), []
    for s in rd.settings():
        key = (s["method"], s["bulk"])
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def test_grid_has_the_agreed_size():
    s = rd.settings()
    counts = {m: sum(x["method"] == m for x in s) for m in rd.METHODS}
    assert counts == {"Population share": 1, "Nearest analogues": 12, "Log-linear regression": 72, "Gradient boosting": 24}


@pytest.mark.parametrize("setting", one_per_method(), ids=lambda s: f"{s['method']}-{s['bulk']}")
def test_held_out_region_never_enters_its_own_prediction(setting):
    panel = toy_panel()
    before = rd.loo_predict(panel, setting, 3)
    panel.monthly[3] *= 7.0
    panel.monthly[3, :, :, 1] += 1000.0
    after = rd.loo_predict(panel, setting, 3)
    np.testing.assert_allclose(before, after)


@pytest.mark.parametrize("setting", one_per_method(), ids=lambda s: f"{s['method']}-{s['bulk']}")
def test_validation_never_reads_years_4_and_5(setting):
    panel = toy_panel()
    before = rd.loo_predict(panel, setting, 5)
    panel.monthly[:, :, 36:, :] *= 3.0
    after = rd.loo_predict(panel, setting, 5)
    np.testing.assert_allclose(before, after)


def test_population_share_is_population_times_the_network_rate():
    panel = toy_panel()
    s = {"method": "Population share", "bulk": "in label"}
    y = panel.yearly
    train = np.array([0, 1, 2, 4, 5, 6, 7])
    pop = panel.regions["population_m"].to_numpy()
    expected = pop[3] * y[train][:, :, 2, :].sum(axis=(0, 2)) / pop[train].sum()
    np.testing.assert_allclose(rd.loo_predict(panel, s, 3), expected)


def test_log_linear_regression_recovers_an_exactly_log_linear_world():
    panel = toy_panel(exact=True)
    s = {"method": "Log-linear regression", "bulk": "in label", "features": "income, urban", "log_population": False, "alpha": 1e-6}
    # Adoption is linear in urban share, so the log-linear fit is close but not exact.
    for h in range(8):
        pred = rd.loo_predict(panel, s, h)
        actual = panel.yearly[h, :, 3, :].sum(axis=-1)
        assert np.abs(pred / actual - 1).max() < 0.03


def test_draws_are_centred_and_widened_by_the_t_distribution():
    rng = np.random.default_rng(0)
    point = np.full(rd.N_CAT, 1000.0)
    shape = np.full((rd.N_CAT, 12), 1 / 12)
    em = rd.ErrorModel(sigma_region=0.2, sigma_cell=0.0, phi=0.0, n=8)
    yearly, months = rd.draws(point, shape, em, rng, n=200_000)
    tot = np.log(yearly.sum(axis=1) / point.sum())
    assert abs(np.median(tot)) < 0.005
    # t with 7 degrees of freedom × √(1 + 1/8): P90 at 1.415 × 1.061 × 0.2 = 0.300 (normal 0.256).
    assert abs(np.quantile(tot, 0.9) - 0.300) < 0.006
    np.testing.assert_allclose(months.sum(axis=1), yearly.sum(axis=1), rtol=1e-9)


def test_error_model_splits_region_and_cell_errors():
    point = np.full((8, rd.N_CAT), 100.0)
    region = np.linspace(-0.2, 0.2, 8)
    actual = (point + 1) * np.exp(region)[:, None] - 1
    months = actual.sum(axis=1, keepdims=True) * np.full((1, 12), 1 / 12)
    em = rd.error_model(actual, point, months, np.full((8, rd.N_CAT, 12), 1 / 12))
    assert abs(em.sigma_region - np.sqrt(np.mean(region ** 2))) < 1e-9
    assert em.sigma_cell < 1e-9 and em.phi < 1e-9


def test_capacity_plan_is_the_one_year_optimum():
    rng = np.random.default_rng(1)
    demand = rng.lognormal(np.log(1000), 0.3, (400, 12))
    for rho in (0.3, 0.8, 3.0):
        m = 1.0
        c = rho * m
        k = capacity_plan(demand, rho)
        grid = np.linspace(demand.min(), demand.max(), 2001)
        cost = [c * g + m * np.maximum(demand - g, 0).sum(axis=1).mean() for g in grid]
        best = grid[int(np.argmin(cost))]
        assert c * k + m * np.maximum(demand - k, 0).sum(axis=1).mean() <= min(cost) + 1e-6 * abs(min(cost))
        assert abs(k - best) / best < 0.02


def test_test_region_design():
    existing = pd.DataFrame({"population_m": [0.9, 6.0, 2.0], "income_index": [0.8, 1.25, 1.0], "urban_share": [0.42, 0.82, 0.6]})
    reg = rs.design_test_regions(existing)
    assert len(reg) == 30 and reg["inside_range"].sum() == 20 and (reg["landmass"] == "island").sum() == 10
    pd.testing.assert_frame_equal(reg, rs.design_test_regions(existing))
    for r in reg.itertuples():
        out = [r.population_m < 0.9 * 0.9995 or r.population_m > 6.0 * 1.0005,
               not 0.8 - 5e-4 <= r.income_index <= 1.25 + 5e-4, not 0.42 - 5e-4 <= r.urban_share <= 0.82 + 5e-4]
        assert sum(out) == (0 if r.inside_range else 1)


@needs_world
def test_simulated_copies_match_the_existing_regions_and_leave_the_world_untouched():
    files = sorted(WORLD.glob("*.parquet")) + [WORLD / "world.json"]
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    w = rs.prepare(WORLD)
    cube = rs.simulate_year(w, w["sim_existing"], 3, np.random.default_rng([7, 99, 0]))
    lines = pd.read_parquet(WORLD / "order_lines.parquet")
    real = lines[(lines.day > 730) & (lines.day <= 1095) & (lines.quantity < rs.BULK_QTY)].groupby("region_id").quantity.sum().to_numpy()
    ratio = cube[..., 0].sum(axis=(1, 2)) / real
    assert np.abs(ratio - 1).max() < 0.08, ratio
    after = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    assert before == after


@pytest.mark.skipif(not (OUT / "overview.json").exists(), reason="model not built")
def test_committed_outputs():
    o = json.loads((OUT / "overview.json").read_text())
    assert o["chosen"] in rd.METHODS
    assert o["chosen"] == min(rd.METHODS, key=lambda m: o["validation"][m]["score"])
    assert len(o["test_regions"]) == 30 and o["settings_tried"] == 109
    for m in rd.METHODS:
        t = o["test"][m]
        assert 0 <= t["coverage_cells"] <= 1 and t["score"] > 0
    assert set(o["capacity"]["plans"]) >= {"Our forecast", "Our P50 only", "Ground truth"}
