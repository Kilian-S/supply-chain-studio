"""Network plan (model O): the programme and the evaluation on hand-made networks, and the committed outputs."""

import json
from pathlib import Path

import numpy as np
import pytest

from scgame.ml import network_plan as npl

OUT = Path(__file__).resolve().parents[2] / "data" / "models" / "network-plan"


def toy_net(k0: float = 10_000.0) -> dict:
    """The real distance classes: 8 regions (Calopeia first; Fardo, Veloria, Austra islands) and three new
    markets (two on the continent, one island), every unit cost the same."""
    land = ["c", "c", "c", "c", "f", "c", "v", "a", "c", "c", "n"]
    mult = np.array([[1.0 if a == b else 1.4 if land[a] == land[b] else 2.2 for b in range(11)] for a in range(11)])
    keep = np.vectorize(npl.KEEP.get)(mult)
    return {"mult": mult, "keep": keep, "delivery": np.full(11, 2.4), "margin": np.full(11, 11.5), "k0": k0}


def demand_only(region: int, units: float, months: int = 48) -> np.ndarray:
    D = np.zeros((1, 11, months))
    D[0, region] = units
    return D


def test_a_cheap_fc_opens_where_serving_from_afar_is_dear():
    # Only Fardo (an island) has demand; capacity is cheap and opening free, so Fardo gets its own FC sized to it.
    net = toy_net()
    p = npl.solve_plan(demand_only(4, 1000.0), net, {"c": 1.0, "F": 0.0})
    assert p["open"][4, 1:].all() and not p["open"][[1, 2, 3, 5, 6, 7, 8, 9, 10], 1:].any()
    assert p["K"][4, 1:] == pytest.approx(1000.0, rel=1e-4)
    assert p["K"][npl.CENTRAL].tolist() == [10_000.0] * 5  # today's FC is never reduced


def test_a_dear_opening_cost_keeps_the_central_fc_alone():
    net = toy_net()
    p = npl.solve_plan(demand_only(4, 1000.0), net, {"c": 1.0, "F": 1e9})
    assert not p["open"][1:].any()
    assert p["K"][npl.CENTRAL].tolist() == [10_000.0] * 5


def test_capacity_beyond_today_is_built_only_when_it_pays():
    # 12,000 units a month in Calopeia against 10,000 of capacity: with capacity cheaper than the lost margin
    # the central FC grows to 12,000; with it dearer the excess is lost.
    net = toy_net()
    cheap = npl.solve_plan(demand_only(0, 12_000.0), net, {"c": 1.0, "F": 1e9})
    assert cheap["K"][npl.CENTRAL, 1:] == pytest.approx(12_000.0, rel=1e-4)
    dear = npl.solve_plan(demand_only(0, 12_000.0), net, {"c": 12 * 11.5 + 1, "F": 1e9})
    assert dear["K"][npl.CENTRAL, 1:] == pytest.approx(10_000.0)


def test_evaluation_reproduces_the_programmes_second_stage():
    rng = np.random.default_rng(0)
    net = toy_net(k0=8_000.0)
    D = rng.uniform(200, 2_000, size=(1, 11, 48))
    costs = {"c": 9.0, "F": 2_700.0}
    p = npl.solve_plan(D, net, costs)
    full = np.concatenate([np.zeros((11, 12)), D[0]], axis=1)  # year 6 without demand
    ev = npl.Evaluator(net)(p["K"], full)
    second = ev["delivery"] + ev["slow_delivery"] + ev["lost_sales"]
    fs = npl.first_stage_cost(p, costs)
    years_7_10 = fs["capacity"] - costs["c"] * p["K"][:, 0].sum()
    assert years_7_10 + fs["opening"] + second == pytest.approx(p["objective"], rel=2e-3)


def test_evaluation_accounts_for_every_unit_of_demand():
    net = toy_net(k0=5_000.0)
    K = np.zeros((11, 5))
    K[npl.CENTRAL] = 5_000.0
    D = np.full((11, 60), 800.0)
    ev = npl.Evaluator(net)(K, D)
    assert ev["served"] <= 5_000.0 * 60 + 1e-6
    assert ev["served"] == pytest.approx(5_000.0 * 60, rel=1e-6)  # demand exceeds capacity every month
    assert 0 < ev["fill_rate"] < 1


def test_fixed_plans_never_shrink():
    p = npl.fixed_plan(toy_net(), np.array([10_000.0, 9_000.0, 12_000.0, 11_000.0, 13_000.0]))
    assert p["K"][npl.CENTRAL].tolist() == [10_000.0, 10_000.0, 12_000.0, 12_000.0, 13_000.0]


@pytest.mark.skipif(not (OUT / "overview.json").exists(), reason="outputs not built")
def test_outputs():
    o = json.loads((OUT / "overview.json").read_text())
    assert o["chosen_S"] in npl.S_GRID
    assert set(npl.PLANS) <= set(o["plans"])
    for f in ("first", "second"):
        for name, e in o["evaluation"][f].items():
            assert 0 < e["replications"]["fill_rate"] <= 1, name
    assert len(o["ranges"]["capacity_cost_ratio"]) == len(npl.RHO_GRID)
    assert len(o["ranges"]["opening_cost"]) == len(npl.OPENING_MULTIPLES)
    assert not (OUT / "card.json").exists()
