"""Split orders (model H): the simulation, the features and the scores on hand-made cases and on the
generated world, and the committed outputs."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from scgame.ml import split_risk as sr
from scgame.ml import split_sim as ss

ROOT = Path(__file__).resolve().parents[2]
WORLD, MODELS = ROOT / "data" / "world_v1", ROOT / "data" / "models"
OUT = MODELS / "split-risk"
needs_world = pytest.mark.skipif(not (WORLD / "order_lines.parquet").exists(), reason="world v1 not generated")


@pytest.fixture(scope="module")
def world():
    return ss.load_world(WORLD, MODELS)


NET = ss.Network((2, 3, 6), 0.4, "best_sellers", 0.95, 1)


@needs_world
def test_with_unlimited_stock_the_plain_rule_is_exact(world):
    for net in (NET, ss.Network((4, 7), 0.6, "communities", 0.9, 2)):
        p = ss.plan(world, net, 3)
        sim = ss.simulate(world, p, 3, unlimited=True)
        multi = sim.n_skus > 1
        assert (sim.split[multi] == sim.rule[multi]).all()


@needs_world
def test_lines_ship_only_from_fcs_that_list_the_sku_and_hold_stock(world):
    p = ss.plan(world, NET, 3)
    sim = ss.simulate(world, p, 3)
    sl = world.year_lines(3)
    shipped = sim.ship >= 0
    assert p.listed[sim.ship[shipped], world.sku[sl][shipped]].all()
    assert (p.S[sim.ship[shipped], world.sku[sl][shipped]] > 0).all()  # never shipped where nothing is stocked
    assert (sim.fills <= sim.attempts).all()
    assert sim.parcels.max() <= len(NET.fcs)


@needs_world
def test_listings_have_the_planned_size_and_the_central_fc_lists_everything(world):
    for rule in ss.RULES:
        listed = ss.listing(world, dataclasses.replace(NET, rule=rule), 4)
        assert listed[0].all()
        assert (listed[1:].sum(axis=1) == round(0.4 * world.n_sku)).all()


@needs_world
def test_the_plan_and_the_features_read_nothing_from_the_scored_year(world):
    lift = np.ones((world.n_sku, world.n_sku))
    stats = sr.sku_stats(world, 4)
    p = ss.plan(world, NET, 5)
    X = sr.features(world, p, 5, lift, stats)
    sl = world.year_lines(5)
    o = world.order[sl] - world.order[sl].min()
    keep = o < 500  # the first 500 orders of year 5 stay; every other line of year 5 changes
    qty = world.qty.copy()
    qty[sl] = np.where(keep, qty[sl], qty[sl] * 7)
    w2 = dataclasses.replace(world, qty=qty)
    p2 = ss.plan(w2, NET, 5)
    assert np.array_equal(p.S, p2.S) and np.array_equal(p.listed, p2.listed)
    X2 = sr.features(w2, p2, 5, lift, sr.sku_stats(w2, 4))
    assert X.iloc[:500].equals(X2.iloc[:500])


@needs_world
def test_replay_probability_equals_the_plain_rule_when_always_in_stock(world):
    p = ss.plan(world, NET, 4)
    sim = ss.simulate(world, p, 4)
    pr = sr.replay_probability(world, p, 4, np.ones_like(sim.attempts))
    multi = sim.n_skus > 1
    assert np.allclose(pr[multi], sim.rule[multi])


def test_replay_probability_of_two_independent_coin_flips():
    # One region, two FCs (home and central), one order of two SKUs listed at both; in stock with 0.5 at home, 1 centrally.
    # Split unless both ship from home (0.25) or both from the centre (0.25), so 0.5.
    fields = {f.name: None for f in dataclasses.fields(ss.World)}
    w = ss.World(**{**fields, "order": np.array([0, 0]), "sku": np.array([0, 1]), "qty": np.ones(2), "day": np.array([1, 1]),
                    "region": np.array([0, 0])})
    net = ss.Network((2,), 1.0, "best_sellers", 0.95)  # FC 0 is the central FC, FC 1 the home FC
    p = ss.Plan(net, 1, np.ones((2, 2), dtype=bool), np.ones((2, 2)), np.array([[1, 0]]), np.zeros((2, 2, 12)), 0)
    rates = np.stack([np.ones((2, 2, 12)), np.full((2, 2, 12), 0.5)])  # (FC, SKU, route class, month)
    assert sr.replay_probability(w, p, 1, rates)[0] == pytest.approx(0.5)


def test_sku_risk_adds_up_to_order_risk_times_basket_size():
    order = np.array([0, 0, 1, 2, 2, 2])
    sku = np.array([0, 1, 1, 0, 1, 2])
    multi = np.array([True, False, True])
    p = np.array([0.2, 0.5])
    risk = sr.sku_risk(order, sku, multi, p, 3)
    assert risk.tolist() == pytest.approx([0.7, 0.7, 0.5])
    assert risk.sum() == pytest.approx(0.2 * 2 + 0.5 * 3)


def test_calibration_maps_fix_a_shifted_model_and_keep_a_calibrated_one():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.05, 0.6, 40_000)
    y = (rng.random(40_000) < truth).astype(float)
    half = np.arange(40_000) % 2
    assert sr.crossfit("none", truth, y, half) < sr.crossfit("none", truth**2, y, half)
    fixed = sr.crossfit("isotonic", truth**2, y, half)
    assert fixed < sr.crossfit("none", truth**2, y, half)
    assert fixed == pytest.approx(sr.crossfit("none", truth, y, half), abs=5e-3)
    assert sr.reliability(y, truth)["ece"] < 0.01


def test_networks_are_balanced_and_test_site_sets_are_unseen():
    nets = sr.draw_networks(np.random.default_rng(sr.SEED))
    sites = {k: {n.sites for n in v} for k, v in nets.items()}
    assert not sites["test"] & (sites["training"] | sites["validation"])
    assert not sites["validation"] & sites["training"]
    assert not set(sr.DECISION_SITES) & (sites["training"] | sites["validation"] | sites["test"])
    assert len(nets["training"]) == sr.N_TRAIN and len(nets["test"]) == sr.N_TEST
    rules = [n.rule for n in nets["training"]]
    assert all(rules.count(r) == sr.N_TRAIN // len(ss.RULES) for r in ss.RULES)


@pytest.mark.skipif(not (OUT / "overview.json").exists(), reason="outputs not built")
def test_outputs():
    o = json.loads((OUT / "overview.json").read_text())
    assert o["validation"]["chosen_method"] in sr.METHODS
    for m in sr.METHODS:
        t = o["test"]["methods"][m]
        assert 0 < t["log_loss"] < 1 and 0 <= t["average_precision"] <= 1
        assert sum(t["reliability"]["count"]) == o["test"]["rows"]
        assert o["validation"]["chosen"][m]["map"] in sr.MAPS
    assert len(o["business"]["candidates"]) == sr.N_TEST + len(sr.DECISION_SITES) * len(ss.RULES)
    assert not (OUT / "card.json").exists()


# ---- Hand-made worlds for the inventory and cost logic -------------------------------------------

def tiny_world(n_sku: int, lines: list[tuple[int, int, int, float]]) -> ss.World:
    """Two regions (id 1 is the central site, id 2 has the regional FC). lines: (order, sku, day, qty), all
    from region 2, sorted by day and order."""
    import pandas as pd

    order = np.array([x[0] for x in lines])
    first = np.unique(order, return_index=True)[1]
    day = np.array([x[2] for x in lines])
    fields = {f.name: None for f in dataclasses.fields(ss.World)}
    return ss.World(**{**fields, "skus": pd.DataFrame({"sku_id": range(1, n_sku + 1)}), "region_ids": np.array([1, 2]),
                       "mult": np.array([[1.0, 1.4], [1.4, 1.0]]), "order": order, "sku": np.array([x[1] for x in lines]),
                       "qty": np.array([x[3] for x in lines]), "day": day, "region": np.ones(len(lines), dtype=int),
                       "order_day": day[first], "order_region": np.ones(len(first), dtype=int), "order_id": np.arange(len(first)),
                       "weight": np.array([2.0, 1.0][:n_sku])})


def test_stock_runs_out_and_is_replenished_lead_time_after_the_weekly_review():
    # One unit a day at the regional FC (level 3) with the central FC (level 5) behind it.
    w = tiny_world(1, [(d - 1, 0, d, 1.0) for d in range(1, 16)])
    net = ss.Network((2,), 1.0, "best_sellers", 0.95)
    p = ss.Plan(net, 1, np.ones((2, 1), dtype=bool), np.array([[5.0], [3.0]]), np.array([[0, 1], [1, 0]]), np.zeros((2, 1, 12)), 0)
    sim = ss.simulate(w, p, 1)
    # Days 1-3 from home; 4-7 from the centre; day 8 the review of day 1 arrives (1 unit); day 9 the centre's
    # last unit; days 10-14 lost; day 15 both reviews of day 8 arrive.
    assert sim.ship.tolist() == [1, 1, 1, 0, 0, 0, 0, 1, 0, -1, -1, -1, -1, -1, 1]
    assert sim.attempts[1, 0, 0].sum() == 15 and sim.fills[1, 0, 0].sum() == 5  # home, first choice
    assert sim.attempts[0, 0, 1].sum() == 10 and sim.fills[0, 0, 1].sum() == 5  # centre, later choice


def test_parcel_cost_and_the_plain_rule_on_a_split_order():
    # One order of two SKUs; the regional FC lists only the first (2 kg), the second (1 kg) comes from the centre.
    w = tiny_world(2, [(0, 0, 1, 1.0), (0, 1, 1, 1.0)])
    net = ss.Network((2,), 0.5, "best_sellers", 0.95)
    listed = np.array([[True, True], [True, False]])
    p = ss.Plan(net, 1, listed, np.full((2, 2), 9.0), np.array([[0, 1], [1, 0]]), np.zeros((2, 2, 12)), 0)
    sim = ss.simulate(w, p, 1)
    assert sim.split.tolist() == [True] and sim.rule.tolist() == [True] and sim.parcels.tolist() == [2]
    assert sim.cost[0] == pytest.approx((1.5 + 0.35 * 2) * 1.0 + (1.5 + 0.35 * 1) * 1.4)
    assert sim.cost_one[0] == pytest.approx((1.5 + 0.35 * 3) * 1.0)


@needs_world
def test_the_replay_plans_the_same_network_with_levels_from_the_year_before(world):
    real = ss.plan(world, NET, 5)
    replay = ss.plan(world, NET, 5, levels_from=3)
    assert np.array_equal(real.listed, replay.listed)
    cube = ss.demand_at_fcs(world, NET, real.listed, 3)
    assert np.array_equal(replay.S, ss.levels(cube, real.listed, NET.service))
    assert not np.array_equal(real.S, replay.S)


@needs_world
def test_replay_rates_keep_first_choices_apart_from_later_choices(world):
    p = ss.plan(world, NET, 3)
    sim = ss.simulate(world, p, 3)
    rates = sr.replay_rates(sim)
    assert rates.shape == (len(NET.fcs), world.n_sku, ss.ROUTE_CLASSES, 12)
    first = sim.fills[1:, :, 0].sum() / sim.attempts[1:, :, 0].sum()
    later = sim.fills[1:, :, 1].sum() / sim.attempts[1:, :, 1].sum()
    assert first > later  # lines arriving after their first FC ran out find less stock
