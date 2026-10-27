"""Scheduled events change the rules from their start day (day d covers [d - 1, d))."""

import pathlib

import pytest

from scgame.demand import Order, daily_demand
from scgame.engine import Engine
from scgame.scenario import Event, Scenario
from test_engine import A, B, DAY, ISLAND, make, route

FIRST = DAY + 1  # first play day in the test scenarios


def ev(kind, start=FIRST, end=None, **kw):
    return Event(id=f"{kind}-{start}", name=kind, kind=kind, start_day=start, end_day=end, **kw)


def with_events(sc, *events):
    return sc.model_copy(update={"events": list(events)})


def classic():
    root = pathlib.Path(__file__).resolve().parents[2] / "scenarios"
    return Scenario.load(root / "classic-network.json")


def test_demand_shock_scales_daily_demand_from_its_start():
    base = classic()
    shocked = with_events(base, ev("demand", start=1095, factor=0.7))
    a, b = daily_demand(base, 0), daily_demand(shocked, 0)
    assert all(a[r][d] == b[r][d] for r in a for d in range(731, 1095))
    before = sum(a["Calopeia"][d] for d in range(1095, 1400))
    after = sum(b["Calopeia"][d] for d in range(1095, 1400))
    assert after == pytest.approx(0.7 * before, rel=0.03)
    # Fixed-size customers keep their order size and order less often.
    assert set(b["Entworpe"][d] for d in range(1095, 1400)) <= {0, 250, 500}


def test_market_opening_and_end_of_life():
    sc = with_events(classic(), ev("market_opening", start=900, regions=[2], ramp_days=30),
                     ev("end_of_life", start=1300, end=1400))
    d = daily_demand(sc, 0)
    assert all(d["Sorange"][x] == 0 for x in range(731, 900))
    assert all(d[r][x] == 0 for r in d for x in range(1400, 1461))
    assert sum(d["Calopeia"][x] for x in range(1300, 1310)) > sum(d["Calopeia"][x] for x in range(1390, 1400))


def test_cost_and_price_events_apply_while_active():
    sc = with_events(make(warehouses=((A, 100),), routes=[(A, A, route(100, 10))]),
                     ev("cost", start=FIRST, end=FIRST, factor=2.0, cost_items=["production"]),
                     ev("price", start=FIRST + 1, factor=0.5))
    e = Engine(sc, orders=[Order(DAY + 0.5, A, 10), Order(DAY + 1.5, A, 10)])
    e.run(until_day=FIRST + 1)
    assert e.records[0].production_cost == pytest.approx(2 * (2_000 + 10 * 900))
    assert e.records[0].revenue == 10 * 1450 and e.records[1].revenue == 10 * 725


def test_interest_rate_change():
    sc = with_events(make(cash=365_000, warehouses=((A, 0),)), ev("interest", start=FIRST + 1, rate=0.0))
    e = Engine(sc, orders=[])
    e.run(until_day=FIRST + 1)
    assert e.records[0].interest == pytest.approx(100.0) and e.records[1].interest == 0


def test_closed_lane_blocks_supply_and_fulfilment():
    sc = with_events(make(warehouses=((A, 0), (ISLAND, 50)), routes=[(A, ISLAND, route(100, 10))]),
                     ev("lane", end=FIRST + 1, regions=[ISLAND], closed=True))
    e = Engine(sc, orders=[Order(DAY + 0.5, A, 5)])
    assert e.factories[A].batch is None  # no production for the island while cut off
    e.step_day()
    assert e.records[0].lost[A] == 5  # island stock cannot reach the mainland either
    e.run(until_day=FIRST + 2)
    assert sum(r.batches_started for r in e.records) == 1  # resumes once the lane reopens


def test_slower_lane_lengthens_transit():
    sc = with_events(make(routes=[(A, B, route(0, 10, "mail"))], warehouses=((A, 0), (B, 0))),
                     ev("lane", regions=[B], lead_time_factor=3.0))
    e = Engine(sc, orders=[])
    e.run(until_day=FIRST + 1)  # 10 drums at capacity 10: done at DAY + 1, then 3 days by mail
    assert e.warehouses[B].inbound[0].arrive == pytest.approx(DAY + 4)


def test_factory_outage_pauses_production():
    sc = with_events(make(routes=[(A, A, route(100, 100))]), ev("outage", start=FIRST + 2, end=FIRST + 6, regions=[A]))
    e = Engine(sc, orders=[])
    # 10 days of work started at DAY, paused for days FIRST+2..FIRST+6 (5 days): done at DAY + 15.
    assert e.factories[A].batch.end == pytest.approx(DAY + 15)


def test_warehouse_outage_stops_shipping():
    sc = with_events(make(warehouses=((A, 100),)), ev("outage", end=FIRST, regions=[A], facility="warehouse"))
    e = Engine(sc, orders=[Order(DAY + 0.5, A, 10), Order(DAY + 1.5, A, 10)])
    e.run(until_day=FIRST + 1)
    assert e.records[0].lost[A] == 10 and e.records[1].fulfilled[A] == 10


def test_surprises_are_hidden_until_they_start():
    sc = with_events(make(), ev("demand", start=FIRST + 3, factor=0.5, visibility="surprise"),
                     ev("price", start=FIRST + 10, factor=1.1))
    e = Engine(sc, orders=[])
    assert [x.kind for x, _ in e.visible_events()] == ["price"]
    e.run(until_day=FIRST + 2)
    assert [(x.kind, on) for x, on in e.visible_events()] == [("demand", True), ("price", False)]
