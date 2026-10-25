"""Rule-by-rule tests of the engine against the original game's mechanics."""

import pytest

from scgame.demand import Order
from scgame.engine import DecisionError, Engine
from scgame.library import CLASSIC_LANES
from scgame.scenario import (
    Economics, Region, RouteSettings, Rules, Scenario, StartFactory, StartRoute, StartState,
    StartWarehouse, Timeline,
)

A, B, ISLAND = 1, 2, 3
DAY = 730


def make(cash=10_000_000, factories=((A, 10),), warehouses=((A, 0),), routes=(), economics=None,
         rules=None, end_day=DAY + 200, policy="nearest"):
    return Scenario(
        id="test",
        name="test",
        regions=[Region(id=A, name="A"), Region(id=B, name="B"), Region(id=ISLAND, name="I", landmass="island")],
        lanes=CLASSIC_LANES,
        economics=economics or Economics(),
        rules=rules or Rules(),
        demand={},
        start=StartState(
            cash=cash,
            factories=[StartFactory(region=r, capacity=c) for r, c in factories],
            warehouses=[StartWarehouse(region=r, inventory=i, serves=[A, B, ISLAND]) for r, i in warehouses],
            routes=[StartRoute(factory_region=f, warehouse_region=w, settings=RouteSettings(**s)) for f, w, s in routes],
            fulfilment_policy=policy,
        ),
        timeline=Timeline(start_day=DAY + 1, end_day=end_day),  # first day covers [DAY, DAY + 1)
    )


def route(order_point, quantity, mode="truck", priority=0):
    return dict(order_point=order_point, quantity=quantity, mode=mode, priority=priority)


# --- production -------------------------------------------------------------------------------

def test_batch_starts_when_position_at_or_below_order_point_and_charges_at_start():
    e = Engine(make(routes=[(A, A, route(100, 100))]), orders=[])
    # 100 drums at capacity 10 take 10 days; production $2,000 + 100 x $900 and one truck $15,000.
    assert e.factories[A].batch is not None
    assert e.factories[A].batch.end == pytest.approx(DAY + 10)
    assert e.cash == pytest.approx(10_000_000 - (2_000 + 90_000) - 15_000)


def test_wip_does_not_count_towards_inventory_position():
    e = Engine(make(routes=[(A, A, route(50, 100))]), orders=[])
    e.run(until_day=DAY + 8)
    # Still producing the first batch; the factory is busy, so no second batch despite position 0.
    assert e.inventory_position(A) == 0
    assert sum(r.batches_started for r in e.records) == 1
    e.run(until_day=DAY + 12)
    # Batch shipped on day 740: en route 100 > order point 50, so the factory stays idle.
    assert e.inventory_position(A) == 100
    assert e.factories[A].batch is None


def test_truck_arrives_after_seven_days_mail_after_one():
    for mode, days in (("truck", 7), ("mail", 1)):
        e = Engine(make(routes=[(A, A, route(0, 10, mode))]), orders=[])  # 10 drums: 1 day
        e.run(until_day=DAY + days + 1)  # completes at DAY + 1, arrives at DAY + 1 + days
        assert e.warehouses[A].on_hand == 0
        e.step_day()
        assert e.warehouses[A].on_hand == 10


def test_quantity_zero_never_produces():
    e = Engine(make(routes=[(A, A, route(10_000, 0))]), orders=[])
    e.run()
    assert sum(r.batches_started for r in e.records) == 0


def test_cash_check_blocks_unaffordable_batch():
    e = Engine(make(cash=50_000, routes=[(A, A, route(100, 100))]), orders=[])
    assert e.factories[A].batch is None
    e2 = Engine(make(cash=50_000, routes=[(A, A, route(100, 100))],
                     rules=Rules(cash_check_on_production=False)), orders=[])
    assert e2.factories[A].batch is not None


def test_highest_priority_triggering_warehouse_wins():
    sc = make(warehouses=((A, 0), (B, 0)),
              routes=[(A, A, route(100, 10, priority=1)), (A, B, route(100, 10, priority=5))])
    e = Engine(sc, orders=[])
    assert e.factories[A].batch.warehouse == B


def test_truck_cost_is_per_started_truck_and_mail_per_drum():
    e = Engine(make(routes=[(A, B, route(0, 0))], warehouses=((A, 0), (B, 0))), orders=[])
    production, shipping = e._batch_cost(A, B, RouteSettings(**route(0, 201)))
    assert shipping == 2 * 20_000  # two trucks across the continent
    production, shipping = e._batch_cost(A, ISLAND, RouteSettings(**route(0, 10, "mail")))
    assert shipping == 10 * 400


# --- fulfilment -------------------------------------------------------------------------------

def test_order_filled_from_stock_with_revenue_and_fulfilment_cost():
    e = Engine(make(warehouses=((A, 100),)), orders=[Order(DAY + 0.5, B, 10)])
    e.step_day()
    rec = e.records[0]
    assert rec.fulfilled[B] == 10
    assert rec.revenue == 10 * 1450
    assert rec.fulfilment_cost == 10 * 200  # warehouse in A, customer in B: across the continent


def test_order_without_stock_is_lost():
    e = Engine(make(warehouses=((A, 3),)), orders=[Order(DAY + 0.5, A, 10)])
    e.step_day()
    assert e.records[0].fulfilled[A] == 3
    assert e.records[0].lost[A] == 7


def test_order_split_across_warehouses_nearest_first():
    e = Engine(make(warehouses=((A, 4), (ISLAND, 100))), orders=[Order(DAY + 0.5, A, 10)])
    e.step_day()
    assert e.records[0].shipped_from == {A: 4, ISLAND: 6}


def _inbound_case(policy):
    # Warehouse A is empty but a mail shipment lands at 730.9; the island warehouse has stock.
    sc = make(factories=((A, 100),), warehouses=((A, 0), (ISLAND, 50)),
              routes=[(A, A, route(0, 10, "mail"))], policy=policy)
    e = Engine(sc, orders=[Order(DAY + 0.85, A, 10)])
    e.factories[A].batch = None
    e._events = [ev for ev in e._events if ev[1] != 2]  # drop the auto-started batch
    e.routes.clear()
    from scgame.engine import Shipment, _ARRIVAL
    s = Shipment(A, A, 10, "mail", DAY - 0.1, DAY + 0.9)
    e.warehouses[A].inbound.append(s)
    e._push(s.arrive, _ARRIVAL, s)
    e.step_day()
    return e.records[0]


def test_nearest_waits_for_inbound_stock_within_window():
    rec = _inbound_case("nearest")
    assert rec.shipped_from == {A: 10}
    assert rec.fulfilment_cost == 10 * 150


def test_soonest_uses_stock_on_hand_elsewhere():
    rec = _inbound_case("soonest")
    assert rec.shipped_from == {ISLAND: 10}
    assert rec.fulfilment_cost == 10 * 400


# --- money ------------------------------------------------------------------------------------

def test_holding_cost_on_warehouse_and_transit_stock():
    e = Engine(make(cash=0, warehouses=((A, 365),)), orders=[])
    e.step_day()
    assert e.records[0].holding_cost == pytest.approx(90.0)  # 365 drum-days x $90 / 365


def test_interest_nominal_daily():
    e = Engine(make(cash=365_000, warehouses=((A, 0),)), orders=[])
    e.step_day()
    assert e.records[0].interest == pytest.approx(100.0)
    e2 = Engine(make(cash=365_000, warehouses=((A, 0),), economics=Economics(interest_compounding="effective")), orders=[])
    e2.step_day()
    assert e2.records[0].interest == pytest.approx(365_000 * (1.1 ** (1 / 365) - 1))


# --- building ---------------------------------------------------------------------------------

def test_build_factory_costs_and_comes_online_after_90_days():
    e = Engine(make(warehouses=((A, 0), (B, 0))), orders=[])
    e.build_factory(B, 20)
    assert e.cash == pytest.approx(10_000_000 - 500_000 - 20 * 50_000)
    e.set_route(B, B, order_point=100, quantity=20)
    e.run(until_day=DAY + 90)
    assert e.factories[B].batch is None and e.factories[B].capacity == 0
    e.step_day()
    assert e.factories[B].capacity == 20 and e.factories[B].batch is not None


def test_capacity_addition_and_limits():
    e = Engine(make(warehouses=((A, 0),)), orders=[])
    e.add_capacity(A, 5)
    with pytest.raises(DecisionError):
        e.add_capacity(A, -1)
    with pytest.raises(DecisionError):
        e.build_factory(A, 10)  # one factory per region
    e.run(until_day=DAY + 91)
    assert e.factories[A].capacity == 15


def test_warehouse_build_and_insufficient_cash():
    e = Engine(make(cash=150_000), orders=[])
    e.build_warehouse(B)
    assert not e.warehouses[B].operational(e.t)
    with pytest.raises(DecisionError):
        e.build_warehouse(ISLAND)  # only $50,000 left
    e.run(until_day=DAY + 60)
    assert e.warehouses[B].operational(e.t)


def test_runs_are_deterministic_for_a_seed():
    from scgame.scenario import Scenario as S
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "scenarios"
    sc = S.load(root / "classic-network.json")
    a, b = Engine(sc, seed=3), Engine(sc, seed=3)
    a.run(); b.run()
    assert a.summary() == b.summary()
    # Daily demand totals equal the recorded series whatever the seed.
    c = Engine(sc, seed=4); c.run()
    assert [r.demand for r in a.records] == [r.demand for r in c.records]


# --- market permissions and capacity totals (from the Run 3 history) ---------------------------

def test_blocked_market_is_not_served():
    e = Engine(make(warehouses=((A, 100),)), orders=[Order(DAY + 0.2, B, 5), Order(DAY + 0.6, B, 5)])
    e.set_service(B, A, allowed=False)
    e.advance_to(DAY + 0.5)
    assert e.records == [] and e.warehouses[A].on_hand == 100  # first order lost
    e.set_service(B, A, allowed=True)  # decision taken mid-day
    e.step_day()
    assert e.records[0].lost[B] == 5 and e.records[0].fulfilled[B] == 5


def test_start_warehouse_serves_only_home_market_by_default():
    sc = make(warehouses=((A, 100),))
    sc.start.warehouses[0].serves = None
    e = Engine(sc, orders=[Order(DAY + 0.5, B, 5), Order(DAY + 0.5, A, 5)])
    e.step_day()
    assert e.records[0].lost[B] == 5 and e.records[0].fulfilled[A] == 5


def test_new_warehouse_serves_all_regions():
    e = Engine(make(), orders=[])
    e.build_warehouse(ISLAND)
    assert {(ISLAND, A), (ISLAND, B), (ISLAND, ISLAND)} <= e.serves


def test_set_capacity_takes_the_new_total():
    e = Engine(make(warehouses=((A, 0),)), orders=[])
    e.set_capacity(B, 17)          # no factory yet: build one with 17
    e.set_capacity(B, 20)          # then add 3 more
    e.set_capacity(A, 10.5)        # existing factory at 10: add 0.5
    assert e.cash == pytest.approx(10_000_000 - (500_000 + 17 * 50_000) - 3 * 50_000 - 0.5 * 50_000)
    e.run(until_day=DAY + 91)
    assert e.factories[B].capacity == 20 and e.factories[A].capacity == 10.5
