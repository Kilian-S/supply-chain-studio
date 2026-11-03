"""Discrete-event simulation of the supply chain game.

Time is continuous in days. As in the original game, day d covers [d - 1, d): the game labels a day
by its end, so "day 731" is the first day after time 730 and the cash shown for day d is the cash at
time d. The player acts between days (the clock is
player-controlled); everything inside a day is driven by events: customer orders, batch completions,
shipment arrivals and facilities coming online.

Rules follow the original game as confirmed on the live site (see reference/GAME_MECHANICS.md):

* A factory starts a batch for a warehouse when all hold: the warehouse's on-hand plus en-route stock
  (WIP excluded) is at or below the order point, the quantity is above zero, cash covers production
  and shipping, and the factory is idle. Among triggering warehouses the highest priority wins.
* Production and shipping are paid when the batch starts; the batch ships as soon as it completes.
* Orders must be shipped within the lost-sale window or are lost. They may be split across
  warehouses. "Nearest" waits for stock arriving at a closer warehouse within the window before
  using a farther one; "soonest" uses stock on hand anywhere first.
* Holding cost accrues on drums in warehouses and (optionally) in transit, never on WIP.
* Interest accrues on cash daily.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from .demand import Order, build_orders, daily_demand
from .scenario import Event, FulfilmentPolicy, RouteSettings, Scenario, ShipMode


class DecisionError(ValueError):
    """A player decision the rules do not allow."""


# Event ranks: at equal times, facilities come online first, then stock arrives, then batches
# complete, then customers order.
_ONLINE, _ARRIVAL, _BATCH_DONE, _ORDER = range(4)


@dataclass(slots=True)
class Batch:
    warehouse: int
    quantity: int
    mode: ShipMode
    start: float
    end: float


@dataclass(slots=True)
class Shipment:
    factory: int
    warehouse: int
    quantity: int
    mode: ShipMode
    depart: float
    arrive: float


@dataclass(slots=True)
class Claim:
    """Part of a customer order waiting for stock that arrives within the window."""

    region: int
    quantity: int
    deadline: float


@dataclass(slots=True)
class FactoryState:
    region: int
    capacity: float = 0.0
    online_at: float = -math.inf
    batch: Batch | None = None
    pending_capacity: list[tuple[float, float]] = field(default_factory=list)

    def operational(self, t: float) -> bool:
        return self.online_at <= t and self.capacity > 0


@dataclass(slots=True)
class WarehouseState:
    region: int
    online_at: float = -math.inf
    on_hand: int = 0
    inbound: list[Shipment] = field(default_factory=list)
    claims: deque[Claim] = field(default_factory=deque)

    def operational(self, t: float) -> bool:
        return self.online_at <= t

    @property
    def en_route(self) -> int:
        return sum(s.quantity for s in self.inbound)

    def en_route_by_mode(self, mode: ShipMode) -> int:
        return sum(s.quantity for s in self.inbound if s.mode is mode)

    def available_inbound(self, deadline: float) -> int:
        """Inbound stock arriving by `deadline` that no waiting order has claimed yet."""
        arriving = sum(s.quantity for s in self.inbound if s.arrive <= deadline)
        return max(0, arriving - sum(c.quantity for c in self.claims))


@dataclass
class DayRecord:
    day: int
    cash: float = 0.0
    revenue: float = 0.0
    production_cost: float = 0.0
    truck_cost: float = 0.0
    mail_cost: float = 0.0
    fulfilment_cost: float = 0.0
    holding_cost: float = 0.0
    capex: float = 0.0
    interest: float = 0.0
    demand: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    fulfilled: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    lost: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    shipped_from: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    on_hand: dict[int, int] = field(default_factory=dict)
    en_route_truck: dict[int, int] = field(default_factory=dict)
    en_route_mail: dict[int, int] = field(default_factory=dict)
    wip: dict[int, int] = field(default_factory=dict)
    capacity: dict[int, float] = field(default_factory=dict)
    batches_started: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("demand", "fulfilled", "lost", "shipped_from"):
            d[key] = dict(d[key])
        return d


@dataclass
class LoggedDecision:
    day: int
    time: float
    kind: str
    params: dict[str, Any]


class Engine:
    def __init__(self, scenario: Scenario, seed: int | None = None, orders: list[Order] | None = None):
        self.scenario = scenario
        self.econ = scenario.economics
        self.rules = scenario.rules
        self.seed = scenario.seed if seed is None else seed
        self.day = scenario.timeline.start_day
        self.t = float(self.day - 1)
        self.cash = scenario.start.cash
        self.policy = scenario.start.fulfilment_policy
        self.finished = False


        self.factories: dict[int, FactoryState] = {
            f.region: FactoryState(region=f.region, capacity=f.capacity) for f in scenario.start.factories
        }
        self.warehouses: dict[int, WarehouseState] = {
            w.region: WarehouseState(region=w.region, on_hand=w.inventory) for w in scenario.start.warehouses
        }
        self.routes: dict[tuple[int, int], RouteSettings] = {
            (r.factory_region, r.warehouse_region): r.settings.model_copy() for r in scenario.start.routes
        }
        # (warehouse region, customer region) pairs the player allows ("satisfy demand in X from Y").
        self.serves: set[tuple[int, int]] = {
            (w.region, c) for w in scenario.start.warehouses for c in (w.serves or [w.region])
        }

        self._events: list[tuple[float, int, int, Any]] = []
        self._seq = 0
        self.daily_demand = daily_demand(scenario, self.seed)
        for order in build_orders(scenario, self.seed, self.daily_demand) if orders is None else orders:
            self._push(order.time, _ORDER, order)

        self.decisions: list[LoggedDecision] = []
        self.records: list[DayRecord] = []
        self._today = DayRecord(day=self.day)
        self._held_drum_days = 0.0
        self._last_t = self.t
        self._check_factories()

    # ============================================================================================
    # Clock
    # ============================================================================================

    def step_day(self) -> DayRecord:
        """Simulate the current day to its end and return its record."""
        if self.finished:
            raise RuntimeError("the run has finished")
        end = float(self.day)
        self._process_until(end)
        self._advance(end)
        record = self._close_day()
        self.day += 1
        if self.day > self.scenario.timeline.end_day:
            self.finished = True
        else:
            self._today = DayRecord(day=self.day)
            self._check_factories()
        return record

    def advance_to(self, time: float) -> None:
        """Move the clock to `time` within the current day, e.g. to replay a decision taken mid-day."""
        if not self.day - 1 <= time < self.day:
            raise ValueError(f"time {time} is outside the current day {self.day}")
        self._process_until(time)
        self._advance(time)

    def _process_until(self, end: float) -> None:
        while self._events and self._events[0][0] < end:
            time, rank, _, payload = heapq.heappop(self._events)
            self._advance(time)
            self._handle(rank, payload)

    def run(self, until_day: int | None = None, bot: Callable[[Engine], None] | None = None) -> None:
        """Step until the end of `until_day` (default: the last day). `bot` acts before each day."""
        last = self.scenario.timeline.end_day if until_day is None else until_day
        while not self.finished and self.day <= last:
            if bot is not None:
                bot(self)
            self.step_day()

    # ============================================================================================
    # Player decisions (taken between days)
    # ============================================================================================

    def apply(self, decision: dict[str, Any]) -> None:
        """Apply a decision given as a dict, e.g. from a saved log: {"kind": "set_route", ...}."""
        params = {k: v for k, v in decision.items() if k not in ("kind", "day", "time")}
        handler = {
            "set_route": self.set_route,
            "set_policy": self.set_policy,
            "build_factory": self.build_factory,
            "add_capacity": self.add_capacity,
            "build_warehouse": self.build_warehouse,
            "set_capacity": self.set_capacity,
            "set_service": self.set_service,
        }.get(decision["kind"])
        if handler is None:
            raise DecisionError(f"unknown decision kind {decision['kind']!r}")
        handler(**params)

    def set_route(self, factory: int, warehouse: int, order_point: int | None = None,
                  quantity: int | None = None, mode: str | ShipMode | None = None,
                  priority: int | None = None) -> None:
        if factory not in self.factories:
            raise DecisionError(f"no factory in region {factory}")
        if warehouse not in self.warehouses:
            raise DecisionError(f"no warehouse in region {warehouse}")
        current = self.routes.get((factory, warehouse), RouteSettings())
        updated = current.model_copy(update={k: v for k, v in {
            "order_point": order_point, "quantity": quantity,
            "mode": ShipMode(mode) if mode is not None else None, "priority": priority,
        }.items() if v is not None})
        try:
            RouteSettings.model_validate(updated.model_dump())
        except ValueError as exc:
            raise DecisionError(str(exc)) from exc
        self.routes[(factory, warehouse)] = updated
        self._log("set_route", factory=factory, warehouse=warehouse, **updated.model_dump(mode="json"))
        self._check_factories()

    def set_policy(self, policy: str | FulfilmentPolicy) -> None:
        policy = FulfilmentPolicy(policy)
        if policy not in self.rules.allowed_policies:
            raise DecisionError(f"policy {policy.value} is not allowed in this scenario")
        self.policy = policy
        self._log("set_policy", policy=policy.value)

    def build_factory(self, region: int, capacity: float) -> None:
        self._check_buildable(region)
        if region in self.factories:
            raise DecisionError(f"region {region} already has a factory")
        if capacity <= 0:
            raise DecisionError("capacity must be positive")
        cost = self.econ.factory_build_cost + capacity * self.econ.capacity_cost_per_unit
        self._pay_capex(cost)
        online = self.t + self.econ.factory_build_days
        self.factories[region] = FactoryState(region=region, capacity=0.0, online_at=online)
        self.factories[region].pending_capacity.append((online, capacity))
        self._push(online, _ONLINE, ("capacity", region, capacity))
        self._log("build_factory", region=region, capacity=capacity, cost=cost)

    def set_capacity(self, region: int, total: float) -> None:
        """The original's "schedule factory capacity change": build a factory, or add up to `total`."""
        if region not in self.factories:
            self.build_factory(region, total)
            return
        f = self.factories[region]
        scheduled = f.capacity + sum(amount for _, amount in f.pending_capacity)
        self.add_capacity(region, total - scheduled)

    def set_service(self, region: int, warehouse: int, allowed: bool) -> None:
        """Allow or block the warehouse in `warehouse` from serving customers in `region`."""
        self.scenario.region(region)
        if warehouse not in self.warehouses:
            raise DecisionError(f"no warehouse in region {warehouse}")
        if allowed:
            self.serves.add((warehouse, region))
        else:
            self.serves.discard((warehouse, region))
        self._log("set_service", region=region, warehouse=warehouse, allowed=allowed)

    def add_capacity(self, region: int, amount: float) -> None:
        if not self.rules.allow_capacity_additions:
            raise DecisionError("capacity additions are not allowed in this scenario")
        if region not in self.factories:
            raise DecisionError(f"no factory in region {region}")
        if amount <= 0:
            raise DecisionError("capacity can only be added, never removed")
        cost = amount * self.econ.capacity_cost_per_unit
        self._pay_capex(cost)
        online = self.t + self.econ.capacity_build_days
        self.factories[region].pending_capacity.append((online, amount))
        self._push(online, _ONLINE, ("capacity", region, amount))
        self._log("add_capacity", region=region, amount=amount, cost=cost)

    def build_warehouse(self, region: int) -> None:
        self._check_buildable(region)
        if region in self.warehouses:
            raise DecisionError(f"region {region} already has a warehouse")
        cost = self.econ.warehouse_build_cost
        self._pay_capex(cost)
        online = self.t + self.econ.warehouse_build_days
        self.warehouses[region] = WarehouseState(region=region, online_at=online)
        if self.rules.new_warehouse_serves_all:
            self.serves |= {(region, r.id) for r in self.scenario.regions}
        self._push(online, _ONLINE, ("warehouse", region, None))
        self._log("build_warehouse", region=region, cost=cost)

    # ============================================================================================
    # Read-only views
    # ============================================================================================

    def inventory_position(self, warehouse: int) -> int:
        w = self.warehouses[warehouse]
        return w.on_hand + w.en_route

    def summary(self) -> dict[str, Any]:
        recs = self.records
        demand = sum(sum(r.demand.values()) for r in recs)
        fulfilled = sum(sum(r.fulfilled.values()) for r in recs)
        lost = sum(sum(r.lost.values()) for r in recs)
        totals = {k: sum(getattr(r, k) for r in recs) for k in (
            "revenue", "production_cost", "truck_cost", "mail_cost", "fulfilment_cost",
            "holding_cost", "capex", "interest")}
        return {
            "scenario": self.scenario.id,
            "days_played": len(recs),
            "final_cash": self.cash,
            "demand": demand,
            "fulfilled": fulfilled,
            "lost": lost,
            "fill_rate": fulfilled / demand if demand else 1.0,
            "leftover_inventory": sum(w.on_hand + w.en_route for w in self.warehouses.values()),
            **totals,
        }

    def visible_events(self) -> list[tuple[Event, bool]]:
        """Events the player may see now, with whether each is under way. Surprises appear when they start."""
        today = self.day
        out = []
        for e in self.scenario.events:
            started = e.start_day <= today
            if e.visibility == "announced" or started:
                out.append((e, started and e.covers(today)))
        return out

    # ============================================================================================
    # Scheduled events
    # ============================================================================================

    @staticmethod
    def _day_at(t: float) -> int:
        return math.floor(t) + 1  # day d covers [d - 1, d)

    def _active(self, kind: str, t: float) -> list[tuple[Event, float]]:
        day = self._day_at(t)
        return [(e, e.strength(day)) for e in self.scenario.events if e.kind == kind and e.covers(day)]

    def _cost_factor(self, item: str, t: float, *regions: int) -> float:
        m = 1.0
        for e, k in self._active("cost", t):
            if item in e.cost_items and (item == "holding" or e.touches(*regions)):
                m *= 1 + (e.factor - 1) * k
        return m

    def _price(self, t: float, region: int) -> float:
        m = 1.0
        for e, k in self._active("price", t):
            if e.touches(region):
                m *= 1 + (e.factor - 1) * k
        return self.econ.price_per_drum * m

    def _lane_effect(self, t: float, a: int, b: int) -> tuple[bool, float, float]:
        """(closed, lead-time factor, cost factor) for the lane between regions a and b."""
        closed, lead, cost = False, 1.0, 1.0
        if a == b:
            return closed, lead, cost
        for e, k in self._active("lane", t):
            if e.touches(a, b):
                closed = closed or e.closed
                lead *= 1 + (e.lead_time_factor - 1) * k
                cost *= 1 + (e.factor - 1) * k
        return closed, lead, cost

    def _out(self, facility: str, region: int, t: float) -> bool:
        return any(e.facility == facility and e.touches(region) for e, _ in self._active("outage", t))

    def _production_end(self, region: int, start: float, duration: float) -> float:
        """Finish time of a batch, pausing through scheduled outages of this factory."""
        windows = sorted((e.start_day - 1.0, float(e.end_day)) for e in self.scenario.events
                         if e.kind == "outage" and e.facility == "factory" and e.touches(region))
        t, left = start, duration
        for a, b in windows:
            if b <= t:
                continue
            if a <= t:
                t = b
                continue
            if a >= t + left:
                break
            left -= a - t
            t = b
        return t + left

    def _daily_rate(self, day: int) -> float:
        rate = self.econ.interest_rate_annual
        for e in self.scenario.events:
            if e.kind == "interest" and e.covers(day):
                rate = e.rate
        return rate / 365 if self.econ.interest_compounding == "nominal" else (1 + rate) ** (1 / 365) - 1

    # ============================================================================================
    # Internals
    # ============================================================================================

    def _push(self, time: float, rank: int, payload: Any) -> None:
        heapq.heappush(self._events, (time, rank, self._seq, payload))
        self._seq += 1

    def _log(self, kind: str, **params: Any) -> None:
        self.decisions.append(LoggedDecision(day=self.day, time=self.t, kind=kind, params=params))

    def _check_buildable(self, region: int) -> None:
        self.scenario.region(region)  # raises KeyError for unknown regions
        allowed = self.rules.buildable_regions
        if allowed is not None and region not in allowed:
            raise DecisionError(f"building is not allowed in region {region}")

    def _pay_capex(self, cost: float) -> None:
        if cost > self.cash:
            raise DecisionError(f"insufficient cash: need {cost:,.2f}, have {self.cash:,.2f}")
        self.cash -= cost
        self._today.capex += cost

    def _held(self) -> int:
        held = sum(w.on_hand for w in self.warehouses.values())
        if self.econ.holding_cost_in_transit:
            held += sum(w.en_route for w in self.warehouses.values())
        return held

    def _advance(self, time: float) -> None:
        if time > self._last_t:
            self._held_drum_days += self._held() * (time - self._last_t)
            self._last_t = time
        self.t = max(self.t, time)

    def _handle(self, rank: int, payload: Any) -> None:
        if rank == _ORDER:
            self._on_order(payload)
        elif rank == _ARRIVAL:
            self._on_arrival(payload)
        elif rank == _BATCH_DONE:
            self._on_batch_done(payload)
        else:
            self._on_online(*payload)
        self._check_factories()

    # --- events ---------------------------------------------------------------------------------

    def _on_order(self, order: Order) -> None:
        self._today.demand[order.region] += order.quantity
        remaining = self._allocate(order.region, order.quantity, order.time + self.rules.lost_sale_window_days)
        if remaining:
            self._today.lost[order.region] += remaining

    def _on_arrival(self, shipment: Shipment) -> None:
        w = self.warehouses[shipment.warehouse]
        w.inbound.remove(shipment)
        w.on_hand += shipment.quantity
        self._serve_claims(w)

    def _on_batch_done(self, factory_region: int) -> None:
        f = self.factories[factory_region]
        batch = f.batch
        assert batch is not None
        f.batch = None
        lane = self.scenario.lane(factory_region, batch.warehouse)
        days = lane.truck_days if batch.mode is ShipMode.TRUCK else lane.mail_days
        days *= self._lane_effect(self.t, factory_region, batch.warehouse)[1]
        shipment = Shipment(factory_region, batch.warehouse, batch.quantity, batch.mode, self.t, self.t + days)
        self.warehouses[batch.warehouse].inbound.append(shipment)
        self._push(shipment.arrive, _ARRIVAL, shipment)

    def _on_online(self, kind: str, region: int, amount: float | None) -> None:
        if kind == "capacity":
            f = self.factories[region]
            f.capacity += amount
            f.pending_capacity = [p for p in f.pending_capacity if p != (self.t, amount)]
        # Warehouses need no action: `operational` compares online_at with the clock.

    # --- fulfilment -----------------------------------------------------------------------------

    def _fulfilment_rank(self, region: int) -> list[WarehouseState]:
        live = [w for w in self.warehouses.values()
                if w.operational(self.t) and (w.region, region) in self.serves
                and not self._out("warehouse", w.region, self.t) and not self._lane_effect(self.t, w.region, region)[0]]
        return sorted(live, key=lambda w: (self.scenario.lane(w.region, region).mail_cost_per_drum, w.region))

    def _allocate(self, region: int, quantity: int, deadline: float) -> int:
        """Fill as much of an order as the policy allows; return the quantity that is lost."""
        remaining = quantity
        ranked = self._fulfilment_rank(region)
        if self.policy is FulfilmentPolicy.NEAREST:
            for w in ranked:
                remaining -= self._ship(w, region, min(remaining, w.on_hand))
                remaining -= self._claim(w, region, remaining, deadline)
                if not remaining:
                    break
        else:
            for w in ranked:
                remaining -= self._ship(w, region, min(remaining, w.on_hand))
                if not remaining:
                    break
            if remaining:
                by_arrival = sorted(
                    (w for w in ranked if w.available_inbound(deadline) > 0),
                    key=lambda w: min(s.arrive for s in w.inbound),
                )
                for w in by_arrival:
                    remaining -= self._claim(w, region, remaining, deadline)
                    if not remaining:
                        break
        return remaining

    def _claim(self, w: WarehouseState, region: int, quantity: int, deadline: float) -> int:
        take = min(quantity, w.available_inbound(deadline))
        if take > 0:
            w.claims.append(Claim(region, take, deadline))
        return max(take, 0)

    def _ship(self, w: WarehouseState, region: int, quantity: int) -> int:
        if quantity <= 0:
            return 0
        w.on_hand -= quantity
        revenue = quantity * self._price(self.t, region)
        cost = (quantity * self.scenario.lane(w.region, region).mail_cost_per_drum
                * self._cost_factor("fulfilment", self.t, w.region, region) * self._lane_effect(self.t, w.region, region)[2])
        self.cash += revenue - cost
        self._today.revenue += revenue
        self._today.fulfilment_cost += cost
        self._today.fulfilled[region] += quantity
        self._today.shipped_from[w.region] += quantity
        return quantity

    def _serve_claims(self, w: WarehouseState) -> None:
        while w.claims and w.on_hand > 0:
            claim = w.claims[0]
            if claim.deadline < self.t:
                w.claims.popleft()
                self._today.lost[claim.region] += claim.quantity
                continue
            given = self._ship(w, claim.region, min(claim.quantity, w.on_hand))
            claim.quantity -= given
            if claim.quantity == 0:
                w.claims.popleft()

    def _expire_claims(self) -> None:
        for w in self.warehouses.values():
            kept: deque[Claim] = deque()
            for claim in w.claims:
                if claim.deadline < self.t:
                    self._today.lost[claim.region] += claim.quantity
                else:
                    kept.append(claim)
            w.claims = kept

    # --- production -----------------------------------------------------------------------------

    def _batch_cost(self, factory: int, warehouse: int, settings: RouteSettings) -> tuple[float, float]:
        production = (self.econ.batch_fixed_cost + settings.quantity * self.econ.unit_cost) \
            * self._cost_factor("production", self.t, factory)
        lane = self.scenario.lane(factory, warehouse)
        if settings.mode is ShipMode.TRUCK:
            shipping = math.ceil(settings.quantity / self.econ.truck_capacity) * lane.truck_cost
        else:
            shipping = settings.quantity * lane.mail_cost_per_drum
        shipping *= self._cost_factor("transport", self.t, factory, warehouse) * self._lane_effect(self.t, factory, warehouse)[2]
        return production, shipping

    def _check_factories(self) -> None:
        for f in sorted(self.factories.values(), key=lambda f: f.region):
            if f.batch is not None or not f.operational(self.t) or self._out("factory", f.region, self.t):
                continue
            best: tuple[int, int, RouteSettings, float, float] | None = None
            for (fr, wr), s in self.routes.items():
                if fr != f.region or s.quantity <= 0:
                    continue
                w = self.warehouses.get(wr)
                if w is None or not w.operational(self.t):
                    continue
                if w.on_hand + w.en_route > s.order_point:
                    continue
                if self._lane_effect(self.t, fr, wr)[0]:
                    continue
                production, shipping = self._batch_cost(fr, wr, s)
                if self.rules.cash_check_on_production and self.cash < production + shipping:
                    continue
                if best is None or (s.priority, -wr) > (best[2].priority, -best[1]):
                    best = (fr, wr, s, production, shipping)
            if best is not None:
                self._start_batch(f, *best[1:])

    def _start_batch(self, f: FactoryState, warehouse: int, s: RouteSettings,
                     production: float, shipping: float) -> None:
        self.cash -= production + shipping
        self._today.production_cost += production
        if s.mode is ShipMode.TRUCK:
            self._today.truck_cost += shipping
        else:
            self._today.mail_cost += shipping
        self._today.batches_started += 1
        f.batch = Batch(warehouse, s.quantity, s.mode, self.t,
                        self._production_end(f.region, self.t, s.quantity / f.capacity))
        self._push(f.batch.end, _BATCH_DONE, f.region)

    # --- end of day -----------------------------------------------------------------------------

    def _close_day(self) -> DayRecord:
        self._expire_claims()
        rec = self._today
        rec.holding_cost = (self._held_drum_days * self.econ.holding_cost_per_drum_year / 365
                            * self._cost_factor("holding", self.t - 0.5))
        self._held_drum_days = 0.0
        self.cash -= rec.holding_cost
        rec.interest = self.cash * self._daily_rate(self.day)
        self.cash += rec.interest
        rec.cash = self.cash
        rec.on_hand = {r: w.on_hand for r, w in self.warehouses.items()}
        rec.en_route_truck = {r: w.en_route_by_mode(ShipMode.TRUCK) for r, w in self.warehouses.items()}
        rec.en_route_mail = {r: w.en_route_by_mode(ShipMode.MAIL) for r, w in self.warehouses.items()}
        rec.wip = {r: (f.batch.quantity if f.batch else 0) for r, f in self.factories.items()}
        rec.capacity = {r: f.capacity for r, f in self.factories.items()}
        self.records.append(rec)
        return rec
