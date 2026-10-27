"""Customer orders for a run.

The original game generates individual orders at random times with random sizes. Our recorded data
only holds daily totals, so orders are rebuilt from them: each day's total is split into orders
(fixed size, or random sizes with the stated mean) arriving at uniformly random times in the day.
Daily totals therefore match the recorded series exactly.
"""

from __future__ import annotations

import csv
import random
import zlib
from dataclasses import dataclass
from functools import lru_cache

from .scenario import RecordedDemand, Scenario


@dataclass(frozen=True, slots=True)
class Order:
    time: float
    region: int
    quantity: int


@lru_cache(maxsize=16)
def _read_csv(path: str) -> dict[str, dict[int, int]]:
    """Return {column name: {day: demand}}."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    columns = [c for c in rows[0] if c.lower() != "day"]
    day_key = next(c for c in rows[0] if c.lower() == "day")
    return {c: {int(r[day_key]): int(float(r[c] or 0)) for r in rows} for c in columns}


def daily_totals(spec: RecordedDemand, region_name: str) -> dict[int, int]:
    return _read_csv(spec.csv)[region_name]


def split_into_orders(total: int, spec: RecordedDemand, rng: random.Random) -> list[int]:
    if total <= 0:
        return []
    if spec.fixed_order_size:
        size = spec.fixed_order_size
        sizes = [size] * (total // size)
        if total % size:
            sizes.append(total % size)
        return sizes
    # Shifted geometric sizes (1, 2, 3, ...) with the requested mean; the last order is truncated.
    p = 1.0 / max(spec.mean_order_size, 1.0)
    sizes: list[int] = []
    remaining = total
    while remaining > 0:
        size = 1
        while rng.random() > p:
            size += 1
        size = min(size, remaining)
        sizes.append(size)
        remaining -= size
    return sizes


def demand_multiplier(scenario: Scenario, region_id: int, day: int) -> float:
    """Combined effect of the scenario's demand events on one region and day."""
    m = 1.0
    for e in scenario.events:
        if not e.touches(region_id):
            continue
        if e.kind == "demand":
            m *= 1 + (e.factor - 1) * e.strength(day)
        elif e.kind == "market_opening":
            m *= 0.0 if day < e.start_day else (e.strength(day) if e.ramp_days else 1.0)
        elif e.kind == "end_of_life" and day >= e.start_day:
            span = max(e.end_day - e.start_day + 1, 1)
            m *= max(0.0, 1 - (day - e.start_day + 1) / span)
    return m


def _scale(total: int, m: float, spec: RecordedDemand, rng: random.Random) -> int:
    """Scale a daily total, rounding at random so the expected value is exact. Fixed-size orders
    stay whole (a 250-drum customer still orders 250 drums, just more or less often)."""
    unit = spec.fixed_order_size or 1
    x = total / unit * m
    whole = int(x)
    return (whole + (rng.random() < x - whole)) * unit


def daily_demand(scenario: Scenario, seed: int | None = None) -> dict[str, dict[int, int]]:
    """Daily demand per region for every recorded day, with the scenario's events applied to play days."""
    seed = scenario.seed if seed is None else seed
    start, end = scenario.timeline.start_day, scenario.timeline.end_day
    out: dict[str, dict[int, int]] = {}
    for region in scenario.regions:
        spec = scenario.demand.get(region.name)
        if spec is None:
            continue
        totals = dict(daily_totals(spec, region.name))
        if scenario.events:
            rng = random.Random(seed * 7_919 + zlib.crc32(region.name.encode()) + 1)
            for day in range(start, end + 1):
                m = demand_multiplier(scenario, region.id, day)
                if m != 1.0:
                    totals[day] = _scale(totals.get(day, 0), m, spec, rng)
        out[region.name] = totals
    return out


def build_orders(scenario: Scenario, seed: int | None = None,
                 daily: dict[str, dict[int, int]] | None = None) -> list[Order]:
    """All orders from the first to the last play day, sorted by time."""
    seed = scenario.seed if seed is None else seed
    daily = daily_demand(scenario, seed) if daily is None else daily
    start, end = scenario.timeline.start_day, scenario.timeline.end_day
    orders: list[Order] = []
    for region in scenario.regions:
        spec = scenario.demand.get(region.name)
        if spec is None:
            continue
        # Per-region stream so adding a region never changes another region's orders.
        rng = random.Random(seed * 1_000_003 + zlib.crc32(region.name.encode()))
        totals = daily[region.name]
        for day in range(start, end + 1):
            sizes = split_into_orders(totals.get(day, 0), spec, rng)
            times = sorted(day - 1 + rng.random() for _ in sizes)  # day d covers [d - 1, d)
            orders.extend(Order(t, region.id, q) for t, q in zip(times, sizes))
    orders.sort(key=lambda o: (o.time, o.region))
    return orders


def history(daily: dict[str, dict[int, int]], up_to_day: int) -> dict[str, dict[int, int]]:
    """Daily demand per region from day 1 up to and including `up_to_day` (what HQ would plot)."""
    return {name: {d: q for d, q in series.items() if d <= up_to_day} for name, series in daily.items()}
