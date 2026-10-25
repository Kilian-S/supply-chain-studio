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


def build_orders(scenario: Scenario, seed: int | None = None) -> list[Order]:
    """All orders from the first to the last play day, sorted by time."""
    seed = scenario.seed if seed is None else seed
    start, end = scenario.timeline.start_day, scenario.timeline.end_day
    orders: list[Order] = []
    for region in scenario.regions:
        spec = scenario.demand.get(region.name)
        if spec is None:
            continue
        # Per-region stream so adding a region never changes another region's orders.
        rng = random.Random(seed * 1_000_003 + zlib.crc32(region.name.encode()))
        totals = daily_totals(spec, region.name)
        for day in range(start, end + 1):
            sizes = split_into_orders(totals.get(day, 0), spec, rng)
            times = sorted(day + rng.random() for _ in sizes)
            orders.extend(Order(t, region.id, q) for t, q in zip(times, sizes))
    orders.sort(key=lambda o: (o.time, o.region))
    return orders


def history(scenario: Scenario, up_to_day: int) -> dict[str, dict[int, int]]:
    """Daily demand per region from day 1 up to and including `up_to_day` (what HQ would plot)."""
    out: dict[str, dict[int, int]] = {}
    for region in scenario.regions:
        spec = scenario.demand.get(region.name)
        if spec is None:
            continue
        totals = daily_totals(spec, region.name)
        out[region.name] = {d: q for d, q in totals.items() if d <= up_to_day}
    return out
