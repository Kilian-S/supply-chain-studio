"""Ground truth for the demand forecast: years 4 to 10 of world v1 re-run many times.

Each replication keeps everything fixed that a forecaster could in principle know or that defines the
world (catalogue with its future launches and delistings, customers, regions, bundles, bulk buyers and
their mean reorder gaps) and redraws every order with a fresh random stream. The spread across
replications is the true uncertainty of demand; their mean is the true expected demand. Used to check
the forecast, never to choose it.

Output: `truth_replications.npz` in the world directory, units per replication, region, SKU and month
for months 37 to 120 (years 4 to 10).

Usage: uv run python -m scgame.world.replicate <world dir> [replications] [workers]
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from scgame.world.future import END_DAY, load_world, simulate_years_6_to_10
from scgame.world.generator import (
    CATEGORIES,
    DAYS,
    OrderSampler,
    bulk_orders,
    expected_rates,
)

FIRST_DAY = 3 * 365 + 1  # start of year 4
FIRST_MONTH = 37
MONTHS = 120


def month_of_day(day: np.ndarray) -> np.ndarray:
    """Month number 1..120 for day numbers 1..3650 (twelve equal parts of each 365-day year)."""
    day = np.asarray(day)
    doy = (day - 1) % 365
    return (day - 1) // 365 * 12 + np.minimum((doy / 365 * 12).astype(int), 11) + 1


_W: dict = {}


def _init(world_dir: str) -> None:
    world = Path(world_dir)
    w = load_world(world)
    cat = pd.read_parquet(world / "skus_future.parquet")
    cat["category_id"] = cat["category"].map({c.name: i for i, c in enumerate(CATEGORIES)})
    w["cat"] = cat
    w["buyers_future"] = pd.read_parquet(world / "bulk_buyers_future.parquet")
    _W.update(w)


def one_replication(r: int) -> np.ndarray:
    w, cat, p = _W, _W["cat"], _W["params"]
    rng = np.random.default_rng([p.seed, 7, r])
    order_rows: list[np.ndarray] = []
    line_rows: list[np.ndarray] = []
    # Years 4 and 5 under the world v1 rules.
    days = np.arange(FIRST_DAY, DAYS + 1)
    rates = expected_rates(cat, days) * p.demand_scale
    sampler = OrderSampler(cat, w["regions"], w["customers"], w["bundles"], p)
    nxt = 1
    for j, day in enumerate(days):
        nxt = sampler.day(int(day), rates[:, j], rng, nxt, order_rows, line_rows)
    lumpy = cat[(cat["true_type"] == "lumpy") & ~cat["new"]]
    nxt = bulk_orders(lumpy, w["customers"], rng, nxt, order_rows, line_rows,
                      start=np.maximum(lumpy["launch_day"].to_numpy(), FIRST_DAY),
                      stop=np.minimum(lumpy["end_day"].to_numpy(), DAYS + 1), buyers=w["buyers"])
    # Years 6 to 10 under the future rules.
    _o6, l6 = simulate_years_6_to_10(w, cat, rng, nxt, buyers_future=w["buyers_future"])
    lines = np.vstack(line_rows + l6)
    sku_row = np.full(int(cat["sku_id"].max()) + 1, -1)
    sku_row[cat["sku_id"].to_numpy()] = np.arange(len(cat))
    month = month_of_day(lines[:, 3]) - FIRST_MONTH
    keep = (lines[:, 3] >= FIRST_DAY) & (lines[:, 3] <= END_DAY)
    cube = np.zeros((len(w["regions"]), len(cat), MONTHS - FIRST_MONTH + 1), dtype=np.int32)
    np.add.at(cube, (lines[keep, 4] - 1, sku_row[lines[keep, 1]], month[keep]), lines[keep, 2])
    return cube


def replicate(world_dir: str | Path, reps: int = 200, workers: int = 4) -> Path:
    world = Path(world_dir)
    with Pool(workers, initializer=_init, initargs=(str(world),)) as pool:
        cubes = pool.map(one_replication, range(reps))
    cat = pd.read_parquet(world / "skus_future.parquet")
    out = world / "truth_replications.npz"
    np.savez_compressed(out, units=np.stack(cubes), sku_id=cat["sku_id"].to_numpy(),
                        region_id=np.arange(1, cubes[0].shape[0] + 1), months=np.arange(FIRST_MONTH, MONTHS + 1))
    return out


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/world_v1"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    print(replicate(target, reps, workers))
