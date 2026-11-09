"""New regions for the new-region demand model (C): the test regions and the ground truth.

World v1 has only its 8 regions, so a clean test of a cold-start forecast needs regions that never existed.
This module simulates them with the generator's own process (`OrderSampler`, `expected_rates`,
`bulk_orders`, imported unchanged). Nothing in the world files changes.

Rules (agreed 3 November 2026, `docs/design/new-region-demand/PROPOSAL.md` section 8).
- A new region's consumer demand follows the generator's formula, with its weight (population × online
  adoption × √income) divided by the sum of the existing 8 weights. The existing regions are unchanged:
  the new region adds demand, it takes none.
- Customers are drawn as in the generator, 30,000 × the region's share of that sum, with the same order
  frequencies and 1.5% business customers.
- Each lumpy SKU on sale gains bulk buyers among the region's business customers at the world's rate per
  business customer (the generator gives every lumpy SKU 2 buyers on average among all business
  customers), with the generator's gaps and order sizes. The bulk process is memoryless, so starting it
  at the start of the year is exact.
- No ramp-up: every region of the world sells at its full rate from day 1.

The same simulation of an existing region's features gives the expected demand of a region like it that
opens fresh (fresh customers and bulk buyers), the ground truth for the leave-one-region-out validation.

Usage: uv run python -m scgame.ml.region_sim <world dir> <out dir> [replications] [workers]
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from scgame.world.future import load_world
from scgame.world.generator import CATEGORIES, OrderSampler, bulk_orders, expected_rates

N_TEST = 30
N_INSIDE = 20
BULK_QTY = 20  # a line of 20 units or more is a bulk line (as in model B)
STREAM_DESIGN, STREAM_REALISED, STREAM_REPS = 31, 32, 33
YEAR_DAYS = 365
FEATURES = ["population_m", "income_index", "urban_share"]


def online_adoption(urban: np.ndarray) -> np.ndarray:
    return 0.4 + 0.5 * np.asarray(urban)


def region_weight(reg: pd.DataFrame) -> np.ndarray:
    """The generator's customer weight (`build_regions_and_customers`)."""
    return (reg["population_m"] * online_adoption(reg["urban_share"]) * reg["income_index"] ** 0.5).to_numpy()


def design_test_regions(existing: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """30 new regions, fixed before any model is fitted. 20 inside the existing ranges (a Latin hypercube over
    log population, income and urban share), 10 outside the range on one feature by up to a quarter of the
    range (population on the log scale), the other two features inside. 10 of the 30 are islands."""
    rng = np.random.default_rng([seed, STREAM_DESIGN])
    lo = np.array([np.log(existing["population_m"].min()), existing["income_index"].min(), existing["urban_share"].min()])
    hi = np.array([np.log(existing["population_m"].max()), existing["income_index"].max(), existing["urban_share"].max()])

    def lhs(n: int) -> np.ndarray:
        u = np.column_stack([(rng.permutation(n) + rng.random(n)) / n for _ in range(3)])
        return lo + u * (hi - lo)

    inside = lhs(N_INSIDE)
    outside = lhs(N_TEST - N_INSIDE)
    feature = np.arange(N_TEST - N_INSIDE) % 3  # population, income, urban share in turn
    side = rng.permutation(np.arange(N_TEST - N_INSIDE) % 2)  # below or above the range
    beyond = rng.uniform(0.02, 0.25, size=N_TEST - N_INSIDE) * (hi - lo)[feature]
    outside[np.arange(len(feature)), feature] = np.where(side == 0, lo[feature] - beyond, hi[feature] + beyond)
    x = np.vstack([inside, outside])
    island = np.zeros(N_TEST, dtype=bool)
    island[rng.choice(N_TEST, size=N_TEST // 3, replace=False)] = True
    reg = pd.DataFrame({
        "id": np.arange(101, 101 + N_TEST), "name": [f"New region {i}" for i in range(1, N_TEST + 1)],
        "landmass": np.where(island, "island", "continent"),
        "population_m": np.round(np.exp(x[:, 0]), 3), "income_index": np.round(x[:, 1], 3), "urban_share": np.round(x[:, 2], 3),
        "inside_range": np.arange(N_TEST) < N_INSIDE,
        "outside_feature": [None] * N_INSIDE + [FEATURES[f] for f in feature],
    })
    return reg


def as_new(reg: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Region rows for the sampler: online adoption and customer share (weight over the existing 8 weights)."""
    out = reg[["id", "name", "landmass", "population_m", "income_index", "urban_share"]].copy()
    out["online_adoption"] = online_adoption(out["urban_share"])
    out["customer_share"] = region_weight(out) / region_weight(existing).sum()
    return out.reset_index(drop=True)


def draw_customers(reg: pd.DataFrame, p, rng: np.random.Generator) -> pd.DataFrame:
    """Customers per region as in the generator: 30,000 × share, log-normal order frequency, 1.5% business."""
    parts = []
    nxt = 10_000_000
    for r in reg.itertuples():
        n = max(round(p.customers * r.customer_share), 2)
        freq = rng.lognormal(np.log(1.5), 1.1, size=n)
        business = rng.random(n) < p.business_share
        business[0] = False  # at least one consumer, so the sampler can place orders
        parts.append(pd.DataFrame({"customer_id": np.arange(nxt, nxt + n), "region_id": r.id, "orders_per_year": freq.round(3),
                                   "business": business, "signup_day": 1}))
        nxt += n
    return pd.concat(parts, ignore_index=True)


def draw_bulk_buyers(lumpy_ids: np.ndarray, cust: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Each (lumpy SKU, business customer) pair becomes a buyer with probability `rate`, gap U(25, 90) days."""
    biz = cust[cust["business"]]
    rows = []
    for s in lumpy_ids:
        pick = biz[rng.random(len(biz)) < rate]
        for b in pick.itertuples():
            rows.append({"sku_id": int(s), "customer_id": int(b.customer_id), "region_id": int(b.region_id), "gap": float(rng.uniform(25, 90))})
    return pd.DataFrame(rows, columns=["sku_id", "customer_id", "region_id", "gap"])


def simulate_year(w: dict, reg: pd.DataFrame, year: int, rng: np.random.Generator) -> np.ndarray:
    """One draw of `year` for the regions in `reg` (already `as_new`): units per region, category, month (12)
    and kind (0 consumer lines, 1 bulk lines of 20 units or more)."""
    p, cat = w["params"], w["skus"]
    first = (year - 1) * YEAR_DAYS + 1
    days = np.arange(first, first + YEAR_DAYS)
    cust = draw_customers(reg, p, rng)
    rates = expected_rates(cat, days) * p.demand_scale
    sampler = OrderSampler(cat, reg, cust, w["bundles"], p)
    order_rows: list[np.ndarray] = []
    line_rows: list[np.ndarray] = []
    nxt = 1
    for j, day in enumerate(days):
        nxt = sampler.day(int(day), rates[:, j], rng, nxt, order_rows, line_rows)
    lumpy = cat[(cat["true_type"] == "lumpy") & (cat["launch_day"] < first + YEAR_DAYS) & (cat["end_day"] > first)]
    buyers = draw_bulk_buyers(lumpy["sku_id"].to_numpy(), cust, w["bulk_rate"], rng)
    bulk_orders(lumpy, cust, rng, nxt, order_rows, line_rows, start=np.maximum(lumpy["launch_day"].to_numpy(), first),
                stop=np.minimum(lumpy["end_day"].to_numpy(), first + YEAR_DAYS), buyers=buyers)
    lines = np.vstack(line_rows) if line_rows else np.zeros((0, 5), dtype=int)
    cube = np.zeros((len(reg), len(CATEGORIES), 12, 2), dtype=np.int32)
    if len(lines):
        row_of = {rid: i for i, rid in enumerate(reg["id"])}
        r_idx = np.array([row_of[r] for r in lines[:, 4]])
        c_idx = w["cat_of_sku"][lines[:, 1]]
        doy = (lines[:, 3] - 1) % YEAR_DAYS
        m_idx = np.minimum((doy / YEAR_DAYS * 12).astype(int), 11)
        kind = (lines[:, 2] >= BULK_QTY).astype(int)
        np.add.at(cube, (r_idx, c_idx, m_idx, kind), lines[:, 2])
    return cube


_W: dict = {}


def prepare(world_dir: str | Path) -> dict:
    w = load_world(Path(world_dir))
    cat = w["skus"]
    w["cat_of_sku"] = np.full(int(cat["sku_id"].max()) + 1, -1)
    w["cat_of_sku"][cat["sku_id"].to_numpy()] = cat["category_id"].to_numpy()
    n_biz = int(w["customers"]["business"].sum())
    w["bulk_rate"] = 2.0 / n_biz  # integers(1, 4) buyers per lumpy SKU in the generator, mean 2
    w["test_regions"] = design_test_regions(w["regions"], w["params"].seed)
    w["sim_test"] = as_new(w["test_regions"], w["regions"])
    w["sim_existing"] = as_new(w["regions"], w["regions"])
    return w


def _init(world_dir: str) -> None:
    _W.update(prepare(world_dir))


def _one(args: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    stream, r = args
    rng = np.random.default_rng([_W["params"].seed, stream, r])
    test = simulate_year(_W, _W["sim_test"], 5, rng)
    existing = simulate_year(_W, _W["sim_existing"], 4, rng)
    return test, existing


def simulate(world_dir: str | Path, out_dir: str | Path, reps: int = 200, workers: int = 4, log=print) -> Path:
    """Writes `simulation.npz` and `test_regions.csv` into `out_dir`.
    test_realised (30, 8 categories, 12 months, 2 kinds), year 5 of the new regions, stream 32;
    test_reps (reps, 30, 8, 12, 2) and existing_reps (reps, 8, 8, 12, 2), year 5 and year 4, stream 33."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(str(world_dir),)) as pool:
        jobs = [(STREAM_REALISED, 0)] + [(STREAM_REPS, r) for r in range(reps)]
        res = pool.map(_one, jobs)
    w = prepare(world_dir)
    w["test_regions"].to_csv(out / "test_regions.csv", index=False)
    path = out / "simulation.npz"
    np.savez_compressed(path, test_realised=res[0][0], test_reps=np.stack([r[0] for r in res[1:]]),
                        existing_reps=np.stack([r[1] for r in res[1:]]), test_ids=w["test_regions"]["id"].to_numpy(),
                        existing_ids=w["regions"]["id"].to_numpy())
    log(f"simulated {reps} replications of 30 new regions (year 5) and 8 existing regions as new (year 4) in {time.time() - t0:.0f} s")
    return path


if __name__ == "__main__":
    world_dir, out_dir = sys.argv[1], sys.argv[2]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    simulate(world_dir, out_dir, reps, workers)
