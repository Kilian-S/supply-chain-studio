"""World v1, years 6 to 10: the future the demand forecast is tested on.

Years 1 to 5 (`generator.py`) are left untouched. Years 6 to 10 follow the same rules, with three
additions fixed before any forecasting model was fitted (`design/demand-forecast/PROPOSAL.md`, section 2).

- New SKUs launch at the average rate of years 1 to 5 (37 launches in five years), drawn with the original
  category sizes, type mix and parameter ranges, ids from 301.
- At the start of each year the SKUs with the lowest expected sales over the past year are delisted, as
  many as launch that year, so the catalogue stays near 300. Only SKUs on sale for at least a year count.
- A growing SKU's growth phase ends at an age drawn between 4 and 7 years (or at its age at the end of
  year 5, if older), after which its trend stays flat.

Bulk buyers of the lumpy SKUs keep buying (same customers, same mean gaps); new lumpy SKUs draw their own.
A bundle partner is drawn only among members on sale that day. Everything uses a random stream separate
from years 1 to 5.

Usage: uv run python -m scgame.world.future <world dir> [stream] [out dir]
(the first future is stream 6 in the world directory; the second is stream 16 in <world dir>/future2)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scgame.world.generator import (
    CATEGORIES,
    DAYS,
    TYPES,
    OrderSampler,
    WorldParams,
    bulk_orders,
    draw_sku,
    expected_rates,
    type_share,
)

YEARS = 10
END_DAY = YEARS * 365  # last day of year 10
OPEN = 10**6  # stop day of a SKU that is never delisted
LAUNCHES_PER_YEAR = 37 / 5  # launches after day 1 in years 1 to 5 of world v1
GROWTH_STOP_YEARS = (4.0, 7.0)
# Expected bulk units per lumpy SKU per year: 2 buyers on average, E[1/gap] for gap ~ U(25, 90), and
# E[max(20, lognormal(log 80, 0.5))] units per order.
_BULK_QTY = float(np.mean(np.maximum(20, np.random.default_rng(0).lognormal(np.log(80), 0.5, 200_000)).astype(int)))
BULK_UNITS_PER_YEAR = 2 * 365 * np.log(90 / 25) / 65 * _BULK_QTY


def load_world(world: Path) -> dict:
    meta = json.loads((world / "world.json").read_text())
    p = WorldParams(**{k: v for k, v in meta["params"].items() if k in WorldParams.__dataclass_fields__})
    skus = pd.read_parquet(world / "skus.parquet")
    skus["category_id"] = skus["category"].map({c.name: i for i, c in enumerate(CATEGORIES)})
    buyers_path = world / "bulk_buyers.parquet"
    if not buyers_path.exists():
        raise FileNotFoundError(f"{buyers_path} is missing; rerun `python -m scgame.world.generator` (world v1 is unchanged)")
    return {
        "params": p, "skus": skus, "regions": pd.read_parquet(world / "regions.parquet"),
        "customers": pd.read_parquet(world / "customers.parquet"), "bundles": pd.read_parquet(world / "bundles.parquet"),
        "buyers": pd.read_parquet(buyers_path), "orders": pd.read_parquet(world / "orders.parquet", columns=["order_id"]),
    }


def year_start(year: int) -> int:
    return (year - 1) * 365 + 1


def future_catalogue(skus: pd.DataFrame, rng: np.random.Generator, demand_scale: float) -> tuple[pd.DataFrame, dict]:
    """The full catalogue for years 1 to 10, with `stop_day` and `growth_stop_age` (the hidden life cycle)."""
    cat = skus.copy()
    cat["new"] = False
    cat["stop_day"] = np.where(cat["end_day"] <= DAYS, cat["end_day"], OPEN)
    age_end = (DAYS - cat["launch_day"]) / 365
    draw = rng.uniform(*GROWTH_STOP_YEARS, size=len(cat))
    cat["growth_stop_age"] = np.where(cat["true_type"] == "trending", np.maximum(draw, age_end), np.nan)

    sizes = np.array([c.skus for c in CATEGORIES], dtype=float)
    rows = []
    next_id = int(cat["sku_id"].max()) + 1
    launches: dict[int, int] = {}
    for year in range(6, YEARS + 1):
        n = int(rng.poisson(LAUNCHES_PER_YEAR))
        launches[year] = n
        for _ in range(n):
            ci = int(rng.choice(len(CATEGORIES), p=sizes / sizes.sum()))
            kind = TYPES[int(rng.choice(len(TYPES), p=type_share(ci)))]
            row = draw_sku(rng, next_id, ci, kind)
            row["launch_day"] = int(rng.integers(year_start(year), year_start(year) + 365))
            row["end_day"] = OPEN
            row["stop_day"] = OPEN
            row["growth_stop_age"] = float(rng.uniform(*GROWTH_STOP_YEARS)) if kind == "trending" else np.nan
            row["new"] = True
            rows.append(row)
            next_id += 1
    if rows:
        cat = pd.concat([cat, pd.DataFrame(rows)], ignore_index=True)

    # Delisting: at the start of each year, the slowest sellers by expected units over the past year.
    delisted: dict[int, list[int]] = {}
    for year in range(6, YEARS + 1):
        d0 = year_start(year)
        past = np.arange(d0 - 365, d0)
        units = expected_rates(cat, past).sum(axis=1) * demand_scale
        units = units + np.where(cat["true_type"] == "lumpy", BULK_UNITS_PER_YEAR, 0.0)
        eligible = (cat["launch_day"] <= d0 - 365) & (cat["stop_day"] > d0)
        order = np.argsort(np.where(eligible, units, np.inf), kind="stable")[: min(launches[year], int(eligible.sum()))]
        cat.loc[order, "stop_day"] = d0
        delisted[year] = cat.loc[order, "sku_id"].astype(int).tolist()
    return cat, {"launches": launches, "delisted": delisted}


def on_sale(cat: pd.DataFrame, day: int) -> int:
    return int(((cat["launch_day"] <= day) & (cat["stop_day"] > day)).sum())


def simulate_years_6_to_10(w: dict, cat: pd.DataFrame, rng: np.random.Generator, first_order: int,
                           buyers_future: pd.DataFrame | None = None, record: list[dict] | None = None) -> tuple[list, list]:
    """Orders for days 1826 to 3650 under the given catalogue."""
    p = w["params"]
    days = np.arange(DAYS + 1, END_DAY + 1)
    rates = expected_rates(cat, days) * p.demand_scale
    sampler = OrderSampler(cat, w["regions"], w["customers"], w["bundles"], p, listed_bundle_follow=True)
    order_rows: list[np.ndarray] = []
    line_rows: list[np.ndarray] = []
    nxt = first_order
    for j, day in enumerate(days):
        nxt = sampler.day(int(day), rates[:, j], rng, nxt, order_rows, line_rows)
    lumpy = cat[cat["true_type"] == "lumpy"]
    start = np.maximum(lumpy["launch_day"].to_numpy(), DAYS + 1)
    stop = np.minimum(lumpy["stop_day"].to_numpy(), END_DAY + 1)
    if buyers_future is None:
        # Existing lumpy SKUs keep their buyers; new ones draw theirs (recorded).
        old = lumpy[~lumpy["new"]]
        nxt = bulk_orders(old, w["customers"], rng, nxt, order_rows, line_rows, start[~lumpy["new"].to_numpy()],
                          stop[~lumpy["new"].to_numpy()], buyers=w["buyers"])
        if record is not None:
            record.extend(w["buyers"][w["buyers"]["sku_id"].isin(old["sku_id"])].to_dict("records"))
        new = lumpy[lumpy["new"]]
        nxt = bulk_orders(new, w["customers"], rng, nxt, order_rows, line_rows, start[lumpy["new"].to_numpy()],
                          stop[lumpy["new"].to_numpy()], record=record)
    else:
        nxt = bulk_orders(lumpy, w["customers"], rng, nxt, order_rows, line_rows, start, stop, buyers=buyers_future)
    return order_rows, line_rows


def generate_future(world_dir: str | Path, stream: int = 6, out_dir: str | Path | None = None) -> dict:
    """Years 6 to 10 from random stream [seed, stream], written to `out_dir` (default the world directory).
    The first future uses stream 6; a second draw under the same rules uses another stream."""
    world = Path(world_dir)
    out = Path(out_dir) if out_dir is not None else world
    out.mkdir(parents=True, exist_ok=True)
    w = load_world(world)
    p = w["params"]
    rng = np.random.default_rng([p.seed, stream])
    cat, events = future_catalogue(w["skus"], rng, p.demand_scale)
    record: list[dict] = []
    first = int(w["orders"]["order_id"].max()) + 1
    order_rows, line_rows = simulate_years_6_to_10(w, cat, rng, first, record=record)

    orders = pd.DataFrame(np.vstack(order_rows), columns=["order_id", "day", "customer_id", "region_id"]).sort_values(["day", "order_id"])
    lines = pd.DataFrame(np.vstack(line_rows), columns=["order_id", "sku_id", "quantity", "day", "region_id"]).sort_values(["day", "order_id"])
    for df in (orders, lines):
        for c in df.columns:
            df[c] = df[c].astype("int32")
    out_cat = cat.drop(columns=["category_id"])
    out_cat.to_parquet(out / "skus_future.parquet", index=False)
    pd.DataFrame(record).to_parquet(out / "bulk_buyers_future.parquet", index=False)
    orders.to_parquet(out / "orders_future.parquet", index=False)
    lines.to_parquet(out / "order_lines_future.parquet", index=False)

    meta = {
        "version": 1,
        "first_day": DAYS + 1,
        "last_day": END_DAY,
        "rules": {"launches_per_year": LAUNCHES_PER_YEAR, "growth_stop_years": list(GROWTH_STOP_YEARS),
                  "bulk_units_per_year_for_delisting": BULK_UNITS_PER_YEAR, "random_stream": [p.seed, stream]},
        "launches": events["launches"],
        "delisted": events["delisted"],
        "on_sale_at_year_start": {y: on_sale(cat, year_start(y)) for y in range(1, YEARS + 1)},
        "counts": {"skus_total": len(cat), "new_skus": int(cat["new"].sum()), "orders": len(orders), "order_lines": len(lines)},
    }
    (out / "world_future.json").write_text(json.dumps(meta, indent=2, default=int) + "\n")
    return meta


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/world_v1"
    stream = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out_dir = sys.argv[3] if len(sys.argv) > 3 else None
    m = generate_future(target, stream, out_dir)
    print(json.dumps({k: m[k] for k in ("launches", "on_sale_at_year_start", "counts")}, indent=2))
