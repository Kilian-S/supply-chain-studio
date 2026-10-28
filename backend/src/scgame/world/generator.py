"""World generator v1: a multi-product online retailer on Pangea.

Generates five years of customer orders for a catalogue of SKUs across eight regions. Every
behaviour comes from parameters we set (the "truth"), so models trained on the data can be checked
against what really drives it: each SKU's hidden demand type, the hidden bundles of products bought
together, and each region's features.

Calibration (see `reference/calibration/README.md`): revenue concentration, the shape of retail
seasonality and the spread of customer order frequency follow UCI Online Retail II. Basket sizes and
the demand-type mix are set for a general consumer retailer instead, because that dataset is mostly
wholesale.

Usage: uv run python -m scgame.world.generator <output dir> [seed]
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DAYS = 5 * 365  # day 1 .. 1825; day-of-year = (day - 1) % 365

# Monthly revenue index from UCI Online Retail II (Jan..Dec), damped by half for a broader retailer.
UCI_MONTHLY = [0.786, 0.629, 0.907, 0.713, 0.836, 0.885, 0.801, 0.852, 1.159, 1.356, 1.739, 1.337]
RETAIL_SEASON = [1 + 0.5 * (m - 1) for m in UCI_MONTHLY]


@dataclass(frozen=True)
class RegionSpec:
    id: int
    name: str
    landmass: str
    population_m: float  # millions
    income_index: float  # 1.0 = average
    urban_share: float


REGIONS = [
    RegionSpec(1, "Calopeia", "continent", 6.0, 1.10, 0.78),
    RegionSpec(2, "Sorange", "continent", 4.5, 1.25, 0.82),
    RegionSpec(3, "Tyran", "continent", 3.2, 0.95, 0.61),
    RegionSpec(4, "Entworpe", "continent", 2.6, 0.85, 0.55),
    RegionSpec(5, "Fardo", "fardo", 1.1, 0.90, 0.48),
    RegionSpec(6, "Nordhal", "continent", 2.0, 1.15, 0.70),
    RegionSpec(7, "Veloria", "veloria", 1.6, 1.00, 0.66),
    RegionSpec(8, "Austra", "austra", 0.9, 0.80, 0.42),
]


@dataclass(frozen=True)
class CategorySpec:
    name: str
    skus: int
    price_median: float  # $ per unit
    margin: tuple[float, float]  # gross margin range
    volume_l_median: float  # litres per unit
    density_kg_l: float
    seasonal_peak_doy: int  # day of year of the category's seasonal peak
    seasonal_bias: float  # multiplies the chance a SKU is seasonal
    income_elasticity: float  # how much richer regions buy more
    qty_extra_p: float  # chance of each extra unit on a line (geometric)


CATEGORIES = [
    CategorySpec("Electronics", 40, 120, (0.15, 0.30), 3.0, 0.30, 330, 1.3, 0.8, 0.10),
    CategorySpec("Home & kitchen", 45, 35, (0.30, 0.45), 8.0, 0.25, 300, 0.8, 0.4, 0.15),
    CategorySpec("Grocery", 40, 6, (0.10, 0.25), 2.0, 0.80, 350, 0.5, 0.0, 0.45),
    CategorySpec("Pet supplies", 45, 25, (0.25, 0.40), 12.0, 0.70, 100, 0.5, 0.2, 0.30),
    CategorySpec("Toys", 35, 22, (0.30, 0.50), 6.0, 0.20, 345, 2.0, 0.3, 0.15),
    CategorySpec("Beauty", 35, 15, (0.40, 0.60), 0.5, 0.90, 320, 0.8, 0.5, 0.25),
    CategorySpec("Sports", 30, 45, (0.30, 0.45), 10.0, 0.20, 160, 1.8, 0.5, 0.10),
    CategorySpec("Books", 30, 14, (0.20, 0.35), 1.0, 0.70, 340, 0.7, 0.3, 0.10),
]

# Hidden demand types and their base shares in the catalogue.
TYPES = ["steady", "seasonal", "trending", "declining", "intermittent", "lumpy"]
TYPE_SHARE = np.array([0.18, 0.14, 0.10, 0.10, 0.38, 0.10])
# Median network-wide consumer units per day for each type (log-normal around it).
TYPE_RATE = {"steady": 18.0, "seasonal": 9.0, "trending": 10.0, "declining": 10.0, "intermittent": 0.35, "lumpy": 0.15}


@dataclass
class WorldParams:
    seed: int = 7
    customers: int = 30_000
    business_share: float = 0.015  # customers placing bulk orders for lumpy SKUs
    extra_lines_mean: float = 1.4  # basket size = 1 + negative binomial with this mean
    bundle_count: int = 45
    bundle_follow_p: float = 0.45  # chance an extra basket item comes from the first item's bundle
    # Scales all consumer demand so orders per customer per year match UCI Online Retail II (mean about 3.1).
    demand_scale: float = 0.33
    regions: list[dict] = field(default_factory=lambda: [asdict(r) for r in REGIONS])
    categories: list[dict] = field(default_factory=lambda: [asdict(c) for c in CATEGORIES])


def _seasonal_profile(peak: int, amplitude: float) -> np.ndarray:
    doy = np.arange(365)
    shape = 1 + amplitude * np.cos(2 * np.pi * (doy - peak) / 365)
    return shape / shape.mean()


def build_catalogue(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    sku = 0
    for ci, cat in enumerate(CATEGORIES):
        share = TYPE_SHARE.copy()
        share[1] *= cat.seasonal_bias
        share /= share.sum()
        types = rng.choice(len(TYPES), size=cat.skus, p=share)
        for t in types:
            sku += 1
            kind = TYPES[t]
            bulky = rng.random() < 0.25
            volume = cat.volume_l_median * rng.lognormal(0, 0.6) * (3.0 if bulky else 1.0)
            price = cat.price_median * rng.lognormal(0, 0.5) * (1.6 if bulky else 1.0)
            margin = rng.uniform(*cat.margin)
            rate = TYPE_RATE[kind] * rng.lognormal(0, 0.7)
            launch = 1
            end = DAYS + 1
            if kind == "trending":
                launch = int(rng.integers(1, int(DAYS * 0.55)))
            if kind == "declining" and rng.random() < 0.4:
                end = int(rng.integers(int(DAYS * 0.6), DAYS))
            if kind not in ("trending", "declining") and rng.random() < 0.08:
                launch = int(rng.integers(1, int(DAYS * 0.4)))  # some later launches in every type
            rows.append(dict(
                sku_id=sku, name=f"{cat.name[:3].upper()}-{sku:04d}", category=cat.name, category_id=ci,
                true_type=kind, bulky=bulky, price=round(price, 2), unit_cost=round(price * (1 - margin), 2),
                margin=round(margin, 4), volume_l=round(volume, 3), weight_kg=round(volume * cat.density_kg_l * rng.lognormal(0, 0.2), 3),
                base_rate=rate, launch_day=launch, end_day=end,
                season_amp=float(rng.uniform(0.6, 0.95)) if kind == "seasonal" else float(rng.uniform(0.0, 0.15)),
                season_peak=int((cat.seasonal_peak_doy + rng.normal(0, 20)) % 365),
                growth=float(rng.uniform(4, 10)) if kind == "trending" else 0.0,
                decay=float(rng.uniform(1.5, 3.5)) if kind == "declining" else 0.0,
            ))
    return pd.DataFrame(rows)


def daily_rates(cat: pd.DataFrame) -> np.ndarray:
    """Expected consumer units per day for every SKU (rows) and day (columns), network-wide."""
    days = np.arange(1, DAYS + 1)
    doy = (days - 1) % 365
    month = np.minimum((doy / 365 * 12).astype(int), 11)
    retail = np.array(RETAIL_SEASON)[month]
    out = np.zeros((len(cat), DAYS))
    for i, s in enumerate(cat.itertuples()):
        r = s.base_rate * retail * _seasonal_profile(s.season_peak, s.season_amp)[doy]
        age = np.clip(days - s.launch_day, 0, None) / 365
        if s.true_type == "trending":
            r = r * (1 / (1 + np.exp(-(age - 1.2) * 2.2))) * (1 + s.growth * age / 5) / 1.5
        elif s.true_type == "declining":
            r = r * np.exp(-s.decay * days / DAYS)
        else:
            r = r * np.minimum(1.0, age * 4 + 0.25)  # short ramp-up after launch
        r[(days < s.launch_day) | (days >= s.end_day)] = 0
        out[i] = r
    return out


def build_regions_and_customers(p: WorldParams, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    reg = pd.DataFrame(p.regions)
    reg["online_adoption"] = 0.4 + 0.5 * reg["urban_share"]
    weight = reg["population_m"] * reg["online_adoption"] * reg["income_index"] ** 0.5
    reg["customer_share"] = weight / weight.sum()
    region = rng.choice(reg["id"].to_numpy(), size=p.customers, p=reg["customer_share"].to_numpy())
    # Orders per year: log-normal with the spread of UCI Online Retail II (median 1.5, p90 6.4).
    freq = rng.lognormal(np.log(1.5), 1.1, size=p.customers)
    business = rng.random(p.customers) < p.business_share
    signup = np.where(rng.random(p.customers) < 0.6, 1, rng.integers(1, DAYS, size=p.customers))
    cust = pd.DataFrame(dict(customer_id=np.arange(1, p.customers + 1), region_id=region,
                             orders_per_year=freq.round(3), business=business, signup_day=signup))
    return reg, cust


def build_bundles(cat: pd.DataFrame, p: WorldParams, rng: np.random.Generator) -> pd.DataFrame:
    """Hidden groups of products bought together, mostly within a category."""
    free = set(cat["sku_id"])
    rows = []
    by_cat = {c: list(g["sku_id"]) for c, g in cat.groupby("category")}
    for b in range(1, p.bundle_count + 1):
        size = int(rng.integers(3, 7))
        home = rng.choice(list(by_cat))
        pool = [s for s in by_cat[home] if s in free]
        cross = [s for s in free if s not in by_cat[home]]
        members = list(rng.choice(pool, size=min(len(pool), size - 1), replace=False)) if pool else []
        if cross and rng.random() < 0.6:
            members.append(int(rng.choice(cross)))
        for s in members:
            free.discard(int(s))
            rows.append(dict(bundle_id=b, sku_id=int(s)))
    return pd.DataFrame(rows)


def generate(out_dir: str | Path, p: WorldParams | None = None) -> dict:
    p = p or WorldParams()
    rng = np.random.default_rng(p.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cat = build_catalogue(rng)
    reg, cust = build_regions_and_customers(p, rng)
    bundles = build_bundles(cat, p, rng)
    rates = daily_rates(cat) * p.demand_scale  # (sku, day)

    n_sku = len(cat)
    sku_ids = cat["sku_id"].to_numpy()
    cat_idx = cat["category_id"].to_numpy()
    qty_p = np.array([CATEGORIES[c].qty_extra_p for c in cat_idx])
    # Regional taste: richer regions buy more of income-elastic categories.
    pref = np.array([[r.income_index ** CATEGORIES[c].income_elasticity for c in range(len(CATEGORIES))] for r in reg.itertuples()])
    bundle_of = np.full(n_sku + 1, 0)
    bundle_of[bundles["sku_id"].to_numpy()] = bundles["bundle_id"].to_numpy()
    members = {b: g["sku_id"].to_numpy() for b, g in bundles.groupby("bundle_id")}

    consumers = cust[~cust["business"]]
    by_region = {r: consumers[consumers["region_id"] == r] for r in reg["id"]}
    cust_pick = {r: (g["customer_id"].to_numpy(), (g["orders_per_year"] / g["orders_per_year"].sum()).to_numpy()) for r, g in by_region.items()}

    mean_lines = 1 + p.extra_lines_mean
    mean_qty = float(np.mean(1 / (1 - qty_p)))
    order_rows: list[np.ndarray] = []
    line_rows: list[np.ndarray] = []
    next_order = 1
    nb_n, nb_p = 1.0, 1.0 / (1 + p.extra_lines_mean)  # negative binomial: mean = n(1-p)/p

    for day in range(1, DAYS + 1):
        lam = rates[:, day - 1]
        for ri, r in enumerate(reg.itertuples()):
            w = lam * pref[ri, cat_idx] * r.customer_share
            total = w.sum()
            if total <= 0:
                continue
            n = rng.poisson(total / (mean_lines * mean_qty))
            if n == 0:
                continue
            probs = w / total
            ids, wts = cust_pick[r.id]
            customers = rng.choice(ids, size=n, p=wts)
            sizes = 1 + rng.negative_binomial(nb_n, nb_p, size=n)
            first = rng.choice(n_sku, size=n, p=probs)
            order_ids = np.arange(next_order, next_order + n)
            next_order += n
            order_rows.append(np.column_stack([order_ids, np.full(n, day), customers, np.full(n, r.id)]))
            items = [first]
            owners = [np.arange(n)]
            extra_owner = np.repeat(np.arange(n), sizes - 1)
            if len(extra_owner):
                fb = bundle_of[sku_ids[first[extra_owner]]]
                follow = (fb > 0) & (rng.random(len(extra_owner)) < p.bundle_follow_p)
                extra = rng.choice(n_sku, size=len(extra_owner), p=probs)
                for k in np.nonzero(follow)[0]:
                    extra[k] = rng.choice(members[fb[k]]) - 1
                items.append(extra)
                owners.append(extra_owner)
            it = np.concatenate(items)
            ow = np.concatenate(owners)
            # One line per distinct product in an order.
            key = np.unique(np.column_stack([ow, it]), axis=0)
            qty = 1 + rng.geometric(1 - qty_p[key[:, 1]]) - 1
            line_rows.append(np.column_stack([order_ids[key[:, 0]], sku_ids[key[:, 1]], qty, np.full(len(key), day), np.full(len(key), r.id)]))

    # Bulk orders for lumpy SKUs from business customers (like the original game's Entworpe buyer).
    biz = cust[cust["business"]]
    for s in cat[cat["true_type"] == "lumpy"].itertuples():
        buyers = biz.sample(n=int(rng.integers(1, 4)), random_state=int(rng.integers(1 << 31)))
        for b in buyers.itertuples():
            gap = rng.uniform(25, 90)
            day = s.launch_day + rng.exponential(gap)
            while day < min(s.end_day, DAYS + 1):
                q = int(max(20, rng.lognormal(np.log(80), 0.5)))
                d = int(day)
                order_rows.append(np.array([[next_order, d, b.customer_id, b.region_id]]))
                line_rows.append(np.array([[next_order, s.sku_id, q, d, b.region_id]]))
                next_order += 1
                day += rng.exponential(gap)

    orders = pd.DataFrame(np.vstack(order_rows), columns=["order_id", "day", "customer_id", "region_id"]).sort_values(["day", "order_id"])
    lines = pd.DataFrame(np.vstack(line_rows), columns=["order_id", "sku_id", "quantity", "day", "region_id"]).sort_values(["day", "order_id"])
    for df in (orders, lines):
        for c in df.columns:
            df[c] = df[c].astype("int32")

    cat_out = cat.drop(columns=["category_id"])
    cat_out.to_parquet(out / "skus.parquet", index=False)
    reg.to_parquet(out / "regions.parquet", index=False)
    cust.to_parquet(out / "customers.parquet", index=False)
    bundles.to_parquet(out / "bundles.parquet", index=False)
    orders.to_parquet(out / "orders.parquet", index=False)
    lines.to_parquet(out / "order_lines.parquet", index=False)

    rev = lines.merge(cat[["sku_id", "price"]], on="sku_id").assign(r=lambda d: d.quantity * d.price).groupby("sku_id")["r"].sum().sort_values(ascending=False)
    lines_per_order = lines.groupby("order_id").size()
    meta = {
        "version": 1,
        "days": DAYS,
        "params": asdict(p),
        "counts": {"regions": len(reg), "skus": n_sku, "customers": len(cust), "orders": len(orders), "order_lines": len(lines),
                   "bundles": int(bundles["bundle_id"].nunique()), "skus_in_bundles": len(bundles)},
        "checks": {
            "revenue_share_top20pct_skus": float(rev.iloc[: int(len(rev) * 0.2)].sum() / rev.sum()),
            "mean_lines_per_order": float(lines_per_order.mean()),
            "share_single_line_orders": float((lines_per_order == 1).mean()),
            "orders_per_customer_per_year_mean": float(len(orders) / len(cust) / (DAYS / 365)),
            "orders_per_customer_per_year_median": float(orders.groupby("customer_id").size().median() / (DAYS / 365)),
            "true_type_counts": cat["true_type"].value_counts().to_dict(),
        },
    }
    (out / "world.json").write_text(json.dumps(meta, indent=2, default=int) + "\n")
    return meta


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/world_v1"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    print(json.dumps(generate(target, WorldParams(seed=seed))["counts"], indent=2))
