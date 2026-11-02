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


def draw_sku(rng: np.random.Generator, sku: int, ci: int, kind: str) -> dict:
    """One SKU's hidden parameters. The order of random draws is part of world v1, do not reorder."""
    cat = CATEGORIES[ci]
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
    return dict(
        sku_id=sku, name=f"{cat.name[:3].upper()}-{sku:04d}", category=cat.name, category_id=ci,
        true_type=kind, bulky=bulky, price=round(price, 2), unit_cost=round(price * (1 - margin), 2),
        margin=round(margin, 4), volume_l=round(volume, 3), weight_kg=round(volume * cat.density_kg_l * rng.lognormal(0, 0.2), 3),
        base_rate=rate, launch_day=launch, end_day=end,
        season_amp=float(rng.uniform(0.6, 0.95)) if kind == "seasonal" else float(rng.uniform(0.0, 0.15)),
        season_peak=int((cat.seasonal_peak_doy + rng.normal(0, 20)) % 365),
        growth=float(rng.uniform(4, 10)) if kind == "trending" else 0.0,
        decay=float(rng.uniform(1.5, 3.5)) if kind == "declining" else 0.0,
    )


def type_share(ci: int) -> np.ndarray:
    share = TYPE_SHARE.copy()
    share[1] *= CATEGORIES[ci].seasonal_bias
    return share / share.sum()


def build_catalogue(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    sku = 0
    for ci, cat in enumerate(CATEGORIES):
        types = rng.choice(len(TYPES), size=cat.skus, p=type_share(ci))
        for t in types:
            sku += 1
            rows.append(draw_sku(rng, sku, ci, TYPES[t]))
    return pd.DataFrame(rows)


def expected_rates(cat: pd.DataFrame, days: np.ndarray) -> np.ndarray:
    """Expected consumer units per day for every SKU (rows) on the given days (columns), network-wide,
    before `demand_scale`. Optional columns `growth_stop_age` (years) and `stop_day` extend the life cycle
    beyond year 5 (`scgame.world.future`); without them this is exactly the world v1 process."""
    doy = (days - 1) % 365
    month = np.minimum((doy / 365 * 12).astype(int), 11)
    retail = np.array(RETAIL_SEASON)[month]
    out = np.zeros((len(cat), len(days)))
    has_stop_age = "growth_stop_age" in cat.columns
    has_stop_day = "stop_day" in cat.columns
    for i, s in enumerate(cat.itertuples()):
        r = s.base_rate * retail * _seasonal_profile(s.season_peak, s.season_amp)[doy]
        age = np.clip(days - s.launch_day, 0, None) / 365
        if s.true_type == "trending":
            if has_stop_age:
                age = np.minimum(age, s.growth_stop_age)
            r = r * (1 / (1 + np.exp(-(age - 1.2) * 2.2))) * (1 + s.growth * age / 5) / 1.5
        elif s.true_type == "declining":
            r = r * np.exp(-s.decay * (days - s.launch_day + 1) / DAYS)  # launch day 1 in years 1 to 5
        else:
            r = r * np.minimum(1.0, age * 4 + 0.25)  # short ramp-up after launch
        stop = s.stop_day if has_stop_day else s.end_day
        r[(days < s.launch_day) | (days >= stop)] = 0
        out[i] = r
    return out


def daily_rates(cat: pd.DataFrame) -> np.ndarray:
    """Expected consumer units per day for every SKU (rows) and day (columns), network-wide."""
    return expected_rates(cat, np.arange(1, DAYS + 1))


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


class OrderSampler:
    """Draws one day of consumer orders. The order of random draws is part of world v1, do not reorder."""

    def __init__(self, cat: pd.DataFrame, reg: pd.DataFrame, cust: pd.DataFrame, bundles: pd.DataFrame, p: WorldParams,
                 listed_bundle_follow: bool = False):
        self.p = p
        self.reg = reg
        self.n_sku = len(cat)
        self.sku_ids = cat["sku_id"].to_numpy()
        self.cat_idx = (cat["category_id"] if "category_id" in cat else cat["category"].map({c.name: i for i, c in enumerate(CATEGORIES)})).to_numpy()
        self.qty_p = np.array([CATEGORIES[c].qty_extra_p for c in self.cat_idx])
        # Regional taste: richer regions buy more of income-elastic categories.
        self.pref = np.array([[r.income_index ** CATEGORIES[c].income_elasticity for c in range(len(CATEGORIES))] for r in reg.itertuples()])
        self.bundle_of = np.full(int(self.sku_ids.max()) + 1, 0)
        self.bundle_of[bundles["sku_id"].to_numpy()] = bundles["bundle_id"].to_numpy()
        self.members = {b: g["sku_id"].to_numpy() for b, g in bundles.groupby("bundle_id")}
        self.row_of = np.full(int(self.sku_ids.max()) + 1, -1)
        self.row_of[self.sku_ids] = np.arange(self.n_sku)
        consumers = cust[~cust["business"]]
        by_region = {r: consumers[consumers["region_id"] == r] for r in reg["id"]}
        self.cust_pick = {r: (g["customer_id"].to_numpy(), (g["orders_per_year"] / g["orders_per_year"].sum()).to_numpy()) for r, g in by_region.items()}
        self.mean_lines = 1 + p.extra_lines_mean
        self.mean_qty = float(np.mean(1 / (1 - self.qty_p[: min(self.n_sku, 300)])))  # the world v1 catalogue sets order sizes
        self.nb_n, self.nb_p = 1.0, 1.0 / (1 + p.extra_lines_mean)  # negative binomial: mean = n(1-p)/p
        # Years 6 to 10 only: a bundle partner is drawn among members on sale that day (world v1 also drew
        # members before launch or after their end, 0.13% of units).
        self.listed_bundle_follow = listed_bundle_follow

    def day(self, day: int, lam: np.ndarray, rng: np.random.Generator, next_order: int,
            order_rows: list[np.ndarray], line_rows: list[np.ndarray]) -> int:
        p = self.p
        for ri, r in enumerate(self.reg.itertuples()):
            w = lam * self.pref[ri, self.cat_idx] * r.customer_share
            total = w.sum()
            if total <= 0:
                continue
            n = rng.poisson(total / (self.mean_lines * self.mean_qty))
            if n == 0:
                continue
            probs = w / total
            ids, wts = self.cust_pick[r.id]
            customers = rng.choice(ids, size=n, p=wts)
            sizes = 1 + rng.negative_binomial(self.nb_n, self.nb_p, size=n)
            first = rng.choice(self.n_sku, size=n, p=probs)
            order_ids = np.arange(next_order, next_order + n)
            next_order += n
            order_rows.append(np.column_stack([order_ids, np.full(n, day), customers, np.full(n, r.id)]))
            items = [first]
            owners = [np.arange(n)]
            extra_owner = np.repeat(np.arange(n), sizes - 1)
            if len(extra_owner):
                fb = self.bundle_of[self.sku_ids[first[extra_owner]]]
                follow = (fb > 0) & (rng.random(len(extra_owner)) < p.bundle_follow_p)
                extra = rng.choice(self.n_sku, size=len(extra_owner), p=probs)
                for k in np.nonzero(follow)[0]:
                    members = self.members[fb[k]]
                    if self.listed_bundle_follow:
                        members = members[lam[self.row_of[members]] > 0]
                        if len(members) == 0:
                            continue
                    extra[k] = self.row_of[rng.choice(members)]
                items.append(extra)
                owners.append(extra_owner)
            it = np.concatenate(items)
            ow = np.concatenate(owners)
            # One line per distinct product in an order.
            key = np.unique(np.column_stack([ow, it]), axis=0)
            qty = 1 + rng.geometric(1 - self.qty_p[key[:, 1]]) - 1
            line_rows.append(np.column_stack([order_ids[key[:, 0]], self.sku_ids[key[:, 1]], qty, np.full(len(key), day), np.full(len(key), r.id)]))
        return next_order


def bulk_orders(lumpy: pd.DataFrame, cust: pd.DataFrame, rng: np.random.Generator, next_order: int,
                order_rows: list[np.ndarray], line_rows: list[np.ndarray], start: np.ndarray, stop: np.ndarray,
                buyers: pd.DataFrame | None = None, record: list[dict] | None = None) -> int:
    """Bulk orders for lumpy SKUs from business customers (like the original game's Entworpe buyer).
    Each SKU gets 1 to 3 buyers, each reordering after exponential gaps from `start` until `stop`.
    `buyers` (sku_id, customer_id, region_id, gap) fixes the buyers and their mean gaps instead of drawing
    them; `record` collects the ones drawn."""
    biz = cust[cust["business"]]
    for s, first_day, last_day in zip(lumpy.itertuples(), start, stop):
        if buyers is None:
            drawn = biz.sample(n=int(rng.integers(1, 4)), random_state=int(rng.integers(1 << 31)))
            plan = [(b.customer_id, b.region_id, None) for b in drawn.itertuples()]
        else:
            mine = buyers[buyers["sku_id"] == s.sku_id]
            plan = list(zip(mine["customer_id"], mine["region_id"], mine["gap"]))
        for customer_id, region_id, fixed_gap in plan:
            gap = rng.uniform(25, 90) if fixed_gap is None else float(fixed_gap)
            if record is not None:
                record.append({"sku_id": int(s.sku_id), "customer_id": int(customer_id), "region_id": int(region_id), "gap": float(gap)})
            day = first_day + rng.exponential(gap)
            while day < last_day:
                q = int(max(20, rng.lognormal(np.log(80), 0.5)))
                d = int(day)
                order_rows.append(np.array([[next_order, d, customer_id, region_id]]))
                line_rows.append(np.array([[next_order, s.sku_id, q, d, region_id]]))
                next_order += 1
                day += rng.exponential(gap)
    return next_order


def generate(out_dir: str | Path, p: WorldParams | None = None) -> dict:
    p = p or WorldParams()
    rng = np.random.default_rng(p.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cat = build_catalogue(rng)
    reg, cust = build_regions_and_customers(p, rng)
    bundles = build_bundles(cat, p, rng)
    rates = daily_rates(cat) * p.demand_scale  # (sku, day)

    sampler = OrderSampler(cat, reg, cust, bundles, p)
    order_rows: list[np.ndarray] = []
    line_rows: list[np.ndarray] = []
    next_order = 1
    for day in range(1, DAYS + 1):
        next_order = sampler.day(day, rates[:, day - 1], rng, next_order, order_rows, line_rows)
    lumpy = cat[cat["true_type"] == "lumpy"]
    buyers: list[dict] = []
    next_order = bulk_orders(lumpy, cust, rng, next_order, order_rows, line_rows,
                             start=lumpy["launch_day"].to_numpy(), stop=np.minimum(lumpy["end_day"].to_numpy(), DAYS + 1), record=buyers)

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
    pd.DataFrame(buyers).to_parquet(out / "bulk_buyers.parquet", index=False)  # hidden truth, used by scgame.world.future
    orders.to_parquet(out / "orders.parquet", index=False)
    lines.to_parquet(out / "order_lines.parquet", index=False)

    rev = lines.merge(cat[["sku_id", "price"]], on="sku_id").assign(r=lambda d: d.quantity * d.price).groupby("sku_id")["r"].sum().sort_values(ascending=False)
    lines_per_order = lines.groupby("order_id").size()
    meta = {
        "version": 1,
        "days": DAYS,
        "params": asdict(p),
        "counts": {"regions": len(reg), "skus": len(cat), "customers": len(cust), "orders": len(orders), "order_lines": len(lines),
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
