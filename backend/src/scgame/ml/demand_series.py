"""Monthly demand series for the demand forecast: six levels, two hierarchies, scaled windows.

Months are twelve equal parts of each 365-day year, numbered 1 to 120 over the ten years of world v1
(years 6 to 10 come from `scgame.world.future`). Units ordered per month are the demand.

Levels (in this order): Total, Region, Category, Region × category, SKU, Region × SKU. The upper
hierarchy (the first four levels, bottom Region × category) is forecast five years ahead. SKU and
Region × SKU are forecast two years ahead; their series are the SKUs and Region × SKU pairs with a sale
in the fitting window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scgame.world.generator import CATEGORIES
from scgame.world.replicate import month_of_day

LEVELS = ["Total", "Region", "Category", "Region × category", "SKU", "Region × SKU"]
UPPER = LEVELS[:4]
MONTHS = 120
BULK_QTY = 20  # order lines of this many units or more count as bulk
# Months since first sale are capped: training windows of a three-year fit never see more than 24, and a
# network would extrapolate beyond the range it was trained on (later origins and the recursion go to 108).
MONTHS_ON_SALE_CAP = 24


@dataclass
class World:
    skus: pd.DataFrame  # every SKU of years 1 to 10, with its row in `units`
    regions: pd.DataFrame
    units: np.ndarray  # (region, sku row, month 1..120) units ordered, index month - 1
    bulk: np.ndarray  # same shape, units in bulk lines


def load(world_dir: str | Path) -> World:
    world = Path(world_dir)
    skus = pd.read_parquet(world / "skus_future.parquet").reset_index(drop=True)
    skus["category_id"] = skus["category"].map({c.name: i for i, c in enumerate(CATEGORIES)})
    regions = pd.read_parquet(world / "regions.parquet")
    lines = pd.concat([pd.read_parquet(world / "order_lines.parquet"), pd.read_parquet(world / "order_lines_future.parquet")])
    row = np.full(int(skus["sku_id"].max()) + 1, -1)
    row[skus["sku_id"].to_numpy()] = np.arange(len(skus))
    r = lines["region_id"].to_numpy() - 1
    s = row[lines["sku_id"].to_numpy()]
    m = month_of_day(lines["day"].to_numpy()) - 1
    q = lines["quantity"].to_numpy()
    units = np.zeros((len(regions), len(skus), MONTHS), dtype=np.float64)
    bulk = np.zeros_like(units)
    np.add.at(units, (r, s, m), q)
    big = q >= BULK_QTY
    np.add.at(bulk, (r[big], s[big], m[big]), q[big])
    return World(skus, regions, units, bulk)


@dataclass
class Series:
    """All series for one fitting window, with their values over all 120 months and static features."""

    meta: pd.DataFrame  # one row per series: level, region (1..8 or -1), category (0..7 or -1), sku_id (or -1)
    y: np.ndarray  # (series, 120)
    bulk: np.ndarray  # (series, 120)
    S_upper: np.ndarray  # (81, 64) summing matrix of the upper hierarchy, rows in meta order
    upper_idx: np.ndarray  # rows of meta in the upper hierarchy (81)
    rc_idx: np.ndarray  # rows of the 64 Region × category series (bottom of the upper hierarchy)
    pair_idx: np.ndarray  # rows of the Region × SKU series
    pair_parent: np.ndarray  # for each pair, its position in rc_idx
    sku_idx: np.ndarray  # rows of the SKU series
    pair_sku: np.ndarray  # for each pair, its position in sku_idx


def build_series(w: World, fit_end: int) -> Series:
    """Series for a fitting window of months 1..fit_end. SKU and Region × SKU series are those with a sale
    in the window; the upper hierarchy always has 81 series."""
    n_reg = len(w.regions)
    cat = w.skus["category_id"].to_numpy()
    rows, ys, bs = [], [], []

    def add(level, region, category, sku_id, y, b):
        rows.append({"level": level, "region": region, "category": category, "sku_id": sku_id})
        ys.append(y)
        bs.append(b)

    U, B = w.units, w.bulk
    add("Total", -1, -1, -1, U.sum((0, 1)), B.sum((0, 1)))
    for r in range(n_reg):
        add("Region", r + 1, -1, -1, U[r].sum(0), B[r].sum(0))
    for c in range(len(CATEGORIES)):
        add("Category", -1, c, -1, U[:, cat == c].sum((0, 1)), B[:, cat == c].sum((0, 1)))
    for r in range(n_reg):
        for c in range(len(CATEGORIES)):
            add("Region × category", r + 1, c, -1, U[r, cat == c].sum(0), B[r, cat == c].sum(0))
    sold = U[:, :, :fit_end].sum(2) > 0  # (region, sku)
    sku_sold = sold.any(0)
    for s in np.nonzero(sku_sold)[0]:
        add("SKU", -1, int(cat[s]), int(w.skus["sku_id"].iloc[s]), U[:, s].sum(0), B[:, s].sum(0))
    for r in range(n_reg):
        for s in np.nonzero(sold[r])[0]:
            add("Region × SKU", r + 1, int(cat[s]), int(w.skus["sku_id"].iloc[s]), U[r, s], B[r, s])
    meta = pd.DataFrame(rows)
    y = np.vstack(ys)
    bulk = np.vstack(bs)

    lev = meta["level"].to_numpy()
    upper_idx = np.nonzero(np.isin(lev, UPPER))[0]
    rc_idx = np.nonzero(lev == "Region × category")[0]
    S = np.zeros((len(upper_idx), len(rc_idx)))
    um = meta.iloc[upper_idx]
    rcm = meta.iloc[rc_idx]
    for i, u in enumerate(um.itertuples()):
        S[i] = ((u.region < 0) | (rcm["region"].to_numpy() == u.region)) & ((u.category < 0) | (rcm["category"].to_numpy() == u.category))
    pair_idx = np.nonzero(lev == "Region × SKU")[0]
    sku_idx = np.nonzero(lev == "SKU")[0]
    pm = meta.iloc[pair_idx]
    rc_pos = {(r, c): i for i, (r, c) in enumerate(zip(rcm["region"], rcm["category"]))}
    sku_pos = {s: i for i, s in enumerate(meta.iloc[sku_idx]["sku_id"])}
    pair_parent = np.array([rc_pos[(r, c)] for r, c in zip(pm["region"], pm["category"])])
    pair_sku = np.array([sku_pos[s] for s in pm["sku_id"]])
    return Series(meta, y, bulk, S, upper_idx, rc_idx, pair_idx, pair_parent, sku_idx, pair_sku)


def first_sale(y: np.ndarray, upto: int) -> np.ndarray:
    """Month index (0-based) of each series' first sale up to month `upto`, or `upto` if none."""
    sold = y[:, :upto] > 0
    return np.where(sold.any(1), sold.argmax(1), upto)


def static_features(ser: Series, w: World) -> pd.DataFrame:
    """Static inputs per series (NaN where a level has no such trait)."""
    m = ser.meta
    reg = w.regions.set_index("id")
    sk = w.skus.set_index("sku_id")
    out = pd.DataFrame(index=m.index)
    out["level"] = m["level"].map({lv: i for i, lv in enumerate(LEVELS)})
    out["region"] = m["region"].where(m["region"] > 0)
    out["category"] = m["category"].where(m["category"] >= 0)
    for col, name in [("population_m", "region_population"), ("income_index", "region_income"), ("urban_share", "region_urban")]:
        out[name] = m["region"].map(reg[col]).where(m["region"] > 0)
    has_sku = m["sku_id"] > 0
    out["log_price"] = np.log(m["sku_id"].map(sk["price"])).where(has_sku)
    out["log_volume"] = np.log(m["sku_id"].map(sk["volume_l"])).where(has_sku)
    return out


@dataclass
class Windows:
    """Scaled one-year windows at forecast origins: inputs, and targets where the next year is known."""

    series: np.ndarray  # series row per window
    origin: np.ndarray  # origin month T (1-based: the last observed month)
    scale: np.ndarray
    lags: np.ndarray  # (window, 12) months T-11..T divided by scale
    dyn: pd.DataFrame  # growth, months_on_sale, empty_share, bulk_share, origin_month (1..12)
    target: np.ndarray | None  # (window, 12) months T+1..T+12 divided by scale
    ok: np.ndarray  # False where the series falls back to seasonal naive (fewer than 3 months on sale)


def windows(y: np.ndarray, bulk: np.ndarray, origins: list[int], with_target: bool,
            months_on_sale: np.ndarray | None = None, empty_share: np.ndarray | None = None) -> Windows:
    """Windows for every series at each origin T. `y` holds actual months and, beyond the data, forecasts
    already made (the recursion); `months_on_sale` and `empty_share` override the computed values then."""
    rows = []
    n = y.shape[0]
    for T in origins:
        hist = y[:, T - 12:T]
        fs = first_sale(y, T)
        on_sale = T - fs if months_on_sale is None else months_on_sale
        start = np.maximum(T - 12, fs)
        cnt = np.clip(T - start, 1, 12)
        mask = np.arange(T - 12, T)[None, :] >= start[:, None]
        scale = np.where(mask, hist, 0).sum(1) / cnt
        ok = (on_sale >= 3) & (scale > 0)
        safe = np.where(scale > 0, scale, 1.0)
        prev = y[:, T - 24:T - 12].sum(1) if T >= 24 else np.full(n, np.nan)
        growth = np.where((on_sale >= 24) & (prev > 0), hist.sum(1) / np.where(prev > 0, prev, 1), np.nan)
        es = (np.where(mask, hist == 0, False).sum(1) / cnt) if empty_share is None else empty_share
        bs = bulk[:, :T].sum(1) / np.maximum(y[:, :T].sum(1), 1e-9)
        rows.append({
            "series": np.arange(n), "origin": np.full(n, T), "scale": scale, "lags": hist / safe[:, None],
            "growth": growth, "months_on_sale": np.minimum(on_sale, MONTHS_ON_SALE_CAP).astype(float), "empty_share": es, "bulk_share": np.minimum(bs, 1.0),
            "origin_month": np.full(n, (T - 1) % 12 + 1), "ok": ok,
            "target": (y[:, T:T + 12] / safe[:, None]) if with_target else None,
        })
    def cat(k: str) -> np.ndarray:
        return np.concatenate([r[k] for r in rows])

    dyn = pd.DataFrame({k: cat(k) for k in ("growth", "months_on_sale", "empty_share", "bulk_share", "origin_month")})
    return Windows(cat("series"), cat("origin"), cat("scale"), np.vstack([r["lags"] for r in rows]), dyn,
                   np.vstack([r["target"] for r in rows]) if with_target else None, cat("ok"))


def level_weights(meta: pd.DataFrame, series: np.ndarray, scale: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Training weight per window: proportional to the series' volume within its level and origin, each
    level summing to the same total (agreed choice A5)."""
    lev = meta["level"].to_numpy()[series]
    key = pd.Series(lev).astype(str) + "|" + pd.Series(origin).astype(str)
    tot = pd.Series(scale).groupby(key).transform("sum").to_numpy()
    return np.where(tot > 0, scale / np.where(tot > 0, tot, 1), 0.0)
