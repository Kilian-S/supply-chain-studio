"""Multi-SKU inventory simulation used to price stocking policies (model J's business check).

Network: one fulfilment centre (FC) per region; Calopeia's FC doubles as the central FC. Each SKU is
stocked either in every regional FC ("distributed") or only centrally ("central"), at a target
service level. Weekly review, order-up-to level S = mu (L + R) + z sigma sqrt(L + R) from the
previous year's daily demand at that FC, lead time 7 days, lost sales when out of stock.

Costs per SKU: holding, replenishment order lines, a yearly slot cost per stocking point, outbound
delivery (by weight and distance), and the margin of lost sales. Lost sales come from stockouts and,
for central stocking, from slower delivery: some customers outside the central region do not wait. `cost_table` simulates every SKU under every (mode, service) option at once,
so any policy is just a choice per SKU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

LEAD, REVIEW = 7, 7
ORDER_COST = 5.0  # $ per replenishment order line (one SKU to one FC)
SLOT_COST = 150.0  # $ per year for each SKU stocked in an FC (space, handling, systems)
# Share of demand lost when delivery comes from the central FC instead of the customer's own region.
SLOW_DELIVERY_LOSS = {"same_landmass": 0.03, "cross_water": 0.10}
CENTRAL = 1  # region id of the central FC
SERVICE_LEVELS = [0.80, 0.85, 0.90, 0.95, 0.98, 0.995]
MODES = ["distributed", "central"]


def distance_multiplier(regions: pd.DataFrame) -> np.ndarray:
    """(from FC region, to customer region) multiplier: same region 1.0, same landmass 1.4, across water 2.2."""
    land = regions.set_index("id")["landmass"]
    ids = regions["id"].to_numpy()
    m = np.empty((len(ids), len(ids)))
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            m[i, j] = 1.0 if a == b else 1.4 if land[a] == land[b] else 2.2
    return m


def demand_cube(lines: pd.DataFrame, skus: pd.DataFrame, regions: pd.DataFrame, first: int, last: int) -> np.ndarray:
    """Units per (sku, region, day) for days first..last."""
    sub = lines[(lines["day"] >= first) & (lines["day"] <= last)]
    si = {s: i for i, s in enumerate(skus["sku_id"])}
    ri = {r: i for i, r in enumerate(regions["id"])}
    cube = np.zeros((len(skus), len(regions), last - first + 1))
    agg = sub.groupby(["sku_id", "region_id", "day"])["quantity"].sum()
    s = np.array([si[k[0]] for k in agg.index])
    r = np.array([ri[k[1]] for k in agg.index])
    d = np.array([k[2] - first for k in agg.index])
    cube[s, r, d] = agg.to_numpy()
    return cube


def simulate(test: np.ndarray, estimate: np.ndarray, skus: pd.DataFrame, regions: pd.DataFrame,
             mode: str, service: float) -> dict[str, np.ndarray]:
    """Per-SKU costs and service for one (mode, service) applied to all SKUs.

    test, estimate: (sku, region, day) demand for the simulated year and for the year the
    order-up-to levels are estimated from.
    """
    n_sku, n_reg, days = test.shape
    central = int(np.nonzero(regions["id"].to_numpy() == CENTRAL)[0][0])
    mult = distance_multiplier(regions)
    delivery_lost = np.zeros(n_sku)
    if mode == "central":
        keep = np.where(mult[central] == 1.0, 1.0, np.where(mult[central] == 1.4,
                        1 - SLOW_DELIVERY_LOSS["same_landmass"], 1 - SLOW_DELIVERY_LOSS["cross_water"]))
        delivery_lost = (test * (1 - keep)[None, :, None]).sum(axis=(1, 2))
        test = test * keep[None, :, None]
        estimate = estimate * keep[None, :, None]
        fc_demand = test.sum(axis=1, keepdims=True)  # all remaining regions served from one FC
        est = estimate.sum(axis=1, keepdims=True)
        out_mult = (test.sum(axis=2) * mult[central][None, :]).sum(axis=1) / np.maximum(test.sum(axis=(1, 2)), 1e-9)
    else:
        fc_demand, est = test, estimate
        out_mult = np.ones(n_sku)
    mu = est.mean(axis=2)
    sd = est.std(axis=2)
    z = norm.ppf(service)
    S = np.ceil(mu * (LEAD + REVIEW) + z * sd * np.sqrt(LEAD + REVIEW))
    S[mu == 0] = 0
    on_hand = S.copy()
    pipeline = np.zeros((LEAD,) + S.shape)
    h = ((0.25 * skus["unit_cost"].to_numpy() + 2.0 * skus["volume_l"].to_numpy()) / 365)[:, None]
    held = np.zeros(S.shape)
    served = np.zeros(S.shape)
    lost = np.zeros(S.shape)
    orders = np.zeros(S.shape)
    for t in range(days):
        on_hand += pipeline[t % LEAD]
        pipeline[t % LEAD] = 0
        d = fc_demand[:, :, t]
        sale = np.minimum(on_hand, d)
        on_hand -= sale
        served += sale
        lost += d - sale
        held += on_hand
        if t % REVIEW == 0:
            position = on_hand + pipeline.sum(axis=0)
            q = np.maximum(S - position, 0)
            orders += q > 0
            pipeline[(t + LEAD) % LEAD] += q  # arrives LEAD days later (slot reused after this day)
    price = skus["price"].to_numpy()
    margin = skus["margin"].to_numpy()
    weight = skus["weight_kg"].to_numpy()
    units = served.sum(axis=1)
    points = (S > 0).sum(axis=1).astype(float)
    return {
        "holding": (held * h).sum(axis=1),
        "ordering": orders.sum(axis=1) * ORDER_COST,
        "slots": points * SLOT_COST * days / 365,
        "outbound": units * (1.5 + 0.35 * weight) * out_mult,
        "lost_margin": (lost.sum(axis=1) + delivery_lost) * price * margin,
        "served": units,
        "demand": fc_demand.sum(axis=(1, 2)) + delivery_lost,
        "stock_value": (held.sum(axis=1) / days) * skus["unit_cost"].to_numpy(),
        "points": points,
    }


def cost_table(lines: pd.DataFrame, skus: pd.DataFrame, regions: pd.DataFrame, year: int) -> dict:
    """Simulate year `year` (1-based) with levels estimated from the year before, for every option."""
    first, last = (year - 1) * 365 + 1, year * 365
    test = demand_cube(lines, skus, regions, first, last)
    est = demand_cube(lines, skus, regions, first - 365, first - 1)
    table = {}
    for mode in MODES:
        for s in SERVICE_LEVELS:
            r = simulate(test, est, skus, regions, mode, s)
            r["total"] = r["holding"] + r["ordering"] + r["slots"] + r["outbound"] + r["lost_margin"]
            table[(mode, s)] = r
    return table


def evaluate(table: dict, choice: list[tuple[str, float]]) -> dict[str, float]:
    """Totals for a policy given as one (mode, service) per SKU."""
    keys = ["holding", "ordering", "slots", "outbound", "lost_margin", "total", "served", "demand", "stock_value", "points"]
    out = {k: 0.0 for k in keys}
    for i, opt in enumerate(choice):
        r = table[opt]
        for k in keys:
            out[k] += float(r[k][i])
    out["fill_rate"] = out["served"] / out["demand"] if out["demand"] else 1.0
    return out


def best_per_group(table: dict, groups: np.ndarray) -> dict:
    """For each group, the (mode, service) with the lowest total cost over its SKUs in `table`."""
    best = {}
    for g in np.unique(groups):
        idx = groups == g
        best[g] = min(table, key=lambda opt: table[opt]["total"][idx].sum())
    return best
