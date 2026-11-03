"""Order-level fulfilment simulation for model H (split orders).

A network is the central FC (full range) plus regional FCs at some of the other region sites, each listing
a share of the SKUs. Every FC stocks its listed SKUs under the stocking policy of model J's page
(`policy_sim`): weekly review, 7-day lead time, order-up-to level S = mu (L + R) + z sigma sqrt(L + R)
from the previous year's daily demand that would reach the FC under the network, lost sales when no FC
has stock.

Routing. A region's home FC is its own FC if it has one, otherwise the central FC. Each order line tries
its region's FCs in a fixed order (home FC, central FC, then the others by distance multiplier and id)
and ships from the first that lists the SKU and has stock. Within a day the orders arrive in random
order and FCs serve lines in rounds (first choices, then second choices, ...), first come first served
within a round; once an FC runs short of a SKU on a day, later lines of that day go elsewhere. An order
that ships from two or more FCs is a split order.

Planning. The network for year y is planned at the end of year y - 1. Listings rank SKUs by the order
lines of the FC's own region in year y - 1; order-up-to levels come from the daily demand of year
`levels_from` (y - 1 for the real year, y - 2 for the replay simulation of year y - 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import policy_sim as ps
from .product_affinity import PARCEL_FEE, PER_KG

YEAR = 365
RULES = ["best_sellers", "clusters", "communities", "categories", "random"]
SHARES = [0.2, 0.4, 0.6, 0.8]
SERVICES = [0.90, 0.95, 0.98]
COVER_CAP = 20.0  # cover above this many lead-time-plus-review demands counts as this many


@dataclass(frozen=True)
class Network:
    """One planned network. sites: region ids of the regional FCs (the central FC is always open)."""

    sites: tuple[int, ...]
    share: float
    rule: str
    service: float
    seed: int = 0

    @property
    def fcs(self) -> tuple[int, ...]:
        return (ps.CENTRAL, *self.sites)

    def as_dict(self) -> dict:
        return {"sites": list(self.sites), "share": self.share, "rule": self.rule, "service": self.service, "seed": self.seed}


@dataclass
class World:
    """World v1 years 1 to 5 as arrays. Lines are sorted by day, then order id."""

    skus: pd.DataFrame
    regions: pd.DataFrame
    region_ids: np.ndarray
    mult: np.ndarray  # (from region index, to region index) distance multiplier
    order: np.ndarray  # per line: order index (0 .. orders - 1, increasing with day)
    sku: np.ndarray  # per line: SKU index
    qty: np.ndarray
    day: np.ndarray
    region: np.ndarray  # per line: customer region index
    order_day: np.ndarray  # per order
    order_region: np.ndarray  # per order: region index
    order_id: np.ndarray  # per order: id in order_lines.parquet
    weight: np.ndarray  # per SKU, kg
    cluster: np.ndarray  # per SKU: model J cluster (as committed)
    cluster_distributed: np.ndarray  # per cluster: model J's stocking policy is distributed
    community: np.ndarray  # per SKU: model K community (as committed)
    alone: np.ndarray  # per SKU: model K found no linked pair
    category: np.ndarray  # per SKU: category index

    @property
    def n_sku(self) -> int:
        return len(self.skus)

    def year_lines(self, year: int) -> slice:
        a, b = np.searchsorted(self.day, [(year - 1) * YEAR + 1, year * YEAR + 1])
        return slice(int(a), int(b))


def load_world(world_dir: str | Path, models_dir: str | Path) -> World:
    world, models = Path(world_dir), Path(models_dir)
    skus = pd.read_parquet(world / "skus.parquet")
    regions = pd.read_parquet(world / "regions.parquet")
    lines = pd.read_parquet(world / "order_lines.parquet").sort_values(["day", "order_id", "sku_id"], kind="stable")
    sku_index = {int(s): i for i, s in enumerate(skus["sku_id"])}
    region_ids = regions["id"].to_numpy()
    rix = {int(r): i for i, r in enumerate(region_ids)}
    codes, uniques = pd.factorize(lines["order_id"])  # orders numbered in day order
    first = np.unique(codes, return_index=True)[1]
    day = lines["day"].to_numpy()
    reg = lines["region_id"].map(rix).to_numpy()

    import json

    seg = json.loads((models / "sku-segmentation" / "overview.json").read_text())["default"]
    variant = json.loads((models / "sku-segmentation" / "variants" / f"m{seg['mask']}.json").read_text())
    clustering = variant["clusterings"][f"{seg['model']}-{seg['k']}"]
    graph = json.loads((models / "product-affinity" / "graph.json").read_text())
    node = {n["id"]: n for n in graph["nodes"]}
    return World(
        skus=skus, regions=regions, region_ids=region_ids, mult=ps.distance_multiplier(regions),
        order=codes.astype(np.int64), sku=lines["sku_id"].map(sku_index).to_numpy(), qty=lines["quantity"].to_numpy().astype(float),
        day=day, region=reg, order_day=day[first], order_region=reg[first], order_id=np.asarray(uniques),
        weight=skus["weight_kg"].to_numpy(), cluster=np.array(clustering["labels"]),
        cluster_distributed=np.array([p["mode"] == "distributed" for p in clustering["policy"]]),
        community=np.array([node[int(s)]["community"] for s in skus["sku_id"]]),
        alone=np.array([node[int(s)]["alone"] for s in skus["sku_id"]]),
        category=pd.factorize(skus["category"])[0],
    )


# =================================================================================================
# Network plan: routing order, listing, order-up-to levels
# =================================================================================================

def preferences(w: World, net: Network) -> np.ndarray:
    """(customer region index, k) -> index into net.fcs of the k-th FC a line tries."""
    fc_reg = [int(np.nonzero(w.region_ids == f)[0][0]) for f in net.fcs]
    pref = np.zeros((len(w.region_ids), len(fc_reg)), dtype=int)
    for r in range(len(w.region_ids)):
        home = fc_reg.index(r) if r in fc_reg else 0
        rest = sorted((i for i in range(len(fc_reg)) if i not in (home, 0)), key=lambda i: (w.mult[fc_reg[i], r], net.fcs[i]))
        pref[r] = [home] + ([0] if home != 0 else []) + rest
    return pref


def home_fc(w: World, net: Network) -> np.ndarray:
    return preferences(w, net)[:, 0]


def lines_per_sku(w: World, year: int, region_index: int | None = None) -> np.ndarray:
    sl = w.year_lines(year)
    s = w.sku[sl]
    if region_index is not None:
        s = s[w.region[sl] == region_index]
    return np.bincount(s, minlength=w.n_sku).astype(float)


def listing(w: World, net: Network, year: int) -> np.ndarray:
    """(FC, SKU) listed, for the network planned for `year` (ranked on year - 1). The central FC lists all."""
    listed = np.zeros((len(net.fcs), w.n_sku), dtype=bool)
    listed[0] = True
    n = round(net.share * w.n_sku)
    rng = np.random.default_rng([net.seed, year])
    for i, site in enumerate(net.sites, start=1):
        r = int(np.nonzero(w.region_ids == site)[0][0])
        load = lines_per_sku(w, year - 1, r)
        best = np.argsort(-load, kind="stable")
        chosen = np.zeros(w.n_sku, dtype=bool)
        if net.rule == "random":
            p = (load + 1) / (load + 1).sum()
            chosen[rng.choice(w.n_sku, size=n, replace=False, p=p)] = True
        else:
            if net.rule != "best_sellers":
                groups = {"clusters": w.cluster, "categories": w.category,
                          "communities": np.where(w.alone, -1, w.community)}[net.rule]
                units = [np.nonzero(groups == g)[0] for g in np.unique(groups) if g >= 0]
                units = [u for u in units if len(u) > 1 or net.rule != "communities"]
                units.sort(key=lambda u: (-load[u].mean(), u[0]))
                for u in units:
                    if chosen.sum() + len(u) <= n:
                        chosen[u] = True
            for s in best:  # fill the remaining slots with the region's best sellers
                if chosen.sum() >= n:
                    break
                chosen[s] = True
        listed[i] = chosen
    return listed


def demand_at_fcs(w: World, net: Network, listed: np.ndarray, year: int) -> np.ndarray:
    """(FC, SKU, day) units of `year` if every line shipped from the first FC on its route that lists it."""
    sl = w.year_lines(year)
    pref = preferences(w, net)
    s, r, q, d = w.sku[sl], w.region[sl], w.qty[sl], w.day[sl] - (year - 1) * YEAR - 1
    target = np.full(len(s), -1)
    for k in range(pref.shape[1] - 1, -1, -1):
        f = pref[r, k]
        target = np.where(listed[f, s], f, target)
    cube = np.zeros((len(net.fcs), w.n_sku, YEAR))
    np.add.at(cube, (target, s, d), q)
    return cube


def levels(cube: np.ndarray, listed: np.ndarray, service: float) -> np.ndarray:
    """Order-up-to levels as in policy_sim, from a year of daily demand at each FC."""
    mu, sd = cube.mean(axis=2), cube.std(axis=2)
    h = ps.LEAD + ps.REVIEW
    S = np.ceil(mu * h + norm.ppf(service) * sd * np.sqrt(h))
    S[(mu == 0) | ~listed] = 0
    return S


@dataclass
class Plan:
    net: Network
    year: int  # the year the network is planned for (listing ranked on year - 1)
    listed: np.ndarray
    S: np.ndarray
    pref: np.ndarray
    monthly_mu: np.ndarray  # (FC, SKU, month) mean daily demand in the year levels come from
    levels_from: int


def month_of_doy(doy: np.ndarray) -> np.ndarray:
    return np.minimum((doy / YEAR * 12).astype(int), 11)


def plan(w: World, net: Network, year: int, levels_from: int | None = None) -> Plan:
    levels_from = year - 1 if levels_from is None else levels_from
    listed = listing(w, net, year)
    cube = demand_at_fcs(w, net, listed, levels_from)
    month = month_of_doy(np.arange(YEAR))
    monthly = np.stack([cube[:, :, month == m].mean(axis=2) for m in range(12)], axis=2)
    return Plan(net, year, listed, levels(cube, listed, net.service), preferences(w, net), monthly, levels_from)


# =================================================================================================
# Simulation
# =================================================================================================

@dataclass
class Outcome:
    """One simulated year. Per line of the year: FC index shipped from (-1 lost). Per order of the year
    (index relative to the year's first order): SKUs, parcels, split, cost and the plain rule (the home FC
    lists some but not all of the order's SKUs, exact when stock never runs out)."""

    year: int
    ship: np.ndarray
    first_order: int
    n_skus: np.ndarray
    parcels: np.ndarray
    split: np.ndarray
    rule: np.ndarray
    cost: np.ndarray
    cost_one: np.ndarray  # one parcel of the whole order from its home FC
    lost_lines: np.ndarray
    attempts: np.ndarray  # (FC, SKU, month) lines that reached the FC with the SKU listed
    fills: np.ndarray  # (FC, SKU, month) of those, lines filled


def simulate(w: World, p: Plan, year: int, seed: int = 0, unlimited: bool = False) -> Outcome:
    """Run the orders of `year` through the plan. unlimited: every listed SKU always in stock (for checks)."""
    sl = w.year_lines(year)
    n_fc = len(p.net.fcs)
    s, r, q, d = w.sku[sl], w.region[sl], w.qty[sl], w.day[sl] - (year - 1) * YEAR - 1
    o = w.order[sl]
    # Random arrival order of the day's orders, kept for all of an order's lines.
    rng = np.random.default_rng([seed, year, 17])
    rank = rng.permutation(o.max() - o.min() + 1)[o - o.min()]
    seq = np.lexsort((rank, d))
    s, r, q, d, o = s[seq], r[seq], q[seq], d[seq], o[seq]
    starts = np.searchsorted(d, np.arange(YEAR + 1))
    S = p.S.ravel()
    on_hand = np.full(S.shape, np.inf) if unlimited else S.copy()
    pipe = np.zeros((ps.LEAD, len(S)))
    listed = p.listed.ravel()
    ship = np.full(len(s), -2)
    month = month_of_doy(np.arange(YEAR))
    attempts = np.zeros((n_fc * w.n_sku, 12))
    fills = np.zeros((n_fc * w.n_sku, 12))
    for t in range(YEAR):
        on_hand += pipe[t % ps.LEAD]
        pipe[t % ps.LEAD] = 0
        a, b = starts[t], starts[t + 1]
        idx = np.arange(a, b)
        for k in range(n_fc):
            idx = idx[ship[idx] == -2]
            if not len(idx):
                break
            key = p.pref[r[idx], k] * w.n_sku + s[idx]
            ok_list = listed[key]
            ii, kk = idx[ok_list], key[ok_list]
            if not len(ii):
                continue
            order = np.argsort(kk, kind="stable")
            ii, kk = ii[order], kk[order]
            qq = q[ii]
            cs = np.cumsum(qq)
            new = np.r_[True, kk[1:] != kk[:-1]]
            within = cs - np.maximum.accumulate(np.where(new, cs - qq, 0))
            ok = within <= on_hand[kk]
            ship[ii[ok]] = kk[ok] // w.n_sku
            np.add.at(attempts[:, month[t]], kk, 1)
            np.add.at(fills[:, month[t]], kk[ok], 1)
            if not unlimited:
                np.subtract.at(on_hand, kk[ok], qq[ok])
        ship[idx[ship[idx] == -2]] = -1
        if t % ps.REVIEW == 0 and not unlimited:
            position = on_hand + pipe.sum(axis=0)
            pipe[t % ps.LEAD] += np.maximum(S - position, 0)  # arrives LEAD days later
    back = np.empty_like(seq)
    back[seq] = np.arange(len(seq))
    ship = ship[back]  # in the World's line order again
    return outcome(w, p, year, ship, attempts.reshape(n_fc, w.n_sku, 12), fills.reshape(n_fc, w.n_sku, 12))


def outcome(w: World, p: Plan, year: int, ship: np.ndarray, attempts: np.ndarray, fills: np.ndarray) -> Outcome:
    sl = w.year_lines(year)
    o = w.order[sl]
    first = int(o.min())
    oi = o - first
    n_orders = int(oi.max()) + 1
    s = w.sku[sl]
    kg = w.qty[sl] * w.weight[s]
    region = w.order_region[first:first + n_orders]
    home = p.pref[:, 0]
    fc_reg = np.array([int(np.nonzero(w.region_ids == f)[0][0]) for f in p.net.fcs])
    n_skus = np.bincount(oi, minlength=n_orders)
    shipped = ship >= 0
    pair = np.unique(oi[shipped] * len(p.net.fcs) + ship[shipped])
    parcels = np.bincount(pair // len(p.net.fcs), minlength=n_orders)
    pkg = np.bincount(oi[shipped] * len(p.net.fcs) + ship[shipped], weights=kg[shipped], minlength=n_orders * len(p.net.fcs))
    pkg = pkg.reshape(n_orders, len(p.net.fcs))
    m = w.mult[fc_reg][:, region].T  # (order, FC)
    cost = ((pkg > 0) * (PARCEL_FEE + PER_KG * pkg) * m).sum(axis=1)
    cost_one = (PARCEL_FEE + PER_KG * pkg.sum(axis=1)) * w.mult[fc_reg[home[region]], region] * (parcels > 0)
    listed_home = np.bincount(oi, weights=p.listed[home[w.region[sl]], s], minlength=n_orders)
    rule = (listed_home > 0) & (listed_home < n_skus)  # the plain rule: home FC lists some, not all
    lost = np.bincount(oi, weights=~shipped, minlength=n_orders)
    return Outcome(year, ship, first, n_skus, parcels, parcels > 1, rule, cost, cost_one, lost, attempts, fills)
