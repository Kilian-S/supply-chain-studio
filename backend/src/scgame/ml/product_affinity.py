"""Product affinity: which SKUs are bought together, and does storing them together prevent split orders?

Usage: uv run python -m scgame.ml.product_affinity <world dir> <output dir>

Method
  1. Baskets of the training set (years 1 to 3). For every pair of SKUs: how many baskets contain both,
     how many chance alone would give within baskets of the same region and size (lift = observed /
     expected), and how surprising the count is under independence (Poisson tail). The
     Benjamini-Hochberg procedure keeps the false discovery rate among accepted pairs at a chosen level.
  2. Accepted pairs with lift above 1 form a graph, weighted by log lift. Louvain communities group
     densely linked SKUs; SKUs without an accepted pair are bought alone.
  3. Business check, a split catalogue: the central site runs 2 to 5 FCs, each SKU is stored in
     exactly one, each FC holds an equal share of the catalogue (5% slack). A basket whose SKUs sit in
     different FCs ships as several parcels (a split order). Assignment rules, built on the training
     set: at random, by category, by popularity (spread evenly), by community, and by the hidden
     bundles (reference only).
  4. The false discovery rate and the Louvain resolution are chosen by delivery cost of the
     community rule on the validation set (year 4). Year 5 (test set) is priced once. The hidden bundles are used only to
     check the communities, never to choose.

Outputs (no card.json until the page exists)
  overview.json  facts, settings grid, chosen setting, check against the hidden bundles, test results
  graph.json     SKUs with their community, accepted pairs with count, lift and p-value
  matrix.json    co-purchase counts and lift for every pair (training set), for the heatmaps
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities
from scipy import sparse
from scipy.stats import poisson
from sklearn.metrics import adjusted_rand_score

from . import policy_sim as ps

TRAIN_END = 3 * 365  # training set years 1 to 3
VAL = (TRAIN_END + 1, 4 * 365)  # validation set year 4
TEST = (4 * 365 + 1, 5 * 365)  # test set year 5
FDR_LEVELS = [0.001, 0.01, 0.05]
RESOLUTIONS = [0.5, 1.0, 1.5, 2.0, 3.0]
FCS = [2, 3, 4, 5]  # FCs the catalogue is split across
SLACK = 0.05  # each FC holds at most (1 + SLACK) x its equal share of SKUs
PARCEL_FEE = 1.5  # $ per shipment, plus $0.35 per kg, times the distance multiplier (as in policy_sim)
PER_KG = 0.35
SEED = 11


# =================================================================================================
# Baskets and pair statistics
# =================================================================================================

def baskets(lines: pd.DataFrame, sku_index: dict[int, int], first: int, last: int) -> dict:
    """Orders in a period as a binary order x SKU matrix, with each order's region and weight."""
    sub = lines[(lines["day"] >= first) & (lines["day"] <= last)]
    codes, uniques = pd.factorize(sub["order_id"])
    cols = sub["sku_id"].map(sku_index).to_numpy()
    X = sparse.csr_matrix((np.ones(len(sub), dtype=np.int32), (codes, cols)), shape=(len(uniques), len(sku_index)))
    X.data[:] = 1
    Q = sparse.csr_matrix((sub["quantity"].to_numpy().astype(float), (codes, cols)), shape=X.shape)
    region = sub.groupby(codes)["region_id"].first().to_numpy()
    return {"X": X, "Q": Q, "region": region}


def pair_stats(X: sparse.csr_matrix, region: np.ndarray) -> dict[str, np.ndarray]:
    """Co-purchase counts, lift and one-sided p-values for every pair.

    Chance is computed within comparable baskets: same region and same basket size (1, 2, 3, 4, 5+).
    Regions sell different SKUs and large baskets hold many, so a shop-wide baseline would link SKUs
    that merely share a region or a basket size. Expected count = sum over strata of
    count_s(a) count_s(b) / n_s; the p-value uses a Poisson tail with that mean (its variance is at
    least the hypergeometric one, so the test is slightly conservative)."""
    n = X.shape[0]
    co = (X.T @ X).toarray().astype(float)
    cnt = np.diag(co).copy()
    np.fill_diagonal(co, 0)
    size = np.minimum(np.asarray(X.sum(axis=1)).ravel(), 5)
    expected = np.zeros_like(co)
    for r in np.unique(region):
        for k in range(2, 6):
            rows = np.nonzero((region == r) & (size == k))[0]
            if len(rows):
                c = np.asarray(X[rows].sum(axis=0)).ravel().astype(float)
                expected += np.outer(c, c) / len(rows)
    np.fill_diagonal(expected, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(expected > 0, co / expected, 0.0)
    p = np.where(co > 0, poisson.sf(co - 1, np.maximum(expected, 1e-12)), 1.0)
    np.fill_diagonal(p, 1.0)
    return {"n": n, "count": co, "sku_count": cnt, "expected": expected, "lift": lift, "p": p}


def bh_accept(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg over all pairs: accepted pairs keep the false discovery rate at q."""
    iu = np.triu_indices(p.shape[0], 1)
    pv = p[iu]
    m = len(pv)
    order = np.argsort(pv)
    passed = pv[order] <= q * np.arange(1, m + 1) / m
    k = np.nonzero(passed)[0].max() + 1 if passed.any() else 0
    acc = np.zeros(m, dtype=bool)
    acc[order[:k]] = True
    out = np.zeros_like(p, dtype=bool)
    out[iu] = acc
    return out | out.T


def communities(accepted: np.ndarray, weight: np.ndarray, resolution: float) -> np.ndarray:
    """Louvain communities on the accepted pairs; every SKU without an accepted pair is its own
    community of one (bought alone). Labels are ordered by community size, largest first."""
    n = accepted.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    a, b = np.nonzero(np.triu(accepted, 1))
    g.add_weighted_edges_from(zip(a.tolist(), b.tolist(), weight[a, b].tolist()))
    groups = louvain_communities(g, weight="weight", resolution=resolution, seed=SEED)
    groups = sorted((sorted(c) for c in groups), key=lambda c: (-len(c), c[0]))
    labels = np.empty(n, dtype=int)
    for i, c in enumerate(groups):
        labels[c] = i
    return labels


# =================================================================================================
# Split catalogue: assignment rules and delivery
# =================================================================================================

def capacity(n_sku: int, k: int) -> int:
    return int(np.ceil(n_sku / k * (1 + SLACK)))


def pack(units: list[np.ndarray], k: int, n_sku: int) -> np.ndarray:
    """Place whole units, largest first, each in the FC with the most free room; SKUs are never split
    from their unit unless a unit is larger than an FC."""
    cap = capacity(n_sku, k)
    fc = np.full(n_sku, -1)
    room = np.full(k, cap)
    for u in sorted(units, key=len, reverse=True):
        f = int(np.argmax(room))
        if len(u) <= room[f]:
            fc[u] = f
            room[f] -= len(u)
        else:  # too large for any FC: spread it
            for s_ in u:
                g = int(np.argmax(room))
                fc[s_] = g
                room[g] -= 1
    return fc


def snake(order: np.ndarray, k: int) -> np.ndarray:
    """Deal SKUs in the given order to FCs 1..k, k..1, 1..k: spreads popular SKUs evenly."""
    fc = np.empty(len(order), dtype=int)
    pattern = np.concatenate([np.arange(k), np.arange(k)[::-1]])
    fc[order] = pattern[np.arange(len(order)) % (2 * k)]
    return fc


def assignments(train: dict, categories: np.ndarray, labels: np.ndarray, truth: np.ndarray) -> dict[str, dict[int, np.ndarray]]:
    """SKU-to-FC assignment per rule and number of FCs, from the training set only (truth only for
    the reference rule). Communities and bundles are packed whole; the remaining SKUs fill the free
    room most popular first."""
    n_sku = train["X"].shape[1]
    pop = popularity_order(train["X"])
    rng = np.random.default_rng(SEED)

    def grouped(groups: np.ndarray) -> list[np.ndarray]:
        return [np.nonzero(groups == g)[0] for g in np.unique(groups)]

    def with_singles(k: int, groups: list[np.ndarray]) -> np.ndarray:
        big = [g for g in groups if len(g) > 1]
        fc = pack(big, k, n_sku)
        room = capacity(n_sku, k) - np.bincount(fc[fc >= 0], minlength=k)
        for s_ in pop:
            if fc[s_] < 0:
                f = int(np.argmax(room))
                fc[s_] = f
                room[f] -= 1
        return fc

    out: dict[str, dict[int, np.ndarray]] = {r: {} for r in ("random", "category", "popularity", "communities", "hidden_bundles")}
    for k in FCS:
        out["random"][k] = snake(rng.permutation(n_sku), k)
        out["category"][k] = pack(grouped(categories), k, n_sku)
        out["popularity"][k] = snake(pop, k)
        out["communities"][k] = with_singles(k, grouped(labels))
        out["hidden_bundles"][k] = with_singles(k, [np.nonzero(truth == b)[0] for b in np.unique(truth[truth > 0])])
    return out


def popularity_order(X: sparse.csr_matrix) -> np.ndarray:
    """SKUs by number of baskets, most popular first."""
    return np.argsort(-np.asarray(X.sum(axis=0)).ravel(), kind="stable")


def deliver(b: dict, fc: np.ndarray, k: int, weight: np.ndarray, mult: np.ndarray, region_ids: list[int]) -> dict[str, float]:
    """Parcels and delivery cost of a period's baskets when SKU i is stored in FC fc[i]. All FCs are at
    the central site; each FC that holds part of a basket ships one parcel."""
    X, Q, region = b["X"], b["Q"], b["region"]
    kg = Q.multiply(weight[None, :]).tocsr()
    ri = {r: i for i, r in enumerate(region_ids)}
    m = np.array([mult[ri[ps.CENTRAL], ri[r]] for r in region])
    parcels = np.zeros(X.shape[0])
    cost = np.zeros(X.shape[0])
    for f in range(k):
        on = (fc == f).astype(float)
        has = (X @ on) > 0
        parcels += has
        cost += has * (PARCEL_FEE + PER_KG * (kg @ on)) * m
    multi = np.asarray(X.sum(axis=1)).ravel() > 1
    split = parcels > 1
    return {"fcs": k, "orders": int(X.shape[0]), "multi": int(multi.sum()), "split": int(split.sum()),
            "parcels": int(parcels.sum()), "cost": float(cost.sum()),
            "split_share": float(split.sum() / X.shape[0]), "multi_split_share": float(split.sum() / max(multi.sum(), 1))}


def price(rule: dict[int, np.ndarray], period: dict, weight: np.ndarray, mult: np.ndarray, region_ids: list[int]) -> list[dict]:
    return [deliver(period, rule[k], k, weight, mult, region_ids) for k in FCS]


# =================================================================================================
# Checks against the hidden bundles
# =================================================================================================

def pair_check(labels: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Pairs put together versus pairs truly bundled. truth: bundle id, 0 for no bundle."""
    iu = np.triu_indices(len(labels), 1)
    together = labels[iu[0]] == labels[iu[1]]
    true = (truth[iu[0]] == truth[iu[1]]) & (truth[iu[0]] > 0)
    tp = int((together & true).sum())
    unique_truth = np.where(truth > 0, truth, -np.arange(1, len(truth) + 1))  # unbundled SKUs alone
    return {"precision": tp / max(int(together.sum()), 1), "recall": tp / max(int(true.sum()), 1),
            "pairs_together": int(together.sum()), "pairs_true": int(true.sum()),
            "ari": float(adjusted_rand_score(unique_truth, labels))}


# =================================================================================================
# Main
# =================================================================================================

def main(world_dir: str, out_dir: str) -> None:
    t0 = time.time()
    world, out = Path(world_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    skus = pd.read_parquet(world / "skus.parquet")
    regions = pd.read_parquet(world / "regions.parquet")
    lines = pd.read_parquet(world / "order_lines.parquet")
    bundles = pd.read_parquet(world / "bundles.parquet")
    sku_index = {int(s): i for i, s in enumerate(skus["sku_id"])}
    region_ids = regions["id"].tolist()
    weight = skus["weight_kg"].to_numpy()
    categories = pd.factorize(skus["category"])[0]
    mult = ps.distance_multiplier(regions)
    truth = np.zeros(len(skus), dtype=int)
    truth[[sku_index[int(s)] for s in bundles["sku_id"]]] = bundles["bundle_id"].to_numpy()

    train = baskets(lines, sku_index, 1, TRAIN_END)
    val = baskets(lines, sku_index, *VAL)
    test = baskets(lines, sku_index, *TEST)
    st = pair_stats(train["X"], train["region"])
    log_lift = np.log(np.maximum(st["lift"], 1e-12))
    print(f"pair statistics ready ({time.time() - t0:.0f}s)")

    # Settings grid: false discovery rate x resolution, chosen by delivery cost on the validation set.
    grid = []
    for q in FDR_LEVELS:
        accepted = bh_accept(st["p"], q) & (st["lift"] > 1)
        for res in RESOLUTIONS:
            labels = communities(accepted, log_lift, res)
            rule = assignments(train, categories, labels, truth)["communities"]
            v = price(rule, val, weight, mult, region_ids)
            sizes = np.bincount(labels)
            grid.append({"fdr": q, "resolution": res, "edges": int(np.triu(accepted, 1).sum()),
                         "communities": int((sizes > 1).sum()), "alone": int((sizes == 1).sum()),
                         "val_cost": float(np.mean([x["cost"] for x in v])),
                         "val_split_share": float(np.mean([x["split_share"] for x in v])),
                         "truth": pair_check(labels, truth), "labels": labels.tolist()})
            print(f"  fdr {q} resolution {res}: {grid[-1]['communities']} communities, val cost ${grid[-1]['val_cost']:,.0f} ({time.time() - t0:.0f}s)")
    best = min(grid, key=lambda g: g["val_cost"])
    labels = np.array(best["labels"])
    accepted = bh_accept(st["p"], best["fdr"]) & (st["lift"] > 1)
    print(f"chose fdr {best['fdr']} resolution {best['resolution']}")

    # Baselines for the check against the hidden bundles.
    n_edges = best["edges"]
    iu = np.triu_indices(len(skus), 1)
    top = np.argsort(-st["count"][iu])[:n_edges]
    by_count = np.zeros_like(accepted)
    by_count[iu[0][top], iu[1][top]] = True
    by_count = by_count | by_count.T
    count_labels = communities(by_count, st["count"], best["resolution"])
    check = {"communities": pair_check(labels, truth), "count_communities": pair_check(count_labels, truth),
             "categories": pair_check(categories, truth)}

    # Test set, priced once for every assignment rule.
    bundle_units = [np.nonzero(truth == b)[0] for b in np.unique(truth[truth > 0])]
    rules = assignments(train, categories, labels, truth)
    results = {name: price(rule, test, weight, mult, region_ids) for name, rule in rules.items()}
    print(f"test set priced ({time.time() - t0:.0f}s)")

    X = train["X"]
    basket_sizes = np.asarray(X.sum(axis=1)).ravel()
    top200_count = np.argsort(-st["count"][iu])[:200]
    top200_lift = np.argsort(-np.where(st["count"][iu] >= 5, st["lift"][iu], 0))[:200]
    same_bundle = (truth[iu[0]] == truth[iu[1]]) & (truth[iu[0]] > 0)
    overview = {
        "model": {"id": "product-affinity", "title": "Bought together"},
        "periods": {"train_end": TRAIN_END, "validation": VAL, "test": TEST},
        "facts": {"orders": int(X.shape[0]), "multi_share": float((basket_sizes > 1).mean()),
                  "skus_per_order": float(basket_sizes.mean()), "pairs_seen": int((st["count"][iu] > 0).sum()),
                  "pairs_possible": int(len(iu[0])),
                  "top200_count_true_share": float(same_bundle[top200_count].mean()),
                  "top200_lift_true_share": float(same_bundle[top200_lift].mean()),
                  "basket_size_counts": np.bincount(basket_sizes).tolist()},
        "settings": {"fdr_levels": FDR_LEVELS, "resolutions": RESOLUTIONS, "fcs": FCS, "slack": SLACK,
                     "parcel_fee": PARCEL_FEE, "per_kg": PER_KG, "central_region": ps.CENTRAL},
        "grid": [{k: v for k, v in g.items() if k != "labels"} for g in grid],
        "chosen": {"fdr": best["fdr"], "resolution": best["resolution"]},
        "truth": {"bundles": int(len(bundle_units)), "skus_in_bundles": int((truth > 0).sum()), "check": check},
        "results": results,
        "assignments": {name: {str(k): v.tolist() for k, v in rule.items()} for name, rule in rules.items()},
        "built_seconds": round(time.time() - t0, 1),
    }
    (out / "overview.json").write_text(json.dumps(overview, separators=(",", ":")))

    a, b = np.nonzero(np.triu(accepted, 1))
    graph = {
        "nodes": [{"id": int(s.sku_id), "name": s.name, "category": s.category, "baskets": int(st["sku_count"][i]),
                   "community": int(labels[i]), "alone": bool((labels == labels[i]).sum() == 1), "bundle": int(truth[i])}
                  for i, s in enumerate(skus.itertuples())],
        "edges": [{"a": int(i), "b": int(j), "count": int(st["count"][i, j]), "lift": round(float(st["lift"][i, j]), 2),
                   "p": float(st["p"][i, j])} for i, j in zip(a, b)],
    }
    (out / "graph.json").write_text(json.dumps(graph, separators=(",", ":")))
    matrix = {"count": st["count"].astype(int).tolist(), "lift": np.round(st["lift"], 2).tolist()}
    (out / "matrix.json").write_text(json.dumps(matrix, separators=(",", ":")))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
