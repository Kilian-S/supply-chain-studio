"""Product affinity: which SKUs are bought together, and does storing them together prevent split orders?

Usage: uv run python -m scgame.ml.product_affinity <world dir> <output dir>

Method
  1. Baskets of the training set (years 1 to 3). For every pair of SKUs: how many baskets contain both,
     how many chance alone would give within baskets of the same region, size and quarter (lift = observed /
     expected), and how surprising the count is under independence (Poisson tail). The
     Benjamini-Hochberg procedure keeps the false discovery rate among accepted pairs at a chosen level.
  2. Accepted pairs with lift above 1 form a graph, weighted by log lift. Louvain communities group
     densely linked SKUs; SKUs without an accepted pair are bought alone.
  3. Business check, a split catalogue: the central site runs 2 to 5 FCs, each SKU is stored in
     exactly one, each FC holds an equal share of the SKUs and of the workload (order lines), 5% slack. A basket whose SKUs sit in
     different FCs ships as several parcels (a split order). Assignment rules, built on the training
     set: at random, by category, by popularity (spread evenly), best sellers together, by community,
     and by the hidden bundles (reference only).
     The quarter in the chance baseline was added after the check against the hidden bundles showed
     timing links; overview.json reports the same choice without it (quarter_note).
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
    quarter = (sub.groupby(codes)["day"].first().to_numpy() - 1) // 91
    return {"X": X, "Q": Q, "region": region, "quarter": quarter}


def pair_stats(X: sparse.csr_matrix, region: np.ndarray, quarter: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Co-purchase counts, lift and one-sided p-values for every pair.

    Chance is computed within comparable baskets: same region, same basket size (1, 2, 3, 4, 5+) and
    same quarter. Regions sell different SKUs, large baskets hold many, and growing or seasonal SKUs
    sell in the same months, so a shop-wide baseline would link SKUs that merely share a region, a
    basket size or a season. Expected count = sum over strata of
    count_s(a) count_s(b) / n_s; the p-value uses a Poisson tail with that mean (its variance is at
    least the hypergeometric one, so the test is slightly conservative)."""
    n = X.shape[0]
    co = (X.T @ X).toarray().astype(float)
    cnt = np.diag(co).copy()
    np.fill_diagonal(co, 0)
    size = np.minimum(np.asarray(X.sum(axis=1)).ravel(), 5)
    quarter = np.zeros(n, dtype=int) if quarter is None else quarter
    expected = np.zeros_like(co)
    stratum = pd.Series(list(zip(region, size, quarter)))
    for _, rows in stratum.groupby(stratum).groups.items():
        rows = np.asarray(rows)
        if size[rows[0]] >= 2:
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

def place(units: list[np.ndarray], load: np.ndarray, k: int, sequential: bool = False) -> np.ndarray:
    """Put units (groups of SKUs) into k FCs, in the given order. Every FC is limited to an equal share
    of the SKUs and of the workload (order lines), each with SLACK. Balanced placement takes the FC
    left with the most room; sequential placement fills FC 1 first, then FC 2, and so on. A unit that
    fits nowhere whole is placed SKU by SKU, heaviest first."""
    n = len(load)
    sku_cap, load_cap = np.ceil(n / k * (1 + SLACK)), load.sum() / k * (1 + SLACK)
    fc = np.full(n, -1)
    skus, work = np.zeros(k), np.zeros(k)

    def room(f: int, u: np.ndarray) -> float:
        return min((sku_cap - skus[f] - len(u)) / sku_cap, (load_cap - work[f] - load[u].sum()) / load_cap)

    def put(f: int, u: np.ndarray) -> None:
        fc[u] = f
        skus[f] += len(u)
        work[f] += load[u].sum()

    queue = list(units)
    while queue:
        u = queue.pop(0)
        fits = [f for f in range(k) if room(f, u) >= 0]
        if fits:
            put(fits[0] if sequential else max(fits, key=lambda f: room(f, u)), u)
        elif len(u) > 1:
            queue = [np.array([s_]) for s_ in sorted(u, key=lambda s_: -load[s_])] + queue
        else:  # a single SKU too heavy for any FC's remaining room: least overflow
            put(max(range(k), key=lambda f: room(f, u)), u)
    return fc


def assignments(train: dict, categories: np.ndarray, labels: np.ndarray, truth: np.ndarray) -> dict[str, dict[int, np.ndarray]]:
    """SKU-to-FC assignment per rule and number of FCs, from the training set only (truth only for
    the reference rule). Workload is each SKU's number of order lines in the training set."""
    load = np.asarray(train["X"].sum(axis=0)).ravel().astype(float)
    n_sku = len(load)
    pop = popularity_order(train["X"])
    singles = [np.array([s_]) for s_ in pop]
    rng = np.random.default_rng(SEED)
    shuffled = [np.array([s_]) for s_ in rng.permutation(n_sku)]

    def whole_then_singles(groups: np.ndarray, keep: np.ndarray | None = None) -> list[np.ndarray]:
        big = [np.nonzero(groups == g)[0] for g in np.unique(groups) if (groups == g).sum() > 1 and (keep is None or keep[groups == g].all())]
        big.sort(key=lambda u: -load[u].sum())
        placed = np.zeros(n_sku, dtype=bool)
        for u in big:
            placed[u] = True
        return big + [u for u in singles if not placed[u[0]]]

    cats = sorted((np.nonzero(categories == c)[0] for c in np.unique(categories)), key=lambda u: -load[u].sum())
    bundled = np.where(truth > 0, truth, -np.arange(1, n_sku + 1))
    out: dict[str, dict[int, np.ndarray]] = {r: {} for r in ("random", "category", "popularity", "best_sellers", "communities", "hidden_bundles")}
    for k in FCS:
        out["random"][k] = place(shuffled, load, k)
        out["category"][k] = place(cats, load, k)
        out["popularity"][k] = place(singles, load, k)
        out["best_sellers"][k] = place(singles, load, k, sequential=True)
        out["communities"][k] = place(whole_then_singles(labels), load, k)
        out["hidden_bundles"][k] = place(whole_then_singles(bundled), load, k)
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
    lines_per_sku = np.asarray(X.sum(axis=0)).ravel()
    load_share = [float(lines_per_sku[fc == f].sum() / lines_per_sku.sum()) for f in range(k)]
    return {"fcs": k, "orders": int(X.shape[0]), "multi": int(multi.sum()), "split": int(split.sum()), "load_share": load_share,
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
    st = pair_stats(train["X"], train["region"], train["quarter"])
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

    # Disclosure: the quarter was added to the chance baseline after the check against the hidden
    # bundles exposed timing links. The same choice without it, for comparison.
    st_nq = pair_stats(train["X"], train["region"])
    nq = []
    for q in FDR_LEVELS:
        acc_nq = bh_accept(st_nq["p"], q) & (st_nq["lift"] > 1)
        lab_nq = communities(acc_nq, np.log(np.maximum(st_nq["lift"], 1e-12)), 1.0)
        v = price(assignments(train, categories, lab_nq, truth)["communities"], val, weight, mult, region_ids)
        nq.append({"fdr": q, "val_cost": float(np.mean([x["cost"] for x in v])), "truth": pair_check(lab_nq, truth)})
    without_quarter = min(nq, key=lambda g: g["val_cost"])

    # Baselines for the check against the hidden bundles.
    n_edges = best["edges"]
    iu = np.triu_indices(len(skus), 1)
    top = np.argsort(-st["count"][iu])[:n_edges]
    by_count = np.zeros_like(accepted)
    by_count[iu[0][top], iu[1][top]] = True
    by_count = by_count | by_count.T
    # The raw-count baseline gets the same treatment as ours: its resolution chosen on the validation set.
    count_grid = []
    for res in RESOLUTIONS:
        cl = communities(by_count, st["count"], res)
        v = price(assignments(train, categories, cl, truth)["communities"], val, weight, mult, region_ids)
        count_grid.append((float(np.mean([x["cost"] for x in v])), res, cl))
    _, count_res, count_labels = min(count_grid, key=lambda t: t[0])
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
        "chosen": {"fdr": best["fdr"], "resolution": best["resolution"], "count_baseline_resolution": count_res},
        "quarter_note": {"added_after_truth_check": True, "with_quarter": {"val_cost": best["val_cost"], "truth": best["truth"]},
                         "without_quarter": without_quarter},
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
