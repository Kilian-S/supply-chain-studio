"""Model J: SKU segmentation. Builds every result the model page shows and writes it as JSON.

Usage: uv run python -m scgame.ml.sku_segmentation <world dir> <output dir>

Outputs:
  card.json              summary for the models home page
  overview.json          data, transformations, features, baselines, selection, profiles, policies
  skus.json              one record per SKU (features, classes, weekly series) for the explorer
  variants/m<mask>.json  per feature-group subset: embeddings, clusterings, metrics, neighbours
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from . import policy_sim as ps
from .sku_features import FEATURES, _seasonal_strength, GROUP_LABELS, GROUPS, TRAIN_END, build_features, transform, weekly_matrix

K_RANGE = list(range(2, 11))
MODELS = ["kmeans", "gmm"]
BOOTSTRAPS = 8
SEED = 11
TYPE_ORDER = ["steady", "seasonal", "trending", "declining", "intermittent", "lumpy"]


# =================================================================================================
# Baselines
# =================================================================================================

def abc_xyz(features: pd.DataFrame, weekly: np.ndarray, on_sale: np.ndarray) -> pd.DataFrame:
    rev = features["revenue"].to_numpy()
    order = np.argsort(-rev)
    cum = np.cumsum(rev[order]) / rev.sum()
    abc = np.empty(len(rev), dtype=object)
    abc[order] = np.where(cum <= 0.80, "A", np.where(cum <= 0.95, "B", "C"))
    cv = []
    for i in range(len(rev)):
        d = weekly[i][on_sale[i]]
        cv.append(d.std() / d.mean() if len(d) and d.mean() > 0 else np.inf)
    cv = np.array(cv)
    xyz = np.where(cv < 0.5, "X", np.where(cv < 1.0, "Y", "Z"))
    adi, cv2 = features["adi"].to_numpy(), features["cv2"].to_numpy()
    sbc = np.where(adi < 1.32, np.where(cv2 < 0.49, "smooth", "erratic"), np.where(cv2 < 0.49, "intermittent", "lumpy"))
    return pd.DataFrame({"abc": abc, "xyz": xyz, "abc_xyz": [a + x for a, x in zip(abc, xyz)], "sbc": sbc, "cv_weekly": cv})


# =================================================================================================
# Clustering for one feature-group subset
# =================================================================================================

def _fit(model: str, k: int, X: np.ndarray, seed: int):
    if model == "kmeans":
        m = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        return m.labels_, None, float(m.inertia_), None
    m = GaussianMixture(n_components=k, covariance_type="diag", n_init=3, reg_covar=1e-3, random_state=seed).fit(X)
    prob = m.predict_proba(X)
    return prob.argmax(axis=1), prob.max(axis=1), None, float(m.bic(X))


def _relabel(labels: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Stable cluster numbering: order clusters by their mean first principal coordinate, so that
    colours do not jump when switching k or model."""
    pc = PCA(n_components=1, random_state=0).fit_transform(X)[:, 0]
    order = sorted(np.unique(labels), key=lambda c: pc[labels == c].mean())
    remap = {c: i for i, c in enumerate(order)}
    return np.array([remap[c] for c in labels])


def run_variant(mask: int, features: pd.DataFrame, truth: np.ndarray, full_z: np.ndarray) -> dict:
    import umap  # imported in the worker; numba compiles once per process

    groups = [g for i, g in enumerate(GROUPS) if mask >> i & 1]
    X, cols = transform(features, groups)
    rng = np.random.default_rng(SEED + mask)
    emb = {}
    if X.shape[1] >= 3:
        emb["umap3d"] = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.15, random_state=SEED).fit_transform(X)
        pca = PCA(n_components=3, random_state=0).fit(X)
        emb["pca3d"] = pca.transform(X)
        explained = pca.explained_variance_ratio_.tolist()
    else:
        pad = np.zeros((len(X), 3))
        pad[:, : X.shape[1]] = X
        emb["umap3d"] = emb["pca3d"] = pad
        explained = [1.0 / X.shape[1]] * X.shape[1] + [0.0] * (3 - X.shape[1])
    emb["umap2d"] = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.15, random_state=SEED).fit_transform(X) if X.shape[1] >= 2 \
        else np.column_stack([X[:, 0], np.zeros(len(X))])

    clusterings = {}
    for model in MODELS:
        for k in K_RANGE:
            labels, prob, inertia, bic = _fit(model, k, X, SEED)
            labels = _relabel(labels, X)
            stab = []
            for _ in range(BOOTSTRAPS):
                idx = rng.choice(len(X), size=int(0.8 * len(X)), replace=False)
                lb, *_ = _fit(model, k, X[idx], int(rng.integers(1 << 30)))
                stab.append(adjusted_rand_score(labels[idx], lb))
            profile = [full_z[labels == c].mean(axis=0).round(3).tolist() for c in range(k)]
            clusterings[f"{model}-{k}"] = {
                "labels": labels.tolist(),
                "prob": None if prob is None else np.round(prob, 3).tolist(),
                "metrics": {
                    "silhouette": float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else 0.0,
                    "inertia": inertia, "bic": bic,
                    "stability": float(np.mean(stab)), "stability_sd": float(np.std(stab)),
                    "ari_truth": float(adjusted_rand_score(truth, labels)),
                    "nmi_truth": float(normalized_mutual_info_score(truth, labels)),
                },
                "sizes": np.bincount(labels, minlength=k).tolist(),
                "profile": profile,
            }
    nn = NearestNeighbors(n_neighbors=9).fit(X)
    dist, ind = nn.kneighbors(X)
    return {
        "mask": mask, "groups": groups, "features": cols, "explained_variance": explained,
        "embeddings": {k: np.round(v, 4).tolist() for k, v in emb.items()},
        "clusterings": clusterings,
        "neighbours": ind[:, 1:].tolist(), "neighbour_dist": np.round(dist[:, 1:], 3).tolist(),
    }


# =================================================================================================
# Naming and policies
# =================================================================================================

# Pattern words, the same six as the demand types on the page. Each SKU gets one from fixed cut-offs
# on its own features (checked in this order); the ground truth is never used.
BULK_UNITS_PER_LINE = 5.0  # units per order line
INTERMITTENT_ZERO_SHARE = 0.15  # share of weeks on sale without a sale
GROWING_PER_YEAR, FADING_PER_YEAR = 0.30, -0.20  # growth per year
SEASONAL_SHUFFLES = 19  # seasonal only if stronger than every shuffled copy of its own sales (p < 0.05)
MAJORITY = 0.6  # a cluster takes a pattern word only if this share of its SKUs carry it


def seasonal_test(series: np.ndarray, seed: int) -> bool:
    """Is the yearly pattern stronger than in every random reshuffle of the same weekly sales?"""
    real = _seasonal_strength(series)
    if real == 0.0:
        return False
    rng = np.random.default_rng(seed)
    return all(_seasonal_strength(rng.permutation(series)) < real for _ in range(SEASONAL_SHUFFLES))


def sku_pattern(f: pd.Series, seasonal: bool) -> str:
    growth = np.exp(f["trend"]) - 1
    if f["qty_per_line"] >= BULK_UNITS_PER_LINE:
        return "Bulk orders"
    # Life cycle before intermittency: a SKU ramping up from zero has many empty weeks early on.
    if growth >= GROWING_PER_YEAR:
        return "Growing"
    if growth <= FADING_PER_YEAR:
        return "Fading"
    if f["zero_share"] >= INTERMITTENT_ZERO_SHARE:
        return "Intermittent"
    if seasonal:
        return "Seasonal"
    return "Steady"


# Traits for names, each tied to the feature group it comes from: (group, features, word if high, word if low).
TRAITS = [
    ("volume", ("mean_weekly_units",), "fast", "slow"),
    ("variability", ("cv2",), "erratic", "stable"),
    ("intermittency", ("zero_share",), "sporadic", "regular"),
    ("physical", ("volume_l", "weight_kg"), "bulky", "small"),
    ("economic", ("price",), "premium", "budget"),
    ("economic", ("margin",), "high margin", "low margin"),
    ("basket", ("solo_share",), "bought alone", "bought with others"),
]


def names_for(patterns: np.ndarray, ranks: pd.DataFrame, labels: np.ndarray, k: int, groups: list[str]) -> list[str]:
    """A cluster where most SKUs share a pattern word takes that word, plus the trait that sets it apart
    from clusters with the same word. Any other cluster is named by the two traits that set it apart
    from the average SKU. Traits come from the feature groups the clustering used."""
    # Traits from the groups the clustering used come first; the others only when those run out.
    traits = [t for t in TRAITS if t[0] in groups] + [t for t in TRAITS if t[0] not in groups]
    n_own = max(1, sum(1 for t in TRAITS if t[0] in groups))
    level = np.array([[ranks.loc[labels == c, list(cols)].median().mean() for _, cols, _, _ in traits] for c in range(k)])

    def ranked_words(row: np.ndarray) -> list[str]:
        own = [i for i in np.argsort(-np.abs(row[:n_own]))]
        rest = [n_own + i for i in np.argsort(-np.abs(row[n_own:]))]
        return [traits[i][2] if row[i] > 0 else traits[i][3] for i in own + rest]

    base: list[str | None] = []
    for c in range(k):
        words, counts = np.unique(patterns[labels == c], return_counts=True)
        base.append(str(words[counts.argmax()]) if counts.max() >= MAJORITY * counts.sum() else None)
    names = [""] * k
    used: set[str] = set()

    def first_free(options) -> str:
        name = next(o for o in options if o not in used)
        used.add(name)
        return name

    for word in sorted({b for b in base if b}):
        siblings = [c for c in range(k) if base[c] == word]
        if len(siblings) == 1:
            names[siblings[0]] = first_free([word])
            continue
        rel = level[siblings] - level[siblings].mean(axis=0)
        for row, c in sorted(zip(rel, siblings), key=lambda rc: -np.abs(rc[0]).max()):
            w = ranked_words(row)
            names[c] = first_free([f"{word}, {a}" for a in w] + [f"{word}, {a}, {b}" for a in w for b in w if a != b])
    for c in (c for c in range(k) if base[c] is None):
        w = ranked_words(level[c] - 0.5)
        names[c] = first_free([f"{a.capitalize()}, {b}" for a in w for b in w if a != b] + [f"{a.capitalize()}, {b}, {d}" for a in w for b in w for d in w if len({a, b, d}) == 3])
    return names


STAT_KEYS = ["mean_weekly_units", "zero_share", "seasonality", "trend", "price", "volume_l", "weight_kg", "margin", "solo_share", "qty_per_line"]


def group_stats(features: pd.DataFrame, labels: np.ndarray, k: int) -> list[dict[str, float]]:
    """Median of plain features per group, in everyday units."""
    out = []
    for c in range(k):
        g = features[labels == c]
        row = {key: round(float(g[key].median()), 4) for key in STAT_KEYS}
        row["sells_in_weeks"] = round(1 - row.pop("zero_share"), 4)
        row["growth_per_year"] = round(float(np.exp(row.pop("trend")) - 1), 4)
        out.append(row)
    return out


def purity(truth: np.ndarray, labels) -> float:
    """Share of products that sit in a group whose most common true type is their own."""
    labels = np.asarray(labels)
    hits = 0
    for c in np.unique(labels):
        _, counts = np.unique(truth[labels == c], return_counts=True)
        hits += counts.max()
    return float(hits / len(truth))


def priced(train: dict, table: dict, groups: np.ndarray) -> tuple[dict, dict]:
    """Choose a (mode, service) per group on the training year, then price that choice on `table`."""
    best = ps.best_per_group(train, groups)
    return best, ps.evaluate(table, [best[g] for g in groups])


def policy_results(train: dict, test: dict, groupings: dict[str, np.ndarray]) -> dict:
    """Policies chosen per group on the training year, priced once on the test year."""
    out = {}
    n = len(next(iter(test.values()))["total"])
    for name, groups in groupings.items():
        best, res = priced(train, test, groups)
        res["rules"] = len(best)
        res["choices"] = {str(g): {"mode": m, "service": s, "skus": int((groups == g).sum())} for g, (m, s) in best.items()}
        out[name] = res
    central = ps.evaluate(test, [("central", 0.95)] * n)
    central["rules"] = 1
    out["all_central"] = central
    return out


# =================================================================================================
# Main
# =================================================================================================

def main(world_dir: str, out_dir: str) -> None:
    t0 = time.time()
    world = Path(world_dir)
    out = Path(out_dir)
    (out / "variants").mkdir(parents=True, exist_ok=True)
    skus = pd.read_parquet(world / "skus.parquet")
    regions = pd.read_parquet(world / "regions.parquet")
    lines = pd.read_parquet(world / "order_lines.parquet")
    orders = pd.read_parquet(world / "orders.parquet")
    meta = json.loads((world / "world.json").read_text())

    features, weekly = build_features(lines, orders, skus)
    weekly_train, on_sale = weekly_matrix(lines, skus, TRAIN_END)
    weekly_all, _ = weekly_matrix(lines, skus, meta["days"])
    truth_names = skus["true_type"].to_numpy()
    truth = np.array([TYPE_ORDER.index(t) for t in truth_names])
    base = abc_xyz(features, weekly_train, on_sale)
    print(f"features ready ({time.time() - t0:.0f}s)")

    # Unweighted z-scores of all features: used for cluster profiles and naming.
    full_X, keys = transform(features, GROUPS)
    group_size = {g: sum(1 for f in FEATURES if f.group == g) for g in GROUPS}
    full_z = full_X * np.array([np.sqrt(group_size[f.group]) for f in FEATURES])

    masks = list(range(1, 1 << len(GROUPS)))
    variants = Parallel(n_jobs=-1, verbose=0)(delayed(run_variant)(m, features, truth, full_z) for m in masks)
    full_mask = masks[-1]
    # Seasonal means a yearly pattern of its own: sales are divided by the shop's weekly index (all
    # SKUs together) before the test, so the shop-wide calendar alone does not count.
    shop = weekly_train.sum(axis=0)
    own = weekly_train / np.maximum(shop / shop.mean(), 1e-9)
    seasonal = Parallel(n_jobs=-1)(delayed(seasonal_test)(own[i][on_sale[i]], SEED + i) for i in range(len(skus)))
    patterns = np.array([sku_pattern(row, seasonal[i]) for i, (_, row) in enumerate(features.iterrows())])
    ranks = features[sorted({col for _, cols, _, _ in TRAITS for col in cols})].rank(pct=True)
    for v in variants:
        for c in v["clusterings"].values():
            c["names"] = names_for(patterns, ranks, np.array(c["labels"]), len(c["sizes"]), v["groups"])
            lab = np.array(c["labels"])
            c["metrics"]["purity"] = purity(truth, lab)
            c["stats"] = group_stats(features, lab, len(c["sizes"]))
    full = next(v for v in variants if v["mask"] == full_mask)
    leaderboard = []
    for v in variants:
        key, c = max(v["clusterings"].items(), key=lambda kv: kv[1]["metrics"]["ari_truth"])
        leaderboard.append({"mask": v["mask"], "groups": v["groups"], "best": key, **{m: c["metrics"][m] for m in ("ari_truth", "silhouette", "stability", "purity")}})
    leaderboard.sort(key=lambda r: -r["ari_truth"])
    print(f"variants ready ({time.time() - t0:.0f}s)")

    # Train, validate, test. Policies are chosen per group on the training set (year 3, order-up-to
    # levels from year 2). Every setting (feature groups, method, k) is priced on the validation set
    # (year 4) and the cheapest is chosen. Only that choice and the baselines are priced on the test
    # set (year 5), once. The ground truth is never used to choose.
    table3 = ps.cost_table(lines, skus, regions, 3)
    table4 = ps.cost_table(lines, skus, regions, 4)
    table5 = ps.cost_table(lines, skus, regions, 5)
    _, abc_xyz_val = priced(table3, table4, base["abc_xyz"].to_numpy())
    for v in variants:
        for c in v["clusterings"].values():
            best, res = priced(table3, table4, np.array(c["labels"]))
            c["metrics"]["val_cost"] = res["total"]
            c["metrics"]["val_vs_abc_xyz"] = res["total"] / abc_xyz_val["total"] - 1
            c["policy"] = [{"mode": best[g][0], "service": best[g][1]} for g in range(len(c["sizes"]))]
        (out / "variants" / f"m{v['mask']}.json").write_text(json.dumps(v, separators=(",", ":")))
    for row in leaderboard:
        v = next(x for x in variants if x["mask"] == row["mask"])
        row["best_policy"], c = min(v["clusterings"].items(), key=lambda kv: kv[1]["metrics"]["val_cost"])
        row["val_vs_abc_xyz"] = c["metrics"]["val_vs_abc_xyz"]
    chosen_v, chosen_key = min(((v, key) for v in variants for key in v["clusterings"]),
                               key=lambda vk: vk[0]["clusterings"][vk[1]]["metrics"]["val_cost"])
    model_c, k_c = chosen_key.split("-")
    best_k = int(k_c)
    default = {"mask": chosen_v["mask"], "model": model_c, "k": best_k, "embedding": "umap3d"}
    chosen = chosen_v["clusterings"][chosen_key]

    groupings = {
        "uniform": np.zeros(len(skus), dtype=int),
        "abc": base["abc"].to_numpy(),
        "abc_xyz": base["abc_xyz"].to_numpy(),
        "sbc": base["sbc"].to_numpy(),
        "true_type": truth_names,
        "per_sku": np.arange(len(skus)),
        f"cluster-{chosen_key}": np.array(chosen["labels"]),
    }
    policies = policy_results(table3, table5, groupings)
    # Trade-off between stocking cost (validation set) and match with the demand types, every setting.
    tradeoff = {
        "settings": [[round(c["metrics"]["val_cost"]), round(c["metrics"]["purity"], 4), round(c["metrics"]["ari_truth"], 4), v["mask"], key]
                     for v in variants for key, c in v["clusterings"].items()],
        "chosen": [default["mask"], chosen_key],
        "baselines": {name: {"val_cost": round(priced(table3, table4, groupings[name])[1]["total"]), "purity": purity(truth, groupings[name]),
                             "ari": float(adjusted_rand_score(truth, groupings[name]))}
                      for name in ("uniform", "abc", "abc_xyz", "sbc", "true_type")},
    }
    print(f"policies ready ({time.time() - t0:.0f}s): chose {default}")

    # Data section: sample rows, size, transformation examples.
    sample = (lines[lines["day"] <= TRAIN_END].sample(12, random_state=3)
              .merge(skus[["sku_id", "name", "category", "price"]], on="sku_id")
              .merge(orders[["order_id", "customer_id"]], on="order_id")
              .sort_values(["day", "order_id"]))
    examples = []
    for t in TYPE_ORDER:
        # A typical SKU of each type: the median by demand rate among those on sale from day 1.
        pool = skus[(skus["true_type"] == t) & (skus["launch_day"] == 1)].sort_values("base_rate")
        pool = pool if len(pool) else skus[skus["true_type"] == t].sort_values("base_rate")
        sid = int(pool["sku_id"].iloc[len(pool) // 2])
        daily = lines[(lines["sku_id"] == sid) & (lines["day"] <= TRAIN_END)].groupby("day")["quantity"].sum().reindex(range(1, TRAIN_END + 1), fill_value=0)
        i = int(np.nonzero(skus["sku_id"].to_numpy() == sid)[0][0])
        examples.append({"sku_id": sid, "name": skus.loc[i, "name"], "true_type": t,
                         "daily": daily.astype(int).tolist(), "weekly": weekly_train[i].astype(int).tolist()})

    def hist(x: np.ndarray, bins: int = 24) -> dict:
        c, e = np.histogram(x[np.isfinite(x)], bins=bins)
        return {"counts": c.tolist(), "edges": np.round(e, 4).tolist()}

    feature_info = []
    for f, col in zip(FEATURES, full_z.T):
        raw = features[f.key].to_numpy(dtype=float)
        feature_info.append({
            "key": f.key, "group": f.group, "label": f.label, "formula": f.formula, "description": f.description,
            "log": f.log, "raw_hist": hist(raw), "z_hist": hist(col),
            "raw_summary": {"min": float(np.min(raw)), "median": float(np.median(raw)), "max": float(np.max(raw))},
        })
    corr = np.corrcoef(full_z.T)

    def crosstab(a, b, rows, cols) -> list[list[int]]:
        return [[int(((a == r) & (b == c)).sum()) for c in cols] for r in rows]

    overview = {
        "model": {"id": "sku-segmentation", "title": "Product groups"},
        "world": {"version": meta["version"], "counts": meta["counts"], "checks": meta["checks"], "days": meta["days"],
                  "train_days": TRAIN_END, "categories": sorted(skus["category"].unique().tolist())},
        "data_sample": sample[["day", "order_id", "customer_id", "region_id", "sku_id", "name", "category", "quantity", "price"]].to_dict("records"),
        "examples": examples,
        "groups": [{"key": g, "label": GROUP_LABELS[g], "features": [f.key for f in FEATURES if f.group == g]} for g in GROUPS],
        "features": feature_info,
        "correlation": np.round(corr, 3).tolist(),
        "truth": {"types": TYPE_ORDER, "counts": [int((truth == i).sum()) for i in range(len(TYPE_ORDER))]},
        "baselines": {
            "abc_xyz_grid": {"rows": ["A", "B", "C"], "cols": ["X", "Y", "Z"],
                             "counts": crosstab(base["abc"].to_numpy(), base["xyz"].to_numpy(), ["A", "B", "C"], ["X", "Y", "Z"])},
            "sbc": {"classes": ["smooth", "erratic", "intermittent", "lumpy"],
                    "counts": [int((base["sbc"] == c).sum()) for c in ["smooth", "erratic", "intermittent", "lumpy"]]},
            "sbc_points": {"adi": features["adi"].round(3).tolist(), "cv2": features["cv2"].round(3).tolist()},
            "vs_truth": {
                name: {"ari": float(adjusted_rand_score(truth, base[col])), "nmi": float(normalized_mutual_info_score(truth, base[col])),
                       "purity": purity(truth, base[col].to_numpy()), "groups": int(base[col].nunique()),
                       "crosstab": crosstab(base[col].to_numpy(), truth_names, sorted(base[col].unique()), TYPE_ORDER),
                       "rows": sorted(base[col].unique().tolist())}
                for name, col in [("abc", "abc"), ("abc_xyz", "abc_xyz"), ("sbc", "sbc")]
            },
        },
        "selection": {f"{m}-{k}": full["clusterings"][f"{m}-{k}"]["metrics"] for m in MODELS for k in K_RANGE},
        "default": default,
        "policies": policies,
        "policy_settings": {"lead_days": ps.LEAD, "review_days": ps.REVIEW, "order_cost": ps.ORDER_COST,
                            "slot_cost": ps.SLOT_COST, "slow_delivery_loss": ps.SLOW_DELIVERY_LOSS,
                            "service_levels": ps.SERVICE_LEVELS, "central_region": ps.CENTRAL,
                            "train_year": 3, "validation_year": 4, "test_year": 5},
        "k_range": K_RANGE, "models": MODELS,
        "subset_leaderboard": leaderboard,
        "tradeoff": tradeoff,
        "built_seconds": round(time.time() - t0, 1),
    }
    (out / "overview.json").write_text(json.dumps(overview, separators=(",", ":"), default=str))

    sku_records = []
    for i, s in enumerate(skus.itertuples()):
        sku_records.append({
            "id": int(s.sku_id), "name": s.name, "category": s.category, "true_type": s.true_type, "bulky": bool(s.bulky),
            "price": float(s.price), "margin": float(s.margin), "launch_day": int(s.launch_day),
            "features": {f.key: round(float(features.loc[i, f.key]), 4) for f in FEATURES},
            "abc": base.loc[i, "abc"], "xyz": base.loc[i, "xyz"], "sbc": base.loc[i, "sbc"],
            "pattern": str(patterns[i]),
            "weekly": weekly_all[i].astype(int).tolist(),
        })
    (out / "skus.json").write_text(json.dumps(sku_records, separators=(",", ":")))

    pol = policies[f"cluster-{chosen_key}"]
    card = {
        "id": "sku-segmentation", "title": "Product clustering", "status": "ready",
        "question": "Which SKUs sell alike, so each cluster can get its own stocking policy?",
        "method": "Clustering",
        "headline": f"{best_k} clusters, stocking cost {abs(pol['total'] / policies['uniform']['total'] - 1):.0%} lower than one rule for all",
    }
    (out / "card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
