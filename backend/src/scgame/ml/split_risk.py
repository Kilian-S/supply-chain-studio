"""Model H: split orders. Which orders will ship in more than one parcel under a planned network?

Usage: uv run python -m scgame.ml.split_risk <world dir> <models dir> <output dir>

The design agreed with Kilian is in design/split-risk/PROPOSAL.md (section 6a); the handover in
design/split-risk/HANDOVER.md describes the run.

  1. Networks. 60 training, 20 validation and 20 test networks drawn at random (sites, listing share,
     listing rule, service level); the test site sets appear in no other set. Three more site sets are
     kept for the decision in the business check.
  2. Labels. `split_sim` runs the orders of a year through each network. Training years 2 and 3,
     validation year 4, test year 5 (year 1 only sets the first stock levels and listings).
  3. Features. What a planner knows at the end of the year before (basket, region, listing, stock risk,
     model J clusters, model K communities). Never the name of the network's rule or site set.
  4. Methods. The plain rule (two probabilities), the replay simulation (last year through the planned
     network, in-stock rates per FC, SKU and month, SKUs independent), logistic regression, LightGBM.
  5. Choice on validation only. Hyperparameters and calibration map (none, Platt, isotonic) by log loss
     with cross-fitting over two halves of the validation networks; the chosen map is then fitted on
     all of them. The test set is scored once.
  6. Business check. Split rate of candidate networks for year 5 estimated at the end of year 4, against
     the simulated year 5, and the regret of each method's choice of listing rule.

Outputs (no card.json until the page exists)
  overview.json  facts, networks, settings, validation grid, test scores, calibration, SHAP, business check
  skus.json      per test network and SKU, predicted and simulated split orders
  examples.json  example orders for the page
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve

from . import product_affinity as pa
from . import split_sim as ss

N_TRAIN, N_VAL, N_TEST = 60, 20, 20
TRAIN_YEARS, VAL_YEAR, TEST_YEAR = (2, 3), 4, 5
ROWS_PER_NETWORK = 20_000  # training orders sampled per network and year
SITES = [2, 3, 4, 5, 6, 7, 8]  # regional FC sites (Calopeia, region 1, is the central FC)
DECISION_SITES = [(2,), (2, 3, 6), (2, 3, 4, 6, 7, 8)]
DECISION_SHARE, DECISION_SERVICE = 0.4, 0.95
LR_C = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # inverse L2 penalty
GBM_GRID = {"num_leaves": [15, 63], "min_data_in_leaf": [100, 1000], "sampling": [0.8, 1.0]}
GBM_TREES = [250, 500, 1000]
GBM_RATE = 0.05
MAPS = ["none", "platt", "isotonic"]
NOISE_SEEDS = [1, 2, 3, 4, 5]
BOOTSTRAP = 2000
SHAP_ROWS = 50_000
EPS = 1e-6
SEED = 23
METHODS = ["rule", "replay", "logistic", "gbm"]
METHOD_LABELS = {"rule": "Plain rule", "replay": "Replay simulation", "logistic": "Logistic regression", "gbm": "Gradient boosting"}

# name, group, transform for logistic regression (log: log1p; num: as is; bin: 0/1; cat: one-hot)
FEATURES = [
    ("n_skus", "Basket", "log"), ("units", "Basket", "log"), ("max_qty", "Basket", "log"), ("bulk", "Basket", "bin"),
    ("kg", "Basket", "log"), ("litres", "Basket", "log"), ("n_categories", "Basket", "num"), ("month", "Basket", "cat"),
    ("region", "Region", "cat"), ("home_central", "Region", "bin"), ("home_mult", "Region", "num"),
    ("n_unlisted", "Listing", "num"), ("share_unlisted", "Listing", "num"), ("some_not_all", "Listing", "bin"),
    ("n_listed_elsewhere", "Listing", "num"), ("n_open", "Listing", "num"), ("listing_share", "Listing", "num"),
    ("service", "Listing", "num"),
    ("cover_min", "Stock risk", "log"), ("cover_mean", "Stock risk", "log"), ("cv2_max", "Stock risk", "log"),
    ("adi_max", "Stock risk", "log"), ("growth_max", "Stock risk", "num"), ("demand_min", "Stock risk", "log"),
    ("cover_min_central", "Stock risk", "log"), ("growth_max_central", "Stock risk", "num"),
    *[(f"cluster_{c}", "SKU clusters", "num") for c in range(9)], ("share_distributed", "SKU clusters", "num"),
    ("n_communities", "Communities", "num"), ("community_cut", "Communities", "bin"), ("max_lift", "Communities", "log"),
    ("share_alone", "Communities", "num"),
]
NAMES = [f[0] for f in FEATURES]
CATS = [f[0] for f in FEATURES if f[2] == "cat"]


# =================================================================================================
# Networks
# =================================================================================================

def draw_networks(rng: np.random.Generator) -> dict[str, list[ss.Network]]:
    """Training, validation and test networks on disjoint site sets, settings balanced within each set."""
    sets = [c for n in range(1, 7) for c in itertools.combinations(SITES, n) if c not in DECISION_SITES]
    order = rng.permutation(len(sets))
    groups = {"test": order[:N_TEST], "validation": order[N_TEST:N_TEST + N_VAL], "training": order[N_TEST + N_VAL:N_TEST + N_VAL + N_TRAIN]}
    out, seed = {}, 0
    for name in ("training", "validation", "test"):
        idx = groups[name]
        rules = rng.permutation(np.resize(ss.RULES, len(idx)))
        shares = rng.permutation(np.resize(ss.SHARES, len(idx)))
        services = rng.permutation(np.resize(ss.SERVICES, len(idx)))
        nets = []
        for i, j in enumerate(idx):
            seed += 1
            nets.append(ss.Network(sets[j], float(shares[i]), str(rules[i]), float(services[i]), seed))
        out[name] = nets
    out["decision"] = [ss.Network(s, DECISION_SHARE, r, DECISION_SERVICE, 1000 + i)
                       for i, (s, r) in enumerate(itertools.product(DECISION_SITES, ss.RULES))]
    return out


# =================================================================================================
# Features
# =================================================================================================

def sku_stats(w: ss.World, year: int) -> dict[str, np.ndarray]:
    """Per SKU from one year of network-wide daily units: CV squared of non-zero days, ADI (days per day
    with a sale) and growth within the year (log of second-half over first-half units, minus the shop's)."""
    sl = w.year_lines(year)
    d = w.day[sl] - (year - 1) * ss.YEAR - 1
    daily = np.zeros((w.n_sku, ss.YEAR))
    np.add.at(daily, (w.sku[sl], d), w.qty[sl])
    days = (daily > 0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        nz_mean = daily.sum(axis=1) / days
        nz_var = np.array([daily[i][daily[i] > 0].var() if days[i] else np.nan for i in range(w.n_sku)])
        cv2 = nz_var / nz_mean**2
        adi = ss.YEAR / days
    h1, h2 = daily[:, :182].sum(axis=1), daily[:, 182:].sum(axis=1)
    growth = np.log((h2 + 1) / (h1 + 1)) - np.log(h2.sum() / h1.sum())
    return {"cv2": np.where(days > 0, cv2, np.nan), "adi": np.where(days > 0, adi, np.nan), "growth": growth}


def max_pair_lift(w: ss.World, oi: np.ndarray, s: np.ndarray, starts: np.ndarray, n_skus: np.ndarray, lift: np.ndarray) -> np.ndarray:
    """Largest lift between two SKUs of each order (0 for one-SKU orders)."""
    out = np.zeros(len(starts))
    for n in np.unique(n_skus):
        if n < 2:
            continue
        rows = np.nonzero(n_skus == n)[0]
        m = s[starts[rows][:, None] + np.arange(n)[None, :]]
        best = np.zeros(len(rows))
        for a, b in itertools.combinations(range(n), 2):
            best = np.maximum(best, lift[m[:, a], m[:, b]])
        out[rows] = best
    return out


def features(w: ss.World, p: ss.Plan, order_year: int, lift: np.ndarray, stats: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per order of `order_year` under plan p (planned for p.year). Reads only the order's own lines,
    the plan (built from years before p.year), SKU stats of p.year - 1 and pair lifts of the training set."""
    sl = w.year_lines(order_year)
    o, s, q, r = w.order[sl], w.sku[sl], w.qty[sl], w.region[sl]
    oi = o - o.min()
    starts = np.r_[0, np.nonzero(np.diff(oi))[0] + 1]
    n = len(starts)
    doy = w.day[sl] - (order_year - 1) * ss.YEAR - 1
    month = ss.month_of_doy(doy)

    def total(v):
        return np.add.reduceat(np.asarray(v, dtype=float), starts)

    def vmax(v, mask=None):
        v = np.where(mask, v, -np.inf) if mask is not None else v
        out = np.maximum.reduceat(np.asarray(v, dtype=float), starts)
        return np.where(np.isfinite(out), out, np.nan)

    def vmin(v, mask=None):
        v = np.where(mask, v, np.inf) if mask is not None else v
        out = np.minimum.reduceat(np.asarray(v, dtype=float), starts)
        return np.where(np.isfinite(out), out, np.nan)

    n_skus = total(np.ones(len(s)))
    home = p.pref[r, 0]
    listed_home = p.listed[home, s]
    target = np.where(listed_home, home, 0)  # unlisted lines go to the central FC, which lists every SKU
    h = ss.ps.LEAD + ss.ps.REVIEW
    mu_m = p.monthly_mu[target, s, month]
    cover = np.minimum(p.S[target, s] / np.maximum(h * mu_m, 0.1), ss.COVER_CAP)
    demand = p.monthly_mu[target, s].mean(axis=1)
    other = p.listed.sum(axis=0)[s] - p.listed[target, s]
    growth = np.nan_to_num(stats["growth"][s], nan=0.0)
    comm_size = np.bincount(w.community, minlength=w.community.max() + 1)
    listed_in_comm = np.stack([np.bincount(w.community, weights=p.listed[f], minlength=len(comm_size)) for f in range(len(p.net.fcs))])
    cut = ~w.alone[s] & (listed_in_comm[home, w.community[s]] > 0) & (listed_in_comm[home, w.community[s]] < comm_size[w.community[s]])
    region = r[starts]
    fc_reg = np.array([int(np.nonzero(w.region_ids == f)[0][0]) for f in p.net.fcs])
    n_listed = total(listed_home)
    cats = np.zeros((n, w.category.max() + 1))
    np.add.at(cats, (oi, w.category[s]), 1)
    clusters = np.zeros((n, 9))
    np.add.at(clusters, (oi, w.cluster[s]), 1)
    df = pd.DataFrame({
        "n_skus": n_skus, "units": total(q), "max_qty": vmax(q), "bulk": (vmax(q) >= 20).astype(float),
        "kg": total(q * w.weight[s]), "litres": total(q * w.skus["volume_l"].to_numpy()[s]),
        "n_categories": (cats > 0).sum(axis=1).astype(float), "month": month[starts],
        "region": region, "home_central": (home[starts] == 0).astype(float),
        "home_mult": w.mult[fc_reg[home[starts]], region],
        "n_unlisted": n_skus - n_listed, "share_unlisted": (n_skus - n_listed) / n_skus,
        "some_not_all": ((n_listed > 0) & (n_listed < n_skus)).astype(float),
        "n_listed_elsewhere": total(other > 0), "n_open": float(len(p.net.sites)), "listing_share": p.net.share,
        "service": p.net.service,
        "cover_min": vmin(cover, listed_home), "cover_mean": total(np.where(listed_home, cover, 0)) / np.maximum(n_listed, 1),
        "cv2_max": vmax(stats["cv2"][s], listed_home & np.isfinite(stats["cv2"][s])),
        "adi_max": vmax(stats["adi"][s], listed_home & np.isfinite(stats["adi"][s])),
        "growth_max": vmax(growth, listed_home), "demand_min": vmin(demand, listed_home),
        "cover_min_central": vmin(cover, ~listed_home), "growth_max_central": vmax(growth, ~listed_home),
        **{f"cluster_{c}": clusters[:, c] for c in range(9)},
        "share_distributed": total(w.cluster_distributed[w.cluster[s]]) / n_skus,
        "n_communities": pd.Series(np.where(w.alone[s], np.nan, w.community[s])).groupby(oi).nunique().to_numpy().astype(float),
        "community_cut": (total(cut) > 0).astype(float), "max_lift": max_pair_lift(w, oi, s, starts, n_skus.astype(int), lift),
        "share_alone": total(w.alone[s]) / n_skus,
    })
    df["cover_mean"] = np.where(n_listed > 0, df["cover_mean"], np.nan)
    return df[NAMES]


# =================================================================================================
# Methods
# =================================================================================================

def clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, EPS, 1 - EPS)


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def replay_rates(sim: ss.Outcome) -> np.ndarray:
    """In-stock rate per (FC, SKU, month) from a simulated year, falling back to the SKU's yearly rate at
    the FC, then the FC's rate, then 1."""
    att, fil = sim.attempts, sim.fills
    with np.errstate(divide="ignore", invalid="ignore"):
        monthly = fil / att
        yearly = fil.sum(axis=2) / att.sum(axis=2)
        fc = fil.sum(axis=(1, 2)) / att.sum(axis=(1, 2))
    fc = np.where(np.isfinite(fc), fc, 1.0)
    yearly = np.where(np.isfinite(yearly), yearly, fc[:, None])
    return np.where(np.isfinite(monthly), monthly, yearly[:, :, None])


def replay_probability(w: ss.World, p: ss.Plan, order_year: int, rates: np.ndarray) -> np.ndarray:
    """Split probability per order of `order_year` if each line's stock-outs were independent coin flips
    with the replayed in-stock rates. Lines try their FCs in route order."""
    sl = w.year_lines(order_year)
    o, s, r = w.order[sl], w.sku[sl], w.region[sl]
    oi = o - o.min()
    starts = np.r_[0, np.nonzero(np.diff(oi))[0] + 1]
    month = ss.month_of_doy(w.day[sl] - (order_year - 1) * ss.YEAR - 1)
    n_fc = len(p.net.fcs)
    ship = np.zeros((len(s), n_fc))
    remain = np.ones(len(s))
    for k in range(n_fc):
        f = p.pref[r, k]
        a = np.where(p.listed[f, s], rates[f, s, month], 0.0)
        ship[np.arange(len(s)), f] = remain * a
        remain = remain * (1 - a)
    lost_all = np.multiply.reduceat(remain, starts)
    one = sum(np.multiply.reduceat(ship[:, f] + remain, starts) - lost_all for f in range(n_fc))
    return 1 - (one + lost_all)


def lr_matrix(df: pd.DataFrame, ref: dict | None = None) -> tuple[np.ndarray, dict]:
    """Logistic regression inputs: log1p where marked, missing as 0 plus a flag, standardised with the
    training means and deviations, categories one-hot."""
    cols, names = [], []
    for name, _, kind in FEATURES:
        v = df[name].to_numpy(dtype=float)
        if kind == "cat":
            for c in range(12 if name == "month" else 8):
                cols.append((v == c).astype(float))
                names.append(f"{name}={c}")
            continue
        if kind == "log":
            v = np.log1p(np.maximum(v, 0))
        miss = np.isnan(v)
        if (ref is None and miss.any()) or (ref is not None and f"{name}?" in ref["names"]):
            cols.append(miss.astype(float))
            names.append(f"{name}?")
        cols.append(np.where(miss, 0.0, v))
        names.append(name)
    X = np.column_stack(cols)
    if ref is None:
        ref = {"names": names, "mean": X.mean(axis=0), "sd": X.std(axis=0) + 1e-9}
    X = (X - ref["mean"]) / ref["sd"]
    return X.astype(np.float32), ref


def gbm_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CATS:
        out[c] = pd.Categorical(out[c].astype(int), categories=list(range(12 if c == "month" else 8)))
    return out


def fit_map(kind: str, p: np.ndarray, y: np.ndarray):
    """A calibration map from validation predictions, returned as a function."""
    if kind == "none":
        return lambda x: clip(x)
    if kind == "platt":
        z = np.log(clip(p) / (1 - clip(p)))[:, None]
        m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        return lambda x: clip(m.predict_proba(np.log(clip(x) / (1 - clip(x)))[:, None])[:, 1])
    iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(p, y)
    return lambda x: clip(iso.predict(x))


def crossfit(kind: str, p: np.ndarray, y: np.ndarray, half: np.ndarray) -> float:
    """Log loss of a calibration map fitted on one half of the validation networks and scored on the other."""
    total = 0.0
    for h in (0, 1):
        fit, score = half == h, half != h
        total += log_loss(y[score], fit_map(kind, p[fit], y[fit])(p[score])) * score.sum()
    return total / len(y)


def sku_risk(order: np.ndarray, sku: np.ndarray, multi: np.ndarray, p: np.ndarray, n_sku: int) -> np.ndarray:
    """Expected split orders per SKU, the sum of the probabilities of the multi-SKU orders containing it.
    order: per line, the order's index in the year; multi: per order; p: per multi-SKU order."""
    pos = np.cumsum(multi) - 1
    keep = multi[order]
    return np.bincount(sku[keep], weights=p[pos[order[keep]]], minlength=n_sku)


# =================================================================================================
# Scores
# =================================================================================================

def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    b = np.minimum((p * bins).astype(int), bins - 1)
    n = np.bincount(b, minlength=bins)
    pred = np.bincount(b, weights=p, minlength=bins) / np.maximum(n, 1)
    obs = np.bincount(b, weights=y, minlength=bins) / np.maximum(n, 1)
    return {"count": n.tolist(), "predicted": pred.round(5).tolist(), "observed": obs.round(5).tolist(),
            "ece": float(np.sum(n * np.abs(pred - obs)) / n.sum())}


def scores(y: np.ndarray, p: np.ndarray, rule: np.ndarray, net: np.ndarray) -> dict:
    """Test scores of one method. rule: the plain rule's 0/1 flags (its recall is the operating point)."""
    rule_recall = float(rule[y == 1].mean())
    prec, rec, thr = precision_recall_curve(y, p)
    at = np.nonzero(rec[:-1] >= rule_recall)[0]
    i = at[-1] if len(at) else 0  # highest threshold that still reaches the rule's recall
    keep = np.unique(np.linspace(0, len(prec) - 1, 200).astype(int))
    per_net = [{"network": int(k), "orders": int((net == k).sum()), "simulated": float(y[net == k].mean()),
                "predicted": float(p[net == k].mean())} for k in np.unique(net)]
    return {
        "log_loss": log_loss(y, p), "brier": float(np.mean((p - y) ** 2)), "average_precision": float(average_precision_score(y, p)),
        "prediction_bias": float(p.mean() - y.mean()), "reliability": reliability(y, p),
        "at_rule_recall": {"recall": float(rec[i]), "precision": float(prec[i]), "threshold": float(thr[min(i, len(thr) - 1)])},
        "pr_curve": {"precision": prec[keep].round(4).tolist(), "recall": rec[keep].round(4).tolist()},
        "per_network": per_net,
        "split_rate_mae": float(np.mean([abs(n["predicted"] - n["simulated"]) for n in per_net])),
    }


def bootstrap(y: np.ndarray, probs: dict[str, np.ndarray], net: np.ndarray, rng: np.random.Generator) -> dict:
    """95% intervals over resampled test networks for log loss and Brier score of each method, and for the
    differences between gradient boosting and the others."""
    nets = np.unique(net)
    idx = np.searchsorted(nets, net)
    count = np.bincount(idx).astype(float)
    sums = {}
    for m, p in probs.items():
        pc = clip(p)
        sums[m] = {"log_loss": np.bincount(idx, weights=-(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
                   "brier": np.bincount(idx, weights=(p - y) ** 2)}
    draws = rng.integers(0, len(nets), size=(BOOTSTRAP, len(nets)))
    n = count[draws].sum(axis=1)
    out = {}
    for metric in ("log_loss", "brier"):
        vals = {m: sums[m][metric][draws].sum(axis=1) / n for m in probs}
        out[metric] = {m: np.percentile(v, [2.5, 97.5]).tolist() for m, v in vals.items()}
        out[f"{metric}_gbm_minus"] = {m: np.percentile(vals["gbm"] - vals[m], [2.5, 97.5]).tolist() for m in probs if m != "gbm"}
    return out


# =================================================================================================
# Main
# =================================================================================================

class Runner:
    """Holds the world and caches plans and simulations."""

    def __init__(self, world_dir: str, models_dir: str):
        self.w = ss.load_world(world_dir, models_dir)
        lines = pd.read_parquet(Path(world_dir) / "order_lines.parquet")
        sku_index = {int(s): i for i, s in enumerate(self.w.skus["sku_id"])}
        train = pa.baskets(lines, sku_index, 1, 3 * ss.YEAR)
        self.lift = pa.pair_stats(train["X"], train["region"], train["quarter"])["lift"]
        self.stats = {y: sku_stats(self.w, y) for y in range(1, 5)}

    def labelled(self, net: ss.Network, year: int, seed: int = 0) -> tuple[ss.Plan, ss.Outcome, pd.DataFrame]:
        p = ss.plan(self.w, net, year)
        sim = ss.simulate(self.w, p, year, seed)
        X = features(self.w, p, year, self.lift, self.stats[year - 1])
        return p, sim, X


def main(world_dir: str, models_dir: str, out_dir: str) -> None:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    R = Runner(world_dir, models_dir)
    w = R.w
    nets = draw_networks(rng)
    print(f"world and pair lifts ready ({time.time() - t0:.0f}s)")

    def log(msg: str) -> None:
        print(f"{msg} ({time.time() - t0:.0f}s)", flush=True)

    # ---- Training rows -------------------------------------------------------------------------
    Xs, ys, rules_tr, nets_tr = [], [], [], []
    month_counts = np.zeros((12, 5))  # orders, split, rule, split not flagged, flagged not split
    extra, n_split_tr = 0.0, 0
    examples_src = None
    for k, net in enumerate(nets["training"]):
        for year in TRAIN_YEARS:
            p, sim, X = R.labelled(net, year)
            multi = sim.n_skus > 1
            y, rule = sim.split[multi], sim.rule[multi]
            m = X["month"].to_numpy()[multi]
            for col, v in enumerate([np.ones_like(y), y, rule, y & ~rule, rule & ~y]):
                month_counts[:, col] += np.bincount(m, weights=v, minlength=12)
            extra += float((sim.cost - sim.cost_one)[sim.split].sum())
            n_split_tr += int(sim.split.sum())
            if k == 0 and year == 3:
                examples_src = (net, p, sim, X)
            pick = rng.choice(np.nonzero(multi)[0], size=min(ROWS_PER_NETWORK, int(multi.sum())), replace=False)
            pick.sort()
            Xs.append(X.iloc[pick].reset_index(drop=True))
            ys.append(sim.split[pick])
            rules_tr.append(sim.rule[pick])
            nets_tr.append(np.full(len(pick), k))
    Xtr = pd.concat(Xs, ignore_index=True)
    ytr = np.concatenate(ys).astype(float)
    rule_tr = np.concatenate(rules_tr)
    del Xs
    log(f"training rows {len(ytr):,}, split share {ytr.mean():.3f}")

    # ---- Validation and test rows, with the replay simulation ----------------------------------
    def scored_set(name: str, year: int) -> dict:
        Xl, yl, rl, nl, rp, plans, sims = [], [], [], [], [], [], []
        for k, net in enumerate(nets[name]):
            p, sim, X = R.labelled(net, year)
            replay = ss.simulate(w, ss.plan(w, net, year, levels_from=year - 2), year - 1)
            rates = replay_rates(replay)
            multi = sim.n_skus > 1
            Xl.append(X[multi].reset_index(drop=True))
            yl.append(sim.split[multi])
            rl.append(sim.rule[multi])
            nl.append(np.full(int(multi.sum()), k))
            rp.append(replay_probability(w, p, year, rates)[multi])
            plans.append(p)
            sims.append((sim, replay))
        return {"X": pd.concat(Xl, ignore_index=True), "y": np.concatenate(yl).astype(float), "rule": np.concatenate(rl),
                "net": np.concatenate(nl), "replay": clip(np.concatenate(rp)), "plans": plans, "sims": sims}

    V = scored_set("validation", VAL_YEAR)
    log(f"validation rows {len(V['y']):,}")
    T = scored_set("test", TEST_YEAR)
    log(f"test rows {len(T['y']):,}")

    # ---- Fit the methods ---------------------------------------------------------------------------
    rule_p = {True: float(ytr[rule_tr].mean()), False: float(ytr[~rule_tr].mean())}

    def rule_prob(flags: np.ndarray) -> np.ndarray:
        return np.where(flags, rule_p[True], rule_p[False])

    Xlr, ref = lr_matrix(Xtr)
    Vlr, _ = lr_matrix(V["X"], ref)
    lr_models, val_raw = {}, {"rule": {"-": rule_prob(V["rule"])}, "replay": {"-": V["replay"]}, "logistic": {}, "gbm": {}}
    for c in LR_C:
        m = LogisticRegression(C=c, max_iter=300).fit(Xlr, ytr)
        lr_models[c] = m
        val_raw["logistic"][f"C={c}"] = clip(m.predict_proba(Vlr)[:, 1])
        log(f"logistic regression C={c}")
    Gtr, Gv = gbm_frame(Xtr), gbm_frame(V["X"])
    gbm_models = {}
    for leaves, rows, sampling in itertools.product(*GBM_GRID.values()):
        params = {"objective": "binary", "learning_rate": GBM_RATE, "num_leaves": leaves, "min_data_in_leaf": rows,
                  "feature_fraction": sampling, "bagging_fraction": sampling, "bagging_freq": 1 if sampling < 1 else 0,
                  "seed": SEED, "deterministic": True, "num_threads": 4, "verbose": -1}
        booster = lgb.train(params, lgb.Dataset(Gtr, ytr, categorical_feature=CATS), num_boost_round=max(GBM_TREES))
        gbm_models[(leaves, rows, sampling)] = booster
        for trees in GBM_TREES:
            val_raw["gbm"][f"leaves={leaves},rows={rows},sampling={sampling},trees={trees}"] = clip(booster.predict(Gv, num_iteration=trees))
        log(f"LightGBM leaves {leaves} rows {rows} sampling {sampling}")

    # ---- Choose settings and calibration maps on validation (cross-fitted) ---------------------
    half = (V["net"] % 2).astype(int)
    grid, chosen = {}, {}
    for method, cands in val_raw.items():
        grid[method] = []
        for setting, p in cands.items():
            row = {"setting": setting, "raw_log_loss": log_loss(V["y"], p)}
            for kind in MAPS:
                row[kind] = crossfit(kind, p, V["y"], half)
            grid[method].append(row)
        best = min(((r["setting"], kind, r[kind]) for r in grid[method] for kind in MAPS), key=lambda t: t[2])
        chosen[method] = {"setting": best[0], "map": best[1], "crossfit_log_loss": best[2]}
    chosen_method = min(chosen, key=lambda m: chosen[m]["crossfit_log_loss"])
    maps = {m: fit_map(chosen[m]["map"], val_raw[m][chosen[m]["setting"]], V["y"]) for m in METHODS}
    log(f"chosen {chosen}; best method {chosen_method}")

    def gbm_of(setting: str) -> tuple[lgb.Booster, int]:
        kv = dict(x.split("=") for x in setting.split(","))
        return gbm_models[(int(kv["leaves"]), int(kv["rows"]), float(kv["sampling"]))], int(kv["trees"])

    def predictor(method: str):
        """Calibrated probabilities of a method for a feature frame (the replay and the rule need more)."""
        setting = chosen[method]["setting"]
        if method == "logistic":
            c = float(setting.split("=")[1])
            return lambda X: maps[method](clip(lr_models[c].predict_proba(lr_matrix(X, ref)[0])[:, 1]))
        if method == "gbm":
            booster, trees = gbm_of(setting)
            return lambda X: maps[method](clip(booster.predict(gbm_frame(X), num_iteration=trees)))
        raise ValueError(method)

    predict = {m: predictor(m) for m in ("logistic", "gbm")}

    # ---- Test set, scored once -----------------------------------------------------------------
    test_p = {"rule": maps["rule"](rule_prob(T["rule"])), "replay": maps["replay"](T["replay"]),
              "logistic": predict["logistic"](T["X"]), "gbm": predict["gbm"](T["X"])}
    booster, trees = gbm_of(chosen["gbm"]["setting"])
    raw_gbm_test = clip(booster.predict(gbm_frame(T["X"]), num_iteration=trees))
    test = {m: scores(T["y"], p, T["rule"], T["net"]) for m, p in test_p.items()}
    test["gbm_uncalibrated"] = {"reliability": reliability(T["y"], raw_gbm_test), "log_loss": log_loss(T["y"], raw_gbm_test)}
    rule_flags = {"precision": float(T["y"][T["rule"]].mean()), "recall": float(T["rule"][T["y"] == 1].mean()),
                  "flagged": float(T["rule"].mean()), "wrong": float((T["rule"] != (T["y"] == 1)).mean())}
    intervals = bootstrap(T["y"], test_p, T["net"], np.random.default_rng(SEED + 1))
    log("test set scored")

    # ---- Noise floor: other random orders of a day ---------------------------------------------
    flips, sds = [], []
    for k, net in enumerate(nets["test"]):
        p = T["plans"][k]
        base = T["sims"][k][0]
        multi = base.n_skus > 1
        runs = np.stack([base.split[multi]] + [ss.simulate(w, p, TEST_YEAR, seed).split[multi] for seed in NOISE_SEEDS])
        flips.append(((runs != runs[0]).any(axis=0)).mean())
        sds.append(runs.mean(axis=1).std())
    noise = {"seeds": NOISE_SEEDS, "orders_flipping": float(np.mean(flips)), "split_rate_sd": float(np.mean(sds))}
    log("noise floor done")

    # ---- SHAP values ---------------------------------------------------------------------------
    sample = np.random.default_rng(SEED + 2).choice(len(T["y"]), size=min(SHAP_ROWS, len(T["y"])), replace=False)
    contrib = booster.predict(gbm_frame(T["X"].iloc[sample]), num_iteration=trees, pred_contrib=True)
    mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
    shap = {"features": {n: float(v) for n, v in zip(NAMES, mean_abs)},
            "groups": {g: float(sum(v for (n, gg, _), v in zip(FEATURES, mean_abs) if gg == g)) for g in dict.fromkeys(f[1] for f in FEATURES)},
            "rows": len(sample)}

    # ---- SKUs at risk on the test networks -----------------------------------------------------
    sl = w.year_lines(TEST_YEAR)
    o = w.order[sl] - w.order[sl].min()
    sku_rows, sku_summary = [], {}
    offset = 0
    all_pred = {m: [] for m in METHODS}
    all_sim = []
    for k, net in enumerate(nets["test"]):
        sim = T["sims"][k][0]
        multi = sim.n_skus > 1
        n_multi = int(multi.sum())
        simulated = sku_risk(o, w.sku[sl], multi, T["y"][offset:offset + n_multi], w.n_sku)
        rec = {"network": k, "simulated": simulated.round(0).astype(int).tolist()}
        for m in METHODS:
            pred = sku_risk(o, w.sku[sl], multi, test_p[m][offset:offset + n_multi], w.n_sku)
            rec[m] = pred.round(2).tolist()
            all_pred[m].append(pred)
        all_sim.append(simulated)
        sku_rows.append(rec)
        offset += n_multi
    sim_all = np.concatenate(all_sim)
    for m in METHODS:
        pr = np.concatenate(all_pred[m])
        has = sim_all + pr > 0
        sku_summary[m] = {"mae": float(np.abs(pr - sim_all)[has].mean()), "correlation": float(np.corrcoef(pr[has], sim_all[has])[0, 1]),
                          "total_predicted": float(pr.sum()), "total_simulated": float(sim_all.sum())}
    log("SKU risk done")

    # ---- Business check: candidate networks for year 5 estimated at the end of year 4 ----------
    extra_per_split = extra / max(n_split_tr, 1)
    candidates = []
    for name in ("test", "decision"):
        for k, net in enumerate(nets[name]):
            if name == "test":
                p, sim = T["plans"][k], T["sims"][k][0]
            else:
                p = ss.plan(w, net, TEST_YEAR)
                sim = ss.simulate(w, p, TEST_YEAR)
            multi = sim.n_skus > 1
            truth = float(sim.split[multi].mean())
            # Year-4 baskets under the network planned for year 5.
            X4 = features(w, p, VAL_YEAR, R.lift, R.stats[TEST_YEAR - 1])
            sl4 = w.year_lines(VAL_YEAR)
            oi4 = w.order[sl4] - w.order[sl4].min()
            n4 = np.bincount(oi4)
            m4 = n4 > 1
            flags4 = X4["some_not_all"].to_numpy()[m4] > 0
            replay = ss.simulate(w, ss.plan(w, net, TEST_YEAR, levels_from=VAL_YEAR - 1), VAL_YEAR)
            est = {"rule": float(maps["rule"](rule_prob(flags4)).mean()),
                   "replay": float(replay.split[replay.n_skus > 1].mean()),
                   "logistic": float(predict["logistic"](X4[m4]).mean()),
                   "gbm": float(predict["gbm"](X4[m4]).mean())}
            candidates.append({"set": name, "network": k, **net.as_dict(), "simulated": truth, "estimates": est,
                               "rule_flagged_share": float(flags4.mean()), "multi_orders": int(multi.sum()),
                               "split_orders": int(sim.split.sum()), "parcel_cost": float(sim.cost.sum()),
                               "extra_parcel_cost": float((sim.cost - sim.cost_one)[sim.split].sum())})
    log("business candidates done")
    estimate_error = {}
    for m in METHODS:
        err = np.array([c["estimates"][m] - c["simulated"] for c in candidates])
        orders = np.array([c["multi_orders"] for c in candidates])
        estimate_error[m] = {"mae_points": float(np.abs(err).mean()), "bias_points": float(err.mean()),
                             "mae_orders": float(np.abs(err * orders).mean()),
                             "mae_dollars": float(np.abs(err * orders).mean() * extra_per_split),
                             "test_networks_mae_points": float(np.abs(err[:N_TEST]).mean())}
    decisions = []
    for sites in DECISION_SITES:
        group = [c for c in candidates if c["set"] == "decision" and tuple(c["sites"]) == sites]
        best = min(group, key=lambda c: c["simulated"])
        row = {"sites": list(sites), "best_rule": best["rule"], "best_split": best["simulated"], "choices": {}}
        for m in METHODS:
            pick = min(group, key=lambda c: c["estimates"][m])
            row["choices"][m] = {"rule": pick["rule"], "regret_points": pick["simulated"] - best["simulated"],
                                 "regret_orders": (pick["simulated"] - best["simulated"]) * pick["multi_orders"]}
        decisions.append(row)

    # ---- Examples for the page -----------------------------------------------------------------
    net0, p0, sim0, X0 = examples_src
    multi0 = sim0.n_skus > 1
    y0, r0 = sim0.split, sim0.rule
    home0 = X0["home_central"].to_numpy() > 0
    kinds = {"split, flagged": y0 & r0, "split, not flagged, regional home": y0 & ~r0 & ~home0,
             "split, not flagged, central home": y0 & ~r0 & home0, "whole, flagged": ~y0 & r0,
             "whole, not flagged, 3 or more SKUs": ~y0 & ~r0 & (sim0.n_skus >= 3), "whole, not flagged, 2 SKUs": ~y0 & ~r0 & (sim0.n_skus == 2)}
    p_gbm0 = predict["gbm"](X0)
    examples = []
    for label, mask in kinds.items():
        i = int(np.nonzero(mask & multi0)[0][0])
        examples.append({"kind": label, "order_id": int(w.order_id[sim0.first_order + i]), "split": bool(y0[i]), "rule": bool(r0[i]),
                         "gbm": float(p_gbm0[i]), "features": {n: (None if pd.isna(v) else float(v)) for n, v in X0.iloc[i].items()}})
    sl3 = w.year_lines(3)
    o3 = w.order[sl3] - w.order[sl3].min()
    trace = None
    for i in np.nonzero(y0 & r0 & ~home0 & (sim0.n_skus >= 3))[0]:
        lines_i = np.nonzero(o3 == i)[0]
        hf = p0.pref[w.order_region[sim0.first_order + i], 0]
        s_i, ship_i = w.sku[sl3][lines_i], sim0.ship[lines_i]
        listed_i = p0.listed[hf, s_i]
        if (~listed_i).any() and (listed_i & (ship_i != hf)).any():
            trace = {"order_id": int(w.order_id[sim0.first_order + i]), "region": int(w.region_ids[w.order_region[sim0.first_order + i]]),
                     "home_fc": int(net0.fcs[hf]), "network": net0.as_dict(),
                     "lines": [{"sku": str(w.skus["name"].iloc[s]), "listed_at_home": bool(li), "shipped_from": int(net0.fcs[sh]) if sh >= 0 else None}
                               for s, li, sh in zip(s_i, listed_i, ship_i)]}
            break

    # ---- Write ---------------------------------------------------------------------------------
    n_orders = len(w.order_region)
    sizes = np.bincount(w.order)
    train_orders = np.searchsorted(w.order_day, 3 * ss.YEAR + 1)
    overview = {
        "model": {"id": "split-risk", "title": "Split orders"},
        "periods": {"training_years": list(TRAIN_YEARS), "validation_year": VAL_YEAR, "test_year": TEST_YEAR},
        "facts": {"orders_training": int(train_orders), "multi_share_training": float((sizes[:train_orders] > 1).mean()),
                  "orders_all": int(n_orders), "skus": int(w.n_sku), "fc_sites": len(w.region_ids)},
        "settings": {"rows_per_network": ROWS_PER_NETWORK, "lr_c": LR_C, "gbm_grid": GBM_GRID, "gbm_trees": GBM_TREES,
                     "gbm_learning_rate": GBM_RATE, "maps": MAPS, "decision_sites": [list(s) for s in DECISION_SITES],
                     "decision_share": DECISION_SHARE, "decision_service": DECISION_SERVICE, "bootstrap": BOOTSTRAP,
                     "lead_days": ss.ps.LEAD, "review_days": ss.ps.REVIEW, "parcel_fee": pa.PARCEL_FEE, "per_kg": pa.PER_KG,
                     "cover_cap": ss.COVER_CAP, "seed": SEED},
        "networks": {name: [n.as_dict() for n in lst] for name, lst in nets.items()},
        "features": [{"name": n, "group": g, "transform": t} for n, g, t in FEATURES],
        "training": {"rows": len(ytr), "split_share": float(ytr.mean()), "rule_probabilities": {"flagged": rule_p[True], "not_flagged": rule_p[False]},
                     "causes_by_month": {k: month_counts[:, i].astype(int).tolist() for i, k in enumerate(["orders", "split", "flagged", "split_not_flagged", "flagged_not_split"])},
                     "extra_parcel_cost_per_split": extra_per_split},
        "validation": {"rows": len(V["y"]), "split_share": float(V["y"].mean()), "grid": grid, "chosen": chosen,
                       "chosen_method": chosen_method, "halves": "validation networks with even index against odd index"},
        "test": {"rows": len(T["y"]), "split_share": float(T["y"].mean()), "methods": test, "rule_flags": rule_flags,
                 "intervals": intervals, "noise": noise},
        "shap": shap,
        "sku_risk": sku_summary,
        "business": {"candidates": candidates, "estimate_error": estimate_error, "decisions": decisions,
                     "extra_parcel_cost_per_split": extra_per_split},
        "labels": METHOD_LABELS,
        "runtime_s": round(time.time() - t0, 1),
    }
    (out / "overview.json").write_text(json.dumps(overview, separators=(",", ":")))
    (out / "skus.json").write_text(json.dumps({"skus": w.skus["name"].tolist(), "networks": sku_rows}, separators=(",", ":")))
    (out / "examples.json").write_text(json.dumps({"examples": examples, "trace": trace}, separators=(",", ":")))
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
