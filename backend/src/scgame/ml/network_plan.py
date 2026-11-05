"""Model O: network plan. Where should the shop open or expand FCs over years 6 to 10, and by how much?

Usage: uv run python -m scgame.ml.network_plan <world dir> <models dir> <output dir>

The design agreed with Kilian is in design/network-plan/PROPOSAL.md (sections 1 to 9, decisions in section
14); design/network-plan/HANDOVER.md describes the run.

  1. Demand paths. Model B's test paths for the 8 regions (`network_paths.write_b_paths`, read from
     b_paths.npz), and for three new markets model C's log-linear index refitted on years 1 to 5, scaled month
     by month by B's category paths (C's network rate taken from B), with C's error model. 1,000 joint paths.
  2. The programme. A two-stage stochastic mixed-integer programme by sample average approximation. First
     stage, open (binary) and capacity (units per month) per site for years 7 to 10, never closed or reduced;
     year 6 runs on today's network. Second stage per scenario and month, units served from each FC to each
     region and units lost. HiGHS through scipy.optimize.milp.
  3. Settings on the forecasts only. The number of scenarios S by in-sample against out-of-sample cost on B's
     and C's own paths (plans from paths 0 to S - 1, evaluated on paths 200 to 999).
  4. Plans. Do nothing, expand the busiest FC, P50 plan, P90 plan, our plan, ETS plan (sensitivity), and the
     ground-truth plan (replications 0 to 99, reference).
  5. Check. Every plan costed on replications 100 to 199 of both futures and on both realised futures, each
     month served at the lowest cost given the plan's capacity (the second stage solved for that future).
  6. Ranges. Our, P50, P90 and ground-truth plans re-solved at 9 capacity cost ratios and 6 opening costs.
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from scgame.ml import demand_forecast as df
from scgame.ml import network_paths as npth
from scgame.ml import policy_sim as ps
from scgame.ml import region_demand as rd
from scgame.ml import region_sim as rs

N_SITES = 11  # the 8 regions (ids 1..8, Calopeia first) and the three new markets
CENTRAL = 0
YEARS = [6, 7, 8, 9, 10]
PLAN_YEARS = 4  # years 7 to 10 (a new FC or capacity takes a year to build)
MONTHS = 60  # months 61..120
S_GRID = [25, 50, 100, 200]
S_TOLERANCE = 0.005  # the smallest S whose out-of-sample cost is within 0.5% of S = 200's
OUT_OF_SAMPLE = (200, 1000)  # paths used to check every S
TRUTH_BUILD, TRUTH_SCORE = range(100), range(100, 200)
RHO_GRID = [float(r) for r in df.RHO[::5]]  # 9 capacity cost ratios, 0.1 to 10
OPENING_MULTIPLES = [0.0, 1.0, 3.0, 10.0, 30.0, 50.0]
OPENING_SHARE = 500_000 / 5 / 1_450  # the game's factory fixed cost per year over five years, in unit prices (68.97)
KEEP = {1.0: 1.0, 1.4: 1 - ps.SLOW_DELIVERY_LOSS["same_landmass"], 2.2: 1 - ps.SLOW_DELIVERY_LOSS["cross_water"]}
GAP = 1e-3
TIME_LIMIT = 1200.0
SEED = 29
PLANS = ["Do nothing", "Expand the busiest FC", "P50 plan", "P90 plan", "Our plan"]
WORKERS = 4


# =================================================================================================
# Network, costs and demand
# =================================================================================================

def network(world: Path) -> dict:
    """Sites, distance multipliers, the share of demand kept, and year-5 costs per unit."""
    reg = pd.read_parquet(world / "regions.parquet").sort_values("id").reset_index(drop=True)
    test = rs.design_test_regions(reg, 7).set_index("id").loc[npth.NEW_MARKETS].reset_index()
    land = reg["landmass"].tolist() + [lm if lm == "continent" else f"island {i}" for i, lm in zip(test["id"], test["landmass"])]
    ids = reg["id"].tolist() + test["id"].tolist()
    mult = np.array([[1.0 if a == b else 1.4 if land[a] == land[b] else 2.2 for b in range(N_SITES)] for a in range(N_SITES)])
    keep = np.vectorize(KEEP.get)(mult)
    lines = pd.read_parquet(world / "order_lines.parquet")
    sk = pd.read_parquet(world / "skus.parquet")
    y5 = lines[lines["day"] > 4 * 365].merge(sk[["sku_id", "weight_kg", "price", "margin"]], on="sku_id")
    y5 = y5.assign(kg=y5.quantity * y5.weight_kg, rev=y5.quantity * y5.price, mar=y5.quantity * y5.price * y5.margin)
    g = y5.groupby("region_id").agg(units=("quantity", "sum"), kg=("kg", "sum"), orders=("order_id", "nunique"), rev=("rev", "sum"), mar=("mar", "sum"))
    local = (1.5 * g.orders + 0.35 * g.kg) / g.units  # the shop's parcel formula ($1.50 per parcel, $0.35 per kg) per unit, local lane
    tot = g.sum()
    net = {"local": float((1.5 * tot.orders + 0.35 * tot.kg) / tot.units), "margin": float(tot.mar / tot.units), "price": float(tot.rev / tot.units)}
    d = np.r_[local.to_numpy(), np.full(3, net["local"])]  # new markets: the network's average (no history)
    m = np.r_[(g.mar / g.units).to_numpy(), np.full(3, net["margin"])]
    month = df.ds.month_of_day(y5["day"].to_numpy())
    peak = float(pd.Series(y5["quantity"].to_numpy()).groupby(month).sum().max())
    return {"ids": ids, "names": reg["name"].tolist() + [f"New region {i - 100}" for i in test["id"]], "landmass": land,
            "mult": mult, "keep": keep, "delivery": d, "margin": m, "network": net, "k0": peak,
            "population": np.r_[reg["population_m"].to_numpy(), test["population_m"].to_numpy()], "test_rows": test}


def cost_terms(net: dict, rho: float | None = None, opening_multiple: float = 1.0) -> dict:
    """Capacity cost c per unit of monthly capacity per year and opening cost F per FC per year. rho None is
    the game's ratio (B's GAME_CAPACITY_SHARE of the price)."""
    c_game = df.GAME_CAPACITY_SHARE * net["network"]["price"]
    c = c_game if rho is None else rho * net["network"]["margin"]
    return {"c": float(c), "F": float(OPENING_SHARE * net["network"]["price"] * opening_multiple),
            "rho": float(c / net["network"]["margin"]), "rho_game": float(c_game / net["network"]["margin"])}


def unit_costs(net: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per unit served from site i to region j (delivery plus the margin of demand lost to slower delivery),
    and per unit lost."""
    x = net["delivery"][None, :] * net["mult"] + net["margin"][None, :] * (1 / net["keep"] - 1)
    return x, net["margin"].copy()


def market_paths(world: Path, models: Path, b: dict, key: str, net: dict, seed: int = SEED) -> np.ndarray:
    """(paths, 3, 60) units of the new markets: C's log-linear index refitted on years 1 to 5, with the network
    rate of each category and month taken from B's category paths, and C's error model (region and cell
    errors fixed over the five years, month noise each month)."""
    co = json.loads((models / "new-region-demand" / "overview.json").read_text())
    v = co["validation"][co["chosen"]]
    s, em = v["setting"], v["errors"]
    panel = rd.load_panel(world)
    yearly = panel.yearly  # (8, 8, 5, 2)
    fit = [0, 1, 2, 3, 4]
    U = yearly[:, :, fit, :].sum(axis=2)
    y = U.sum(axis=-1) if s["bulk"] == "in label" else U[..., 0]
    pop_k = panel.regions["population_m"].to_numpy() * 1000
    rate = y / pop_k[:, None]
    label = np.log(np.maximum(rate / (y.sum(axis=0) / pop_k.sum()), 1e-9))
    weight = y / y.sum(axis=1, keepdims=True)
    f_all = rd.features(panel.regions)
    f_new = rd.features(net["test_rows"])
    p = rd.fit_predict_log_index(s, f_all, label, weight, f_new)  # (3, 8)
    cat = b[f"{key}_category"].astype(float)  # (paths, 8 categories, 60)
    y5 = yearly[:, :, 4, :].sum(axis=0)  # (8, 2) year 5 consumer and bulk units per category
    bulk_share = y5[:, 1] / y5.sum(axis=1) if s["bulk"] == "network average" else np.zeros(8)
    cons = cat * (1 - bulk_share)[None, :, None] / pop_k.sum()
    bulk = cat * bulk_share[None, :, None] / pop_k.sum()
    if s["bulk"] == "in label":
        cons, bulk = cons + bulk, np.zeros_like(bulk)
    rng = np.random.default_rng(seed)
    n = cat.shape[0]
    out = np.zeros((n, 3, MONTHS))
    for j in range(3):
        popk = net["population"][8 + j] * 1000
        point = np.exp(p[j]) * popk * cons.mean(axis=(0, 2)) * 12  # mix for centring the cell errors
        mix = point / point.sum()
        a = em["sigma_region"] * np.sqrt(1 + 1 / em["regions"]) * rng.standard_t(em["regions"] - 1, size=n)
        u = rng.normal(0, 1, size=(n, 8))
        eps = (u - (u @ mix)[:, None]) * em["sigma_cell"] / np.sqrt(max(1 - (mix ** 2).sum(), 1e-9))
        units = popk * (np.exp(p[j][None, :, None] + a[:, None, None] + eps[:, :, None]) * cons + bulk)  # (n, 8, 60)
        months = units.sum(axis=1)
        expected = months.mean(axis=0, keepdims=True)
        out[:, j] = months * np.exp(rng.normal(0, 1, size=months.shape) * np.sqrt(em["month_dispersion"] / np.maximum(expected, 1.0)))
    return out


def futures(world: Path, out: Path) -> dict:
    """Demand of the 11 regions in months 61..120 of every future: (replications, 11, 60) and realised (11, 60)."""
    mk = np.load(out / "markets.npz")
    res = {}
    for name, sub in (("first", world), ("second", world / df.SECOND_FUTURE)):
        w = df.ds.load(world, sub)
        realised = w.units.sum(axis=1)[:, 60:120]
        reps = np.load(sub / "truth_replications.npz")["units"]  # (rep, region, sku, months 37..120)
        existing = reps.sum(axis=2)[:, :, 24:84]
        m_reps = mk[f"{name}_replications"].sum(axis=2).reshape(-1, 3, MONTHS)
        m_real = mk[f"{name}_realised"].sum(axis=1).reshape(3, MONTHS)
        res[name] = {"replications": np.concatenate([existing, m_reps], axis=1).astype(float),
                     "realised": np.concatenate([realised, m_real], axis=0).astype(float)}
    return res


# =================================================================================================
# The programme
# =================================================================================================

def solve_plan(D: np.ndarray, net: dict, costs: dict, time_limit: float = TIME_LIMIT) -> dict:
    """Two-stage stochastic MILP over scenarios D (S, 11, 48 months of years 7 to 10). Returns capacity K
    (11, 5 years), open (11, 5), the objective (years 7 to 10) and the solver's report."""
    S = D.shape[0]
    I = J = N_SITES
    Y, M = PLAN_YEARS, 12
    xc, uc = unit_costs(net)
    no, nk = (I - 1) * Y, I * Y
    nb = I * J + J
    blocks = S * Y * M
    n = no + nk + blocks * nb

    def O(i, y):
        return (i - 1) * Y + y

    def K(i, y):
        return no + i * Y + y

    cost = np.zeros(n)
    cost[:no] = costs["F"]
    cost[no:no + nk] = costs["c"]
    cost[no + nk:] = np.tile(np.r_[xc.ravel(), uc] / S, blocks)
    big = float(D.sum(axis=1).max()) * 1.05
    r_, c_, v_, lo, hi = [], [], [], [], []
    row = 0
    for i in range(1, I):
        for y in range(Y):
            r_ += [row, row]; c_ += [K(i, y), O(i, y)]; v_ += [1.0, -big]; lo.append(-np.inf); hi.append(0.0); row += 1
            if y:
                r_ += [row, row]; c_ += [O(i, y), O(i, y - 1)]; v_ += [1.0, -1.0]; lo.append(0.0); hi.append(np.inf); row += 1
    for i in range(I):
        for y in range(1, Y):
            r_ += [row, row]; c_ += [K(i, y), K(i, y - 1)]; v_ += [1.0, -1.0]; lo.append(0.0); hi.append(np.inf); row += 1
    b = np.arange(blocks)
    s_, y_, m_ = b // (Y * M), (b // M) % Y, b % M
    off = no + nk + b * nb
    ii, jj = np.meshgrid(np.arange(I), np.arange(J), indexing="ij")
    xv = off[:, None, None] + ii[None] * J + jj[None]  # (blocks, I, J)
    cap_row = row + b[:, None] * I + np.arange(I)[None]
    dem_row = row + blocks * I + b[:, None] * J + np.arange(J)[None]
    R = np.r_[np.array(r_), np.broadcast_to(cap_row[:, :, None], xv.shape).ravel(), cap_row.ravel(),
              np.broadcast_to(dem_row[:, None, :], xv.shape).ravel(), dem_row.ravel()]
    C = np.r_[np.array(c_), xv.ravel(), (no + np.arange(I)[None] * Y + y_[:, None]).ravel(),
              xv.ravel(), (off[:, None] + I * J + np.arange(J)[None]).ravel()]
    V = np.r_[np.array(v_), np.ones(xv.size), -np.ones(blocks * I), np.broadcast_to(1 / net["keep"][None], xv.shape).ravel(), np.ones(blocks * J)]
    A = sparse.csr_matrix((V, (R, C)), shape=(row + blocks * (I + J), n))
    dem = D[s_, :, y_ * M + m_]
    lo = np.r_[lo, np.full(blocks * I, -np.inf), dem.ravel()]
    hi = np.r_[hi, np.zeros(blocks * I), dem.ravel()]
    lb, ub = np.zeros(n), np.full(n, np.inf)
    ub[:no] = 1
    lb[K(CENTRAL, 0):K(CENTRAL, 0) + Y] = net["k0"]  # today's central capacity is never reduced
    integ = np.zeros(n)
    integ[:no] = 1
    t0 = time.time()
    res = milp(cost, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(lb, ub),
               options={"mip_rel_gap": GAP, "time_limit": time_limit, "disp": False})
    if res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")
    o = np.round(res.x[:no]).reshape(I - 1, Y)
    k = res.x[no:no + nk].reshape(I, Y) * np.r_[1, o.max(axis=1) > 0][:, None]
    k = np.maximum.accumulate(np.where(k < 1e-6, 0.0, k), axis=1)
    Kfull = np.zeros((I, 5))
    Kfull[CENTRAL, 0] = net["k0"]
    Kfull[:, 1:] = k
    Ofull = np.zeros((I, 5), dtype=bool)
    Ofull[CENTRAL] = True
    Ofull[1:, 1:] = (o > 0.5) & (k[1:] > 0)  # a site counts as open when it has capacity (with a free opening cost the solver may open empty sites)
    return {"K": Kfull, "open": Ofull, "objective": float(res.fun), "status": int(res.status), "message": str(res.message),
            "gap": float(getattr(res, "mip_gap", np.nan) or 0.0), "seconds": round(time.time() - t0, 1), "scenarios": int(S)}


def fixed_plan(net: dict, central: np.ndarray) -> dict:
    """A plan with the central FC only, capacity per year given (5 values)."""
    K = np.zeros((N_SITES, 5))
    K[CENTRAL] = np.maximum.accumulate(np.maximum(central, net["k0"]))
    O = np.zeros((N_SITES, 5), dtype=bool)
    O[CENTRAL] = True
    return {"K": K, "open": O}


# =================================================================================================
# Evaluation: the second stage solved for one future
# =================================================================================================

class Evaluator:
    """Serves one future's 60 months at the lowest cost given a plan's capacity (one LP per future)."""

    def __init__(self, net: dict):
        I = J = N_SITES
        xc, uc = unit_costs(net)
        self.net, self.xc, self.uc = net, xc, uc
        nb = I * J + J
        b = np.arange(MONTHS)
        off = b * nb
        ii, jj = np.meshgrid(np.arange(I), np.arange(J), indexing="ij")
        xv = off[:, None, None] + ii[None] * J + jj[None]
        cap = sparse.csr_matrix((np.ones(xv.size), (np.broadcast_to((b[:, None] * I + np.arange(I)[None])[:, :, None], xv.shape).ravel(), xv.ravel())),
                                shape=(MONTHS * I, MONTHS * nb))
        dem = sparse.csr_matrix((np.r_[np.broadcast_to(1 / net["keep"][None], xv.shape).ravel(), np.ones(MONTHS * J)],
                                 (np.r_[np.broadcast_to((b[:, None] * J + np.arange(J)[None])[:, None, :], xv.shape).ravel(), (b[:, None] * J + np.arange(J)[None]).ravel()],
                                  np.r_[xv.ravel(), (off[:, None] + I * J + np.arange(J)[None]).ravel()])), shape=(MONTHS * J, MONTHS * nb))
        self.A_ub, self.A_eq = cap, dem
        self.cost = np.tile(np.r_[xc.ravel(), uc], MONTHS)
        self.nb = nb

    def __call__(self, K: np.ndarray, D: np.ndarray) -> dict:
        """K (11, 5 years), D (11, 60). Second-stage cost parts and service for the five years."""
        cap = np.repeat(K, 12, axis=1).T.ravel()  # (60 x 11) month-major, like the rows
        res = linprog(self.cost, A_ub=self.A_ub, b_ub=cap, A_eq=self.A_eq, b_eq=D.T.ravel(), bounds=(0, None), method="highs")
        if res.status != 0:
            raise RuntimeError(res.message)
        z = res.x.reshape(MONTHS, self.nb)
        x = z[:, :N_SITES * N_SITES].reshape(MONTHS, N_SITES, N_SITES)
        u = z[:, N_SITES * N_SITES:]
        net = self.net
        delivery = float((x * (net["delivery"][None, None, :] * net["mult"][None])).sum())
        slow = float((x * (net["margin"][None, None, :] * (1 / net["keep"] - 1)[None])).sum())
        lost = float((u * net["margin"][None]).sum())
        served = float(x.sum())
        demand = float(D.sum())
        shipped = x.sum(axis=2).reshape(5, 12, N_SITES).sum(axis=1)  # (year, site)
        return {"delivery": delivery, "slow_delivery": slow, "lost_sales": lost, "served": served, "demand": demand,
                "fill_rate": served / demand, "shipped": shipped.T.tolist()}


def first_stage_cost(plan: dict, costs: dict) -> dict:
    K, O = plan["K"], plan["open"]
    opened = O.copy()
    opened[CENTRAL] = False  # today's FC is sunk
    return {"capacity": float(costs["c"] * K.sum()), "opening": float(costs["F"] * opened.sum())}


_EV: dict = {}


def _ev_init(net: dict) -> None:
    _EV["ev"] = Evaluator(net)


def _ev_one(args):
    K, D = args
    return _EV["ev"](K, D)


def evaluate_many(pool: Pool, K: np.ndarray, Ds: np.ndarray) -> list[dict]:
    return pool.map(_ev_one, [(K, D) for D in Ds], chunksize=max(1, len(Ds) // (4 * WORKERS)))


def summarise(plan: dict, rows: list[dict], costs: dict) -> dict:
    fs = first_stage_cost(plan, costs)
    second = np.array([r["delivery"] + r["slow_delivery"] + r["lost_sales"] for r in rows])
    total = fs["capacity"] + fs["opening"] + second
    return {"expected": float(total.mean()), "p90": float(np.percentile(total, 90)), "sd": float(total.std()),
            "parts": {"capacity": fs["capacity"], "opening": fs["opening"],
                      **{k: float(np.mean([r[k] for r in rows])) for k in ("delivery", "slow_delivery", "lost_sales")}},
            "fill_rate": float(np.mean([r["fill_rate"] for r in rows])), "lost_share": float(np.mean([1 - r["served"] / r["demand"] for r in rows])),
            "second_stage": second.tolist()}


def recost(plan: dict, second_stage: list[float], costs: dict) -> float:
    """Expected total under other first-stage prices, the second stage being unchanged for a fixed plan."""
    fs = first_stage_cost(plan, costs)
    return float(fs["capacity"] + fs["opening"] + np.mean(second_stage))


# =================================================================================================
# Main
# =================================================================================================

def _solve_job(args):
    D, net, costs, limit = args
    return solve_plan(D, net, costs, time_limit=limit)


def solve_all(jobs: list[tuple[np.ndarray, dict, dict]], limit: float = TIME_LIMIT) -> list[dict]:
    """Independent programmes in parallel processes (HiGHS runs each on one thread)."""
    with Pool(WORKERS) as sp:
        return sp.map(_solve_job, [(D, net, costs, limit) for D, net, costs in jobs], chunksize=1)


def plan_payload(p: dict) -> dict:
    return {"K": np.round(p["K"], 1).tolist(), "open": p["open"].astype(int).tolist(),
            **{k: p[k] for k in ("objective", "status", "message", "gap", "seconds", "scenarios") if k in p}}


def main(world_dir: str, models_dir: str, out_dir: str) -> None:
    t0 = time.time()
    world, models, out = Path(world_dir), Path(models_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def log(s: str) -> None:
        print(f"{s} ({time.time() - t0:.0f}s)", flush=True)

    net = network(world)
    game = cost_terms(net)
    b = dict(np.load(out / "b_paths.npz"))
    paths = {}
    for key in ("nn", "ets"):
        paths[key] = np.concatenate([b[f"{key}_region"].astype(float), market_paths(world, models, b, key, net)], axis=1)  # (1000, 11, 60)
    fut = futures(world, out)
    log("inputs ready")
    yr = slice(12, 60)  # months of years 7 to 10

    pool = Pool(WORKERS, initializer=_ev_init, initargs=(net,))

    # ---- number of scenarios, on the forecasts' own paths only
    oos = paths["nn"][OUT_OF_SAMPLE[0]:OUT_OF_SAMPLE[1]]
    s_check = []
    for S, p in zip(S_GRID, solve_all([(paths["nn"][:S, :, yr], net, game) for S in S_GRID])):
        rows = evaluate_many(pool, p["K"], oos)
        s_check.append({"S": S, "in_sample": p["objective"], "out_of_sample": summarise(p, rows, game)["expected"], "plan": plan_payload(p)})
        log(f"S = {S}: in-sample {p['objective']:,.0f}, out-of-sample {s_check[-1]['out_of_sample']:,.0f}, {p['seconds']}s")
    ref = s_check[-1]["out_of_sample"]
    S_chosen = next(c["S"] for c in s_check if c["out_of_sample"] <= ref * (1 + S_TOLERANCE))
    log(f"chosen S = {S_chosen}")

    # ---- plans at the game's costs
    q = {k: np.quantile(paths["nn"], k, axis=0) for k in (0.1, 0.5, 0.9)}
    net_p50 = q[0.5].sum(axis=0)  # network P50 per month (sum of regional medians)
    busiest = np.r_[net["k0"], [net_p50[12 * y:12 * (y + 1)].max() for y in range(1, 5)]]
    truth_build = np.concatenate([fut["first"]["replications"][list(TRUTH_BUILD)][:, :, yr]])
    jobs = {"P50 plan": q[0.5][None, :, yr], "P90 plan": q[0.9][None, :, yr], "Our plan": paths["nn"][:S_chosen, :, yr],
            "ETS plan": paths["ets"][:S_chosen, :, yr], "Ground-truth plan": truth_build}
    plans = {"Do nothing": fixed_plan(net, np.full(5, net["k0"])), "Expand the busiest FC": fixed_plan(net, busiest)}
    solved = solve_all([(D, net, game) for D in jobs.values()])
    plans.update(dict(zip(jobs, solved)))
    log("plans solved: " + ", ".join(f"{k} {v.get('seconds', 0)}s" for k, v in plans.items()))

    # ---- check on futures the plans never saw
    score = {}
    for fname in ("first", "second"):
        reps = fut[fname]["replications"][list(TRUTH_SCORE)]
        score[fname] = {}
        for name, p in plans.items():
            rows = evaluate_many(pool, p["K"], reps)
            real = _EV_local(net)(p["K"], fut[fname]["realised"])
            score[fname][name] = {"replications": summarise(p, rows, game), "realised": summarise(p, [real], game)}
        log(f"{fname} future scored")
    # Wait-and-see: the best plan for each scoring replication of the first future, knowing it in advance.
    ws_plans = solve_all([(D[None, :, yr], net, game) for D in fut["first"]["replications"][list(TRUTH_SCORE)]], limit=120)
    ws = [summarise(p, [_EV_local(net)(p["K"], D)], game)["expected"] for p, D in zip(ws_plans, fut["first"]["replications"][list(TRUTH_SCORE)])]
    log("wait-and-see done")

    # ---- ranges: capacity cost ratio and opening cost
    def ranged(costs_list: list[dict]) -> list[dict]:
        out_rows = []
        names = ("P50 plan", "P90 plan", "Our plan", "Ground-truth plan")
        solved_r = solve_all([(jobs[name], net, costs) for costs in costs_list for name in names])
        for ci, costs in enumerate(costs_list):
            row = {"rho": costs["rho"], "F": costs["F"], "plans": {}}
            for name in ("Do nothing", "Expand the busiest FC"):
                row["plans"][name] = {"expected": recost(plans[name], score["first"][name]["replications"]["second_stage"], costs)}
            for ni, name in enumerate(names):
                p = solved_r[ci * len(names) + ni]
                rows = evaluate_many(pool, p["K"], fut["first"]["replications"][list(TRUTH_SCORE)])
                sm = summarise(p, rows, costs)
                row["plans"][name] = {"expected": sm["expected"], "fill_rate": sm["fill_rate"], "sites_open": int(p["open"][:, -1].sum()),
                                      "capacity_year10": float(p["K"][:, -1].sum()), "plan": plan_payload(p)}
            out_rows.append(row)
            log(f"range point rho {costs['rho']:.3f} F {costs['F']:,.0f}")
        return out_rows

    rho_range = ranged([cost_terms(net, rho=r) for r in RHO_GRID])
    opening_range = ranged([cost_terms(net, opening_multiple=f) for f in OPENING_MULTIPLES])
    pool.close()

    def band(a: np.ndarray) -> dict:
        qq = np.quantile(a, [0.1, 0.5, 0.9], axis=0)
        return {"p10": np.round(qq[0]).tolist(), "p50": np.round(qq[1]).tolist(), "p90": np.round(qq[2]).tolist(), "mean": np.round(a.mean(axis=0)).tolist()}

    hist = df.ds.load(world)
    overview = {
        "model": {"id": "network-plan", "title": "Network plan"},
        "sites": {"ids": net["ids"], "names": net["names"], "landmass": net["landmass"], "population_m": net["population"].tolist()},
        "costs": {"delivery_local": net["delivery"].tolist(), "margin": net["margin"].tolist(), "network": net["network"],
                  "mult": net["mult"].tolist(), "keep": net["keep"].tolist(), "game": game, "k0": net["k0"],
                  "opening_share_of_price": OPENING_SHARE, "capacity_share_of_price": df.GAME_CAPACITY_SHARE},
        "settings": {"S_grid": S_GRID, "S_tolerance": S_TOLERANCE, "out_of_sample_paths": list(OUT_OF_SAMPLE), "gap": GAP,
                     "rho_grid": RHO_GRID, "opening_multiples": OPENING_MULTIPLES, "truth_build": [0, 99], "truth_score": [100, 199], "seed": SEED},
        "demand": {"forecast": {n: band(paths["nn"][:, i]) for i, n in enumerate(net["names"])},
                   "forecast_ets": {n: band(paths["ets"][:, i]) for i, n in enumerate(net["names"])},
                   "history_year5": hist.units.sum(axis=1)[:, 48:60].round().tolist()},
        "scenario_check": s_check, "chosen_S": S_chosen,
        "plans": {k: plan_payload(v) for k, v in plans.items()},
        "evaluation": {f: {k: {kk: {x: y for x, y in vv.items() if x != "second_stage"} for kk, vv in v.items()} for k, v in s.items()} for f, s in score.items()},
        "wait_and_see": {"expected": float(np.mean(ws)), "per_replication": ws},
        "ranges": {"capacity_cost_ratio": rho_range, "opening_cost": opening_range},
        "b_check": b_check(b, world),
        "runtime_s": round(time.time() - t0, 1),
    }
    (out / "overview.json").write_text(json.dumps(overview, separators=(",", ":")))
    log("done")


_EVL: dict = {}


def _EV_local(net: dict) -> Evaluator:
    if "ev" not in _EVL:
        _EVL["ev"] = Evaluator(net)
    return _EVL["ev"]


def b_check(b: dict, world: Path) -> dict:
    """B's mean forecast of the network total from these paths' source, as bias by year against the realised
    first future, to compare with B's committed `bias_mean` (a reproduction check)."""
    actual = df.ds.load(world).units.sum(axis=(0, 1))[60:120].reshape(5, 12).sum(axis=1)
    out = {}
    for key in ("nn", "ets"):
        coherent = b[f"{key}_total"].astype(float).mean(axis=0).reshape(-1)[:60].reshape(5, 12).sum(axis=1)  # mean of the reconciled paths
        out[key] = {"bias_mean": ((coherent - actual) / actual).tolist()}
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
