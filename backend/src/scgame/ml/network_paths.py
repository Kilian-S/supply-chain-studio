"""Demand inputs and futures for model O (network plan).

Usage
  uv run python -m scgame.ml.network_paths b <world dir> <out dir>        model B's test paths (about an hour)
  uv run python -m scgame.ml.network_paths markets <world dir> <out dir>  new markets' futures (a few minutes)

Model B's test paths. B's chosen method (and ETS, for the sensitivity) refitted on years 1 to 5 with the
settings B chose on its validation set (read from B's committed `overview.json`), then B's own
`Probabilistic.upper_paths` with B's chosen width and reconciliation and B's random seed for that method, so the
paths are the ones B scores. Nothing in B's code changes; its functions are imported.

The new markets' futures. Three of model C's test regions (rule fixed in the proposal, section 2). Their years
6 to 10 are simulated with C's `region_sim.simulate_year` (imported unchanged) on the catalogue of years 6 to 10
of each future; 200 replications per future and one realised draw each.
"""

from __future__ import annotations

import os

# One thread per process for the linear algebra before numpy loads, as in B's committed run, so B's network
# refit reproduces B's paths. The imports below must follow, hence E402 (as in demand_forecast).
# ruff: noqa: E402, RUF100
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scgame.ml import demand_forecast as df
from scgame.ml import demand_series as ds

B_METHODS = ["Neural network", "ETS"]
NEW_MARKETS = [109, 119, 102]  # PROPOSAL section 2: two most populous continental and the most populous island test regions inside C's range
REPLICATIONS = 200
MARKET_STREAMS = {"first": (40, 41), "second": (42, 43)}  # (replications [7, a, r], realised draw [7, b, 0])
YEARS = list(range(6, 11))


# =================================================================================================
# Model B's test paths
# =================================================================================================

def b_fitted(name: str, b_overview: dict, ser: ds.Series, static: pd.DataFrame) -> df.Fitted:
    """B's method refitted on months 1 to 60 with the setting B chose on its validation set."""
    setting = dict(b_overview["validation"][name].get("setting") or {})
    dummy = df.Fitted(name, ser, df.TEST_FIT, model=SimpleNamespace(setting=setting) if setting else None)
    return df.refit(dummy, ser, static, df.TEST_FIT)


def write_b_paths(world_dir: str | Path, out_dir: str | Path, b_overview_path: str | Path, log=print) -> Path:
    world, out = Path(world_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bo = json.loads(Path(b_overview_path).read_text())
    w = ds.load(world)
    ser = ds.build_series(w, df.TEST_FIT)
    static = ds.static_features(ser, w)
    meta = ser.meta.iloc[ser.upper_idx].reset_index(drop=True)
    lev = meta["level"].to_numpy()
    rows = {"total": np.nonzero(lev == "Total")[0], "region": np.nonzero(lev == "Region")[0], "category": np.nonzero(lev == "Category")[0]}
    assert meta.loc[rows["region"], "region"].tolist() == list(range(1, 9))
    assert meta.loc[rows["category"], "category"].tolist() == list(range(8))
    arrays = {}
    for name in B_METHODS:
        t0 = time.time()
        f = b_fitted(name, bo, ser, static)
        pr = df.Probabilistic(f, df.TEST_YEARS, 200 + df.METHODS.index(name), static)
        best = bo["validation"][name]["best"]
        upper = pr.upper_paths(best["width_upper"], best["reconciliation"])  # (paths, 81, 60)
        mean = pr.mean[ser.upper_idx]  # (81, 60), the unreconciled mean forecast B's paths average
        key = "nn" if name == "Neural network" else "ets"
        for part, idx in rows.items():
            arrays[f"{key}_{part}"] = upper[:, idx, :].astype(np.float32)
            arrays[f"{key}_{part}_mean"] = mean[idx]
        log(f"B paths {name}: {time.time() - t0:.0f}s, fallback {f.fallback}")
    path = out / "b_paths.npz"
    np.savez_compressed(path, months=np.arange(61, 121), **arrays)
    return path


# =================================================================================================
# New markets: simulated futures (ground truth)
# =================================================================================================

def _market_world(world: Path, future: Path) -> dict:
    """C's simulation inputs, with the catalogue of years 6 to 10 of one future. A SKU's sales stop at its
    `stop_day` there (delisting), so `end_day` is set to it for C's bulk-order filter."""
    from scgame.ml import region_sim as rs
    from scgame.world.generator import CATEGORIES

    w = rs.prepare(world)
    cat = pd.read_parquet(future / "skus_future.parquet").reset_index(drop=True)
    cat["category_id"] = cat["category"].map({c.name: i for i, c in enumerate(CATEGORIES)})
    cat["end_day"] = np.minimum(cat["end_day"].to_numpy(), cat["stop_day"].to_numpy())
    w["skus"] = cat
    w["cat_of_sku"] = np.full(int(cat["sku_id"].max()) + 1, -1)
    w["cat_of_sku"][cat["sku_id"].to_numpy()] = cat["category_id"].to_numpy()
    reg = w["test_regions"]
    w["markets"] = rs.as_new(reg[reg["id"].isin(NEW_MARKETS)].set_index("id").loc[NEW_MARKETS].reset_index(), w["regions"])
    return w


_MW: dict = {}


def _init(world: str, future: str) -> None:
    _MW.clear()
    _MW.update(_market_world(Path(world), Path(future)))


def _one(args: tuple[int, int, int]) -> np.ndarray:
    """One draw of years 6 to 10 for the new markets: units (market, category, year, month)."""
    from scgame.ml import region_sim as rs

    a, r, seed = args
    rng = np.random.default_rng([seed, a, r])
    return np.stack([rs.simulate_year(_MW, _MW["markets"], y, rng).sum(axis=-1) for y in YEARS], axis=2)


def simulate_markets(world_dir: str | Path, out_dir: str | Path, workers: int = 4, log=print) -> Path:
    from multiprocessing import Pool

    world, out = Path(world_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = 7
    arrays = {}
    for name, (rep_stream, real_stream) in MARKET_STREAMS.items():
        future = world if name == "first" else world / df.SECOND_FUTURE
        t0 = time.time()
        with Pool(workers, initializer=_init, initargs=(str(world), str(future))) as pool:
            reps = pool.map(_one, [(rep_stream, r, seed) for r in range(REPLICATIONS)])
            real = pool.map(_one, [(real_stream, 0, seed)])[0]
        arrays[f"{name}_replications"] = np.stack(reps).astype(np.int32)  # (rep, market, category, year, month)
        arrays[f"{name}_realised"] = real.astype(np.int32)
        log(f"new markets, {name} future: {time.time() - t0:.0f}s")
    path = out / "markets.npz"
    np.savez_compressed(path, market_id=np.array(NEW_MARKETS), **arrays)
    return path


if __name__ == "__main__":
    what, world_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    if what == "b":
        bo = Path(world_dir).resolve().parents[0] / "models" / "demand-forecast" / "overview.json"
        print(write_b_paths(world_dir, out_dir, bo, log=lambda s: print(s, flush=True)), flush=True)
    elif what == "markets":
        print(simulate_markets(world_dir, out_dir, log=lambda s: print(s, flush=True)), flush=True)
