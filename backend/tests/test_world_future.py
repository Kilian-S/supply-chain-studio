"""World v1 years 6 to 10: years 1 to 5 stay exactly as they were, and the future rules behave."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scgame.world import future as fu
from scgame.world.generator import DAYS, WorldParams, daily_rates, expected_rates, generate

WORLD = Path(__file__).resolve().parents[2] / "data" / "world_v1"

# Content hashes of world v1 (seed 7) recorded on 2 November 2026, before the generator was refactored for
# years 6 to 10. The clustering and Bought together pages rest on exactly these files.
WORLD_V1_HASHES = {
    "skus": 2182386095474902484,
    "regions": 3571222299876388705,
    "customers": 3488399290198957917,
    "bundles": 3909404346419009422,
    "orders": 5720966384723560802,
    "order_lines": 1003551938293606911,
}


def content_hash(df: pd.DataFrame) -> int:
    # Floats rounded to 9 decimals: libm differs in the last bits between Linux and macOS.
    df = df.copy()
    for c in df.select_dtypes("float").columns:
        df[c] = df[c].round(9)
    return int(pd.util.hash_pandas_object(df, index=False).sum() % (1 << 63))


def test_years_1_to_5_are_unchanged(tmp_path):
    generate(tmp_path, WorldParams(seed=7))
    for name, h in WORLD_V1_HASHES.items():
        assert content_hash(pd.read_parquet(tmp_path / f"{name}.parquet")) == h, name


needs_future = pytest.mark.skipif(not (WORLD / "skus_future.parquet").exists(), reason="run scgame.world.future first")


@needs_future
def test_growth_ceiling_leaves_years_1_to_5_alone():
    cat = pd.read_parquet(WORLD / "skus_future.parquet")
    old = cat[~cat["new"]].copy()
    v1 = pd.read_parquet(WORLD / "skus.parquet")
    days = np.arange(1, DAYS + 1)
    assert np.array_equal(expected_rates(old, days), daily_rates(v1))


@needs_future
def test_growth_stops_after_the_drawn_age():
    cat = pd.read_parquet(WORLD / "skus_future.parquet")
    t = cat[cat["true_type"] == "trending"]
    assert (t["growth_stop_age"] >= fu.GROWTH_STOP_YEARS[0]).all()
    # Flat trend after the stop: the same day of year one year apart gives the same rate.
    s = t.iloc[[0]]
    stop_day = int(s["launch_day"].iloc[0] + s["growth_stop_age"].iloc[0] * 365) + 1
    days = np.array([stop_day + 10, stop_day + 375])
    days = days[days <= fu.END_DAY]
    if len(days) == 2:
        r = expected_rates(s, days)[0]
        assert r[0] == pytest.approx(r[1])


FUTURES = [WORLD, WORLD / "future2"]  # the first future and a second draw under the same rules
each_future = pytest.mark.parametrize("future", FUTURES, ids=["first", "second"])


def _skip_missing(future):
    if not (future / "skus_future.parquet").exists():
        pytest.skip("run scgame.ml.demand_forecast (or scgame.world.future) first")


@each_future
def test_catalogue_stays_near_300_and_new_skus_follow_the_rules(future):
    _skip_missing(future)
    meta = json.loads((future / "world_future.json").read_text())
    for year in range(6, 11):
        assert 270 <= meta["on_sale_at_year_start"][str(year)] <= 320
        assert len(meta["delisted"][str(year)]) == meta["launches"][str(year)]
    cat = pd.read_parquet(future / "skus_future.parquet")
    new = cat[cat["new"]]
    assert (new["sku_id"] > 300).all()
    assert new["launch_day"].between(DAYS + 1, fu.END_DAY).all()


@needs_future
def test_no_sku_explodes():
    cat = pd.read_parquet(WORLD / "skus_future.parquet")
    old = cat[~cat["new"]]
    y5 = expected_rates(old, np.arange(4 * 365 + 1, DAYS + 1)).sum(axis=1)
    late = expected_rates(old, np.arange(9 * 365 + 1, fu.END_DAY + 1)).sum(axis=1)
    on = y5 > 0
    assert (late[on] <= 3 * y5[on]).all()


@each_future
def test_future_orders_only_for_skus_on_sale(future):
    _skip_missing(future)
    cat = pd.read_parquet(future / "skus_future.parquet").set_index("sku_id")
    lines = pd.read_parquet(future / "order_lines_future.parquet")
    assert lines["day"].between(DAYS + 1, fu.END_DAY).all()
    life = cat.loc[lines["sku_id"], ["launch_day", "stop_day"]].to_numpy()
    assert ((lines["day"].to_numpy() >= life[:, 0]) & (lines["day"].to_numpy() < life[:, 1])).all()
    orders = pd.read_parquet(WORLD / "orders.parquet", columns=["order_id"])
    assert lines["order_id"].min() > orders["order_id"].max()


def test_the_second_future_is_a_new_draw_of_the_same_years():
    for f in FUTURES:
        _skip_missing(f)
    a, b = (pd.read_parquet(f / "skus_future.parquet") for f in FUTURES)
    assert a[~a["new"]][["sku_id", "base_rate", "launch_day"]].equals(b[~b["new"]][["sku_id", "base_rate", "launch_day"]])
    assert not a["growth_stop_age"].fillna(0).equals(b["growth_stop_age"].fillna(0))  # drawn again
    la, lb = (pd.read_parquet(f / "order_lines_future.parquet") for f in FUTURES)
    assert len(la) != len(lb) or not la.equals(lb)
