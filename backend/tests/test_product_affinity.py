"""Product affinity: the statistics, the packing and the delivery count on small hand-made cases, and
the recovered communities against the hidden bundles on the generated world."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from scgame.ml import product_affinity as pa

OUT = Path(__file__).resolve().parents[2] / "data" / "models" / "product-affinity"


def test_benjamini_hochberg_accepts_only_small_p_values():
    p = np.ones((4, 4))
    p[0, 1] = p[1, 0] = 1e-6
    p[2, 3] = p[3, 2] = 0.04
    acc = pa.bh_accept(p, 0.05)
    assert acc[0, 1] and acc[1, 0]
    assert not acc[2, 3]  # 0.04 fails 0.05 x rank / 6 pairs


def test_lift_is_one_for_independent_skus_and_high_for_a_bundle():
    # 400 two-SKU baskets in one region: SKUs 0 and 1 always together, 2 and 3 mixed at random.
    rows = [[0, 1]] * 100 + [[2, 3]] * 75 + [[2, 4]] * 75 + [[3, 4]] * 75 + [[4, 5]] * 75
    r = np.repeat(np.arange(len(rows)), 2)
    c = np.array(rows).ravel()
    X = sparse.csr_matrix((np.ones(len(c)), (r, c)), shape=(len(rows), 6))
    st = pa.pair_stats(X, np.ones(len(rows), dtype=int))
    assert st["lift"][0, 1] > 3
    assert st["p"][0, 1] < 1e-10
    assert st["count"][0, 2] == 0


def test_pack_keeps_units_whole_within_capacity():
    units = [np.arange(0, 6), np.arange(6, 10), np.arange(10, 13)]
    fc = pa.pack(units, 2, 20)
    for u in units:
        assert len(set(fc[u])) == 1
    assert np.bincount(fc[fc >= 0]).max() <= pa.capacity(20, 2)


def test_a_basket_spread_over_two_fcs_counts_as_split():
    X = sparse.csr_matrix(np.array([[1, 1, 0], [1, 0, 0], [0, 1, 1]], dtype=float))
    b = {"X": X, "Q": X.copy(), "region": np.array([1, 1, 1])}
    fc = np.array([0, 0, 1])
    out = pa.deliver(b, fc, 2, np.ones(3), np.ones((1, 1)), [1])
    assert out["split"] == 1 and out["parcels"] == 4 and out["multi"] == 2


@pytest.mark.skipif(not (OUT / "overview.json").exists(), reason="model outputs not built")
def test_communities_recover_the_hidden_bundles():
    check = json.loads((OUT / "overview.json").read_text())["truth"]["check"]
    assert check["communities"]["recall"] > 0.95
    assert check["communities"]["ari"] > 0.8
    assert check["communities"]["ari"] > check["categories"]["ari"]
