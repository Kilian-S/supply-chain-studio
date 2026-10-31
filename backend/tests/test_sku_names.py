"""Cluster names on the SKU clustering page: unique, and a pattern word only where it is true.

The names are built without the ground truth; this test uses the hidden types only to check them,
for the three settings the page offers as presets.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

OUT = Path(__file__).resolve().parents[2] / "data" / "models" / "sku-segmentation"
WORD = {"steady": "Steady", "seasonal": "Seasonal", "trending": "Growing", "declining": "Fading",
        "intermittent": "Intermittent", "lumpy": "Bulk orders"}


def presets() -> list[tuple[int, str]]:
    o = json.loads((OUT / "overview.json").read_text())
    s = o["tradeoff"]["settings"]
    return [tuple(o["tradeoff"]["chosen"]), tuple(max(s, key=lambda r: r[1])[3:5]), tuple(max(s, key=lambda r: r[2])[3:5])]


@pytest.mark.skipif(not (OUT / "overview.json").exists(), reason="model outputs not built")
@pytest.mark.parametrize("which", [0, 1, 2], ids=["lowest cost", "highest purity", "highest ARI"])
def test_names_unique_and_pattern_words_true(which):
    mask, key = presets()[which]
    c = json.loads((OUT / "variants" / f"m{mask}.json").read_text())["clusterings"][key]
    truth = [WORD[s["true_type"]] for s in json.loads((OUT / "skus.json").read_text())]
    assert len(set(c["names"])) == len(c["names"])
    for k, name in enumerate(c["names"]):
        word = name.split(",")[0]
        if word in WORD.values():
            majority = Counter(t for t, label in zip(truth, c["labels"]) if label == k).most_common(1)[0][0]
            assert word == majority, f"{name}: mostly {majority}"
