"""Regression test: replaying the real Run 3 decision log on the real demand must track the live game.

Live data in reference/data (downloaded from the original game after Run 3).
"""

import csv
from pathlib import Path

import pytest

from scgame.engine import Engine
from scgame.history import read_history_xlsx, replay
from scgame.scenario import Scenario

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "reference" / "data"


@pytest.fixture(scope="module")
def run():
    scenario = Scenario.load(ROOT / "scenarios" / "classic-network.json")
    decisions = read_history_xlsx(DATA / "run3_history.xlsx", {r.name: r.id for r in scenario.regions})
    engine = Engine(scenario, seed=0)
    rejected = replay(engine, decisions)
    live = {int(r["day"]): float(r["cash_thousands"]) * 1000
            for r in csv.DictReader(open(DATA / "run3_live_cash.csv"))}
    return engine, rejected, live


def test_every_real_decision_is_accepted(run):
    _, rejected, _ = run
    assert rejected == []


def test_cash_tracks_live_game_closely_for_the_first_ten_weeks(run):
    engine, _, live = run
    by_day = {r.day: r.cash for r in engine.records}
    gaps = [abs(by_day[d] - live[d]) for d in range(731, 801)]
    # Live cash is shown in thousands. Order times within a day are random in our rebuild, so a
    # batch or a large order can land a day earlier or later: allow a few such days, never more
    # than about one batch ($450k).
    assert sum(g <= 6_000 for g in gaps) >= 0.9 * len(gaps)
    assert max(gaps) < 450_000


def test_final_cash_within_three_percent_of_live(run):
    engine, _, _ = run
    assert engine.cash == pytest.approx(25_952_540.87, rel=0.03)
