"""Replay the exact Run 3 decision history (downloaded from the live game) in the engine.

Live result (coursework report, table 10): final cash $25,952,540.52, 117,305 drums demanded,
3,655 lost, 113,650 sold, 101 drums left over (all in Fardo).

Usage: uv run python verification/run3_history_replay.py [--seeds N] [--policy nearest|soonest]
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from scgame.engine import Engine
from scgame.history import read_history_xlsx, replay
from scgame.scenario import Scenario

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=str(ROOT / "scenarios" / "classic-network.json"))
    parser.add_argument("--history", default=str(ROOT / "reference" / "data" / "run3_history.xlsx"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--policy", default=None, help="fulfilment policy before the first change")
    args = parser.parse_args()

    scenario = Scenario.load(args.scenario)
    if args.policy:
        scenario.start.fulfilment_policy = args.policy
    ids = {r.name: r.id for r in scenario.regions}
    decisions = read_history_xlsx(args.history, ids)
    print(f"{len(decisions)} decisions from day {decisions[0].time} to {decisions[-1].time}")

    results = []
    for seed in range(args.seeds):
        engine = Engine(scenario, seed=seed)
        rejected = replay(engine, decisions)
        s = engine.summary()
        results.append((s, engine))
        print(f"seed {seed}: final cash ${s['final_cash']:,.0f}  demand {s['demand']:,}  lost {s['lost']:,}  "
              f"sold {s['fulfilled']:,}  leftover {s['leftover_inventory']}  rejected {len(rejected)}")
        for item, why in rejected[:5]:
            print(f"    rejected at {item.time}: {item.source} -> {why}")

    cash = [s["final_cash"] for s, _ in results]
    print(f"\nmean final cash ${statistics.mean(cash):,.0f}  (live $25,952,541, "
          f"diff {statistics.mean(cash) / 25_952_540.52 - 1:+.2%}); sd ${statistics.pstdev(cash):,.0f}")
    print(f"mean lost {statistics.mean(s['lost'] for s, _ in results):,.0f}  (live 3,655)")
    s, engine = results[0]
    lost_by_region: dict[int, int] = {}
    for r in engine.records:
        for k, v in r.lost.items():
            lost_by_region[k] = lost_by_region.get(k, 0) + v
    print("lost by region (seed 0):", {scenario.region(k).name: v for k, v in sorted(lost_by_region.items())})
    print("leftover by warehouse (seed 0):",
          {scenario.region(k).name: w.on_hand + w.en_route for k, w in engine.warehouses.items()})
    for k in ("revenue", "production_cost", "truck_cost", "mail_cost", "fulfilment_cost",
              "holding_cost", "capex", "interest"):
        print(f"  {k:16s} ${s[k]:>14,.0f}")


if __name__ == "__main__":
    main()
