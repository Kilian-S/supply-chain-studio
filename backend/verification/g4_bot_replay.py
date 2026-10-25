"""Run the original G4 bot (the code that played the live Run 3) against the new engine.

The bot is imported read-only from the archive and talks to a small adapter that implements the
controller interface it used against the live site. If the engine follows the game's rules, the
outcome should be close to the real result: final cash $25,952,540.52, 3,655 drums lost
(coursework report, table 10).

Usage: uv run python verification/g4_bot_replay.py [--seed N] [--seeds N]
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
G4 = Path("/Users/kilian/Kilian/ICL/Archive/Semester 2/Freight Transport and Logistics/Supply Chain Game/code/G4_automation")

os.environ["DISCORD_WEBHOOK_URL"] = ""
sys.path.insert(0, str(G4))
logging.disable(logging.CRITICAL)

from main import SupplyChainBot  # noqa: E402  (G4 bot)

from scgame.demand import history  # noqa: E402
from scgame.engine import DecisionError, Engine  # noqa: E402
from scgame.scenario import Scenario  # noqa: E402

REGION = {"Calopeia": 1, "Sorange": 2, "Tyran": 3, "Entworpe": 4, "Fardo": 5}
NAMES = list(REGION)

# Run 3 investments, reconstructed from the report's $6.8M capital investment (see PLAN notes).
BUILDS = [
    ("build_factory", dict(region=2, capacity=85)),
    ("build_factory", dict(region=5, capacity=20)),
    ("add_capacity", dict(region=1, amount=5)),
    ("build_warehouse", dict(region=2)),
    ("build_warehouse", dict(region=3)),
    ("build_warehouse", dict(region=5)),
]


def _region(label: str) -> int:
    return REGION[label.split("_")[0]]


class EngineController:
    """The G4 bot's controller interface on top of scgame.Engine."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def get_current_day(self) -> int:
        return self.engine.day

    def get_cash(self) -> float:
        return self.engine.cash

    def get_all_demand(self) -> pd.DataFrame:
        # Completed days only: the live game could not show today's demand in advance.
        hist = history(self.engine.scenario, self.engine.day - 1)
        days = sorted(hist["Calopeia"])
        return pd.DataFrame({"Day": days, **{n: [hist[n].get(d, 0) for d in days] for n in NAMES}})

    def get_warehouse_state(self, warehouse: str) -> dict:
        w = self.engine.warehouses.get(_region(warehouse))
        if w is None or not w.operational(self.engine.t):
            return {"inventory": 0, "mail": 0, "truck": 0}
        from scgame.scenario import ShipMode
        return {"inventory": w.on_hand, "mail": w.en_route_by_mode(ShipMode.MAIL),
                "truck": w.en_route_by_mode(ShipMode.TRUCK)}

    def get_capacity(self, factory: str) -> int:
        f = self.engine.factories.get(_region(factory))
        return int(f.capacity) if f and f.operational(self.engine.t) else 0

    def apply_factory_settings(self, factory: str, routes: list[dict]) -> None:
        f = _region(factory)
        if f not in self.engine.factories:
            return
        for r in routes:
            w = _region(r["warehouse"])
            if w not in self.engine.warehouses:
                continue
            self.engine.set_route(f, w, order_point=int(r["order_point"]), quantity=int(r["quantity"]),
                                  mode=r["shipping_method"].lower(), priority=int(r["priority"]))

    def set_system_mode(self, system: str, mode: str) -> None:
        pass


def run(scenario: Scenario, seed: int) -> Engine:
    engine = Engine(scenario, seed=seed)
    bot = SupplyChainBot(EngineController(engine))
    pending = list(BUILDS)

    def act(e: Engine) -> None:
        while pending:  # build in order as soon as cash allows
            kind, params = pending[0]
            try:
                getattr(e, kind)(**params)
            except DecisionError:
                break
            pending.pop(0)
        bot.run_cycle()

    engine.run(bot=act)
    return engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=str(ROOT / "scenarios" / "classic-network.json"))
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    scenario = Scenario.load(args.scenario)

    results = []
    for seed in range(args.seeds):
        engine = run(scenario, seed)
        s = engine.summary()
        results.append(s)
        build_days = sorted({d.day for d in engine.decisions if d.kind != "set_route"})
        print(f"seed {seed}: final cash ${s['final_cash']:,.0f}  lost {s['lost']:,}  "
              f"fill rate {s['fill_rate']:.2%}  leftover {s['leftover_inventory']}  builds on days {build_days}")
    cash = [r["final_cash"] for r in results]
    print(f"\nmean final cash ${statistics.mean(cash):,.0f} (live game: $25,952,541); "
          f"mean lost {statistics.mean(r['lost'] for r in results):,.0f} (live game: 3,655)")
    last = results[-1]
    for k in ("revenue", "production_cost", "truck_cost", "mail_cost", "fulfilment_cost",
              "holding_cost", "capex", "interest"):
        print(f"  {k:16s} ${last[k]:>14,.0f}")


if __name__ == "__main__":
    main()
