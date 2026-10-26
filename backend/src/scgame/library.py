"""Standard scenarios. `python -m scgame.library <dir>` writes them as JSON files.

Values marked VERIFY are best estimates from our G4 code and still need checking against the live
game's data (cash and inventory at day 730).
"""

from __future__ import annotations

import sys
from pathlib import Path

from .scenario import (
    Economics, LaneTerms, LaneTier, RecordedDemand, Region, RegionMap, RouteSettings, Rules, Scenario,
    StartFactory, StartRoute, StartState, StartWarehouse,
)

CALOPEIA, SORANGE, TYRAN, ENTWORPE, FARDO = 1, 2, 3, 4, 5

def _map(outline, label, factory, warehouse, headquarters=None) -> RegionMap:
    """Our own drawing of the classic continent, scaled from a 900 x 640 sketch to 1000 x 700."""
    def s(p):
        return (round(p[0] * 1000 / 900, 1), round(p[1] * 700 / 640, 1))
    return RegionMap(outline=[s(p) for p in outline], label=s(label), factory=s(factory),
                     warehouse=s(warehouse), headquarters=s(headquarters) if headquarters else None)


CLASSIC_REGIONS = [
    Region(id=CALOPEIA, name="Calopeia", map=_map(
        [(118, 128), (170, 92), (250, 78), (330, 96), (410, 92), (430, 196), (478, 300), (306, 338),
         (138, 302), (114, 220)],
        label=(255, 150), factory=(230, 250), warehouse=(340, 235), headquarters=(170, 180))),
    Region(id=SORANGE, name="Sorange", map=_map(
        [(410, 92), (480, 74), (570, 70), (650, 84), (718, 104), (748, 150), (742, 226), (612, 250),
         (478, 300), (430, 196)],
        label=(575, 125), factory=(530, 190), warehouse=(650, 185))),
    Region(id=TYRAN, name="Tyran", map=_map(
        [(612, 250), (742, 226), (770, 300), (768, 392), (744, 456), (706, 500), (640, 516), (566, 522),
         (520, 500), (486, 476), (478, 300)],
        label=(625, 440), factory=(560, 350), warehouse=(670, 355))),
    Region(id=ENTWORPE, name="Entworpe", map=_map(
        [(138, 302), (306, 338), (478, 300), (486, 476), (420, 486), (340, 474), (262, 452), (198, 420),
         (160, 368)],
        label=(320, 435), factory=(280, 375), warehouse=(400, 375))),
    Region(id=FARDO, name="Fardo", landmass="island", map=_map(
        [(64, 486), (118, 470), (160, 492), (170, 548), (132, 590), (80, 594), (52, 556)],
        label=(108, 615), factory=(95, 515), warehouse=(130, 560))),
]

CLASSIC_LANES = {
    LaneTier.SAME_REGION: LaneTerms(truck_cost=15_000, truck_days=7, mail_cost_per_drum=150, mail_days=1),
    LaneTier.SAME_LANDMASS: LaneTerms(truck_cost=20_000, truck_days=7, mail_cost_per_drum=200, mail_days=1),
    LaneTier.CROSS_WATER: LaneTerms(truck_cost=45_000, truck_days=14, mail_cost_per_drum=400, mail_days=2),
}

# Values the live game actually charged (factory page; confirmed by replaying the G4 bot, which lands
# within 3% of our real final cash). The Imperial course notes listed different numbers.
LIVE_ECONOMICS = Economics(batch_fixed_cost=1500, unit_cost=1000, holding_cost_per_drum_year=100)
IMPERIAL_NOTES_ECONOMICS = Economics(batch_fixed_cost=2000, unit_cost=900, holding_cost_per_drum_year=90)

DEFAULT_ROUTE = RouteSettings(order_point=500, quantity=250, mode="truck", priority=0)


def _classic_demand(csv: str, regions: list[str]) -> dict[str, RecordedDemand]:
    sizes = {"Calopeia": 7.5, "Sorange": 8, "Tyran": 8, "Fardo": 8}
    out = {}
    for name in regions:
        if name == "Entworpe":
            out[name] = RecordedDemand(csv=csv, fixed_order_size=250)
        else:
            out[name] = RecordedDemand(csv=csv, mean_order_size=sizes[name])
    return out


def classic_network(csv: str, economics: Economics, id_: str, name: str, description: str) -> Scenario:
    return Scenario(
        id=id_,
        name=name,
        description=description,
        regions=CLASSIC_REGIONS,
        lanes=CLASSIC_LANES,
        economics=economics,
        rules=Rules(),
        demand=_classic_demand(csv, [r.name for r in CLASSIC_REGIONS]),
        start=StartState(
            cash=6_796_510,  # VERIFY: Run 3 value from the G4 simulator config
            factories=[StartFactory(region=CALOPEIA, capacity=70)],  # VERIFY
            warehouses=[StartWarehouse(region=CALOPEIA, inventory=2528)],  # VERIFY
            routes=[StartRoute(factory_region=CALOPEIA, warehouse_region=CALOPEIA, settings=DEFAULT_ROUTE)],
        ),
    )


def one_region(csv: str) -> Scenario:
    return Scenario(
        id="classic-one-region",
        name="Classic: one region",
        description="The single-region stage: one factory and one warehouse in Calopeia. Decide capacity, "
                    "order point, quantity and transport mode.",
        regions=[CLASSIC_REGIONS[0]],
        lanes=CLASSIC_LANES,
        economics=LIVE_ECONOMICS,
        rules=Rules(buildable_regions=[], allowed_policies=["nearest"]),
        demand=_classic_demand(csv, ["Calopeia"]),
        start=StartState(
            cash=2_000_000,  # VERIFY: estimate from the G4 single-region simulator
            factories=[StartFactory(region=CALOPEIA, capacity=20)],
            warehouses=[StartWarehouse(region=CALOPEIA, inventory=500)],  # VERIFY
            routes=[StartRoute(factory_region=CALOPEIA, warehouse_region=CALOPEIA, settings=DEFAULT_ROUTE)],
        ),
    )


def standard_scenarios(csv: str) -> list[Scenario]:
    return [
        classic_network(
            csv, LIVE_ECONOMICS, "classic-network", "Classic network",
            "Five regions, one factory and warehouse in Calopeia at the start, as in the Imperial 2026 "
            "network run, with the costs the live game charged.",
        ),
        classic_network(
            csv, IMPERIAL_NOTES_ECONOMICS, "classic-network-imperial-notes", "Classic network (course notes costs)",
            "As the classic network, with the production and holding costs printed in the Imperial course "
            "notes ($2,000 + $900 per drum, $90 holding), which differ from what the live game charged.",
        ),
        one_region(csv),
    ]


def main(out_dir: str) -> None:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    csv = str(out / "data" / "classic_demand.csv")
    for scenario in standard_scenarios(csv):
        scenario.save(out / f"{scenario.id}.json", relative_to=out)
        print(f"wrote {scenario.id}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scenarios")
