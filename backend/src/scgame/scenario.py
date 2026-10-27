"""Scenario schema: everything the game designer fixes before a run starts.

A scenario is plain JSON validated by these models. The engine never mutates it; run state lives in
`engine.py` and player decisions are logged separately, so a run can be replayed exactly from
scenario + decision log.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class LaneTier(str, Enum):
    """How far apart two regions are; drives transport and fulfilment cost and time."""

    SAME_REGION = "same_region"
    SAME_LANDMASS = "same_landmass"
    CROSS_WATER = "cross_water"


class ShipMode(str, Enum):
    TRUCK = "truck"
    MAIL = "mail"


class FulfilmentPolicy(str, Enum):
    NEAREST = "nearest"
    SOONEST = "soonest"


class LaneTerms(BaseModel):
    """Transport terms for one lane tier. Customer fulfilment is always by mail."""

    truck_cost: float
    truck_days: float
    mail_cost_per_drum: float
    mail_days: float


Point = tuple[float, float]


class RegionMap(BaseModel):
    """Map geometry in a 1000 x 700 design space. Used only by the front end."""

    outline: list[Point]
    label: Point
    factory: Point
    warehouse: Point
    headquarters: Point | None = None


class Region(BaseModel):
    id: int
    name: str
    landmass: str = "continent"
    map: RegionMap | None = None


class Economics(BaseModel):
    price_per_drum: float = 1450.0
    batch_fixed_cost: float = 2000.0
    unit_cost: float = 900.0
    holding_cost_per_drum_year: float = 90.0
    holding_cost_in_transit: bool = True
    interest_rate_annual: float = 0.10
    # "nominal": daily rate = annual / 365. "effective": (1 + annual) ** (1/365) - 1.
    interest_compounding: Literal["nominal", "effective"] = "nominal"
    capacity_cost_per_unit: float = 50_000.0
    factory_build_cost: float = 500_000.0
    factory_build_days: float = 90.0
    capacity_build_days: float = 90.0
    warehouse_build_cost: float = 100_000.0
    warehouse_build_days: float = 60.0
    truck_capacity: int = 200


class Rules(BaseModel):
    lost_sale_window_days: float = 1.0
    max_factories_per_region: int = 1
    max_warehouses_per_region: int = 1
    allowed_policies: list[FulfilmentPolicy] = [FulfilmentPolicy.NEAREST, FulfilmentPolicy.SOONEST]
    # Regions where the player may build. None means every region.
    buildable_regions: list[int] | None = None
    allow_capacity_additions: bool = True
    # Production only starts if cash covers production and shipping.
    cash_check_on_production: bool = True
    # A newly built warehouse may serve every region until the player blocks some.
    new_warehouse_serves_all: bool = True


class RecordedDemand(BaseModel):
    """Daily demand totals per region, read from a CSV (columns: day, then one per region name)."""

    kind: Literal["recorded"] = "recorded"
    csv: str
    # Orders are rebuilt from the daily totals: either a fixed size, or random sizes with this mean.
    mean_order_size: float = 7.5
    fixed_order_size: int | None = None


class RouteSettings(BaseModel):
    """Per factory-warehouse pair, as in the original factory table."""

    order_point: int = 0
    quantity: int = 0
    mode: ShipMode = ShipMode.TRUCK
    priority: int = 0

    @model_validator(mode="after")
    def _non_negative(self) -> RouteSettings:
        if min(self.order_point, self.quantity, self.priority) < 0:
            raise ValueError("order point, quantity and priority must be integers >= 0")
        return self


class StartFactory(BaseModel):
    region: int
    capacity: float


class StartWarehouse(BaseModel):
    region: int
    inventory: int = 0
    # Customer regions this warehouse may serve at the start. None means its own region only
    # (the company starts by selling only to its home market).
    serves: list[int] | None = None


class StartRoute(BaseModel):
    factory_region: int
    warehouse_region: int
    settings: RouteSettings


class StartState(BaseModel):
    cash: float
    factories: list[StartFactory]
    warehouses: list[StartWarehouse]
    routes: list[StartRoute] = []
    fulfilment_policy: FulfilmentPolicy = FulfilmentPolicy.NEAREST


class Timeline(BaseModel):
    # Day d covers simulated time [d - 1, d), as in the original game. Play runs from the start of
    # start_day to the end of end_day; the classic game is played from time 730 to time 1460.
    start_day: int = 731
    end_day: int = 1460


EventKind = Literal[
    "demand", "market_opening", "end_of_life", "cost", "price", "interest", "lane", "outage",
]
CostItem = Literal["production", "transport", "fulfilment", "holding"]


class Event(BaseModel):
    """Something the designer schedules before the run, e.g. "day 1095: demand falls 30%".

    Days use the game's labels (day d covers [d - 1, d)); `end_day` is the last affected day, or
    None for a permanent change. With `ramp_days` the effect builds up linearly instead of in one step.
    `regions` limits the event to some regions (None means all). Which other fields matter depends
    on `kind`:

    * demand: `factor` multiplies demand (0.7 = -30%).
    * market_opening: no demand before `start_day`, then demand ramps up over `ramp_days`.
    * end_of_life: demand declines linearly from `start_day` to zero at `end_day`.
    * cost: `factor` multiplies the `cost_items` (in the regions involved).
    * price: `factor` multiplies the selling price for customers in `regions`.
    * interest: the annual rate becomes `rate`.
    * lane: lanes touching `regions` (other than within one region) are `closed`, or their lead
      times and costs are multiplied by `lead_time_factor` and `factor`.
    * outage: the `facility` in `regions` stops: a factory produces nothing (a batch in progress
      pauses), a warehouse ships nothing (it still receives stock).
    """

    id: str
    name: str
    kind: EventKind
    start_day: int
    end_day: int | None = None
    ramp_days: int = 0
    visibility: Literal["announced", "surprise"] = "announced"
    description: str = ""
    regions: list[int] | None = None
    factor: float = 1.0
    cost_items: list[CostItem] = ["production", "transport"]
    rate: float | None = None
    closed: bool = False
    lead_time_factor: float = 1.0
    facility: Literal["factory", "warehouse"] = "factory"

    @model_validator(mode="after")
    def _check(self) -> Event:
        if self.end_day is not None and self.end_day < self.start_day:
            raise ValueError(f"event {self.name!r} ends before it starts")
        if self.ramp_days < 0 or self.factor < 0 or self.lead_time_factor <= 0:
            raise ValueError(f"event {self.name!r}: ramp, factor and lead-time factor must be positive")
        if self.kind == "interest" and self.rate is None:
            raise ValueError(f"event {self.name!r}: an interest event needs a rate")
        if self.kind == "end_of_life" and self.end_day is None:
            raise ValueError(f"event {self.name!r}: end of life needs the day demand reaches zero")
        if self.kind == "outage" and self.end_day is None:
            raise ValueError(f"event {self.name!r}: an outage needs an end day")
        return self

    def covers(self, day: int) -> bool:
        return self.start_day <= day and (self.end_day is None or day <= self.end_day)

    def strength(self, day: int) -> float:
        """0 before the event, rising to 1 over the ramp, 1 while active, 0 after it ends."""
        if not self.covers(day):
            return 0.0
        if self.ramp_days <= 0:
            return 1.0
        return min(1.0, (day - self.start_day + 1) / self.ramp_days)

    def touches(self, *regions: int) -> bool:
        return self.regions is None or any(r in self.regions for r in regions)


class Scenario(BaseModel):
    id: str
    name: str
    description: str = ""
    standard: bool = False  # part of the shipped library: read-only, duplicate to edit
    regions: list[Region]
    lanes: dict[LaneTier, LaneTerms]
    economics: Economics = Field(default_factory=Economics)
    rules: Rules = Field(default_factory=Rules)
    demand: dict[str, RecordedDemand]  # keyed by region name
    start: StartState
    timeline: Timeline = Field(default_factory=Timeline)
    events: list[Event] = []
    seed: int = 0

    @model_validator(mode="after")
    def _consistent(self) -> Scenario:
        ids = {r.id for r in self.regions}
        names = {r.name for r in self.regions}
        if len(ids) != len(self.regions) or len(names) != len(self.regions):
            raise ValueError("region ids and names must be unique")
        if missing := set(LaneTier) - set(self.lanes):
            raise ValueError(f"lane terms missing for {sorted(t.value for t in missing)}")
        if unknown := set(self.demand) - names:
            raise ValueError(f"demand given for unknown regions {sorted(unknown)}")
        for f in self.start.factories:
            if f.region not in ids:
                raise ValueError(f"start factory in unknown region {f.region}")
        for w in self.start.warehouses:
            if w.region not in ids:
                raise ValueError(f"start warehouse in unknown region {w.region}")
        if self.timeline.end_day < self.timeline.start_day:
            raise ValueError("timeline ends before it starts")
        for e in self.events:
            if e.regions and (unknown_ids := set(e.regions) - ids):
                raise ValueError(f"event {e.name!r} names unknown regions {sorted(unknown_ids)}")
        if len({e.id for e in self.events}) != len(self.events):
            raise ValueError("event ids must be unique")
        return self

    # --- lookups -----------------------------------------------------------------------------

    def region(self, region_id: int) -> Region:
        for r in self.regions:
            if r.id == region_id:
                return r
        raise KeyError(region_id)

    def tier(self, a: int, b: int) -> LaneTier:
        if a == b:
            return LaneTier.SAME_REGION
        if self.region(a).landmass == self.region(b).landmass:
            return LaneTier.SAME_LANDMASS
        return LaneTier.CROSS_WATER

    def lane(self, a: int, b: int) -> LaneTerms:
        return self.lanes[self.tier(a, b)]

    # --- persistence -------------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        path = Path(path)
        scenario = cls.model_validate_json(path.read_text())
        # Demand CSV paths are stored relative to the scenario file.
        for spec in scenario.demand.values():
            csv = Path(spec.csv)
            if not csv.is_absolute():
                spec.csv = str((path.parent / csv).resolve())
        return scenario

    def save(self, path: str | Path, relative_to: str | Path | None = None) -> None:
        data = self.model_dump(mode="json")
        if relative_to is not None:
            base = Path(relative_to).resolve()
            for spec in data["demand"].values():
                csv = Path(spec["csv"])
                if csv.is_absolute():
                    spec["csv"] = str(csv.relative_to(base)) if csv.is_relative_to(base) else str(csv)
        Path(path).write_text(json.dumps(data, indent=2) + "\n")
