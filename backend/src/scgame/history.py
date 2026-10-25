"""Import the original game's transaction history (History > download) as a decision log."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .engine import DecisionError, Engine


@dataclass
class TimedDecision:
    time: float
    decision: dict[str, Any]
    source: str


_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Order point: (\w+) factory to (\w+) warehouse\."), "order_point"),
    (re.compile(r"Order quantity: (\w+) factory to (\w+) warehouse\."), "quantity"),
    (re.compile(r"Order priority: (\w+) factory to (\w+) warehouse\."), "priority"),
    (re.compile(r"Shipping from (\w+) factory to (\w+) warehouse\."), "mode"),
    (re.compile(r"Schedule factory capacity change in (\w+)"), "capacity"),
    (re.compile(r"Schedule warehouse in (\w+)"), "warehouse"),
    (re.compile(r"Satisfy demand in (\w+) from (\w+) warehouse\."), "service"),
    (re.compile(r"Change ful+fil+ment policy\."), "policy"),
]


def _number(value: Any) -> float:
    return float(str(value).replace(",", ""))


def parse_row(time: float, parameter: str, value: Any, region_ids: dict[str, int]) -> TimedDecision:
    text = re.sub(r"\s+", " ", str(parameter)).strip()
    for pattern, kind in _PATTERNS:
        m = pattern.fullmatch(text)
        if not m:
            continue
        ids = [region_ids[name] for name in m.groups()]
        if kind in ("order_point", "quantity", "priority"):
            d = {"kind": "set_route", "factory": ids[0], "warehouse": ids[1], kind: int(_number(value))}
        elif kind == "mode":
            d = {"kind": "set_route", "factory": ids[0], "warehouse": ids[1], "mode": str(value).strip().lower()}
        elif kind == "capacity":
            d = {"kind": "set_capacity", "region": ids[0], "total": _number(value)}
        elif kind == "warehouse":
            d = {"kind": "build_warehouse", "region": ids[0]}
        elif kind == "service":
            d = {"kind": "set_service", "region": ids[0], "warehouse": ids[1],
                 "allowed": str(value).strip().lower() == "allow"}
        else:
            d = {"kind": "set_policy", "policy": str(value).strip().lower()}
        return TimedDecision(time, d, text)
    raise ValueError(f"unrecognised history entry: {text!r}")


def read_history_xlsx(path: str, region_ids: dict[str, int]) -> list[TimedDecision]:
    import pandas as pd  # only needed for importing exports

    frame = pd.read_excel(path)
    frame.columns = ["day", "parameter", "value"]
    out = [parse_row(_number(r.day), r.parameter, r.value, region_ids) for r in frame.itertuples()]
    return sorted(out, key=lambda d: d.time)


def replay(engine: Engine, decisions: list[TimedDecision]) -> list[tuple[TimedDecision, str]]:
    """Apply decisions at their exact times, then play to the end. Returns rejected decisions."""
    rejected: list[tuple[TimedDecision, str]] = []
    for item in decisions:
        while not engine.finished and engine.day < int(item.time) + 1:  # day d covers [d - 1, d)
            engine.step_day()
        if engine.finished:
            break
        engine.advance_to(max(item.time, engine.t))
        try:
            engine.apply(item.decision)
        except DecisionError as exc:
            rejected.append((item, str(exc)))
    engine.run()
    return rejected
