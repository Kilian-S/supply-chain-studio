"""HTTP API around the engine. Runs live in memory; one process serves one local player.

Run with: uv run uvicorn scgame.api:app --reload
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .demand import daily_demand, demand_multiplier, history
from .engine import DecisionError, Engine
from .scenario import Scenario, ShipMode

SCENARIO_DIR = Path(os.environ.get("SCGAME_SCENARIOS", Path(__file__).resolve().parents[3] / "scenarios"))
MODELS_DIR = Path(os.environ.get("SCGAME_MODELS", Path(__file__).resolve().parents[3] / "data" / "models"))

app = FastAPI(title="Supply Chain Game API", version="0.1.0")
RUNS: dict[str, Engine] = {}


# =================================================================================================
# Request bodies
# =================================================================================================

class NewRun(BaseModel):
    scenario_id: str
    seed: int | None = None


class Step(BaseModel):
    days: int = 1
    until_day: int | None = None


# =================================================================================================
# Serialisation
# =================================================================================================

def _scenario_files() -> dict[str, Path]:
    out = {}
    for path in sorted(SCENARIO_DIR.glob("*.json")):
        out[path.stem] = path
    return out


def _load_scenario(scenario_id: str) -> Scenario:
    path = _scenario_files().get(scenario_id)
    if path is None:
        raise HTTPException(404, f"unknown scenario {scenario_id!r}")
    return Scenario.load(path)


def _run(run_id: str) -> Engine:
    engine = RUNS.get(run_id)
    if engine is None:
        raise HTTPException(404, f"unknown run {run_id!r}")
    return engine


def _state(run_id: str, engine: Engine) -> dict[str, Any]:
    sc = engine.scenario
    t = engine.t
    last = engine.records[-1].to_dict() if engine.records else None
    return {
        "run_id": run_id,
        "scenario": {
            "id": sc.id,
            "name": sc.name,
            "description": sc.description,
            "regions": [r.model_dump(mode="json") for r in sc.regions],
            "lanes": {k.value: v.model_dump(mode="json") for k, v in sc.lanes.items()},
            "economics": sc.economics.model_dump(mode="json"),
            "rules": sc.rules.model_dump(mode="json"),
            "timeline": sc.timeline.model_dump(mode="json"),
        },
        "day": engine.day,
        "time": t,
        "finished": engine.finished,
        "cash": engine.cash,
        "starting_cash": sc.start.cash,
        "policy": engine.policy.value,
        "factories": [
            {
                "region": f.region,
                "capacity": f.capacity,
                "online_at": None if f.online_at == float("-inf") else f.online_at,
                "operational": f.operational(t),
                "pending_capacity": [{"online_at": at, "amount": amount} for at, amount in f.pending_capacity],
                "batch": None if f.batch is None else {
                    "warehouse": f.batch.warehouse, "quantity": f.batch.quantity,
                    "mode": f.batch.mode.value, "start": f.batch.start, "end": f.batch.end,
                },
            }
            for f in sorted(engine.factories.values(), key=lambda f: f.region)
        ],
        "warehouses": [
            {
                "region": w.region,
                "online_at": None if w.online_at == float("-inf") else w.online_at,
                "operational": w.operational(t),
                "on_hand": w.on_hand,
                "en_route_truck": w.en_route_by_mode(ShipMode.TRUCK),
                "en_route_mail": w.en_route_by_mode(ShipMode.MAIL),
                "shipments": [
                    {"factory": s.factory, "quantity": s.quantity, "mode": s.mode.value,
                     "depart": s.depart, "arrive": s.arrive}
                    for s in sorted(w.inbound, key=lambda s: s.arrive)
                ],
                "serves": sorted(c for (wr, c) in engine.serves if wr == w.region),
            }
            for w in sorted(engine.warehouses.values(), key=lambda w: w.region)
        ],
        "routes": [
            {"factory": f, "warehouse": w, **s.model_dump(mode="json")}
            for (f, w), s in sorted(engine.routes.items())
        ],
        "last_record": last,
        "summary": engine.summary(),
        "events": [{**e.model_dump(mode="json"), "active": active} for e, active in engine.visible_events()],
    }


# =================================================================================================
# Endpoints
# =================================================================================================

@app.get("/api/scenarios")
def list_scenarios() -> list[dict[str, Any]]:
    out = []
    for scenario_id, path in _scenario_files().items():
        sc = Scenario.load(path)
        out.append({"id": scenario_id, "name": sc.name, "description": sc.description, "standard": sc.standard,
                    "regions": len(sc.regions), "events": len(sc.events),
                    "start_day": sc.timeline.start_day, "end_day": sc.timeline.end_day})
    # Standard scenarios first, in library order; then the user's own, by name.
    order = {"classic-network": 0, "classic-one-region": 1, "classic-network-imperial-notes": 2}
    return sorted(out, key=lambda s: (not s["standard"], order.get(s["id"], 9), s["name"]))


def _for_client(sc: Scenario) -> dict[str, Any]:
    """Scenario JSON with demand files relative to the scenario folder, as stored on disk."""
    data = sc.model_dump(mode="json")
    for spec in data["demand"].values():
        path = Path(spec["csv"])
        if path.is_absolute() and path.is_relative_to(SCENARIO_DIR.resolve()):
            spec["csv"] = str(path.relative_to(SCENARIO_DIR.resolve()))
    return data


def _from_client(data: dict[str, Any]) -> Scenario:
    try:
        sc = Scenario.model_validate(data)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    for spec in sc.demand.values():
        if not Path(spec.csv).is_absolute():
            spec.csv = str((SCENARIO_DIR / spec.csv).resolve())
        if not Path(spec.csv).is_file():
            raise HTTPException(422, f"demand file not found: {spec.csv}")
    return sc


@app.get("/api/scenarios/{scenario_id}")
def get_scenario(scenario_id: str) -> dict[str, Any]:
    return _for_client(_load_scenario(scenario_id))


@app.put("/api/scenarios/{scenario_id}")
def save_scenario(scenario_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """Create or overwrite one of the player's own scenarios (standard ones are read-only)."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", scenario_id):
        raise HTTPException(422, "scenario id: lower-case letters, digits and hyphens only")
    existing = _scenario_files().get(scenario_id)
    if existing is not None and Scenario.load(existing).standard:
        raise HTTPException(409, "standard scenarios are read-only; duplicate it under a new id")
    sc = _from_client({**data, "id": scenario_id, "standard": False})
    sc.save(SCENARIO_DIR / f"{scenario_id}.json", relative_to=SCENARIO_DIR)
    return _for_client(sc)


@app.delete("/api/scenarios/{scenario_id}")
def delete_scenario(scenario_id: str) -> dict[str, str]:
    path = _scenario_files().get(scenario_id)
    if path is None:
        raise HTTPException(404, f"unknown scenario {scenario_id!r}")
    if Scenario.load(path).standard:
        raise HTTPException(409, "standard scenarios cannot be deleted")
    path.unlink()
    return {"deleted": scenario_id}


@app.post("/api/scenarios/preview")
def preview_scenario(data: dict[str, Any]) -> dict[str, Any]:
    """What the event timeline does, before playing: demand per region (recorded vs with events) and
    the combined multipliers over the play window."""
    sc = _from_client(data)
    plain = daily_demand(sc.model_copy(update={"events": []}), sc.seed)
    shaped = daily_demand(sc, sc.seed)
    days = list(range(sc.timeline.start_day, sc.timeline.end_day + 1))
    probe = Engine(sc.model_copy(update={"events": sc.events}), orders=[])
    any_region = sc.regions[0].id
    return {
        "days": days,
        "baseline": {n: [plain[n].get(d, 0) for d in days] for n in plain},
        "with_events": {n: [shaped[n].get(d, 0) for d in days] for n in shaped},
        "demand_multiplier": {r.name: [demand_multiplier(sc, r.id, d) for d in days] for r in sc.regions},
        "multipliers": {
            item: [probe._cost_factor(item, d - 0.5, *[r.id for r in sc.regions]) for d in days]
            for item in ("production", "transport", "fulfilment", "holding")
        } | {
            "price": [probe._price(d - 0.5, any_region) / sc.economics.price_per_drum for d in days],
            "interest_rate": [probe._daily_rate(d) * 365 for d in days],
        },
    }


@app.post("/api/runs")
def new_run(body: NewRun) -> dict[str, Any]:
    engine = Engine(_load_scenario(body.scenario_id), seed=body.seed)
    run_id = uuid.uuid4().hex[:12]
    RUNS[run_id] = engine
    return _state(run_id, engine)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    return _state(run_id, _run(run_id))


@app.post("/api/runs/{run_id}/step")
def step(run_id: str, body: Step) -> dict[str, Any]:
    engine = _run(run_id)
    if body.until_day is not None:
        engine.run(until_day=body.until_day)
    else:
        for _ in range(max(body.days, 0)):
            if engine.finished:
                break
            engine.step_day()
    return _state(run_id, engine)


@app.post("/api/runs/{run_id}/decisions")
def decide(run_id: str, decision: dict[str, Any]) -> dict[str, Any]:
    engine = _run(run_id)
    if engine.finished:
        raise HTTPException(409, "the run has finished")
    try:
        engine.apply(decision)
    except (DecisionError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return _state(run_id, engine)


@app.get("/api/runs/{run_id}/records")
def records(run_id: str, since_day: int = 0) -> list[dict[str, Any]]:
    return [r.to_dict() for r in _run(run_id).records if r.day >= since_day]


@app.get("/api/runs/{run_id}/log")
def decision_log(run_id: str) -> list[dict[str, Any]]:
    return [{"day": d.day, "time": d.time, "kind": d.kind, **d.params} for d in _run(run_id).decisions]


@app.get("/api/runs/{run_id}/demand")
def demand(run_id: str) -> dict[str, Any]:
    """Daily demand and lost demand per region, as HQ plots them (completed days only)."""
    engine = _run(run_id)
    names = {r.id: r.name for r in engine.scenario.regions}
    played = {r.day: r for r in engine.records}
    hist = history(engine.daily_demand, engine.day - 1)
    days = sorted({d for series in hist.values() for d in series})
    out = {"days": days, "demand": {}, "lost": {}}
    for name, series in hist.items():
        out["demand"][name] = [series.get(d, 0) for d in days]
    for rid, name in names.items():
        # Before the player takes over, lost demand is unknown here; report it only for played days.
        out["lost"][name] = [played[d].lost.get(rid, 0) if d in played else None for d in days]
    return out


@app.get("/api/runs/{run_id}/cash-statement")
def cash_statement(run_id: str) -> dict[str, Any]:
    engine = _run(run_id)
    s = engine.summary()
    return {
        "starting_cash": engine.scenario.start.cash,
        "sources": {"revenue": s["revenue"], "interest": s["interest"]},
        "uses": {
            "capacity_and_buildings": s["capex"],
            "production": s["production_cost"],
            "inbound_shipping_truck": s["truck_cost"],
            "inbound_shipping_mail": s["mail_cost"],
            "outbound_shipping": s["fulfilment_cost"],
            "holding": s["holding_cost"],
        },
        "cash": engine.cash,
    }


# =================================================================================================
# Prepared model results (written by the scgame.ml pipelines)
# =================================================================================================

@app.get("/api/models")
def list_models() -> list[dict[str, Any]]:
    import json
    return [json.loads(card.read_text()) for card in sorted(MODELS_DIR.glob("*/card.json"))]


@app.get("/api/models/{model_id}/{path:path}")
def model_file(model_id: str, path: str):
    from fastapi.responses import FileResponse
    if not re.fullmatch(r"[a-z0-9-]+", model_id) or not re.fullmatch(r"[A-Za-z0-9_/.-]+\.json", path) or ".." in path:
        raise HTTPException(400, "bad path")
    target = (MODELS_DIR / model_id / path).resolve()
    if not target.is_file() or not target.is_relative_to(MODELS_DIR.resolve()):
        raise HTTPException(404, "not found")
    return FileResponse(target, media_type="application/json", headers={"Cache-Control": "max-age=300"})
