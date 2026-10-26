from fastapi.testclient import TestClient

from scgame.api import app

client = TestClient(app)


def test_scenarios_listed():
    ids = {s["id"] for s in client.get("/api/scenarios").json()}
    assert {"classic-network", "classic-one-region"} <= ids


def test_play_a_run():
    state = client.post("/api/runs", json={"scenario_id": "classic-network", "seed": 1}).json()
    run = state["run_id"]
    assert state["day"] == 731 and state["scenario"]["regions"][0]["map"]["outline"]

    state = client.post(f"/api/runs/{run}/decisions", json={"kind": "build_warehouse", "region": 3}).json()
    assert any(w["region"] == 3 and not w["operational"] for w in state["warehouses"])

    bad = client.post(f"/api/runs/{run}/decisions", json={"kind": "build_warehouse", "region": 3})
    assert bad.status_code == 400 and "already has a warehouse" in bad.json()["detail"]

    state = client.post(f"/api/runs/{run}/step", json={"days": 5}).json()
    assert state["day"] == 736 and state["last_record"]["day"] == 735
    assert len(client.get(f"/api/runs/{run}/records").json()) == 5
    assert client.get(f"/api/runs/{run}/log").json()[0]["kind"] == "build_warehouse"

    demand = client.get(f"/api/runs/{run}/demand").json()
    assert demand["days"][-1] == 735 and len(demand["demand"]["Calopeia"]) == 735

    statement = client.get(f"/api/runs/{run}/cash-statement").json()
    assert statement["uses"]["capacity_and_buildings"] == 100_000
