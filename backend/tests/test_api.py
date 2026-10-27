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


def test_design_save_preview_and_delete_a_scenario():
    sc = client.get("/api/scenarios/classic-network").json()
    assert sc["standard"] and sc["demand"]["Calopeia"]["csv"] == "data/classic_demand.csv"
    assert client.put("/api/scenarios/classic-network", json=sc).status_code == 409

    sc["name"] = "Test shock"
    sc["events"] = [{"id": "s", "name": "Shock", "kind": "demand", "start_day": 900, "factor": 0.5}]
    preview = client.post("/api/scenarios/preview", json=sc).json()
    i = preview["days"].index(950)
    assert preview["demand_multiplier"]["Tyran"][i] == 0.5 and preview["demand_multiplier"]["Tyran"][0] == 1

    try:
        saved = client.put("/api/scenarios/zz-test-shock", json=sc).json()
        assert saved["id"] == "zz-test-shock" and not saved["standard"]
        listed = {s["id"]: s for s in client.get("/api/scenarios").json()}
        assert listed["zz-test-shock"]["events"] == 1
        run = client.post("/api/runs", json={"scenario_id": "zz-test-shock"}).json()
        assert run["events"][0]["name"] == "Shock" and not run["events"][0]["active"]
    finally:
        assert client.delete("/api/scenarios/zz-test-shock").status_code == 200

    sc["events"][0]["end_day"] = 800  # ends before it starts
    assert client.post("/api/scenarios/preview", json=sc).status_code == 422
