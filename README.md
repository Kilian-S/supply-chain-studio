# Supply Chain Studio

A reimagined version of the Supply Chain Game (played at Imperial College, CIVE70108, 2026): a single
product network of factories and warehouses on the continent of Pangea. Phase 1 emulates the
original game faithfully, with a redesigned interface; scenario design, forecasting and optimisation
follow (see `PLAN.md`).

## Run it

```bash
./dev.sh            # API on :8000, web app on http://localhost:5173; Ctrl+C or closing the tab stops both
```

Requirements: [uv](https://docs.astral.sh/uv/) and Node.js.

## Layout

| Path | What |
|---|---|
| `backend/src/scgame/` | Scenario schema, demand, discrete-event engine, history import, FastAPI app |
| `backend/tests/` | Rule-by-rule engine tests, API test, replay of the real Run 3 against live data |
| `backend/verification/` | Scripts replaying the real Run 3 decision log and the original G4 bot |
| `frontend/` | React + TypeScript web app (map, panels, charts, timeline) |
| `scenarios/` | Standard scenarios as JSON, plus the real demand data |
| `reference/` | Original game documents, mechanics notes (`GAME_MECHANICS.md`), live game exports |

## Fidelity

Replaying all 1,770 decisions of our real Run 3 on the real demand tracks the live game's cash to
within $1k for the first ten weeks and finishes within 1.6% of the live result ($25.95M). Details
in `reference/GAME_MECHANICS.md`.

```bash
cd backend && uv run pytest            # all tests
uv run python verification/run3_history_replay.py
```
