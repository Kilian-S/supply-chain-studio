# Supply Chain Game Studio (working title)

Plan, last updated 27 October 2026. Status: engine verified against the live game; web app with play mode and scenario designer built; generated worlds, forecasting and optimisation to come.

## 1. Purpose

A portfolio project in operations research. It must show,
live on a screen share, that Kilian can take a strategic supply chain question, build the model behind
it, and present a clear answer to senior decision makers.

It is a reimagined, expanded version of the Supply Chain Game played at Imperial (CIVE70108, 2026),
which Group 4 won in the network round (1st of 10, 96.88% fill rate). The original game was developed
at Kellogg and is published by Responsive Learning Technologies; see `reference/GAME_MECHANICS.md`.

Strategic questions for senior decision makers drive the design (for example, how
expanding one FC changes the share of orders it can fulfil on its own). The app is therefore built
around scenarios and what-if comparison, not only around playing the game.

## 2. Decisions made

| Topic | Decision |
|---|---|
| Phase 1 focus | **The simulation and the game.** Forecasting and optimisation (ML/OR) come later |
| Platform | Web app: Python FastAPI backend, React + TypeScript frontend |
| Players | Single player; the admin role is a separate "Scenario designer" mode, no accounts |
| Clock | Player-controlled: pause, play at a chosen speed, step a day, fast-forward to a day |
| Admin changes | Designed **before** the run as a scenario with a future event timeline; saved and loaded; nothing is edited during a run |
| Scenario library | A set of standard scenarios ships with the app (section 4.3) |
| World | Single product (chemical drums); original region names kept for this private version |
| Regions | Classic five-region map plus procedurally generated worlds and a world editor |
| Look and feel | C "Pangea hybrid" from the mood board: pastel illustrated map, sober analytics panels |
| Demo format | Live screen share from the Mac, with a scripted demo path |
| Timeline | 2-3 weeks; presentation at least 2 weeks away |
| Repository | Private GitHub repository |
| Faithfulness | Rules checked against the live site, not a 100% reverse engineering; interface redesigned freely |
| Tools | Free and open source only |

## 3. Three layers

1. **Scenario** (designed in advance, by the admin): world, economics, rules, demand, starting state
   and the **event timeline**. A versioned JSON file, validated against a schema.
2. **Player decisions** (during play): build factories and warehouses, add capacity, set reorder
   point, quantity, mode and priority per route, choose the fulfilment policy. Each is logged with
   its day.
3. **Run state** (computed): stock, shipments, WIP, cash, KPIs. Never edited directly.

A run is fully reproducible from scenario + seed + decision log. That allows replay, branching from
any past day ("what if I had built Tyran on day 800?") and fair comparison between strategies.

## 4. Scenario design

### 4.1 What a scenario contains
- **World:** regions (name, map shape, landmass or island), facility limits per region.
- **Lanes:** cost and lead time per lane type and mode; truck capacity. By default derived from the
  region pair (same region, across the continent, across water).
- **Economics:** price, production fixed and variable cost, holding cost, interest, capacity cost,
  factory and warehouse build cost and build time.
- **Rules:** lost-sale window, allowed fulfilment policies, cash check on production, whether
  holding cost applies in transit.
- **Demand:** per region an archetype (seasonal, linear growth, ramp then stable, lumpy fixed-size,
  none) with parameters, order size distribution and a random seed; the end-of-life decline.
- **Timeline:** history start, play start and end.
- **Starting state:** cash, facilities, capacities, stock, route settings.
- **Event timeline** (4.2).

### 4.2 Event timeline (designed before the run)
Each event has a start day, optional end day, shape (step or ramp) and scope:

| Event type | Example |
|---|---|
| Demand change | Day 1095: all regions -30%, permanently |
| Demand surge | Day 900-1000: Sorange ramps up +50% |
| Cost change | Day 1000: production and transport costs +25% |
| Lane disruption | Day 1000-1060: island lanes closed, or lead times doubled |
| Facility outage | Day 950-980: Calopeia factory produces nothing |
| Price change | Day 1200: selling price -10% |
| Interest change | Day 900: interest rate 10% to 4% |
| Market opening | Day 900: a new region starts ordering |
| Early end of life | Decline starts on day 1300 instead of 1430 |

Visibility: the designer marks each event as **announced** (on the player's timeline from day one)
or **surprise** (appears as a news item on the day it happens). All four event families (demand,
cost and price, disruptions, market and lifecycle) are in the first version.

### 4.3 Standard scenario library
1. **Imperial 2026 network run**: costs from the Imperial notes, Run 3 starting state.
2. **Original defaults**: the generic game values ($1,500 + $1,000 production, $100 holding).
3. **One region**: home region only, as in the single-region stage.
4. **Demand shock**: after one year in play, demand falls 30% everywhere.
5. **Boom**: the growth market grows twice as fast.
6. **Cost inflation**: production and transport costs rise 25% halfway through.
7. **Island cut off**: island lanes closed for 60 days.
8. **Factory fire**: the home factory is down for 30 days.
9. **Early obsolescence**: the end-of-life decline starts 130 days early.
10. **Generated world**: a random continent from a seed.

Scenarios are saved as JSON files; users can duplicate a standard scenario and edit it.

## 5. Simulation engine

- Built new in Python, generalised from the G4 `network_engine.py`, for any number of regions,
  factories and warehouses.
- Order-level demand: orders arrive at random with random sizes, generated from the scenario's
  demand model and seed.
- Rules from the live site (see `reference/GAME_MECHANICS.md`): production triggers on on-hand +
  en route (WIP excluded) <= order point, quantity > 0, enough cash, factory idle; production and
  shipping paid at start; priority picks the warehouse when a batch completes; nearest and soonest
  fulfilment, including waiting for stock arriving within the window; partial fills across
  warehouses; holding cost in transit.
- Daily steps with ordered sub-steps; events applied at the start of their day.
- Deterministic for a given seed; fast enough for many runs in a few seconds, for later Monte Carlo.
- Validation: reproduce the G4 Run 3 strategy and compare with our actual result
  ($25.95M final cash in the live game) as a sanity check.

## 6. Front end (phase 1)
- **Play view:** illustrated map, facilities, animated shipments, timeline scrubber with events,
  KPI header, side panels for factories and warehouses with inline editing (no pop-up windows).
- **Scenario designer:** world, economics, demand and the event timeline (a Gantt-style track),
  save and load, start from a standard scenario.
- **Results:** cash, demand vs fulfilled, lost sales, inventory by facility, finance breakdown,
  decision and event log; compare two runs side by side.

## 7. Later phases (not now)
- Forecast lab (baselines, LightGBM, neural net, backtests).
- Network design optimisation (MILP with HiGHS) and a what-if studio with Monte Carlo.
- Benchmark bots, including our G4 "Always On" strategy.

## 8. Architecture

```
imperial supply chain game/
  backend/            FastAPI app
    sim/              simulation engine
    scenario/         schema, standard library, procedural generator
    api/              REST + WebSocket for live play
  frontend/           Vite + React + TypeScript
  scenarios/          saved scenario JSON files
  reference/          original game docs, real demand data, mechanics notes
  tests/
```

## 9. Milestones (phase 1)

| Step | Deliverable |
|---|---|
| 1 | Scenario schema, standard library, demand generator, simulation engine with tests |
| 2 | API and Play view (map, panels, clock) |
| 3 | Scenario designer with the event timeline, save and load |
| 4 | Results and run comparison, polish, demo path |

Mechanics reference: `reference/GAME_MECHANICS.md`.
