# Original game mechanics (our own summary)

Summary of the mechanics of the Supply Chain Game (developed at Kellogg, Northwestern, and published by
Responsive Learning Technologies). Written in our own words from
the course documents in `original_game/`, our G4 simulator code and the saved page snapshots. The
original's text, artwork and names are not reused in the app.

## World
- One product (drums of an industrial chemical). Four continental regions and one island.
- Before the play window, only the home region is served, from one factory and one warehouse there.
- At most one factory and one warehouse per region.

## Timeline
- History: days 1-730. Play window: days 730-1460. New markets tracked from day 640.
- All demand declines linearly to zero from day 1430 to day 1460; inventory and capacity are
  worthless at the end.
- Live game speed: 121 simulated days per real day (one day every 5.04 minutes).

## Demand archetypes
| Archetype | Behaviour |
|---|---|
| Seasonal | Strong yearly seasonality, no trend; random order sizes (mean 7-8) and random arrivals |
| Linear growth | Long-run mean grows linearly to day 1430; order size about 8; no seasonality |
| Ramp then stable | Starts day 640, reaches long-run mean about a month later; order size about 8 |
| Lumpy fixed-size | Customer reorders a fixed 250 units at random times (customer's own reorder point) |
| Island ramp | Like ramp then stable, on the island |

## Costs (course notes values; the generic description differs, see note)
| Item | Value |
|---|---|
| Price | $1,450 per drum |
| Production | $2,000 per batch + $900 per drum |
| Holding | $90 per drum per year (applies in transit and in the warehouse, not to WIP) |
| Capacity | $50,000 per drum/day; 90 days to come online; paid at start; cannot be reduced |
| New factory | $500,000 + $50,000 x capacity; 90 days |
| New warehouse | $100,000; 60 days; unlimited storage |
| Interest | 10% per year on cash, compounded daily |

The generic game description uses $1,500 + $1,000 per drum production and $100 holding; the course
notes override these. Our app makes every cost a scenario parameter.

## Transport (factory to warehouse) and fulfilment (warehouse to customer)
| Lane | Truck (up to 200 drums, same price if part-full) | Mail (per drum) |
|---|---|---|
| Same region | $15,000, 7 days | $150, 1 day |
| Across the continent | $20,000, 7 days | $200, 1 day |
| Continent to island | $45,000, 14 days | $400, 2 days |

- Finished batches ship immediately on completion.
- Fulfilment is always by mail at the lane's mail rate.
- An order must ship within one day or it is lost to a competitor (no backlog, no lost future demand).
- Orders may be partially filled, and one order may be filled from several warehouses.
- Fulfilment policy when several warehouses can serve an order: nearest or soonest.

## Production control (per factory, per destination warehouse)
- Reorder point: when the warehouse's inventory position (on hand + in transit) is at or below it, the
  factory produces a batch for that warehouse.
- Order quantity: batch size (integer drums).
- Shipping method: truck or mail.
- Priority 1-5: which warehouse's batch the factory makes first when several have triggered.
- One batch at a time per factory; batch duration = quantity / capacity.

## Confirmed on the live site (25 October 2026)
- Production triggers only when ALL hold: warehouse on hand + en route (WIP excluded) <= order point;
  quantity > 0; enough cash for production AND shipping; factory idle.
- Production and shipping costs are both charged when production starts.
- When a batch completes, the factory picks the triggering warehouse with the highest priority.
- Changing order point and quantity together is order-dependent in the original (a known pain point).
- "Nearest" fulfilment waits for stock en route to the nearest warehouse if it arrives within the
  one-day window; "soonest" ships from whichever warehouse can deliver first.
- The live factory page states $1,500 + $1,000 per drum production cost; the course notes say
  $2,000 + $900. **Confirmed: the live game charged $1,500 + $1,000 and $100 holding.** Replaying the
  G4 bot in our engine gives $26.8M with these values (live result $25.95M) and $39.4M with the notes'
  values.
- The factory screen lists one row per warehouse in the network (mode, order point, quantity, priority).
- Building a factory asks for total capacity; a warehouse needs no parameters.
- Data plots record state changes at fractional-day timestamps (events), not daily snapshots.

## Confirmed from the Run 3 downloads (history, demand, lost demand, cash, cash status)
- **Day convention:** the game labels a day by its end. "Day 731" covers time 730 to 731; the cash
  shown for day d is the cash at time d. Play runs from time 730.0 to 1460.0.
- **Market permissions:** "Satisfy demand in X from warehouse Y: allow/block" per customer region and
  warehouse. At the start the home warehouse serves only the home market; a newly built warehouse
  serves every region until blocked.
- **Capacity changes** are entered as the new total capacity; fractional values are allowed (85.04).
- **Default fulfilment policy** is changed to nearest on day 731.61 in our log; the default before is unknown.
- **Real demand vs our G4 files:** the G4 demand files for Sorange, Tyran, Entworpe and Fardo after
  day 730 were synthetic (from `multi_region_demand_simulator.py`). Only Calopeia was real. The real
  series is now `reference/data/run3_live_demand.xlsx` and `scenarios/data/classic_demand.csv`.
- **Cash statement** (whole game, from $5M at day 0) in `reference/data/run3_live_cash_status.csv`;
  holding is reported separately for warehouses and the pipeline (in transit).

## Engine verification (October 2026)
Replaying all 1,770 real decisions at their exact times on the real demand
(`backend/verification/run3_history_replay.py`, regression test `tests/test_run3_replay.py`):
- every decision accepted; cash within $1k of the live game on almost every day up to day 800;
- final cash $26.38M vs $25.95M live (+1.6%, five seeds, sd $33k); capacity spend within $842;
- remaining gap: order times within a day are random in our rebuild, so batches and big orders
  shift by a day; the replayed order points then meet slightly different stock levels, and the
  paths drift (most visibly in the endgame, where we lose about 450 Entworpe drums the live game
  served).

## What the G4 code tells us (G4_automation, final version)
- `network_engine.py`: daily-step simulator; arrivals, then fulfilment (nearest by cost), then batch
  completion and shipping, then reorder checks, then finance. Holding charged on warehouse stock only
  and production charged without the cash check: both differ from the live rules above.
- `network_config.py`: topology hard-coded for our Run 3 network (3 factories, 4 warehouses).
- `calculator.py`: "Always On" strategy (reorder point = pipeline + 1,000 so the factory always runs;
  shut down once pipeline covers remaining forecast demand; hard stop day 1440; tiered batch sizes
  200/400/600; Tyran floor 400 with priority switching).
- Forecasts used: Calopeia known series, Sorange linear regression, Tyran/Fardo moving average,
  Entworpe fixed 15 per day. Demand analysis used Syntetos-Boylan (ADI, CV squared), STL, Holt-Winters.
- Run 3 started at day 730 with $6,796,510 cash, Calopeia capacity 70 and 2,528 drums in stock
  (carried over from earlier play), not from the single-region defaults.
- `scenario_runner.py` already compared build schedules and capacities offline: an early version
  of the what-if idea, terminal and matplotlib only.

## Screens in the original (and what we will do instead)
| Original | Contents | Our replacement idea |
|---|---|---|
| Map with clickable regions | Region click opens a pop-up to build a factory or warehouse | Interactive map; side panel, no pop-up windows |
| Headquarters | Plot demand, lost demand, cash balance; fulfilment policy | Dashboard tab with linked charts |
| Factory window | WIP plot, add capacity, table of routes (point, quantity, mode, priority) | Factory panel with inline editing and a preview of the effect |
| Warehouse window | Inventory and shipment plots, inbound route table | Warehouse panel |
| Menu bar | Standings, history of actions, cash sources and uses, refresh, quit | Event log on the timeline, finance waterfall, live updates |

Pain points in the original: separate pop-up windows for every facility, plots in new tabs, manual
refresh, no undo or preview, no what-if, data only via Excel download.
