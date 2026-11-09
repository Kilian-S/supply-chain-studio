# Supply Chain Studio

A discrete-event simulator of a multi-region supply network, and a set of machine learning and
operations research models for the long-term questions a network planner faces. The simulator is
checked against a real run of the classroom game its rules come from. Replaying all 1,770
decisions of that run ends 1.6% from the live final cash of $25.95M.

<img src="docs/screenshots/simulator-network.png" alt="The simulator on day 1,201 of the replayed run" width="100%">

## Status

**Done**

- Simulator and scenario designer, checked against a real run
- Product clustering
- Bought together
- Demand five years out

**In progress**

- Demand in a new region, risk of split orders and a five-year network plan (models built, pages
  next)

**Planned**

- Instant what-if, a surrogate of the simulator

## The simulator

Two years of play across five regions, simulated day by day. Players set factory capacity, which
factory supplies which warehouse, order points and quantities, truck or mail, which warehouse
serves each market and the fulfilment policy, and try to finish with the most cash. The
event-driven engine is written from scratch in Python, with a test for every rule.

## Checked against a real run

The rules follow The Supply Chain Game, a classroom simulation by Responsive Learning
Technologies. My team won its network round during my MSc (first of ten teams, 96.88% fill rate),
largely with a bot we wrote to play around the clock, and the game's decision log became the test.
Replayed at their exact times on the real demand, all 1,770 decisions are accepted. Cash stays
within $6k of the live game on almost every day of the first ten weeks and ends 1.6% above it.
The game does not export order times within a day, so the odd batch shifts by a day and the paths
drift apart in the second year.

<img src="docs/screenshots/replay-cash.png" alt="Cash of the replay against the live game, with the gap below" width="100%">

```bash
cd backend && uv run python verification/run3_history_replay.py
```

## Scenarios and the designer

Nine standard scenarios range from the classic network to a demand shock, a factory fire and a
storm that cuts off the island. The designer edits costs, prices and lead times and places events
on a timeline, announced in advance or as surprises, and redraws demand and costs as you edit.

<img src="docs/screenshots/designer.png" alt="The scenario designer" width="100%">

## Models

The game has a single product, so the studio generates its own world, an online shop with 300
SKUs, 30,000 customers in eight regions and five years of orders. Its hidden structure, such as
each SKU's demand type and the bundles customers buy together, is known. Every model is therefore
scored against the ground truth as well as against the rules a planner would otherwise use, with
settings chosen on a validation set and results taken from a held-out test set.

<img src="docs/screenshots/models-home.png" alt="The models page, with three finished models and four greyed out" width="100%">

### Product clustering

Which SKUs sell alike, so each cluster can get its own stocking policy? k-means and Gaussian
mixtures group the SKUs on up to 16 features of demand, size, price and basket. The setting with
the highest purity recovers the hidden demand types almost exactly.

<img src="docs/screenshots/clustering-map.png" alt="SKUs in feature space at the highest-purity setting" width="100%">

Chosen by stocking cost instead, nine k-means clusters cost $2.03M on the test year, against $2.14M for
ABC/XYZ analysis and $2.19M for one policy for every SKU. Recovering the types and stocking
cheaply turn out to be different goals.

<img src="docs/screenshots/clustering-tradeoff.png" alt="Stocking cost against purity for every setting" width="100%">

### Bought together

Which SKUs are bought together, so they should share an FC? Each pair's co-occurrences are
compared with chance within region, basket size and quarter. A Poisson test with false discovery
rate control keeps the linked pairs, and Louvain community detection groups them into 45
communities, which recover all 45 bundles planted in the shop.

<img src="docs/screenshots/affinity-network.png" alt="The network of linked pairs, coloured by community" width="100%">

Stored together across two FCs, the communities split 34.7% of test-year orders, against 35.8%
by category and 39.4% at random. The gain is modest because most baskets mix unrelated items.

<img src="docs/screenshots/affinity-results.png" alt="Split orders under each assignment rule" width="100%">

### Demand five years out

How much will each region and category sell over five years, and how sure are we? Seasonal naive,
ETS, LightGBM and a small neural network forecast every level from the total down to each SKU in
each region, with bands from 1,000 reconciled sample paths.

<img src="docs/screenshots/forecast-fan.png" alt="The five-year forecast of total demand against the actual" width="100%">

At the total the learned models lead for three years, then ETS, which damps growth, catches up. As capacity
plans, LightGBM costs 4.7% more than a plan built from the ground truth, the chosen neural network
8.1% and rebuilding last year's peak 17.8%. On an earlier draw of the test years last year's peak
was cheapest, and the page says so.

<img src="docs/screenshots/forecast-results.png" alt="Capacity cost of each plan over the ground-truth plan" width="100%">

## Roadmap

- **Demand in a new region.** A regression on region traits for markets with no history.
- **Risk of split orders.** A calibrated gradient-boosted classifier for orders and SKUs under a
  planned network.
- **Network plan.** A multi-period mixed-integer programme for where to open or expand FCs over
  five years.
- **Instant what-if.** A surrogate of the simulator that answers in milliseconds.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Node.js.

```bash
git clone https://github.com/Kilian-S/supply-chain-studio.git
cd supply-chain-studio
./dev.sh            # API on :8000, web app on http://localhost:5173, Ctrl+C stops both
```

Model outputs are committed in `data/models/`, so every page works from a plain checkout. To
rebuild a model, generate the shop first (about three seconds, identical every time). On macOS,
LightGBM also needs `brew install libomp`.

```bash
cd backend
uv run python -m scgame.world.generator ../data/world_v1 7
uv run python -m scgame.ml.sku_segmentation ../data/world_v1 ../data/models/sku-segmentation
```

## Verification

```bash
(cd backend && uv run pytest && uv run ruff check src)
(cd frontend && npx tsc --noEmit -p tsconfig.app.json && npx eslint src/models && npx vitest run)
```

228 Python tests cover every engine rule, the API, the replay of the real run and each model.
Every visible string on the model pages is checked by a wording test.

## Layout

| Path | Contents |
|---|---|
| `backend/src/scgame/` | Engine, scenarios, demand, history import and the FastAPI app |
| `backend/src/scgame/world/` | Generator of the simulated shop |
| `backend/src/scgame/ml/` | The models and the simulations that price their decisions |
| `backend/verification/` | Replays of the real run and the chart above |
| `frontend/` | React and TypeScript app with the simulator, designer and model pages |
| `scenarios/` | Standard scenarios as JSON, plus the recorded demand |
| `data/models/` | Model outputs read by the pages |
| `docs/` | Plan, roadmap, and the design notes and reviews behind each page |
| `reference/` | Game mechanics notes and exports from the live run |

## About

Kilian Xhen Schwarz, MSc Transport, Imperial College London. Code and further work at
[github.com/Kilian-S](https://github.com/Kilian-S).
