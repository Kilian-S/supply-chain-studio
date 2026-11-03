# ML and OR roadmap

Agreed 28 October 2026. Companion to `PLAN.md`. Nothing in this file is implemented yet; each
selected item is discussed in detail before it is built.

## 1. What the demo shows

1. **Reverse engineering of the Imperial supply chain game**, including scenario set-up and the
   event timeline. Done: engine verified against the live Run 3 data (see `reference/GAME_MECHANICS.md`).
2. **Autonomous bots**: how the G4 bots played the game 24/7 on AWS, and how the strategy was
   tested in an offline simulator first.
3. **The core: ML and OR for long-term supply chain decisions**, in a simulated world built on the
   game. Supervised and unsupervised learning, regression, classification, clustering, neural
   networks, and optimisation, aimed at questions a zooplus operations research team faces before
   they arise, not at day-to-day operations.

The Imperial game is the foundation; the playground grows well beyond it.

## 2. Decisions

| Topic | Decision |
|---|---|
| Product world | A diverse catalogue of all sorts of products ("Amazon-like"), with customers and multi-item baskets, still on Pangea |
| Selected items | A, B, C, H, J, K, O (section 6); each discussed before implementation |
| Models in the demo | Prepared: trained beforehand, with results cached; a retrain button where useful |
| Time | Enough time; no need to cut scope yet |
| Data additions | Allowed at our discretion, but every external source is shown and agreed first (section 4) |
| Day-to-day operations | Held constant by one standard replenishment bot, so strategic comparisons are like for like |

## 3. Data the world needs (agreed)

| Change | Why | Proposal |
|---|---|---|
| Longer horizon | Strategic models need history and a long future | About 5 years of history plus a 5-year planning horizon |
| Many products | Assortment, segmentation and affinity need a catalogue | 200-500 SKUs in several categories, with size, weight, margin and life cycle |
| Customers and baskets | Clustering, affinity and split orders need order lines | 20-50k customers; each order is a basket of SKUs |
| More regions and candidate sites | Network design needs real choices | 8-12 regions, several candidate FC sites, distance-based costs |
| Capacity in slots or volume | "Expand FC A or B?" needs a binding limit | Each FC has SKU slots and storage volume |
| Hidden drivers we control | Lets us validate every model against the truth | Region features (population, income), product attributes, price elasticity, seasonality, trends |
| Optional: lost sales hurt future demand | Puts a value on service level | Customers who are let down order less later |

## 4. Candidate external sources (to agree before use)

Approach: use real data to **calibrate** the synthetic generator (distributions and structure such
as basket sizes, category mixes, co-purchase patterns, intermittency, weights and dimensions,
freight against weight and distance), not to import it wholesale. The world stays fictional, the
true process stays known, and licences are respected (attribution kept in the repo).

| Source | What it holds | Would calibrate | Licence (as found) | Status |
|---|---|---|---|---|
| [Olist Brazilian E-Commerce](https://www.kaggle.com/olistbr/brazilian-ecommerce) | About 100k orders 2016-2018, 9 CSV files; per-item freight value, product weight and dimensions, categories, customer and seller locations | Freight cost against weight and distance; product size and weight distributions; multi-seller logistics | Not confirmed by search; check the Kaggle licence field | Licence to verify |
| [M5 Forecasting (Walmart)](https://arxiv.org/pdf/2310.12809) | 3,049 products, 10 stores in 3 US states, 1,941 days (2011-2016); prices, calendar and events; a product and store hierarchy | Long-horizon demand patterns, intermittency, price effects, hierarchical structure (B) | Competition data; terms to verify | Licence to verify |
| [Instacart 2017](https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2) | Over 3 million orders, over 200,000 users, about 50,000 products, aisles and departments | Basket sizes, reorder behaviour, co-purchase structure (K), customer segments | Non-commercial use, as-is; a biased sample of their data | Fine for a private demo |
| [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | UK online retailer, December 2009 to December 2011, about 1.07 million lines, invoices with customer IDs | Customer order frequency, basket composition, cancellations | CC BY 4.0 (UCI page) | Usable with attribution |
| [Amazon Reviews 2023 (McAuley Lab)](https://huggingface.co/datasets/minto1234/Amazon-Reviews-2023) | About 48 million items, 33 categories; metadata with categories and a `bought_together` field | A broad "all sorts" category tree and product attributes; bundles for affinity | Research dataset; terms to verify | Licence to verify |
| [Eurostat regional GDP, `nama_10r_2gdp`](https://db.nomics.world/Eurostat/nama_10r_2gdp) | GDP by NUTS 2 region, including per inhabitant | Realistic region features for C (new-region demand) | Eurostat open data | Usable with attribution |
| Eurostat regional population (NUTS 2 `demo_r_d2jan`, or NUTS 3 `demo_r_pjangrp3`) | Population by region | Region sizes and demand scale | Eurostat open data | Table code to confirm |
| [Eurostat road freight by distance class, `road_go_ta_dc`](https://baselight.app/u/eurostat/dataset/road_go_ta_dc) | Tonnes and tonne-kilometres by distance class | Shape of shipment distances; volumes, not prices | Eurostat open data | Volumes only |
| Freight and parcel rates (EUR per km or per parcel) | Real transport prices | Lane costs | Free sources not yet found | **Gap**: keep the game's cost ratios scaled by distance, or find a published tariff |

## 5. All options considered

**Supervised regression**

| # | Strategic question | Method |
|---|---|---|
| A | Expand FC A or FC B: effect on transport cost, service and split orders | Surrogate model: simulator generates labelled network variants; a neural net or LightGBM predicts KPIs instantly; SHAP; validated by simulation |
| B | Long-range demand by region, category and SKU, with uncertainty | Probabilistic hierarchical forecasting: baselines, LightGBM, global neural net; rolling-origin backtests |
| C | Demand of a new region or site (cold start) | Regression on region features, trained on existing regions |
| D | Demand of a new product (assortment and listing) | Regression on product attributes and similar products |
| E | Cost to serve per order, customer or region | Regression on basket and route features |
| F | Price elasticity for long-term pricing and promotion policy | Regression with price variation, causal angle |
| G | Supplier lead-time risk for safety-stock strategy | Quantile regression |

**Classification**

| # | Question | Method |
|---|---|---|
| H | Which orders, SKUs or regions are at risk of split shipments or stockouts under a planned network | Gradient-boosted classifier with calibrated probabilities |
| I | Does a service failure lose the customer; what is a point of fill rate worth | Churn classifier (needs the lost-sales-hurt-demand option) |

**Unsupervised**

| # | Question | Method |
|---|---|---|
| J | Which SKUs belong everywhere and which only centrally | SKU clustering on demand pattern (volume, CV squared, ADI, seasonality), size and margin; a policy per cluster |
| K | Which products are bought together, so they should be stored together | Association rules, product embeddings, community detection on the co-purchase graph |
| L | Customer and region segments for site choice and service tiers | Clustering on order frequency, basket mix, geography |
| M | Turning many simulated futures into a few planning scenarios | Scenario clustering (scenario reduction) for robust optimisation |
| N | Demand or cost anomalies | Isolation forest (operational) |

**Operations research**

| # | Question | Method |
|---|---|---|
| O | Where to open or expand FCs, and by how much, over 5 years | Multi-period MILP facility location, stochastic over scenarios |
| P | Which SKUs to list in which FC | MILP using affinities from K: minimise split orders and cost within slot limits |
| Q | Where to hold safety stock in the network | Multi-echelon inventory optimisation |
| R | Tuning policy parameters | Simulation optimisation (Bayesian optimisation) |

**Ambitious extras**

- Graph neural network surrogate that predicts KPIs for any network layout.
- Reinforcement learning for replenishment (links to the master's thesis; operational rather than strategic).
- Learned heuristics that warm-start the MILP (for example predicting which sites open).

## 6. Selected items and proposed implementation (for discussion)

Order of the storyline: J and K understand the catalogue, B and C forecast, A answers what-if
questions fast, H flags risk, O decides, and the simulator checks the decision.

### J. SKU segmentation (unsupervised, clustering)
- Features per SKU from 5 years of history: mean daily demand, CV squared, ADI (intermittency),
  seasonality strength, trend, unit volume and weight, margin, share of multi-item baskets.
- Methods: standardise, then k-means and a Gaussian mixture compared; choose k by silhouette and
  stability; baseline is the classic ABC/XYZ grid.
- Output: named clusters (for example "bulky fast movers", "intermittent long tail"), a stocking
  policy per cluster, and how far the clusters recover the generator's true product types.
- Open points: which features; whether to show UMAP or t-SNE projections.

### K. Product affinity (unsupervised, graph)
- Agreed and built 1 November 2026 (`backend/src/scgame/ml/product_affinity.py`, outputs in
  `data/models/product-affinity/`; no page yet).
- Pairs: co-purchase counts on the training set (years 1 to 3); chance computed within baskets of the
  same region, basket size and quarter (without the quarter, growing and same-season SKUs looked linked); lift = observed / expected; Poisson tail p-values; Benjamini-Hochberg
  false discovery rate.
- Groups: graph of accepted pairs weighted by log lift, Louvain communities; SKUs without an accepted
  pair are bought alone. Hidden bundles used only to check (precision, recall, ARI).
- Business check: split catalogue. 2 to 5 FCs at the central site, each SKU in exactly one, each FC
  limited to an equal share of SKUs and of workload (order lines), 5% slack; split orders and delivery
  cost by assignment rule (random, category, popularity spread, best sellers together, communities,
  hidden bundles as reference). The raw-count baseline's resolution is chosen on validation too. The
  quarter was added after the bundle check exposed timing links; this is disclosed (quarter_note). Settings chosen on the validation set (year 4), test
  set (year 5) priced once.
- Dropped: a limited-regional-slots check, because stocking whole communities added nothing over
  picking single SKUs there (coverage, not affinity, drove split orders).

### B. Long-range demand forecasting (supervised regression, neural network)
- Built 2 to 3 November 2026 (code in `backend/src/scgame/ml/demand_*.py` and `forecast_*.py`, world
  years 6 to 10 in `backend/src/scgame/world/future.py`, ground truth in `replicate.py`, outputs in
  `data/models/demand-forecast/`). No page yet. Proposal `design/demand-forecast/PROPOSAL.md`, results
  `RESULTS.md`, build and rebuild `HANDOVER.md` in the same folder.
- **Data.** Monthly units (a month is a twelfth of a 365-day year) at six levels, Total, Region, Category,
  Region × category, SKU and Region × SKU (2,744 series in the validation fit). Bulk orders kept. The world
  is extended to ten years for this model only; years 1 to 5 are unchanged byte for byte. Years 6 to 10
  add about 7 launches a year, delist as many of the slowest sellers, and stop each growing SKU's growth at
  an age between 4 and 7 years.
- **Split.** Fit on years 1 to 3, every setting chosen on years 4 and 5 (forecast from month 36), refit on
  years 1 to 5, test once on years 6 to 10 (forecast from month 60; upper levels five years ahead, SKU
  levels two). This differs from the 1 to 3, 4, 5 split of the other pages, as agreed.
- **Methods.** Seasonal naive; ETS (damped additive trend, additive season on log(1 + y), trend by AIC);
  LightGBM and a scikit-learn multilayer perceptron, both global over all series and levels, reading one
  scaled year and predicting the next, applied five times for five years. Mean forecasts; uncertainty from
  1,000 sample paths built from each method's past one-year errors (level errors carried over the years,
  month noise by calendar month, both centred), with one width factor for the upper levels and one for the
  SKU levels; MinT (shrunk covariance) or bottom-up for the upper levels, middle-out for the SKU levels.
- **Selection.** Hyperparameters by the WAPE of the mean forecast, widths, reconciliation and the method by
  weighted quantile loss (P10, P50, P90), all on years 4 and 5 with equal weight per level and year. The
  neural network was chosen.
- **Ground truth.** 200 replications of years 4 to 10; noise floor and calibration; never used to choose.
- **Business check.** One-off capacity per region for years 7 to 10, decided at the end of year 5, cost
  ratio from the game (0.76 to 0.83), full range of ratios.
- **Headline.** Good one and two years ahead (Total WAPE 2.4% and 4.1% against a noise floor of 1.9% and
  1.3%); over-forecasts by 8 to 25% in years 8 to 10 because growth stops in years 6 to 10, where ETS
  (damped) is best. The capacity plan from our forecast costs 15% more than the ground-truth plan, ETS 3%.
- **Kilian's decisions.**
  - 2 November. Forecast five years ahead, not one (so the world was extended to ten years); a
    five-year forecast at Region and Category levels, two years at SKU levels.
  - 2 November. The catalogue stays near 300 SKUs in years 6 to 10 (delisting matches launches).
  - 2 November. All fourteen recommendations of the proposal agreed (horizon and card "Demand five years
    out", world extension rules, monthly units, bulk orders kept, the four methods and `lightgbm`, joint
    error bootstrap, bottom-up versus MinT, the protocol above, scaled pinball loss with equal weight per
    level and year, 200 replications for checking only, the capacity check, the nine-screen storyline, the
    vocabulary, the agent loop for the page).
  - 2 November. The model design A1 to C7 agreed after a first-principles explanation (global model, scaled
    target, one year in and out with recursion, equal weight per level, mean forecasts, LightGBM with both
    squared error and Tweedie as settings, scikit-learn network rather than PyTorch, ensemble of five).
  - 2 November. Phases 0 to 3 to be done without him, with an independent agent checking after phase 2.
  - 3 November. Stop before the page; no frontend and no `card.json`; hand over to the local session.
- **Decided while working alone, for review** (`RESULTS.md` section 9). Months since first sale capped at
  24; network epochs chosen on validation instead of early stopping; network loss as a setting with the
  learning rate fixed; repeated rows instead of weights; the uncertainty fixes after the independent review
  (made after the first test run, disclosed).
- **Open.** How to present the turn after year 2, which forecast feeds O, out-of-sample error samples for
  the SKU bands (to be scored on a second, untouched future), the capacity wording (`RESULTS.md` section
  10).

### C. New-region demand (supervised regression, cold start)
- Training data: existing regions' demand against region features (population, income, distance
  to the nearest FC, urbanisation, category affinities).
- Models: regularised linear regression as a readable baseline, then gradient boosting; leave-one-
  region-out validation; prediction intervals.
- Output: expected demand and uncertainty for candidate regions and sites, feeding O.
- Open points: how many regions are needed to train credibly (affects the world size).

### A. What-if surrogate (supervised regression, neural network)
- Labels from the simulator: thousands of variants of network and capacity (which FCs, how many
  slots, which assortment where), each run over sampled futures.
- Targets: transport cost, fulfilment cost, split-order rate, fill rate, total cost to serve.
- Models: LightGBM and a small neural net; SHAP for explanations; checked against fresh simulator
  runs it never saw.
- Demo: "expand FC A or B?" answered instantly, then confirmed by a full simulation.
- Open points: input encoding of a network; how many simulator runs we can afford.

### H. Risk classification (supervised classification)
- 3 November 2026. Proposal in `design/split-risk/PROPOSAL.md` (split orders caused by listing or
  stock-outs, simulated networks, plain rule and replay simulation as baselines). Kilian agreed the
  model design (PROPOSAL section 6a) the same day and asked for the build; the page is not yet decided.
  The notes below are the original outline.
- Target: whether an order is split across FCs (or a SKU stocks out) under a planned network.
- Features: basket composition, customer region, SKU clusters from J, affinity groups from K,
  listing pattern.
- Model: gradient-boosted classifier, calibrated probabilities, precision and recall vs a simple
  rule; feature importance.
- Open points: which target is more convincing to zooplus (split orders or stockouts).

### O. Network design over 5 years (operations research)
- Multi-period mixed-integer model: open or expand FCs and capacities by year, assign regions
  and categories, minimise build, holding, transport and fulfilment cost, subject to demand
  quantiles from B and C.
- Solver: HiGHS (free). Robust or two-stage stochastic version over scenarios.
- Validation: replay the plan in the simulator against the do-nothing baseline.
- Open points: level of detail (categories or SKUs); stochastic or robust formulation.
