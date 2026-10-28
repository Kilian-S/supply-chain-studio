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
   networks, and optimisation, aimed at questions an operations research team faces before
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
- Co-purchase counts from baskets; lift and association rules as the baseline; item embeddings
  (item2vec style) or graph community detection (Louvain) for groups.
- Output: affinity communities that feed the listing and network decisions; the split-order rate
  if communities are kept together vs scattered.
- Open points: embeddings or graph communities as the main method.

### B. Long-range demand forecasting (supervised regression, neural network)
- Hierarchy: total, region, category, SKU. Horizon: up to 5 years at monthly or weekly level, plus
  daily for operations.
- Models: seasonal naive and ETS baselines, LightGBM global model, and a neural net (N-BEATS or a
  temporal convolutional network); quantile outputs (P10, P50, P90); hierarchical reconciliation.
- Evaluation: rolling-origin backtests; WAPE, bias, pinball loss; comparison with the true process.
- Open points: granularity of the strategic forecast; how far to go with reconciliation.

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
- Target: whether an order is split across FCs (or a SKU stocks out) under a planned network.
- Features: basket composition, customer region, SKU clusters from J, affinity groups from K,
  listing pattern.
- Model: gradient-boosted classifier, calibrated probabilities, precision and recall vs a simple
  rule; feature importance.
- Open points: which target is more convincing to an operations research team (split orders or stockouts).

### O. Network design over 5 years (operations research)
- Multi-period mixed-integer model: open or expand FCs and capacities by year, assign regions
  and categories, minimise build, holding, transport and fulfilment cost, subject to demand
  quantiles from B and C.
- Solver: HiGHS (free). Robust or two-stage stochastic version over scenarios.
- Validation: replay the plan in the simulator against the do-nothing baseline.
- Open points: level of detail (categories or SKUs); stochastic or robust formulation.
