# Split orders (model H), handover

For the session that reviews the model and designs the page. Written 3 November 2026 at the end of the
cloud session on branch `claude/tender-darwin-6oi5j6`, revised the same day after an independent truth
review (`review-truth-r1.md`). No frontend exists for this model and no `card.json` is written.

Read in this order. This file (what was built, how, and what came out), `review-truth-r1.md` (the
independent review and its findings, all addressed below), `PROPOSAL.md` (the question, sections 1 to 4,
the agreed model design in section 6a, the draft storyline in section 10), `docs/ML_ROADMAP.md` section H.

The question. Before a planner commits to a network (which FCs open, what each lists, at what service
level), how many of next year's orders will ship in more than one parcel, and which orders and SKUs?

---

## 1. The split

The usual split of the clustering and Bought together pages, with one difference. Year 1 gives no labels,
because the first stock levels and listings need a year before them.

| Set | Orders from | Networks | Listings and stock levels from | Used for |
|---|---|---|---|---|
| Training | Years 2 and 3 | 60 training networks | The year before each | Fitting every method |
| Validation | Year 4 | 20 validation networks | Year 3 | Choosing hyperparameters, calibration map and the method |
| Test | Year 5 | 20 test networks (site sets never seen) | Year 4 | Scored (see section 7 on the second run) |

There is no refit on year 4 before the test, so each calibration map belongs to the model it is applied
to. Training, validation and test networks use disjoint sets of FC sites; three more site sets are kept
for the decision in the business check and appear in no other set (tested).

## 2. Inputs

- **World v1 years 1 to 5** (`scgame.world.generator`, seed 7), unchanged. Only `order_lines.parquet`,
  `skus.parquet` and `regions.parquet` are read. Years 6 to 10 and the ground-truth replications of model
  B are not used.
- **Model J as committed** (`data/models/sku-segmentation/`). The chosen clustering (k-means, k = 9,
  feature mask 57) and each cluster's stocking policy (distributed or central). Fitted on years 1 to 3,
  setting chosen on year 4.
- **Model K as committed** (`data/models/product-affinity/graph.json`). Each SKU's community and whether it
  is bought alone. Pair lifts are recomputed with `product_affinity.pair_stats` on the training baskets
  (years 1 to 3, chance within region, basket size and quarter), imported, not copied.
- **Stocking constants** from `policy_sim.py` (lead time 7 days, weekly review, the distance multiplier),
  imported. Parcel prices from `product_affinity.py` ($1.50 per parcel plus $0.35 per kg, times the
  distance multiplier).
- **Orders scored.** Only orders with at least two distinct SKUs (56% of orders); one-SKU orders cannot
  split.

## 3. The method, step by step

Code in `backend/src/scgame/ml/`. `split_sim.py` (networks, listings, stock, routing, labels),
`split_risk.py` (networks drawn, features, methods, calibration, scores, business check, outputs).

1. **Networks.** The central FC (Calopeia, lists every SKU) plus 1 to 6 regional FCs at the other seven
   sites. Listing share 20, 40, 60 or 80% of SKUs per regional FC; listing rule regional best sellers,
   whole model J clusters, whole model K communities, whole categories, or a random draw weighted by
   regional order lines (the group rules fill leftover slots with best sellers); service level 90, 95 or
   98%. Site sets are drawn without replacement from the 123 sets of size 1 to 6 (the three decision
   sets removed), 20 for the test, then 20 for validation, then 60 for training; rules, shares and
   service levels are balanced within each set by shuffled repetition. Seed 23.
2. **Planning a network for year y.** Each regional FC ranks SKUs by the order lines of its own region in
   year y − 1. Every FC stocks its listed SKUs with order-up-to level S = μ(L + R) + zσ√(L + R) from the
   daily demand of year y − 1 that would reach it if every line shipped from the first FC on its route
   that lists the SKU (model J's policy, `policy_sim`). The overflow that stock-outs send to other FCs is
   not planned for (limit 2).
3. **Routing.** A region's home FC is its own FC if it has one, otherwise the central FC. A line tries the
   home FC, then the central FC, then the others by distance multiplier and id, and ships from the first
   that lists the SKU and has stock; otherwise it is lost. The day's orders arrive in random order; FCs
   serve first choices, then second choices, and so on, first come first served within a round.
4. **Label.** An order that ships from two or more FCs is a split order. The plain rule flags an order
   when its home FC lists some but not all of its SKUs; with unlimited stock it is exact (tested).
5. **Features.** One fixed-length row per order and network, 40 inputs in six groups (`O.features`).
   Basket (SKUs, units, largest line, bulk line, kg, litres, categories, month), Region (customer region,
   home FC central, distance multiplier), Listing (unlisted SKUs, their share, the rule's flag, SKUs with a
   second FC that lists them, regional FCs open, listing share, service level), Stock risk (smallest and
   mean cover of the SKUs listed at home, largest CV² and ADI, largest growth, smallest daily demand,
   smallest cover and largest growth of the SKUs that would come from the central FC), SKU clusters (count
   per model J cluster, share in distributed clusters), Communities (distinct model K communities, a basket
   community only partly listed at home, largest pair lift, share bought alone). Cover is S over expected
   demand during lead time plus review in the order's month, from last year's monthly pattern at that FC,
   capped at 20. CV², ADI and growth (log of second-half over first-half units, minus the shop's) come from
   year y − 1. Never the listing rule or the site set (design choice X3).
   `n_listed_elsewhere` counts the basket SKUs that an FC other than the one they would ship from also
   lists. For a customer served by the central FC that is the SKUs a regional FC could ship if the central
   FC runs out; for a customer with a regional home FC it is every SKU listed at home (the central FC lists
   everything) plus unlisted SKUs another regional FC lists, so there it is close to a count of the SKUs
   listed at home.
6. **Training rows.** For each training network and year, 20,000 multi-SKU orders drawn at random
   (2,400,000 rows). No rebalancing.
7. **Plain rule.** Two probabilities, the training share of split orders among flagged (0.875) and
   unflagged (0.071) orders.
8. **Replay simulation.** The network planned for year y run through year y − 1 with stock levels from
   y − 2. In-stock rate per FC, SKU, month and route class (first routing round, or later rounds, which
   are served after the first and are mostly lines whose first FC ran out), falling back to the SKU's
   yearly rate at the FC in that class, then the FC's rate in that class. The split probability of an
   order assumes its SKUs stock out independently along their routes, each FC at the rate of the round the
   line reaches it in. In the business check the replay's own simulated split rate is the estimate.
9. **Logistic regression.** scikit-learn, L2, lbfgs. log1p where marked, missing values as 0 plus a flag,
   standardised with training means, region and month one-hot.
10. **Gradient boosting.** LightGBM binary log loss, learning rate 0.05, region and month as native
    categories, `deterministic`, four threads.
11. **Calibration.** For every method and setting, three maps (none, Platt on the logit, isotonic) scored
    by cross-fitting on validation, fitted on the even-numbered validation networks and scored on the odd
    ones, then swapped. The chosen map is refitted on all validation networks.
12. **Scores.** Log loss (chooses), Brier score, average precision, prediction bias, reliability diagram
    (ten equal bins, with counts) and expected calibration error, precision at the plain rule's recall,
    split rate per network. 95% intervals by resampling the 20 test networks 2,000 times.
13. **Noise floor.** Every test network re-simulated on year 5 with five other random orders of arrival;
    share of orders whose label differs in one rerun (mean over the five) and in at least one.
14. **SHAP values.** LightGBM `pred_contrib` on 50,000 test rows. Per feature the mean absolute value; per
    group the mean absolute value of the group's summed contributions per order (log-odds).
15. **SKUs at risk.** Per test network and SKU, the expected number of split orders containing the SKU
    (sum of order probabilities) against the simulated count.
16. **Business check.** 35 candidate networks for year 5, the 20 test networks and, for three site sets
    ({2}, {2, 3, 6}, {2, 3, 4, 6, 7, 8}), each listing rule at 40% and 95% service. At the end of year 4
    each method estimates the split rate. The rule, logistic regression and gradient boosting score year-4
    multi-SKU orders under the network planned for year 5 (listing, levels and stock features from year
    4); the replay simulates year 4 under that network. The truth is the simulated year 5. Errors are
    turned into orders with year 5's multi-SKU orders and into dollars with each candidate's own simulated
    extra parcel cost per split order ($2.97 to $7.94; parcel cost minus one parcel of the same weight from
    the home FC). Differences between methods get 95% intervals by resampling 23 clusters 2,000 times (each
    test network alone, each decision site set as one cluster). The decision is the listing rule with the
    fewest estimated splits per site set; regret is how many more split orders it really has than the best.

## 4. Baselines

- **Plain rule** (listing only) and **replay simulation** (what an operations research team with a
  simulator would do) go through the same calibration choice and scores as the learned models.
- **Logistic regression** shows how much is linear and what the trees add.
- **Noise floor** shows how much of an order's label is luck of arrival order.

## 5. How every setting was chosen, and on which data

Everything below was chosen on the validation set (year 4, 20 validation networks, 1,079,560 orders),
never on the test set or any ground truth.

| Setting | Candidates | Chosen by | Result |
|---|---|---|---|
| Logistic penalty C | 0.001, 0.01, 0.1, 1, 10, 100 | Cross-fitted log loss | C = 0.1, isotonic (all six within 0.0001) |
| LightGBM | Leaves 15 or 63, rows per leaf 100 or 1,000, sampling 0.8 or 1.0, trees 250, 500 or 1,000 (24) | Cross-fitted log loss | 63 leaves, 100 rows, 0.8, 250 trees, isotonic (all 24 within 0.0014). A corner of the grid; with 63 leaves the loss rises with the tree count, so fewer trees might do slightly better |
| Calibration map | None, Platt, isotonic, per method and setting | Cross-fitted log loss | Rule Platt, replay isotonic, logistic isotonic, LightGBM isotonic |
| Method | The four at their best | Cross-fitted log loss | Gradient boosting 0.2428, replay 0.2612, logistic 0.2655, rule 0.2970 |

Fixed in advance, not chosen. The network family and the 60, 20 and 20 networks, 20,000 training rows per
network and year, the learning rate, the tree counts on the grid, the cover cap of 20, the routing rule,
model J's stocking policy, two route classes for the replay (after the review, section 7), the three
decision site sets, ten reliability bins, 2,000 resamples, seed 23.

## 6. Headline results and their output fields

From `data/models/split-risk/overview.json` (written `O`), `skus.json` and `examples.json`, run 2. Method
keys are `rule`, `replay`, `logistic`, `gbm`; labels in `O.labels`.

| Result | Value | Field |
|---|---|---|
| Split share of multi-SKU orders | Training 17.4%, validation 20.7%, test 16.2% | `O.training.split_share`, `O.validation.split_share`, `O.test.split_share` |
| What the rule misses, by month (training) | Split but not flagged 2.6% of orders in February to 13.0% in November | `O.training.causes_by_month` (counts per month) |
| Plain rule on the test set | Flags 9.8% of orders, precision 92.9%, recall 56.1%, wrong on 7.8% | `O.test.rule_flags` |
| Validation choice | See section 5 | `O.validation.chosen`, `O.validation.chosen_method`, `O.validation.grid` |
| Test log loss | Gradient boosting 0.233, replay 0.243, logistic 0.245, rule 0.275 | `O.test.methods.<m>.log_loss` |
| Gradient boosting minus the others (95% interval over networks) | Rule −0.047 to −0.039, logistic −0.016 to −0.010, replay −0.012 to −0.008 | `O.test.intervals.log_loss_gbm_minus` |
| Test Brier score | Gradient boosting 0.065, logistic 0.067, replay 0.068, rule 0.072 | `O.test.methods.<m>.brier` |
| Test average precision | Gradient boosting 0.780, replay 0.759, logistic 0.748, rule 0.592 | `O.test.methods.<m>.average_precision`, curves in `.pr_curve` |
| Precision at the rule's recall (56%) | Gradient boosting 93.0%, rule 92.9%, logistic 92.7%, replay 87.0% | `O.test.methods.<m>.at_rule_recall` |
| Calibration on the test set (expected calibration error, prediction bias) | Gradient boosting 0.5%, −0.5 points; logistic 0.5%, −0.5; replay 1.2%, +1.1; rule 0.8%, +0.1 | `O.test.methods.<m>.reliability.ece`, `.prediction_bias` |
| Calibration between 0.5 and 0.8 | Gradient boosting 4 to 6 points too low, logistic 14 to 22 points too low, replay 3 to 6 points too high; 83% of orders sit below 0.2 | `O.test.methods.<m>.reliability` (bins with counts) |
| Gradient boosting without its map | Log loss 0.236, calibration error 1.6% | `O.test.methods.gbm_uncalibrated` |
| Split rate per test network, mean absolute error | Gradient boosting 1.06 points, logistic 1.10, replay 1.13, rule 3.37 | `O.test.methods.<m>.split_rate_mae`, per network in `.per_network` |
| Noise floor | 3.6% of orders change label with one other order of arrival (7.8% in at least one of five); network split rate moves 0.05 points | `O.test.noise` |
| SHAP per group (mean absolute summed log-odds) | Listing 1.06, stock risk 0.92, basket 0.34, region 0.15, SKU clusters 0.08, communities 0.03 | `O.shap.groups`, per feature `O.shap.features` |
| Strongest inputs | SKUs with a second FC that lists them, the rule's flag, cover at the central FC, month, cover at home | `O.shap.features` |
| SKUs at risk, error per network and SKU | Replay 11.0 split orders (correlation 0.995), gradient boosting 11.7 (0.989), logistic 12.4 (0.989), rule 27.9 (0.952) | `O.sku_risk.<m>`, per SKU in `skus.json` `networks[k].<m>` and `.simulated` |
| SKUs at risk, totals | Simulated 734,294; replay 780,545 (+6.3%), gradient boosting 706,126 (−3.8%), logistic 708,563 (−3.5%), rule 669,555 (−8.8%) | `O.sku_risk.<m>.total_predicted` |
| Business check, split rate error over 35 candidates | Gradient boosting 0.68 points (403 orders, $1,931 a year), replay 0.92 (541, $2,368), logistic 0.95 (558, $2,625), rule 2.82 (1,661, $7,848) | `O.business.estimate_error.<m>` |
| Gradient boosting minus the others, mean absolute error (95% interval) | Replay −0.23 points (−0.52 to +0.06), logistic −0.26 (−0.44 to +0.01), rule −2.14 (−2.97 to −1.30) | `O.business.mae_difference_intervals` |
| Candidates where another method is closer than gradient boosting | Replay 15 of 35, logistic 14, rule 6 | `O.business.estimate_error.<m>.closer_than_gbm` |
| Bias of the estimates | Gradient boosting −0.24 points, logistic −0.08, replay +0.90, rule +0.45 | `.bias_points` |
| Decision | Every method, the rule included, chooses regional best sellers for all three site sets, the best rule each time (leads of 3.2, 5.9 and 7.4 points); the check does not separate the methods | `O.business.decisions` |
| Candidates in full | Network, simulated rate, each estimate, parcel cost | `O.business.candidates` |
| Examples for the page | Six orders (split or whole, flagged or not, months drawn at random) with features; one order traced line by line | `examples.json` |

What it means. Order by order, gradient boosting is best on log loss, Brier score and average precision,
with intervals that exclude zero against every method; its calibration error is tied with logistic
regression. The fixed replay simulation is a close second on log loss and the best on SKU totals. For a
network's split rate, gradient boosting, logistic regression and the replay cannot be told apart (the
intervals include zero), and all three are far better than the plain rule. No method changes the choice of
listing rule. At the rule's own recall gradient boosting is no more precise than the rule; its value is in
the orders the rule does not flag (stock-outs) and in probabilities that add up.

## 7. Chosen or changed after seeing test results or ground truth

- **Two full runs.** Run 1 (`run-1.log`) was scored on the test set. An independent review
  (`review-truth-r1.md`) then found that the replay's in-stock rates pooled first-choice lines with lines
  arriving after another FC ran out, which understated the home FC's rate (validation bias +7.9 points).
  Run 2 (`run-2.log`) keeps the two route classes apart. This is a code fault in a baseline, justified by
  the code, and confirmed on validation (cross-fitted log loss 0.2798 to 0.2612). Run 2 also changed four
  reports, none of which feeds a choice. Business-check intervals were added and dollars now use each
  network's own cost per split. The noise floor gained the one-rerun figure. SHAP groups use summed
  contributions per order. The page examples are drawn at random instead of the first (all January).
- **What did not change.** Every result of the plain rule, logistic regression and gradient boosting, every
  choice of setting and map, the chosen method, the networks and the labels are identical in both runs.
  Only replay results, intervals that include the replay, and the four reports above differ.
- **Run 1's explanation was wrong.** It said the replay overpredicts because stock-outs at one FC come
  together. The pooled rates were the main cause; with them fixed, the replay's validation bias is +0.7
  points.
- **Ground truth not used.** The hidden demand types and planted bundles are never read. Model K's
  communities match the planted bundles exactly (its own result), so the community features carry what
  model K learned from data.
- **Test scores seen during development.** A smoke run with 4 networks per set and tiny grids wrote test
  scores to a scratch file that was not opened; its validation choices were printed.
- **Explanations measured on test networks.** The reviewer measured the overflow between FCs, the share of
  the rule's misses from customers of the central FC, and the noise floor on year 5 (section 8). These
  explain results and changed nothing.
- **Features differ from the proposal** (section 5 there, agreed as decision 4). Added before any run,
  `n_listed_elsewhere` and `some_not_all` (the rule's flag), now the two strongest inputs. The stock-risk
  features for SKUs coming from the central FC were cut to the smallest cover and largest growth. The
  listing rule was dropped as agreed in design choice X3.
- **Proposal stage.** A prototype on year 3 with four hand-picked networks sized the problem. Its figures
  used a weaker rule ("any SKU not listed at home"), corrected before any run (note in PROPOSAL section 1).

## 8. Known limits and doubts

1. **The stocking policy flatters the classifier.** Order-up-to levels ignore season and growth (model J's
   policy, kept so the pages agree), so stock-outs are frequent (about 7% of lines lost in the networks
   checked during development) and gather in November, which the model learns from the month and cover.
   With a season-aware policy the rule would miss far fewer splits.
2. **Routing and unplanned overflow decide much of what the rule misses.** A line out of stock at its FC
   ships from another FC that has it, but no FC's stock level allows for that overflow. On ten test
   networks a sixth to a half of the line attempts at regional FCs come from other regions, and 48 to 83%
   of the splits the rule misses are orders of customers served by the central FC whose SKU shipped from a
   regional FC (53% on ten training networks). A retailer that waits for stock, cancels the line, or plans
   levels for the overflow would see fewer such splits.
3. **The features use the simulator's own stocking formula** (cover). A planner knows the policy too, but
   real stock has noise the simulator lacks (supplier delays, damaged stock, forecasts).
4. **Arrival order is a small part of the remaining error.** 3.6% of orders change label with another
   order of arrival, which alone would allow a Brier score near 0.02 (review estimate) against 0.065. The
   rest is uncertainty about the year's demand and stock, and limits of the model; this check does not
   separate them. Network rates are stable (0.05 points).
5. **Unseen networks means unseen site sets.** Listing rules, shares and service levels all appear in
   training. A planner's own listing rule is untested, although no network name is a feature.
6. **The business check is thin.** The estimates read year-4 baskets, so all methods miss year 5's volume
   equally (rates are compared). The three model-based estimates cannot be told apart, and the decision is
   easy (best sellers have the fewest splits in every site set, by at least 3 points).
7. **The replay still assumes independent stock-outs** and averages a month's rates. With route classes it
   is nearly unbiased on validation (+0.7 points raw) and beats logistic regression; a replay that sampled
   each day's joint stock state would be stronger and was not built.
8. **Calibration is uneven.** The calibration error is weighted by orders and 83% sit below 0.2. Between
   0.5 and 0.8 gradient boosting is 4 to 6 points too low and logistic regression 14 to 22. Sums over a
   network are fine; a single order's probability in that range is not.
9. **In-sample pieces.** Pair lifts on training rows include the rows' own baskets; models J and K chose
   their settings on year 4, the validation year here (no test information).
10. **Hyperparameters hardly matter.** All 24 LightGBM settings lie within 0.0014 log loss, all six
    penalties within 0.0001.
11. **SKU totals of gradient boosting are 3.8% low,** mostly because it underpredicts split orders on the
    test year by 0.5 points (about 2.6% of the total); orders with many SKUs, which count once per SKU,
    add about 1.3%.
12. **The strongest input means two things** (`n_listed_elsewhere`, section 3 step 5). Its name on the page
    must say which, or the drawer must explain both.
13. **SHAP groups** are mean absolute values of summed contributions, so features in a group that push in
    opposite directions offset each other; per-feature values are kept beside them.
14. **Simplifications.** Lost lines are not splits; a line ships whole or not at all; FCs serve first
    choices first within a day; listing ranks use the FC's own region only (a regional FC serves only its
    region).
15. **Unrelated failing test.** `tests/test_demand_forecast.py::test_outputs` fails on the base branch
    too (it expects `error_samples`, which model B's committed outputs lack). Not touched here.

## 9. Dependencies

None new. LightGBM (added for model B), scikit-learn, pandas, numpy, networkx and scipy were already there.
No frontend change; `npm install` was run only for the checks.

## 10. Rebuild

From a fresh clone (models J and K outputs are committed in `data/models/`).

```bash
cd backend && uv sync
uv run python -m scgame.world.generator ../data/world_v1 7      # about 10 s
uv run python -m scgame.ml.split_risk ../data/world_v1 ../data/models ../data/models/split-risk
```

The model run takes about 23 minutes on 4 cores (1,364 s in run 2, `O.runtime_s`), most of it the eight
LightGBM fits on 2.4 million rows (about 2 minutes each).

Reproducibility, checked. A second full run of the run-1 code on the same world gave identical
`skus.json` and `examples.json`, the same choices and the same test scores; `overview.json` differed only in
float rounding from the seventh significant digit on. Run 2 reproduces every result of run 1 that its
changes do not touch. The world itself was generated in this session, not from a fresh clone.

Checks.

```bash
cd backend && uv run pytest tests/test_split_risk.py && uv run ruff check src
```

`tests/test_split_risk.py` (14 tests, about 5 s) covers the plain rule being exact with unlimited stock,
lines shipping only where listed and stocked, listing sizes, the plan and features reading nothing from the
scored year, the replay planning the same network with levels from the year before, the replay's route
classes, the replay probability (equal to the rule when always in stock, and a hand-made case), stock
running out and arriving a lead time after the review on a hand-made case, parcel cost and the rule on a
hand-made split order, SKU risks adding up, calibration maps, disjoint and balanced networks, and the
committed outputs.

## 11. For the page

- The storyline draft is PROPOSAL section 10 and the vocabulary section 11; neither is decided. The plain
  rule's wording there needs the correction above ("lists some but not all").
- Show the reliability diagram with bin counts and say in its drawer where calibration is good and where it
  is not (limit 8). Name the strongest input precisely (limit 12). Say plainly that the decision check does
  not separate the methods.
- Screen 3 (listing and stock-outs) reads `O.training.causes_by_month`; screen 1 reads `examples.json`
  `trace`; screen 4 reads `examples.json` `examples`.
- Do not write `card.json` before the page exists.
