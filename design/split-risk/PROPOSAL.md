# Split orders, implementation proposal

Model H from `ML_ROADMAP.md`. Written 3 November 2026. Kilian asked the same day for a first-principles
explanation of the models and their design choices, agreed every design choice (section 6a, which
includes two changes to this proposal) and asked for the model to be built. The page (sections 10 and
11, decisions 9 to 11) is not yet decided.

Figures come from world v1, seed 7, training set (years 1 to 3) unless stated. To size the problem I ran
a throwaway prototype of the split simulation (section 4) on year 3 with levels from year 2, for four
hand-picked networks. No later choice may rest on those figures, and no validation or test year was
looked at.

---

## 1. The question, and why an operations research team would care

A multi-FC online retailer rarely stocks its full range in every FC. Each region is served by a home FC;
a basket whose SKUs are not all available there ships from more than one FC, as more than one parcel.
Every extra parcel costs a parcel fee, packaging and handling, and the customer receives the order in
pieces. The split rate is therefore one of the first numbers a planner needs before opening an FC or
changing what an FC carries.

Two things cause a split, and they behave differently.

1. **Listing.** A SKU in the basket is not listed at the home FC. Given the basket and the network, this
   is certain, and a plain rule finds it without any model.
2. **Stock-outs.** A listed SKU is out of stock at the home FC when the order arrives, so it comes from
   another FC. This depends on demand, season, growth and the stocking policy, none of which the plain
   rule sees. A stock-out can also cancel a split (the home FC is short of a listed SKU, so the whole
   basket ships from the central FC).

That is the honest case for a classifier. If stock never ran out, the plain rule would be exact and a
model would add nothing. The prototype shows that it does run out.

| Network (prototype, year 3) | Multi-SKU orders split | Flagged by the plain rule | Rule wrong |
|---|---|---|---|
| 3 regional FCs, 40% of SKUs each, 95% service | 15.5% | 17.2% | 4.9% of orders |
| 3 regional FCs, 60% of SKUs each, 98% service | 6.7% | 6.8% | 3.8% |
| 1 regional FC, 20% of SKUs, 90% service | 14.0% | 17.6% | 4.3% |
| 6 regional FCs, 40% of SKUs each, 95% service | 21.6% | 23.8% | 6.9% |

Correction found while building (3 November 2026). This table used a weaker plain rule ("any SKU not
listed at the home FC"), which wrongly flags orders with no SKU listed at home (they ship whole from the
central FC). The rule as built is "the home FC lists some but not all of the order's SKUs", exact when
stock never runs out. The prototype also routed differently. The built model's figures are in
`HANDOVER.md`.

So the rule is right on about 95% of multi-SKU orders, but on the second network it catches only 72% of
the split orders and only 71% of its flags are real. A classifier is worth what it adds over that rule,
and the page will say how much that is.

There is a second, stronger competitor. An operations research team with a simulator would replay last
year's orders through the planned network and read off each SKU's stock-out rate. The classifier must
beat that replay too, or the page says it does not. Its practical advantages are speed (scoring every
order under hundreds of candidate networks in seconds), a probability per order instead of one
simulated outcome, and an explanation of what drives the risk.

**Recommendation.** The target in section 2, with the plain rule and the replay simulation as the two
baselines the classifier has to beat.

## 2. The target

| Item | Proposal |
|---|---|
| Unit | One order with at least two distinct SKUs, under one planned network, in one year |
| Label | 1 if the order ships from more than one FC (a split order), else 0 |
| Network | The central FC (Calopeia, full range) plus 1 to 6 regional FCs at the other region sites, each with a limited listing |
| Prediction time | End of the year before, when the network is planned. Known are the basket, the customer's region, the network (sites, listings, service level) and every order of past years. Unknown are the year's demand and stock levels |
| Orders with one SKU | Cannot split (44% of orders); probability 0 by definition, left out of training and scores, said on the page |
| SKUs at risk | Not a second model. A SKU's risk at an FC is the expected number of its orders that split there (sum of order probabilities), checked against the simulated count |

The roadmap also listed stock-outs as a possible target. They are a stocking question that model J's
policy simulation already prices, while split orders are new and tie listing, affinity and stock
together.

**Recommendation.** Split orders as above, SKUs by aggregation.

## 3. Planned networks and how the simulator labels them

Labels come from simulating real orders through a network. Each network is drawn at random from a
family, so the classifier is trained on some networks and scored on networks it never saw.

| Element | Values |
|---|---|
| Regional FC sites | A random non-empty subset of the seven sites other than Calopeia (127 possible) |
| Home FC | A region's own FC if it has one, otherwise the central FC |
| Listing share | 20%, 40%, 60% or 80% of SKUs per regional FC (the central FC lists every SKU) |
| Listing rule | Regional best sellers; whole SKU clusters from model J; whole communities from model K, then single SKUs; whole categories; a random draw weighted by regional sales. All ranked by the previous year's order lines of the regions the FC serves |
| Service level | 90%, 95% or 98% for every SKU at every FC |

The listing rules matter because they give the classifier networks that split for different reasons
(a community kept whole splits less than the same number of best sellers). The first three rules are
also the networks a planner would actually consider.

**Sets of networks.** 60 training networks, 20 validation networks, 20 test networks. The test networks
use sets of FC sites that appear in neither training nor validation, so "networks not seen in training"
means new site combinations, not just new draws of the other settings.

**Recommendation.** This family, 60, 20 and 20 networks, unseen site sets in the test.

## 4. The split simulation

A new order-level fulfilment simulation in my own module (`scgame/ml/split_sim.py`), because
`policy_sim.py` works on SKU totals and cannot tell which parcel an order line leaves in. It imports the
stocking constants of `policy_sim.py` so the two pages stock alike.

1. **Stock.** Each FC holds each listed SKU under the stocking policy of model J's page. Weekly review,
   7-day lead time, order-up-to level S = μ(L + R) + zσ√(L + R) from the previous year's daily demand that
   would reach that FC under the network. Lost sales when no FC has stock.
2. **Routing.** Orders of a day are processed in random order. Each line goes to the home FC if the SKU
   is listed and in stock there, otherwise to the nearest FC that lists it and has stock (the central FC
   first among equals, by the distance multiplier of `policy_sim.py`), otherwise the line is lost.
3. **Label.** The number of FCs that ship part of the order. Two or more is a split order.
4. **Cost.** Parcels priced as on the Bought together page ($1.50 per parcel plus $0.35 per kg, times the
   distance multiplier from the shipping FC to the customer's region), so the business check can speak in
   dollars.

Whole-order rerouting (ship everything from one FC that can fill it) is left out. It would hide most
listing splits behind longer deliveries and needs a cost trade-off of its own.

What flatters the classifier. Order-up-to levels come from last year's average, ignoring season and
growth, so stock-outs gather in November and in growing SKUs (6 to 10% of lines lost in the prototype),
which a model can learn from the month and last year's growth. This is the stocking policy of model J,
kept so the pages agree; the page will say so.

**Recommendation.** This simulation, item by item routing, no whole-order rerouting, J's stocking policy.

## 5. Features

Everything is known at the end of the year before the order. Nothing is computed from the scored year.

| Group | Features |
|---|---|
| Basket | SKUs in the order, units, largest line, bulk line (20 units or more), weight (kg), volume (L), categories in the order, month |
| Region | Customer region, home FC is central or regional, distance multiplier from the home FC |
| Listing | Basket SKUs not listed at the home FC (count and share; the plain rule's input), listing share and service level of the network, regional FCs open. The listing rule and the set of sites are not features (design choice X3) |
| Stock risk | For basket SKUs listed at the home FC, the smallest and mean cover (order-up-to level over expected demand in the order's month over L + R, from last year's monthly pattern), largest CV² and ADI, largest growth over last year, smallest daily demand at that FC, service level. The same for the unlisted SKUs at the FC they would come from |
| SKU clusters (model J) | Count of basket SKUs per cluster (9), share whose cluster's stocking policy is distributed |
| Communities (model K) | Distinct communities in the basket, whether a basket community is only partly listed at the home FC, largest lift between two basket SKUs, share of basket SKUs bought alone |

Models J and K are read as committed (`data/models/sku-segmentation/`, `data/models/product-affinity/`).
Both were fitted on years 1 to 3 with settings chosen on year 4, so they hold no test information. Pair
lifts come from `product_affinity.pair_stats` on the training baskets. Within the training years these
features are in-sample (a basket's own pairs count towards its lift); validation and test are clean.

**Recommendation.** These features, J and K as committed.

## 6. Methods

| Method | In plain words | Technical rule | Role |
|---|---|---|---|
| Plain rule | An order splits if any of its SKUs is not listed at its home FC | Indicator, turned into two probabilities by the training share of split orders in each of its two classes | Baseline |
| Replay simulation | Run last year's orders through the planned network and use each SKU's stock-out rate there | Per FC, SKU and month in-stock rate from the simulated past year; split probability from listing and in-stock rates, assuming SKUs stock out independently | Operations research baseline |
| Logistic regression | One weight per feature, added up | L2-regularised logistic regression on standardised features, penalty chosen on validation | Simple model |
| Gradient boosting | Many small decision trees, each correcting the last | LightGBM binary classifier, about 24 settings (leaves, rows per leaf, sampling, trees) | Main model |

Explanations by SHAP values, computed by LightGBM itself (`pred_contrib`), so no new package.

**Recommendation.** These four. No new dependency (`lightgbm` and scikit-learn are already there).

## 6a. Model design (agreed)

Explained from first principles in the chat of 3 November 2026; Kilian agreed every choice below.
X3 and E5 change or extend the proposal above.

| # | Choice | Why |
|---|---|---|
| F1 | An example is one order under one network in one year | The label exists per order; SKU risk is the sum of its orders' probabilities, so both answers agree |
| F2 | Only orders with at least two SKUs | Single-SKU orders cannot split and would flatter every score |
| F3 | Output a probability, no threshold chosen | A planner sums probabilities; a threshold only to compare with the rule at its recall |
| F4 | Labels from the simulation over many random networks | The model must answer for networks it has not seen |
| D1 | At most 20,000 random multi-SKU orders per network and year in training | More networks teach more than more orders per network |
| D2 | No class rebalancing or weights | Reweighting moves probabilities away from true frequencies |
| X1 | A fixed-length vector per order from counts, smallest, largest and mean over its SKUs | One stock-out is enough, so the weakest SKU carries the signal; combining line models would bring back the independence assumption |
| X2 | Engineered stocking features, mainly cover (order-up-to level over expected demand during lead time and review in the order's month) | Gives the stocking physics in one number |
| X3 | Describe the order's situation, never the network's name; listing rule and site set are not features | Names let the model memorise rules that do not carry over to a planner's own rule |
| X4 | The replay's in-stock rates are not a feature | Keeps the comparison clean and the classifier independent of the simulator |
| X5 | Models J and K as committed | Fitted on years 1 to 3, settings chosen on year 4, no test information |
| X6 | Logistic regression on log-transformed, standardised features with one-hot categories; LightGBM on raw values with native categories | Linear models are scale sensitive; trees are not |
| M1 | Plain rule with two probabilities from training | A 0/1 rule would have infinite log loss |
| M2 | Replay simulation, in-stock rates per FC, SKU and month from last year, independence assumed | The natural operations research rival, past data only |
| M3 | Logistic regression, L2 penalty from six values, no hand-built interactions | Interactions are what trees should add |
| M4 | LightGBM, leaves 15 or 63, rows per leaf 100 or 1,000, sampling 0.8 or 1.0, read at 250, 500 and 1,000 trees, learning rate 0.05 | Standard controls; tree counts read from one fit |
| M5 | No neural network | Boosted trees are the standard strong choice for tables; model B shows a network |
| M6 | No monotone constraints | Stock-outs can cancel splits |
| M7 | SHAP values from LightGBM (`pred_contrib`), summed per feature group | Additive in log-odds, no new package |
| C1 | Calibration map fitted on validation (year 4) | Training predictions are overconfident |
| C2 | Isotonic, Platt or none, chosen by log loss with cross-fitting over two halves of the validation networks | The map is chosen on data it was not fitted on |
| C3 | Calibration checked on the test set only after every choice is fixed | Out-of-sample check |
| E1 | Choose by log loss | Proper scoring rule, rewards calibration and ranking |
| E2 | Every split and interval grouped by network | Rows within a network are correlated |
| E3 | Test networks use FC site sets never seen before | Backs the claim about unseen networks |
| E4 | Training labels from years 2 and 3, validation year 4, test year 5, no refit | Year 1 has no year before it; a refit would orphan the calibration map |
| E5 | Intervals on test scores from resampling test networks | Twenty networks is a small sample |
| E6 | Noise floor from re-simulating with other within-day order sequences | Shows how much of the label is luck, to explain only |

## 7. Calibrated probabilities

A planner adds probabilities up into expected split orders and parcels, so they must be calibrated, not
just ranked well.

- Calibration map for logistic regression and gradient boosting, isotonic regression or Platt scaling
  or none, chosen on the validation set by log loss with cross-fitting (fitted on half the validation
  networks, scored on the other half, then swapped). The chosen map is then fitted on all validation
  networks.
- Checked out of sample on the test set (year 5, unseen networks) with a reliability diagram, the
  expected calibration error, and predicted against simulated split orders per network and per SKU.

**Recommendation.** This scheme.

## 8. Training, validation and test

| Set | Orders from | Networks | Stock levels from | Used for |
|---|---|---|---|---|
| Training | Years 2 and 3 | 60 training networks | The year before each | Fitting every model |
| Validation | Year 4 | 20 validation networks | Year 3 | Choosing hyperparameters, calibration map and the model |
| Test | Year 5 | 20 test networks (unseen site sets) | Year 4 | Scored once |

The usual split of years 1 to 3, 4 and 5 holds. Year 1 gives no labels because stock levels need a year
before them; it serves only for the year-2 levels and listings. There is no refit on year 4 before the
test, so the calibration map fitted on year 4 belongs to the model it is applied to. Training rows are
capped at 20,000 multi-SKU orders per network and year (about 2.4 million rows).

Selection metric on validation is log loss, a proper scoring rule that rewards calibrated probabilities.
Reported beside it, never used to choose, are the Brier score, average precision, precision and recall
at the plain rule's recall, and the error of expected split orders per network.

**Noise floor.** The same network and year re-simulated with five other random orders of a day's orders
shows how many labels flip by chance alone. It bounds what any model can reach and is used only to
explain.

**Recommendation.** This protocol, no refit.

## 9. Business check, the split cost of a planned network

At the end of year 4 a planner considers a set of candidate networks for year 5 and needs, for each, the
split orders and their parcel cost before committing. Every estimator uses only information available
then.

| Estimator | How it estimates year 5 |
|---|---|
| Plain rule | Year-4 baskets through the network's listing |
| Replay simulation | Year 4 simulated through the network |
| Logistic regression and gradient boosting | Year-4 baskets scored with year-5 network settings and stock risk, probabilities summed |
| Simulated year 5 | The answer |

Two questions, kept apart.

1. **Estimate.** Error in split orders and in parcel cost per candidate network, in orders and dollars.
2. **Decision.** For each set of FC sites, the listing rule and share with the fewest estimated split
   orders under a fixed slot budget, and how many more split orders the chosen network really has than
   the best one (regret).

Candidates are the 20 test networks plus, for three fixed site sets, every listing rule at 40% (15 more),
so the decision compares like with like.

**Recommendation.** This check. Model quality (section 8) and business value stay on separate screens.

## 10. Draft storyline

Titles are plain noun phrases, captions only where a visual cannot be read without one, validation
before test, results last, no recap.

**Models home card.** Split orders. "Which orders will ship in more than one parcel under a planned
network?"

**Opening.** Title "Split orders". Question "Can a planner see which orders will split before
committing to a network?" Boxes "Orders" (training set), "SKUs" (300), "FC sites" (8).

1. **Split orders.** A map of Pangea with one network and one order traced item by item, from home FC
   to parcels, one SKU unlisted and one out of stock. Drawer with the routing rule.
2. **Planned networks.** The family of networks as a grid of small maps, with listing share, listing
   rule and service level beside each, the year strip, and which networks are training, validation and
   test. Drawer on how they were drawn and why the test site sets are new.
3. **Listing and stock-outs.** Split orders by month on the training set, divided into those the
   listing explains and those stock-outs add or cancel. Shows what the plain rule cannot see.
4. **Features.** A feature-vector heatmap for six example orders, as on the clustering page.
5. **Methods.** The four methods in plain words beside their technical rules.
6. **Validation.** Log loss of every method and setting on year 4, the chosen one marked, calibration
   map choice in the drawer.
7. **Calibration.** Reliability diagram on the test set for each method, with the diagonal drawn.
8. **Precision and recall.** Precision-recall curves on the test set, the plain rule as a point.
9. **Drivers.** SHAP values per feature group, and the SKUs at risk for one test network with their
   predicted and simulated split orders.
10. **Results.** Split orders and parcel cost of the candidate networks, estimated by each method
    against the simulated year 5, and the regret of the choice each method would make.

## 11. Vocabulary

New surface terms to approve. Split order, parcel, home FC, network, listing, listing share, classifier,
logistic regression, gradient boosting (approved for model B), probability, log loss, Brier score,
precision, recall, precision-recall curve, average precision, reliability diagram, SHAP value, replay
simulation, regret.

Drawer only (to add to `GLOSSARY.md`). Isotonic regression, Platt scaling, expected calibration error,
cross-fitting, proper scoring rule.

## 12. Build plan after approval

1. `backend/src/scgame/ml/split_sim.py` (networks, stock, routing, labels) with tests, including that
   with unlimited stock the plain rule is exact.
2. `backend/src/scgame/ml/split_risk.py` (features, models, calibration, scores, business check) with
   tests for leakage (nothing from the scored year) and coherence (SKU risks add up to order risks).
3. Outputs in `data/models/split-risk/`, no `card.json`. One independent review of leakage, fairness of
   baselines, choices on validation only and out-of-sample calibration, then
   `design/split-risk/HANDOVER.md`.

Estimated runtime about 20 minutes (100 networks simulated in about 3 seconds each, the replay runs,
2.4 million training rows).

## 13. Decisions

| # | Decision | Recommendation |
|---|---|---|
| 1 | Target | Split order (more than one shipping FC), caused by listing or stock-outs, on orders with at least two SKUs; SKUs by aggregation |
| 2 | Networks | Central FC plus 1 to 6 regional FCs, five listing rules, four listing shares, three service levels; 60, 20 and 20 networks, unseen site sets in the test |
| 3 | Simulation | New order-level simulation in my module, item by item routing to the nearest FC with stock, J's stocking policy, no whole-order rerouting |
| 4 | Features | Section 5, with models J and K read as committed |
| 5 | Methods | Plain rule, replay simulation, logistic regression, LightGBM; SHAP from LightGBM; no new dependency |
| 6 | Calibration | Isotonic, Platt or none chosen on validation by cross-fitting; checked on the test set |
| 7 | Protocol | Labels from years 2 and 3 for training, year 4 for validation, year 5 for test, no refit; log loss to choose |
| 8 | Business check | Split orders and parcel cost of candidate networks estimated at the end of year 4 against the simulated year 5, plus the regret of each method's choice |
| 9 | Card and title | "Split orders", with the question in section 10 |
| 10 | Storyline | Ten screens as in section 10 |
| 11 | Vocabulary | Section 11 |
| 12 | Process | Build and review the model, write the handover, stop before the page as for model B |
