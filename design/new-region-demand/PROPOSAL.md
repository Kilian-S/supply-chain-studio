# New-region demand, implementation proposal

Model C from `ML_ROADMAP.md`. Written 3 November 2026 on branch `claude/relaxed-brahmagupta-qd6agt`
(from model B's branch `claude/festive-johnson-bl41d7`). Kilian agreed every recommendation of section 13
on 3 November 2026, and the model design D1 to D14 in section 4a after a first-principles explanation
in chat. He asked where urban share comes from (section 3) and kept it as a feature.

Figures come from world v1, seed 7, training set (years 1 to 3). While profiling I printed, for years 1
to 3 only, each region's units, bulk share and units per inhabitant by category, the network's yearly
totals and the count of SKUs per hidden type. Nothing from years 4 and 5 was looked at.

**What I know because we wrote the world.** I read the generator before writing this. A region's
consumer demand there is the network's demand times the region's weight (population × online adoption ×
√income, where online adoption is 0.4 + 0.5 × urban share), times income raised to a category-specific
elasticity (0 for Grocery up to 0.8 for Electronics). Bulk orders come from business customers, drawn at
random. Delivery distance has no effect on demand. This knowledge informed the candidate features, the
log-linear form of one method and the rule for simulating new regions (sections 3, 4 and 8). The page
must disclose it, and section 4 keeps two methods that do not assume the form.

---

## 1. The question and its business use

A shop entering a new country has no sales there to forecast from. Before the first order it must still
size a warehouse, plan stock and judge whether entry pays. The only information is what the new region
looks like (population, income, how urban it is, where it lies) and how the existing regions with
similar traits buy.

**Question.** How many units per category will a new region order in its first year, and how sure are
we, given only its features and the history of the existing regions?

**Uses.** Sizing the new region's FC before opening (the business check, section 9), ranking candidate
regions for entry, and demand inputs with uncertainty for network design (O). For zooplus this is the
country-entry question, with about thirty markets to learn from and every new one different.

**Recommendation.** Card "Demand in a new region", question "How much will a region sell before its
first order?".

## 2. Facts that shape the design

| Fact (training set) | Value | Consequence |
|---|---|---|
| Regions | 8 | Only 7 regions to learn from in each validation fold. Few parameters, strong regularisation |
| Population | 0.9 to 6.0 M (21.9 M in all) | Size explains most of a region's units, so the model works per inhabitant |
| Income index | 0.80 to 1.25 | Narrow range. New regions outside it are extrapolation |
| Urban share | 0.42 to 0.82 | Correlated with income (richer regions are more urban), so their effects are hard to separate |
| Landmass | 5 regions on the continent, 3 islands | Candidate feature only |
| Units per 1,000 inhabitants per year | 7.6 (Austra) to 16.3 (Sorange) | A factor of two between regions, beyond population |
| Same, without bulk orders, as an index (network = 1) | 0.57 to 1.33, nearly the same in every category of a region | Consumer demand is smooth in the features |
| Bulk share of a region's units | 5.1% (Austra) to 23.1% (Entworpe) | Bulk orders come from a handful of business buyers per region. They make single cells jump (Veloria Sports 1.81, Entworpe Toys 1.26) and set the noise floor |
| Region shares of network units | Stable year to year (Veloria 6.9%, 7.4%, 6.9%) | The region's place in the network is the quantity to learn. The network's level comes from elsewhere |
| Network growth | +4.6% in year 2, +7.4% in year 3 | The network's change to the target year matters as much as the region's index |

## 3. Inputs

| Input | Use | Note |
|---|---|---|
| Population (M) | Exposure. Units are modelled per 1,000 inhabitants | A setting decides whether log population also enters as a feature (economies of scale) |
| Income index | Feature | |
| Urban share | Feature | |
| Island (landmass other than the continent) | Candidate feature | Validation decides |
| Category | Feature | Region × category is the unit of the model |
| History of the existing regions | Yearly units per region and category up to the forecast origin, monthly units of the network | Training target and the network forecast |

Left out, and why.

- **`online_adoption` and `customer_share`** in `regions.parquet`. They are the generator's own
  mechanism, and `customer_share` is close to the answer. A real shop could not observe them.
- **Customer counts per region.** A new region has no customers before it opens.
- **Distance to the nearest FC.** Years 1 to 5 have no FC network the demand could react to, and the
  generator gives distance no effect (disclosed knowledge). With 7 regions per fold a spurious
  coefficient is likely. Distance belongs to the cost side in O.

**Recommendation.** Population as exposure, income, urban share, island and category as features.

## 4. Methods and baselines

Each method predicts a region's units per 1,000 inhabitants per category as an index against the
network (network = 1). Units for a target year are index × population × the network's units per 1,000
inhabitants in that year (section 5). Uncertainty is added the same way for all (section 6).

| Method | In plain words | Technical rule | Role |
|---|---|---|---|
| Population share | Every inhabitant buys like the average inhabitant of the existing regions | Index 1 in every category | Baseline, the usual first guess in market sizing |
| Nearest analogues | Copy the regions most like the new one | kNN on standardised income, urban share and island, index averaged over the k nearest, per category | Baseline, the analogue method of retail site selection |
| Log-linear regression | Each feature raises or lowers sales per inhabitant by a fixed percentage | Ridge regression of log index on income, urban share, island, category and category × log income | Main candidate, readable as elasticities |
| Gradient boosting | Trees learn the rules from the regions' features | LightGBM on region × category rows, target log index | The usual machine-learning choice, kept to test it against the readable model |

What I expect, said now so the results cannot rewrite it. With 7 regions per fold, trees can split only
on six distinct values per feature and cannot extrapolate beyond the training range, so gradient
boosting will probably lose. The log-linear form matches the generator's multiplicative process, which
flatters it (disclosed). Nearest analogues is free of that assumption but has only 7 analogues to pick
from.

Hyperparameters, all small grids because 8 folds cannot support large ones.

| Method | Settings |
|---|---|
| Nearest analogues | k 1, 2 or 3; plain or distance-weighted mean |
| Log-linear regression | Ridge penalty (6 values, log-spaced); feature set (income and urban; plus island; plus category × log income); log population as a feature or not |
| Gradient boosting | Leaves 2 or 4; rows per leaf 4 or 8; trees 50, 100 or 200; learning rate fixed at 0.05 |
| All | Target with bulk orders or consumer orders only, with bulk added at the network's bulk units per inhabitant |

The last setting tests whether bulk orders are better predicted as a network average than through the
features, since they come from a few random buyers.

**Recommendation.** These four methods and grids. No new dependency (scikit-learn and LightGBM are
installed).

## 4a. Model design (agreed)

Explained from first principles in the chat of 3 November 2026; each choice below was agreed. D5 and
D13 change sections 3 and 6 of the first draft.

| # | Choice | Why |
|---|---|---|
| D1 | One example per region × category; the 8 regions are the real sample size | A region's rows share its features and its luck |
| D2 | Population as exposure, label per 1,000 inhabitants | Removes a known proportional effect, so 7 regions can learn the rest |
| D3 | Label as an index against the network (network = 1) | Separates how the region buys (C) from how much the shop sells (B); stable over the years |
| D4 | Log of the label and of income | Effects are multiplicative; errors symmetric in ratio; predictions positive |
| D5 | One intercept; category enters only through its cross with log income | The index is centred in every category by construction, so category intercepts would be about zero and cost 7 weights |
| D6 | Rows weighted by their share of the region's units, each region weighing the same | Train on the loss used to choose |
| D7 | Bulk orders in the label, or added at the network average, as a setting | A few random buyers decide them; validation decides |
| D8 | Four models from no learning (population share) to the most flexible (LightGBM) | Shows what each step of complexity buys with this little data; no neural network |
| D9 | Ridge (L2) rather than least squares or Lasso | 12 weights against 7 regions, and log income and urban share correlate at 0.94 |
| D10 | Small grids, about 60 settings in all | 8 folds cannot separate many settings without selection bias |
| D11 | Leave one region out, fitted on years 1 to 3, scored on year 4 | A region the model never saw, and stability over time |
| D12 | WAPE with equal weight per region | Each region is one entry decision |
| D13 | Two-part log-normal error from the leave-one-region-out errors; the region part uses a t distribution with n − 1 degrees of freedom and √(1 + 1/n); no tuned width | Out-of-sample errors, an sd estimated from 8 values, no second set to tune on |
| D14 | Test on 30 simulated new regions, 10 outside the training range | Validation regions were used to choose; only regions that never existed give a clean test |

## 5. Forecast origin and the network's level

A region's index says how it buys relative to the network. Units also need the network's level in the
target year. That level is not the cold-start question, and B forecasts it. B's forecasts start at
months 36 and 60, not at the origins used here, and the network changed by 5 to 7% a year in training,
so the choice matters.

| Option | Network units per 1,000 inhabitants in the target year | Effect |
|---|---|---|
| Last year repeated | The existing regions' last year, category by category | A decision possible at the origin. Its error is common to every method |
| Realised target year | The existing regions' actual target year | Isolates the cold-start error. Uses other regions' target-year data, not the new region's |

**Recommendation.** Last year repeated for every method and for the business check, so every number is a
forecast made at the origin. The realised-network version is reported as a diagnostic in a drawer, to
show how much of the error is the network's and how much the region's. Monthly and SKU demand of a new
region follow the network's monthly and SKU mix within each category (the world has no regional taste
within a category, disclosed).

## 6. Uncertainty

P10, P50 and P90 from a log-normal error around each method's prediction, built from its own
leave-one-region-out errors on the validation set (section 7, a jackknife). The error has two parts, a
region error shared by all its categories (8 values, one per held-out region) and a cell error per
region × category. 1,000 draws per new region keep categories coherent and feed O as demand scenarios
(combined with B's network paths there).

No width factor is tuned. The leave-one-out errors are already out of sample, and with 8 regions there is
no second set to tune a width on. Calibration is checked on the test regions and the ground truth.

**Recommendation.** This two-part log-normal error, no width factor, 1,000 draws.

## 7. Validation and test

Kilian's split (training years 1 to 3, validation year 4, test year 5) holds. The held-out unit is a
region, so validation also leaves regions out, and the test uses regions that never existed.

| Step | Fitted on | Predicts | Scored on | Purpose |
|---|---|---|---|---|
| Validation, 8 folds | 7 regions, years 1 to 3 | The eighth region's year 4, from its features only | Its realised year 4 | Choose every setting and the method |
| Refit | All 8 regions, years 1 to 4 | | | The chosen setting |
| Test | | 30 new regions' year 5, from their features only | Their simulated year 5 (section 8) | Scored once |

The held-out region's data never enter its fold, not for fitting, not for standardising features, not
for the network's level (the network in a fold is the other 7 regions). The test regions' data enter
nothing before scoring.

**Selection metric.** WAPE of the P50 forecast of yearly units, per held-out region over its 8
categories, then averaged with equal weight per region, at two levels (Region total and Region ×
category), equal weight per level. Equal weight per region, because each region is one entry decision
and Calopeia should not decide alone. Reported beside it, never used to choose, are bias, coverage of P10
to P90 and weighted quantile loss.

**Learning curve.** The roadmap asks how many regions are needed to train credibly. For n = 2 to 7
training regions, every subset of the 8 predicts the regions it left out (validation year), giving the
error per method against n. It answers the question without changing the world, and its slope at n = 7
shows whether more markets would still help.

**Caveats for the page.** Choosing among about 60 settings on 8 folds makes the validation error
optimistic (selection bias). The test regions measure that gap. The test refits on year 4, which
validation also used, as in B.

**Recommendation.** This protocol, metric and learning curve.

## 8. Test regions and ground truth

The world has only its 8 regions, so a clean test needs regions that never existed. The simulator can
supply them, using the generator's own process without changing the generator or the existing world.

**New regions.** A new module of mine (`scgame.ml.region_sim`) imports `OrderSampler`, `expected_rates`
and `bulk_orders` from the generator and simulates year 5 for each new region on its own random stream.

| Item | Rule |
|---|---|
| Number | 30 |
| Features | 20 inside the existing ranges (Latin hypercube over population, income and urban share), 10 outside by up to a quarter of the range on one feature, a third of them islands. Fixed before any model is fitted |
| Demand | The generator's formula, with the region's weight divided by the sum of the existing 8 weights. The existing regions are unchanged (the new region adds demand, it does not take any) |
| Customers | Drawn as in the generator, 30,000 × the region's share of that sum, with the same order frequencies and 1.5% business customers |
| Bulk buyers | Each lumpy SKU gains buyers among the region's business customers at the existing rate per business customer, with the same gaps and order sizes |
| Ramp-up | None. Every region of the world sells at full rate from day 1, so no ramp-up could be learned (a limit, disclosed) |

**Ground truth.** For each of the 30 new regions (year 5) and each of the 8 existing regions treated as
new (year 4), 200 replications with fresh customers, bulk buyers and orders. Their mean is the true
expected demand of a region with those features, the best any cold-start forecast could do. It gives the
noise floor (WAPE of the true mean against the realised year), calibration (share of replications inside
our P10 to P90) and, from the generator's parameters, the true income elasticity per category to compare
with the log-linear model's. Used to check, never to choose. About 10 minutes on 4 cores.

What it costs in honesty. The test regions follow exactly the process the training regions follow, which
flatters every method, the log-linear one most. Real countries differ in ways no feature here captures
(competition, culture, logistics). The page says so.

**Recommendation.** 30 simulated new regions with these rules, and 200 replications for checking only.
No external data (Eurostat was listed for C in the roadmap; the ranges of the existing regions are
enough and need no licence check).

## 9. Business check, the new region's FC

At the end of year 4 the shop decides to open in a new region and builds its FC capacity (units per
month) for year 5, as in B's check but for one year and one new region. Capacity costs c per unit for
the year, and each unit of demand above capacity in a month is a lost sale costing its margin m. The best
capacity under a forecast is the (1 − ρ / 12) quantile of its draws' monthly demand, ρ = c / m (sample
average approximation, exact, reusing B's `capacity_plan` and `capacity_cost` unchanged).

| Plan | Capacity from |
|---|---|
| Our forecast | Draws of the chosen method |
| Our P50 only | The busiest month of the P50 forecast, no allowance for uncertainty |
| Population share | Its draws |
| Nearest analogues | Its draws (if not chosen) |
| Ground truth | Replications 0 to 99 of the new region |

Scored for the 30 test regions on the realised year 5 and on replications 100 to 199 (expected cost),
over ρ from 0.1 to 10 with the game's ratio marked (0.76 to 0.83, as in B), cost relative to the
ground-truth plan.

**Recommendation.** This check. It keeps model quality (section 7) and business value apart.

## 10. Draft storyline

Titles are plain noun phrases, no result before its screen, validation before test, ground truth second
to last, the business result last, no recap.

**Opening.** Title "Demand in a new region". Question "How much will a region sell before its first
order?" Boxes "Regions" (8), "Categories" (8), "Inhabitants (M)" (21.9).

1. **Regions.** Table of the 8 regions with population, income index, urban share and landmass, rows by
   population. Drawer on the left-out columns and why.
2. **Units per inhabitant.** Heatmap of 8 regions × 8 categories, units per 1,000 inhabitants per year
   (training set), with a control "All units" or "Without bulk orders". The second shows the smooth
   pattern, the first the jumps that bulk buyers add. Drawer on bulk orders.
3. **Leave one region out.** The year strip (training 1 to 3, validation 4, test 5) and the eight folds,
   each region greyed out in turn. Drawer on why a random split would leak.
4. **Forecasting methods.** Plain words beside the technical rules, then one held-out region's
   validation forecast per category for each method against the actual, with a preset per region and a
   warning that single regions invite cherry-picking.
5. **Validation.** Heatmap of methods by held-out region, WAPE (%) on year 4, the chosen method outlined,
   mean and coverage as last columns.
6. **Number of regions.** Validation WAPE against the number of training regions (2 to 7) per method.
7. **New regions.** The 30 test regions, predicted against simulated units (log scales) with P10 to P90
   bars, regions outside the training range marked by styling. Drawer on how they were simulated.
8. **Forecast versus ground truth.** WAPE of our forecast, the population share and the ground-truth mean
   on the test regions, coverage against the truth. Drawer with learned against true income elasticity
   per category, and what flatters the setting.
9. **Results.** Capacity cost of each plan relative to the ground-truth plan across the cost ratio, the
   game's ratio marked, boxes for our plan at that ratio. Drawer with plain-word rules for each plan.

## 11. Vocabulary

New surface terms to approve. Cold start, region feature, exposure, units per 1,000 inhabitants, index
(network = 1), population share, nearest analogue, log-linear regression, ridge regression,
regularisation, elasticity, leave one region out, extrapolation, learning curve, noise floor. Terms from
B reused as approved there (forecast, forecast origin, P10, P50, P90, prediction interval, coverage,
WAPE, bias, LightGBM, gradient boosting, capacity).

Drawer only (to add to `GLOSSARY.md`). Jackknife, log-normal, Latin hypercube, offset, sample average
approximation.

## 12. Build plan after approval

1. `backend/src/scgame/ml/region_sim.py`, the new regions and the replications, with a test that the
   existing world files are untouched and that a simulated copy of an existing region matches its
   expected units.
2. `backend/src/scgame/ml/region_demand.py`, the methods, the leave-one-region-out validation, the
   learning curve, the test, the ground-truth comparison and the capacity check, with leakage tests in
   `backend/tests/test_region_demand.py`.
3. Outputs in `data/models/new-region-demand/` (`overview.json` and the per-region forecasts), no
   `card.json`.
4. One self-review against the truth rules, then `design/new-region-demand/HANDOVER.md` in the shape of
   B's, and section C of `ML_ROADMAP.md`.

Expected runtime about 15 minutes on 4 cores, most of it the replications. Nothing in the generator, the
simulator, `policy_sim.py`, B's code or any other model changes.

## 13. Decisions

All agreed by Kilian on 3 November 2026.

| # | Decision | Recommendation |
|---|---|---|
| 1 | Question and card | Yearly units per category in a new region's first year, months and SKUs from the network's mix; card "Demand in a new region" |
| 2 | Number of regions | Keep the world's 8 (no world change); answer "how many are needed" with the learning curve |
| 3 | Features | Population as exposure; income, urban share, island, category. Not distance, customer counts or the generator's hidden columns |
| 4 | Bulk orders | Kept in demand; whether to predict them as a network average is a setting chosen on validation |
| 5 | Methods | Population share, nearest analogues, log-linear (ridge) regression, LightGBM, with the grids of section 4 |
| 6 | Network level | Last year repeated for every method; realised network as a diagnostic only |
| 7 | Uncertainty | Two-part log-normal error from the leave-one-region-out errors, no width factor, 1,000 draws |
| 8 | Protocol | Leave one region out on year 4 (fit on years 1 to 3), refit on years 1 to 4, test once on 30 new regions' year 5; WAPE with equal weight per region and level |
| 9 | Test regions | 30 simulated by my own module with the generator's process, 10 outside the training range, no cannibalisation, no ramp-up |
| 10 | Ground truth | 200 replications per region, check only, disclosed with the generator knowledge above |
| 11 | Business check | One-year FC capacity for each new region, game cost ratio, full range shown |
| 12 | External data | None |
| 13 | Storyline and vocabulary | Sections 10 and 11 |
