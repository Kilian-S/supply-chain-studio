# New-region demand (model C), handover

For the local session that reviews the model and designs the page. Written 3 November 2026 at the end of
the cloud session on branch `claude/relaxed-brahmagupta-qd6agt` (started from model B's branch
`claude/festive-johnson-bl41d7`). No frontend exists for this model and no `card.json` is written.

Read in this order. This file (what was built and how), `PROPOSAL.md` (the agreed design, sections 4a,
8, 10 and 13 in particular), `ML_ROADMAP.md` section C (decisions in one place).

---

## 1. The split

Kilian's split holds (training years 1 to 3, validation year 4, test year 5). The held-out unit is a
region, so validation also leaves regions out and the test uses regions that never existed.

| Set | Regions | Years | Used for |
|---|---|---|---|
| Training | 7 of the 8 existing regions per fold | 1 to 3 | Fitting every method in each validation fold |
| Validation | The eighth region, each in turn (8 folds) | 4 | Choosing every setting and the method, and the error model of the bands |
| Refit | All 8 existing regions | 1 to 4 | Refitting the chosen settings before the test |
| Test | 30 simulated new regions | 5 | Scored once |

The network level in the target year is the last observed year repeated (year 3 for validation, year 4
for the test), computed from the training regions only.

## 2. Inputs

- **World v1 years 1 to 5** (`scgame.world.generator`, seed 7), read only. `regions.parquet`,
  `skus.parquet`, `order_lines.parquet`, `customers.parquet`, `bundles.parquet`, `bulk_buyers.parquet`
  (via `scgame.world.future.load_world`). Nothing is written to `data/world_v1/`.
- **Region features** (`region_demand.features`). Log income index, urban share, island (landmass other
  than the continent), log population. Population is also the exposure. Left out on purpose are the
  generator's `online_adoption` and `customer_share` columns (its mechanism, close to the answer),
  customer counts (a new region has none) and distance to an FC (no network in years 1 to 5 to react
  to). Urban share and income are values set by hand in the generator, not calibrated on external data.
- **Quantity.** Yearly units per region × category, split into consumer lines and bulk lines (20 units
  or more, as in B). Monthly units for the month shape and the capacity check. A month is a twelfth of a
  365-day year.
- **Label.** log of the region's units per 1,000 inhabitants over the network's (the training regions'),
  pooled over the fitting years, per category (the index, network = 1).
- **New regions** (`scgame.ml.region_sim`). 30 regions fixed before any model was fitted
  (`test_regions.csv`), ids 101 to 130. 20 inside the existing ranges (a Latin hypercube over log
  population, income and urban share), 10 outside on one feature by 2 to 25% of the range (population on
  the log scale), the other two inside. 10 of the 30 are islands. Each is simulated with the generator's
  `OrderSampler`, `expected_rates` and `bulk_orders`, imported unchanged.
  - Consumer demand follows the generator's formula with the region's weight divided by the sum of the
    existing 8 weights, so the existing regions are unchanged (no cannibalisation).
  - Customers 30,000 × that share, the generator's order frequencies, 1.5% business customers.
  - Every lumpy SKU on sale gains bulk buyers among the region's business customers with probability
    2 / 465 each (the generator's mean of 2 buyers per lumpy SKU over its 465 business customers), gaps
    U(25, 90) days, the generator's order sizes.
  - No ramp-up. The year is year 5 of the world's catalogue (v1 rules, no launches).
- **Ground truth** (`simulation.npz`). 200 replications (stream `[7, 33, r]`) of the 30 new regions'
  year 5 and of the 8 existing regions' year 4 as if they opened fresh (fresh customers, bulk buyers and
  orders, same features). The realised test year is a separate draw (stream `[7, 32, 0]`).

## 3. The method, step by step

Code in `backend/src/scgame/ml/`. `region_sim.py` (new regions, replications), `region_demand.py`
(everything else).

1. **Panel.** Units per region, category, month (60) and kind (consumer, bulk) from the order lines.
2. **Label and weights.** For the training regions and the fitting years, units per 1,000 inhabitants
   per category over the network's, log. With bulk "network average" only consumer units enter the
   label. Each row is weighted by its share of its region's units, so each region weighs the same (D6).
3. **Methods** (`fit_predict_log_index`).
   - Population share. Log index 0.
   - Nearest analogues. Income (log), urban share and island standardised on the training regions;
     the index of the k nearest by Euclidean distance averaged per category, plain or weighted by 1 /
     distance.
   - Log-linear regression. scikit-learn `Ridge` (intercept unpenalised) on the standardised features
     of the chosen set, optionally log population, optionally category one-hot × standardised log
     income (D5), weighted rows.
   - Gradient boosting. `LGBMRegressor` on raw features and category (native categorical), learning
     rate 0.05, `min_data_in_bin` 1, deterministic, weighted rows.
4. **Units.** exp(log index) × population × the network's units per 1,000 inhabitants in the level year
   (last year repeated); with bulk "network average", consumer index × consumer rate plus the network's
   bulk rate.
5. **Validation** (`validate`). Every setting, leave one region out, scored on the held-out region's
   year 4. Score per region is the mean of WAPE over its 8 categories and WAPE of its total; the
   selection score is the mean over the 8 regions. The best setting per method, then the best method.
   Ties go to the first setting in grid order.
6. **Error model** (`error_model`, D13). From the chosen setting's held-out errors in logs, e = log((y +
   1) / (ŷ + 1)). A region part, the units-weighted mean of its cells' errors (8 values, σ as root mean
   square), and a cell part, the rest (σ as units-weighted root mean square). Month noise around the
   year's total spread by the predicted category mix and the network's monthly shape of the level year,
   with variance φ / expected monthly units (count-like noise, φ from the same folds).
7. **Draws** (`draws`). 1,000 per region. Region error σ_region × √(1 + 1/8) × t with 7 degrees of
   freedom, cell errors normal, yearly units = prediction × exp(region + cell), months by the network's
   monthly shape of the level year, times log-normal month noise. P10, P50 and P90 from the draws. The
   P50 is the prediction (errors centred at zero; the root mean square keeps any bias in the width).
8. **Validation coverage.** Each held-out region's band built from the other 7 regions' errors (t with
   6 degrees of freedom), so the reported coverage is out of sample.
9. **Learning curve.** Every subset of 2 to 7 training regions (246 subsets) at each method's chosen
   setting, predicting the regions left out, year 4. Mean, P10 and P90 of the selection score over the
   subsets.
10. **Test.** Each method's chosen setting refitted on years 1 to 4 of the 8 regions; the 30 new regions'
    year 5 predicted from their features, scored once on the realised draw, in total and inside and
    outside the training range.
11. **Ground truth.** The replications' mean as a forecast (noise floor), the share of replications
    inside our P10 to P90 (calibration), the true band width over ours, and the learned income elasticity
    per category of the log-linear regression (its chosen setting refitted on years 1 to 4) against the
    generator's (0.5 + the category's income elasticity), and the learned urban effect against the
    generator's 0.5 / (0.4 + 0.5 × urban share) at the mean urban share.
12. **Capacity check** (`capacity_check`). One FC per new region, capacity K (units per month) for year 5
    decided at the end of year 4. Cost c K + m Σ (D − K)+ over 12 months, c / m = ρ. The best K under a
    set of draws is the (1 − ρ / 12) quantile of their months, B's `capacity_plan` (the same optimum as
    for B's 48 months, tested). Plans are our forecast, our P50 only (busiest P50 month), each other
    method, and the ground truth (replications 0 to 99), costed on the realised year 5 and on
    replications 100 to 199. ρ from 0.1 to 10 (B's 41 values); the game's ratio is 0.2267 × price /
    margin per unit, with the network's year-4 price and margin per unit in the region's predicted
    category mix.

## 4. Baselines

- **Population share** is the forecasting baseline (index 1, no learning), run through the same units,
  bands, scores and capacity check as the learned methods. Nearest analogues is a second, assumption-free
  baseline.
- **Ground-truth mean** is the reference for model quality (what a perfect cold-start model of the
  process scores, the noise floor).
- **Capacity plans** have their own references, our P50 only (ignore uncertainty) and the ground-truth plan
  (the best any forecast could do). There is no "last year's peak" for a region without a last year.

## 5. How every setting was chosen, and on which data

Everything below was chosen on the validation set (each existing region's year 4, held out in turn, models
fitted on the other 7 regions' years 1 to 3), never on the test regions or the ground truth. 109 settings
in all.

| Setting | Candidates | Chosen by | Result |
|---|---|---|---|
| Bulk orders | In the label, or consumer label plus the network's bulk rate (every method but population share) | Selection score | Network average for all three learned methods |
| Nearest analogues | k 1, 2 or 3; plain or distance-weighted (12 with bulk) | Same | k 2, distance-weighted |
| Log-linear regression | Penalty 0.01 to 100 (6, log-spaced); features income and urban, plus island, plus category × log income; log population or not (72 with bulk) | Same | Income, urban, island, log population, penalty 0.01 |
| Gradient boosting | Leaves 2 or 4; rows per leaf 4 or 8; trees 50, 100 or 200 (24 with bulk) | Same | 4 leaves, 4 rows, 200 trees |
| Method | The four at their best settings | Same | Log-linear regression |
| Error model | None tuned. σ_region, σ_cell and month dispersion φ are read from the chosen settings' held-out errors | Fixed in advance (D13) | Log-linear: σ_region 0.097, σ_cell 0.184, φ 19.6 |

The selection score is the mean over the 8 held-out regions of the mean of two WAPEs (over the region's 8
categories, and of its total).

Fixed in advance, not chosen. LightGBM learning rate 0.05, 1,000 draws, the t correction, 200
replications, the 30 test regions and their rules, the capacity cost formula, the error offset of 1 unit.

## 6. Headline results and their output fields

From `data/models/new-region-demand/overview.json` (written `O`) and `regions.json` (written `R`). Methods
are keyed by name ("Population share", "Nearest analogues", "Log-linear regression", "Gradient
boosting"). Scores are fractions (0.103 is 10.3%).

| Result | Value | Field |
|---|---|---|
| Method chosen on validation | Log-linear regression | `O.chosen` |
| Validation score per method | Log-linear 0.103, nearest analogues 0.142, gradient boosting 0.151, population share 0.257 | `O.validation.<method>.score` |
| Validation noise floor (ground-truth mean, year 4) | 0.088 | `O.truth.validation_noise_floor` |
| Chosen settings | Section 5 | `O.validation.<method>.setting` |
| Every trial | | `O.validation.<method>.trials` |
| Per held-out region (WAPE, bias, actual, P10, P50, P90 per category) | Log-linear 8.4% (Nordhal) to 19.5% (Fardo) over categories | `O.validation.<method>.per_region[]` |
| Validation coverage of P10 to P90 (bands from the other 7 regions) | Log-linear cells 91%, region totals 8 of 8; others 81 to 84% and 6 or 7 of 8 | `O.validation.<method>.intervals` |
| Error model | Section 5 | `O.validation.<method>.errors` |
| Learning curve, log-linear, 2 to 7 training regions | 0.182, 0.128, 0.117, 0.110, 0.106, 0.103 | `O.learning_curve.<method>[].mean` (also `p10`, `p90`, `subsets`) |
| Learning curve at 7, other methods | Nearest analogues 0.142, gradient boosting 0.151, population share 0.257 (from 0.244, 0.232, 0.281 at 2) | Same |
| Test score per method, 30 new regions | Log-linear 0.110, nearest analogues 0.111, gradient boosting 0.128, population share 0.155 | `O.test.<method>.score` |
| Test noise floor | 0.081 (inside the range 0.085, outside 0.075) | `O.truth.test_noise_floor`, `_inside`, `_outside` |
| Test, log-linear | WAPE over categories 13.1%, of the total 8.9%, bias of the total −7.8% | `O.test["Log-linear regression"].wape_region_category`, `.wape_region`, `.bias_region` |
| Test inside and outside the training range, log-linear | 0.107 and 0.117 | `O.test.<method>.inside_range.score`, `.outside_range.score` |
| Test coverage of P10 to P90, log-linear | Cells 87%, region totals 90% (others 92 to 95% and 93 to 97%) | `O.test.<method>.coverage_cells`, `.coverage_region` |
| Test with the realised network level (diagnostic) | Log-linear 0.089, nearest analogues 0.098, gradient boosting 0.139, population share 0.175 | `O.test.<method>.realised_network_score` |
| Our P50 against the ground-truth mean | Log-linear 0.086 | `O.truth.error_vs_truth_mean.<method>` |
| Calibration, log-linear | 89% of replicated cells and 93% of replicated totals inside our P10 to P90; the true band is 0.34 of ours | `O.truth.calibration.<method>` |
| Learned against true effects, log-linear | Income elasticity 0.64 in every category (true 0.5 to 1.3); urban share 1.01 per unit (true 0.70 at the mean); island −0.04 and log population −0.04 (true 0) | `O.truth.elasticity` (island and population printed in section 7) |
| Capacity cost over the ground-truth plan at the game ratio, realised and expected | Our forecast +2.2%, +1.6%; our P50 only +2.4%, +1.8%; nearest analogues +4.1%, +3.8%; gradient boosting +7.0%, +7.0%; population share +21.5%, +21.1% | `O.capacity.at_game_ratio.<plan>` against `["Ground truth"]` |
| Capacity cost across ρ 0.1 to 10 | | `O.capacity.curves.<plan>.realised`, `.expected`, against `O.capacity.rho` |
| Game ratio per new region | 0.794 to 0.796 | `O.capacity.rho_game` |
| Per test region at the game ratio | Capacity, realised and expected cost per plan | `O.capacity.per_region[]` |
| Per test region, test | WAPE, bias, P10, P50, P90 per category | `O.test.<method>.per_region[]` |
| Test regions | Features, `inside_range`, `outside_feature` | `O.test_regions` (also `test_regions.csv`) |
| Training heatmap data | Units per 1,000 inhabitants per year by region and category, all and consumer; bulk share per region | `O.training` |
| Monthly test actuals and bands | | `R.test[].months_actual`, `R.test[].months.<method>` (rows P10, P50, P90), `R.test[].truth_mean` |

Worst test regions for the log-linear model (WAPE of the total). New region 11 (22%, rich and rural
island), 18 (21%, small and poor, near the training regions), 24 (20%, 7.9 M inhabitants, outside the
range). Rich and rural regions lie off the training regions' income and urban diagonal (Pearson 0.94),
the extrapolation that matters most.

## 7. Chosen or changed after seeing test results or ground truth

- **Self-review against the truth rules** (one pass after the run). Leakage, every quantity in a fold
  (label, standardisation, network level, monthly shape) comes from the training regions and years only,
  and the tests check it per method. Fair baselines, every method had its settings chosen on the same
  folds by the same score, and the bulk setting was open to all three learned methods. Choices, all on
  validation (section 5). Nothing needed fixing; the doubts it raised are in section 8.

- **Nothing was changed after the test was scored.** The validation was run alone first
  (`--validation-only`), reviewed, and the code committed (`e5a0cc8`) before the single full run. The
  full run was repeated once to check that it is deterministic (identical output).
- **Test scores seen during development.** None before the full run.
- **The ground truth was used only to check.** The replications and the generator's true elasticities
  enter only `O.truth`. The simulator's sanity check compared simulated copies of the existing regions
  with their realised consumer units in year 3 (training set), 1 to 6% apart.
- **Printed after the test, for this handover.** The ridge coefficients refitted on years 1 to 4 (island
  −0.04, log population −0.04 per unit of raw feature), each test region's distance to the nearest
  training region, and the network's growth from year 4 to year 5 (+7.8% in all, +9.0% consumer units,
  −3.4% bulk), to explain the bias.
- **What the generator told me** (disclosed in the proposal and to be disclosed on the page). The
  candidate features, the log-linear form, leaving out distance and the rules for simulating new regions
  were informed by knowing how the generator makes demand.
- **Proposal stage.** For years 1 to 3 only, each region's units, bulk share and units per inhabitant by
  category, the network's yearly totals and the count of SKUs per hidden type.

## 8. Known limits and doubts

1. **Eight regions.** Every result rests on 8 held-out regions. 109 settings chosen on 8 folds make the
   validation score optimistic for the chosen method (0.103 on validation, 0.110 on the test), while
   nearest analogues did better on the test than on validation (0.142, 0.111). The two then tie on the
   test. A likely reason, not tested, is that 2 of the 8 validation regions are extremes (Sorange the
   richest, Austra the poorest and smallest), where analogues must extrapolate, while most test regions
   lie inside the range.
2. **Settings at the edge of the grid.** The log-linear penalty is the smallest tried (0.01), and gradient
   boosting's chosen setting is the most flexible corner (4 leaves, 4 rows, 200 trees). The ridge settings
   are near ties (0.103 to 0.106 for the twelve best), so the choice among them is noise. The grid was not
   extended.
3. **The test follows the training process exactly**, and the log-linear form matches it, which flatters
   every method and the log-linear one most. Real countries differ in ways no feature here captures.
4. **The network level is last year repeated.** The network grew 7.8% from year 4 to year 5, so every
   method under-forecasts the new regions by about that much (log-linear bias −7.8%). With the realised
   network the log-linear score falls to 0.089, near the noise floor of 0.081. Combining C with B's
   network forecast is O's job.
5. **The bands are too wide on the test** (87 to 95% of cells inside P10 to P90, against 80%). The t
   correction and the root-mean-square errors are conservative, and validation coverage was already above
   80% for the log-linear model. Not corrected, because that would tune on the test.
6. **One cell error for all categories.** Small, bulk-heavy categories get bands too narrow and large ones
   too wide.
7. **Collinearity.** The model gives urban share 1.01 per unit (true 0.70) and income 0.64 (true 0.5 to
   1.3, about 0.85 on average). Together they predict well where income and urban share move together, as
   in the training regions, and worst where they do not (new regions 6 and 11, rich and rural).
8. **No category effect on income.** The chosen feature set has no category × income cross, so the model
   gives every category the same income elasticity, while the generator's range from 0.5 (Grocery) to 1.3
   (Electronics). The cross did not win on validation (0.1036 against 0.1029).
9. **No ramp-up and no cannibalisation.** Every region of the world sells at full rate from day 1, and a
   new region adds demand without taking any from the existing ones.
10. **Monthly draws** use the network's monthly shape (the world has no regional seasonality) and a
    count-like month noise added for the capacity check (section 9, decision 1).
11. **The learning curve uses the settings chosen with 7 training regions.** Smaller subsets might prefer
    other settings.
12. **Bulk buyers of new regions** are drawn per pair of lumpy SKU and business customer (probability 2 /
    465), while the generator draws 1 to 3 buyers per SKU among all business customers. The expected rate is
    the same, but the spread differs slightly.
13. **A failing test that is not mine.** `tests/test_demand_forecast.py::test_outputs` fails with
    `KeyError: 'error_samples'`, identically on B's branch `claude/festive-johnson-bl41d7`. B's committed
    `overview.json` predates its last commit. Not touched (outside this model's boundary).

## 9. Decided while working alone, for review

1. **Month noise** (a third error part, not in D13). The capacity check needs monthly draws, and yearly
   draws spread by the network's shape had no monthly noise at all. Variance φ / expected monthly units,
   φ from the held-out regions' monthly totals. Yearly bands are unaffected.
2. **109 settings, not "about 60".** The grids of section 4 of the proposal, which were agreed, count to
   109. The figure in D10 was my miscount.
3. **Validation coverage from the other 7 regions' errors**, so it is out of sample. The test bands use all
   8.
4. **Units-weighted error parts.** The region part and σ_cell weight cells by their share of the region's
   units, like the training rows (D6).
5. **P50 is the prediction.** Errors are not shifted by their mean; the root mean square keeps any bias in
   the width.
6. **Ties go to the first setting in grid order.**
7. **LightGBM details** for 56 rows. `min_data_in_bin` 1, `max_bin` 63, `min_child_weight` 0,
   deterministic, one thread.
8. **Price and margin for the capacity check** are the network's of year 4 in the region's predicted
   category mix, so the game ratio is 0.79 to 0.80 for every new region.
9. **The simulation is committed** (`simulation.npz`, 1.9 MB), so the model rebuilds in seconds.

## 10. Dependencies

None new. scikit-learn (`Ridge`) and `lightgbm` were already in `pyproject.toml`. No frontend change;
`npm install` was run only for the checks.

## 11. Rebuild

From a fresh clone.

```bash
cd backend && uv sync
uv run python -m scgame.world.generator ../data/world_v1 7      # years 1 to 5, about 15 s
uv run python -m scgame.ml.region_demand ../data/world_v1 ../data/models/new-region-demand
```

The model reads the committed `simulation.npz` and `test_regions.csv` and runs in about 12 s
(`O.runtime_s`). If `simulation.npz` is missing it builds it first (520 s on 4 cores). To rebuild the
simulation alone, or to run validation without the test, use these.

```bash
uv run python -m scgame.ml.region_sim ../data/world_v1 ../data/models/new-region-demand 200 4
uv run python -m scgame.ml.region_demand ../data/world_v1 <scratch dir> --validation-only
```

**Verified.** A second full run gave an identical `overview.json` (apart from the runtime), and re-simulating
the realised draw and replications 0 to 2 reproduced the committed arrays exactly.

Checks.

```bash
cd backend && uv run pytest && uv run ruff check src
```

`tests/test_region_demand.py` covers leakage (the held-out region and years 4 and 5 never enter a
validation prediction, per method and bulk setting), the grid size, the population-share baseline, recovery
of an exactly log-linear toy world, the t-widened draws, the error split, the one-year capacity optimum,
the test-region design, the simulator against the existing regions (and that it leaves the world files
untouched), and the committed outputs.

## 12. For the page

- The storyline draft is PROPOSAL section 10, vocabulary section 11. Screen 6 (number of regions) can use
  `O.learning_curve`; screen 8 the realised-network diagnostic and the elasticities in its drawer.
- Results the page must not hide. The test tie between the log-linear model and nearest analogues, the
  under-forecast from the network level, the over-wide bands, and what the generator knowledge flatters.
- Do not write `card.json` before the page exists.
