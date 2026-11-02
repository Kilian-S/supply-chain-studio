# Demand forecast, implementation proposal

Model B from `ML_ROADMAP.md`, revised 2 November 2026 after Kilian asked for a five-year forecast.
Nothing is built yet. Each section ends with a recommendation; the decisions are collected at the end.

Figures come from world v1, seed 7, training set (years 1 to 3) unless stated. While profiling I also
printed yearly unit totals for all five years, the generator's expected units by year, and the
catalogue's launch and end-of-life days. No model choice below rests on them. The launch and end
counts informed the rules for extending the world (section 2), which is disclosed there.

---

## 1. The question and the horizon

Kilian wants demand five years ahead. World v1 holds five years in total, so a five-year forecast can
neither be learned nor tested on it. The world is simulated, so the missing years can be generated.

Five years ahead is a planning question about regions and categories, not SKUs. By then most of
today's SKUs will have faded or been replaced, so a SKU forecast that far out means little. SKU and
Region × SKU forecasts stop at two years.

**Recommendation.** Forecast five years ahead at Total, Region, Category and Region × category; two
years ahead at SKU and Region × SKU. Keep the card "Demand five years out" with the question "How much
will each region and category sell over five years, and how sure are we?" (Kilian's wording, with
"product" changed to "category" because SKUs stop at two years).

## 2. Extending the world to ten years

| Item | Proposal |
|---|---|
| Years 1 to 5 | Byte for byte unchanged, so the clustering and Bought together pages and their results stand |
| Years 6 to 10 | Generated after the existing run with a separate random stream, same customers, regions, prices and bundles |
| Files | `orders_future.parquet`, `order_lines_future.parquet` and `skus_future.parquet` (new SKUs only) in `data/world_v1/`, so models that read `order_lines.parquet` see nothing new |
| Same rules as years 1 to 5 | Retail season, SKU seasonality, regional taste, basket sizes, bulk orders, no market growth, no prices or promotions |
| New SKUs | About 7 launches a year (the average of years 1 to 5, 37 launches in 5 years), drawn with the original type mix and parameter ranges, ids from 301 |
| Ends of life | Fading SKUs end as in years 1 to 5 (40% of them at a random day); every SKU launched in years 6 to 10 follows the same rules |
| Growth ceiling | A growing SKU's growth phase ends at a random age between 4 and 7 years, then it stays flat. Before day 1,825 the formula is unchanged |

Why these rules. The generator launches no SKUs after about year 3 and lets growing SKUs grow without
limit, so a plain continuation would shrink the catalogue and explode a few SKUs. Turnover at the
historical average rate keeps the catalogue near 300 SKUs.

What it costs in honesty. We write the rules of the years we then test on, which could suit the model
by design. The rules above are fixed before any model is fitted and are not changed after results are
seen; the page discloses them. One quirk stays visible. Years 4 and 5 have no launches (a side effect
of the original generator), while years 6 to 10 do, so the test asks the forecast to cope with a launch
rate the last two years did not show. The forecast cannot see a single new SKU; growth from future
launches reaches it only through the trend of the totals.

**Recommendation.** These rules, a test that years 1 to 5 stay identical, and disclosure on the page.

## 3. Inputs

| Item | Proposal | Fact |
|---|---|---|
| Quantity | Units ordered per month (demand, since the shop has no stock-outs) | 13,214 to 34,804 units per month in training, November the peak each year |
| Time step | Month, twelve equal parts of the 365-day year | 36 months of training, 60 of history at the test origin |
| Five-year levels | Total, Region, Category, Region × category | 81 series, Region × category at the bottom, no empty months |
| Two-year levels | Adds SKU and Region × SKU | 2,744 series, Region × SKU at the bottom, 34% of its months empty |
| Bulk orders | Kept in demand | Lines of 20 or more units are 11.6% of units |
| Other features | SKU price, unit volume, category, months since first sale; region population, income, urban share | All in the world files |

Weekly and daily forecasts are left out. Strategic questions need months, and day-to-day operations are
held constant by the standard replenishment bot.

**Recommendation.** Monthly units, the two sets of levels above, bulk orders kept.

## 4. Methods

Every method forecasts the mean of each series. Uncertainty is added the same way for all (section 5),
so the comparison is about the forecast, not about how each method makes intervals.

| Method | In plain words | Technical rule | Role |
|---|---|---|---|
| Seasonal naive | Every future year repeats the last one, month by month | ŷ(t + h) = y(t + h − 12k) | Baseline |
| ETS | Level, trend and season, updated as each month arrives; the trend flattens over time | Exponential smoothing with damped additive trend and multiplicative season, per series, settings by AIC; log(1 + y) where a series has empty months | Statistical baseline, also linear |
| LightGBM | One model learns from all series at once which recent patterns lead to which next year | Gradient-boosted trees on the target divided by the series' mean over the 12 months before the origin; features are the last 12 scaled months, growth over the last year, months since first sale, target month, level, category, region and region features, SKU price and volume | Main candidate |
| Neural network | The same task for a small network that reads the last 12 months and writes the next 12 | Global multilayer perceptron on the same scaled windows (scikit-learn), in the spirit of N-BEATS | Main candidate |

Each model forecasts one year from the year before it. Years further out come from applying it again,
feeding its own forecast back in, five times for five years. Training examples come from rolling
origins inside the fitting window (one per month with a year before and a year after it).

What the setting rewards and punishes, for the drawers. The same seasonal shape every year, constant
regional shares within a category, and no prices, promotions, events or market growth flatter every
method, the seasonal ones most. Growth comes only from product life cycles (41 SKUs sold over 50% more
in year 3 than in year 2, 33 sold a third less). Abrupt ends of life and new launches cannot be seen
coming.

**Recommendation.** The four methods. New backend dependency `lightgbm` only; the network uses
scikit-learn, so no PyTorch.

## 5. Uncertainty

Quantiles P10, P50 and P90, read from 1,000 sample paths per method. A path multiplies the mean
forecast by the method's own past errors, one year at a time. For each forecast year, one backtest
origin is drawn and its twelve months of errors are applied to every series together, so regions,
categories and months move together. Errors compound over the five years, so the band widens with
the horizon. One width factor per method, chosen on the validation set, corrects bands that come out
too narrow or too wide. The paths are also the demand scenarios O will need.

**Recommendation.** This joint error bootstrap, with the width factor as a hyperparameter.

## 6. Coherence across levels

Separate forecasts per level do not add up. Two ways to make them coherent, chosen on the validation
set. Bottom-up forecasts the bottom level and sums. MinT reconciliation forecasts every level and finds
the coherent set closest to all of them, weighting each by its past errors (shrunk covariance,
Wickramasuriya, Athanasopoulos and Hyndman, 2019). Each sample path is reconciled, so the quantiles are
coherent too.

**Recommendation.** Bottom-up versus MinT per method, chosen on the validation set.

## 7. Validation and test

| Step | Fitted on | Origin | Scored on | Purpose |
|---|---|---|---|---|
| Fit | Years 1 to 3 | | | Models, error samples, rolling origins |
| Validation | Years 1 to 3 | End of year 3 | Years 4 and 5, 1 to 24 months ahead | Choose hyperparameters, reconciliation, width factor and the method |
| Test | Years 1 to 5, refitted with the chosen setting | End of year 5 | Years 6 to 10, 1 to 60 months ahead (SKU levels to 24) | Scored once, by horizon year |

This model therefore uses year 5 as validation, not test. The agreed split (years 1 to 3, 4, 5) still
holds for the clustering and Bought together pages, and for this model years 6 to 10 play the part
of the test set.

Selection metric on the validation set. Pinball loss over P10, P50 and P90, divided by each series'
actual units (a percentage, equal to WAPE at P50), averaged with equal weight per level and per
forecast year. Reported beside it, never used to choose, are WAPE, bias and coverage (share of actual
months inside P10 to P90, 80% when calibrated).

Caveats for the page. Years 3 to 5 ahead are first checked on the test set, so the width of the band
there rests on the bootstrap, not on a check. The test refits on year 5, which the validation set also
used.

**Recommendation.** This protocol.

## 8. Comparison with the ground truth

Re-run the order stream 200 times with fresh random draws for years 4 to 10, same catalogue (with the
same future launches), customers, regions and bundles. That gives the true distribution of demand per
series and month.

- **Ground-truth mean as a forecast.** The error that remains with the process known. Our error above
  it is what a model could still gain; the rest is noise.
- **Calibration against the truth.** Whether our P10 and P90 match the true ones, year by year.
- **Expected cost of decisions** in section 9, free of the luck of one realised future.

About 40 minutes of compute, run once. Used to check, never to choose.

**Recommendation.** 200 replications, disclosed on the page.

## 9. Business check, a one-off capacity build

At the end of year 5, each region builds its FC capacity (units per month) for years 7 to 10. Building
takes two years and capacity cannot be reduced, as in the game. Each unit of capacity costs a fixed
amount per year; each unit of demand above capacity in a month is a lost sale costing its gross
margin. The best capacity under a forecast minimises the average cost over its sample paths, one small
stochastic programme per region (sample average approximation, solved exactly by sorting). Staged
expansion over the years is left to O.

| Plan | Capacity from |
|---|---|
| Our forecast | Its sample paths for years 7 to 10 |
| Our P50 only | The peak month of the P50 forecast, no allowance for uncertainty |
| Seasonal naive | Its sample paths, which repeat year 5 |
| Last year's peak | Year 5's peak month, the do-nothing plan |
| Ground truth | The 200 replications, the best any forecast could do |

Scored on years 7 to 10 twice, realised cost and expected cost under the true process. The cost ratio
is the main assumption. In the game a unit of daily capacity costs $50,000 against a $1,450 drum,
which over a five-year life is a yearly cost of about 23% of the unit price per unit of monthly
capacity, against a lost margin of 10 to 60% of the price by category. The results show the whole range
of ratios with the game's value marked.

**Recommendation.** This check, costs from the game's ratio, results across the full range.

## 10. Draft storyline

Titles are plain noun phrases, captions only where a visual cannot be read without one, validation
before test, ground truth second to last, the business result last, no recap.

**Models home card.** Demand five years out. "How much will each region and category sell over five
years, and how sure are we?"

**Opening.** Title "Demand five years out". Question "How far ahead can five years of sales see?"
Boxes "Series" (2,744), "Levels" (6), "Years of history" (5).

1. **Monthly sales.** Six small line charts (3 × 2), one per level, each the median-volume series of
   its level over the training set. Caption "Median-volume series of each level, training set".
2. **Monthly sales per SKU.** Heatmap of 300 SKUs by 36 months, colour the month's units over the SKU's
   training mean, rows ordered by growth from year 2 to year 3. Launches, growth, fades and the
   November peak in one picture.
3. **Forecast origins and horizons.** A ten-year strip with both origins, years 4 and 5 as validation,
   years 6 to 10 as the test. Drawer on how years 6 to 10 were generated (section 2).
4. **Forecasting methods.** Plain words beside the technical rules, and a 2 × 2 grid of validation
   forecasts (P50 line, P10 to P90 band, actuals) for the median-volume series of four levels, with a
   warning that single series invite cherry-picking.
5. **Reconciliation.** Eight region forecasts stacked against the total forecast, before and after.
6. **Validation.** Heatmap of methods and reconciliation by level, pinball loss (%) on years 4 and 5,
   the chosen setting outlined, coverage as a last column.
7. **Five-year forecast.** The test fan (P10 to P90, P50) from the end of year 5 against years 6 to
   10, for the total, with presets for each region.
8. **Forecast versus ground truth.** WAPE by forecast year (1 to 5) per level for the ground-truth
   mean, our forecast and seasonal naive. Drawer on calibration and on what flatters the setting.
9. **Results.** Capacity cost of each plan across the cost ratio, the game's ratio marked; boxes for
   our plan at that ratio. Drawer with the per-region table and plain-word rules for each plan.

## 11. Vocabulary

New surface terms to approve. Forecast, forecast origin, horizon, forecast year, series, level,
seasonal naive, ETS (exponential smoothing), LightGBM, gradient boosting, neural network, quantile,
P10, P50, P90, prediction interval, coverage, pinball loss, WAPE, bias, reconciliation, coherent,
bottom-up, sample path, capacity, stochastic programme.

Drawer only (to add to `GLOSSARY.md`). MinT, shrinkage, AIC, damped trend, rolling origin, joint
bootstrap, N-BEATS, multilayer perceptron, sample average approximation, critical ratio.

## 12. Build plan after approval

1. Generator extension for years 6 to 10 and the ground-truth replications, with a test that years 1
   to 5 are unchanged.
2. `backend/src/scgame/ml/demand_forecast.py` with tests; outputs in `data/models/demand-forecast/`, no
   `card.json` until the page exists.
3. Storyline spec, critics and page as for Bought together, then `frontend/src/models/forecast/` with
   `copy.ts` under `lexicon.test.ts`.
4. A `DESIGN_REVIEW.md` entry per review round.

## 13. Decisions for Kilian

Each with my recommendation; "yes to all" works.

| # | Decision | Recommendation |
|---|---|---|
| 1 | Horizon and card | Five years at region and category levels, two at SKU levels; card "Demand five years out" |
| 2 | World extension | Years 6 to 10 by the rules in section 2, fixed before fitting, years 1 to 5 unchanged |
| 3 | Granularity | Monthly units; no weekly or daily |
| 4 | Bulk orders | Kept in demand |
| 5 | Methods | Seasonal naive, ETS, LightGBM, neural network (scikit-learn); add `lightgbm` |
| 6 | Uncertainty | Joint error bootstrap, 1,000 paths, width factor chosen on validation |
| 7 | Reconciliation | Bottom-up versus MinT, chosen on validation |
| 8 | Protocol | Fit on years 1 to 3, validate on years 4 and 5, test on years 6 to 10 after a refit |
| 9 | Selection metric | Scaled pinball loss, equal weight per level and forecast year |
| 10 | Ground truth | 200 replications of years 4 to 10, check only, disclosed |
| 11 | Business check | One-off regional capacity build for years 7 to 10, game cost ratio, full range shown |
| 12 | Storyline | Nine screens as in section 10 |
| 13 | Vocabulary | Section 11 |
| 14 | Process | Same agent loop as Bought together, then your screen-by-screen review |
