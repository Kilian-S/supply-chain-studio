# Demand forecast, implementation proposal

Model B from `ML_ROADMAP.md`, prepared 2 November 2026 for Kilian's review. Nothing is built yet.
Each section ends with a recommendation; the decisions to take are collected at the end.

Figures come from world v1, seed 7, training set (years 1 to 3) unless stated. While profiling I also
printed yearly unit totals for all five years and the generator's expected units by year. No choice
below rests on them, and the page will not quote them.

---

## 1. The question and the horizon

The models home card promises "Demand five years out". The world holds five years in total, and the
agreed split keeps years 4 and 5 for validation and test, so the longest horizon we can test is two
years (a forecast made at the end of year 3 for year 5). Two years is also the horizon that matters for
the network. Building or expanding an FC takes about two years, so capacity for year 5 is decided at
the end of year 3.

Years 6 to 10 (the five-year plan for O) have no data at all. The same model can be refitted on years
1 to 5 and run forward, but nothing can check it except the generator. And the generator has no
launches after about year 3, while its growing SKUs grow without limit, so its years 6 to 10 would
need work first. That belongs with O.

**Recommendation.** Test forecasts up to 24 months ahead. Rename the card "Demand forecast", with
the question "How much will each region and category sell two years ahead, and how sure can we be?".
Leave years 6 to 10 to O.

## 2. Inputs

| Item | Proposal | Training-set fact |
|---|---|---|
| Quantity | Units ordered per month (demand, since the shop has no stock-outs yet) | 13,214 to 34,804 units per month, November the peak each year |
| Time step | Month, twelve equal parts of the 365-day year | 36 months of training |
| Bottom series | Region × SKU | 2,363 series with a sale, 34% of their months without one |
| Levels | Total, Region, Category, Region × category, SKU, Region × SKU | 2,744 series in all |
| Strategic outputs | Region and Region × category (capacity and mix) | 8 and 64 series, no empty months |
| Bulk orders | Kept in the demand, not cleaned out | Lines of 20 or more units are 11.6% of units |
| Other features | SKU price, unit volume, category, months since first sale; region population, income, urban share | All in the world files |

Weekly and daily forecasts are left out. Strategic questions need months, and day-to-day operations are
held constant by the standard replenishment bot.

**Recommendation.** Monthly units, six levels with Region × SKU at the bottom, bulk orders kept.

## 3. Methods

Every method forecasts the mean of each series. Uncertainty is added the same way for all of them
(section 4), so the comparison is about the forecast and not about how each method happens to make
intervals.

| Method | In plain words | Technical rule | Role |
|---|---|---|---|
| Seasonal naive | Next year repeats the last one, month by month | ŷ(t + h) = y(t + h − 12k) | Baseline |
| ETS | Level, trend and season, updated as each month arrives; the trend flattens over time | Exponential smoothing with damped additive trend and multiplicative season, per series, settings by AIC; log(1 + y) where a series has empty months | Statistical baseline |
| LightGBM | One model learns from all 2,744 series at once which recent patterns lead to which future | Gradient-boosted trees on the target divided by the series' mean over the 12 months before the forecast origin; features are the last 12 scaled months, growth over the last year, months since first sale, target month, level, category, region and region features, SKU price and volume | Main candidate |
| Neural network | The same task for a small network that reads the last 12 months and writes the next 12 | Global multilayer perceptron on the same scaled windows (scikit-learn), in the spirit of N-BEATS | Main candidate |

Training examples for the two global models come from rolling forecast origins inside the training
set (months 12 to 24, so every example has a year of history and a year of future). Forecasts 13 to
24 months ahead apply the 12-month model twice, feeding its own first year back in. Three years of
history hold only one or two full "next year" examples per series, which is the real limit of this
problem and will be said on the page.

What the setting rewards and what it punishes, for the page's drawers. The generator repeats the same
seasonal shape every year, keeps each region's share of a category constant, and has no prices,
promotions, events or market growth, which flatters every method, the seasonal ones most. Growth
comes only from product life cycles (41 SKUs sold over 50% more in year 3 than in year 2, 33 sold a
third less). Ends of life arrive abruptly in the generator and no method can see them coming.

**Recommendation.** The four methods above. New backend dependency `lightgbm` only (a few MB); the
network uses scikit-learn, so no PyTorch.

## 4. Uncertainty

Quantiles P10, P50 and P90 at every level, read from 1,000 sample paths per method. A path is the
mean forecast multiplied by one year of the method's own past errors, taken from one backtest origin
in the training set for all 2,744 series and twelve months together. Drawing whole years keeps
regions, categories and months moving together, which is what O needs later (its scenarios are these
paths). One width factor per method, chosen on the validation set, corrects intervals that come out
too narrow or too wide.

**Recommendation.** Sample paths by this joint error bootstrap, with the width factor as a
hyperparameter.

## 5. Coherence across levels

Forecasts made separately per level do not add up (eight region forecasts rarely sum to the total
forecast). Two ways to make them coherent, chosen on the validation set.

- Bottom-up, which forecasts Region × SKU and sums.
- MinT reconciliation, which forecasts every level and finds the coherent set closest to all of them,
  weighting each by its past errors (shrunk covariance, Wickramasuriya, Athanasopoulos and Hyndman,
  2019).

Each sample path is reconciled, so the quantiles are coherent too (a linear projection keeps the
paths valid).

**Recommendation.** Compare bottom-up and MinT per method on the validation set and keep the better.

## 6. Validation and test

| Step | Data used to fit | Forecast origin | Scored on | Purpose |
|---|---|---|---|---|
| Fit | Months 1 to 36 | | | Models, error samples, rolling origins 12 to 24 |
| Validation | Months 1 to 36 | End of year 3 | Year 4, 1 to 12 months ahead | Choose hyperparameters, reconciliation, width factor and the method |
| Test, two years ahead | Months 1 to 36, the chosen setting unchanged | End of year 3 | Year 5, 13 to 24 months ahead | The capacity decision, scored once |
| Test, one year ahead | Months 1 to 48, refitted with the chosen setting | End of year 4 | Year 5, 1 to 12 months ahead | How accuracy falls with the horizon, scored once |

Selection metric on the validation set. Pinball loss over P10, P50 and P90, divided by each series'
actual units (so it reads as a percentage, and at P50 equals WAPE), averaged with equal weight per
level so neither the 2,363 small series nor the single total decide alone. Reported next to it, never
used to choose, are WAPE, bias and coverage (share of actual months inside P10 to P90, 80% when
calibrated).

Two caveats for the page. The validation set only checks 1 to 12 months ahead, so the two-year
forecast is first checked on the test set. And the one-year test forecast is refitted on year 4, which
the validation set also used.

**Recommendation.** This protocol, including the refit for the one-year test forecast.

## 7. Comparison with the ground truth

The generator knows each SKU's process, so the true distribution of demand can be computed. Re-run
the order stream for years 4 and 5 200 times with the same catalogue, customers, regions and bundles
and fresh random draws. That gives, for every series and month, the true mean and quantiles.

- **Ground-truth mean as a forecast.** The error that remains even with the process known. Our error
  above it is what the model could still gain; the rest is noise.
- **Calibration against the truth.** Whether our P10 and P90 match the true ones.
- **Expected cost of decisions** in section 8, free of the luck of one realised year 5.

The replications need a small function beside the generator that reuses its order loop with a second
random stream. World v1's files stay byte for byte identical, so the clustering and Bought together
pages are untouched. About 20 minutes of compute, run once.

**Recommendation.** 200 replications, used to check and never to choose, disclosed on the page.

## 8. Business check, FC capacity two years ahead

At the end of year 3, choose each region's capacity for year 5 in units per month. Each unit of
capacity costs a fixed amount per year; each unit of demand above capacity in a month is a lost sale
and costs the unit's gross margin (the game's rule, an order not shipped is lost). The best capacity
under a forecast minimises the average cost over its sample paths, one small stochastic programme per
region (sample average approximation, solved exactly by sorting).

| Plan | Capacity from |
|---|---|
| Our forecast | Its sample paths |
| Our P50 only | The peak month of the P50 forecast, no allowance for uncertainty |
| Seasonal naive | Its sample paths, same uncertainty method |
| Last year's peak | Year 3's peak month, the do-nothing plan |
| Ground truth | The 200 replications, the best any forecast could do |

Scored on year 5 twice, realised cost on the actual year (one draw, eight regions, so noisy) and
expected cost under the true process. The cost ratio is the main assumption. From the game, a unit of
daily capacity costs $50,000 against a $1,450 drum, which over a five-year life is a yearly cost of
about 23% of the unit price per unit of monthly capacity, against a lost margin of 10 to 60% of the
price by category. Because the answer depends on that ratio, the results show the whole range of
ratios with the game's value marked.

What the check can and cannot say. It values the forecast's uncertainty, not only its accuracy, which
is the point for capacity. It ignores the network (each region is treated as served by its own FC),
which is O's job.

**Recommendation.** This check, costs from the game's ratio, results across the full range of ratios.

## 9. Draft storyline

Under the design rules. Titles are plain noun phrases; captions only where a visual cannot be read
without one; validation before test, ground truth second to last, the business result last; no recap.

**Models home card.** Demand forecast. "How much will each region and category sell two years ahead,
and how sure can we be?"

**Opening.** Title "Demand forecast". Question "Can three years of sales say how much each region will
sell two years later?" Boxes "Series" (2,744), "Levels" (6), "Months of training" (36).

1. **Monthly sales.** Six small line charts in a 3 × 2 grid, one per level, each the median-volume
   series of its level over the training set. The smooth total and the sparse Region × SKU side by
   side show why one method must serve both. Caption "Median-volume series of each level, training
   set". Drawer with series counts and empty-month shares per level.
2. **Monthly sales per SKU.** Heatmap of 300 SKUs by 36 months, colour the month's units over the
   SKU's training mean, rows ordered by growth from year 2 to year 3. Launches, growth, fades and the
   November peak in one picture. Drawer explains that the total's growth is the sum of these life
   cycles.
3. **Forecast origins and horizons.** The year strip with the two origins and their arrows (section
   6). Drawer on the two-year FC lead time, why months, and what is chosen where.
4. **Forecasting methods.** The four methods in plain words beside their technical rules, and a 2 × 2
   grid of one series' validation forecasts (P50 line, P10 to P90 band, year 4 actuals). Presets
   Total, a region, a SKU and a Region × SKU, each the median-volume series of its level, so the
   presets are not picked for fit. A warning that single series invite cherry-picking.
5. **Reconciliation.** Eight region forecasts stacked against the total forecast for year 4, before
   and after reconciliation, so the step from separate to coherent forecasts is visible. "Coherent"
   grows out of "add up".
6. **Validation.** Heatmap of methods and reconciliation (rows) by level (columns), cells the pinball
   loss (%) on year 4, the chosen setting outlined; coverage as a last column.
7. **Forecast versus ground truth.** Test set. Per level, WAPE of the ground-truth mean, of our
   forecast one year ahead and two years ahead, and of seasonal naive, as a dot plot. Drawer on
   calibration against the true quantiles and on what flatters the setting (section 3).
8. **Results.** Capacity cost of each plan in section 8 against the cost ratio, the game's ratio
   marked, realised and expected cost; boxes for our plan at the game's ratio. Drawer with the
   per-region table and the plain-word rules for each plan.

## 10. Vocabulary

New surface terms to approve. Forecast, forecast origin, horizon (months ahead), series, level,
seasonal naive, ETS (exponential smoothing), LightGBM, gradient boosting, neural network, quantile,
P10, P50, P90, prediction interval, coverage, pinball loss, WAPE, bias, reconciliation, coherent,
bottom-up, sample path, capacity, stochastic programme.

Drawer only (to add to `GLOSSARY.md`). MinT, shrinkage, AIC, damped trend, rolling origin, joint
bootstrap, N-BEATS, multilayer perceptron, sample average approximation, critical ratio.

## 11. Build plan after approval

1. `backend/src/scgame/ml/demand_forecast.py` (series, four methods, paths, reconciliation, selection,
   test) and `backend/src/scgame/world/replicate.py` (ground-truth replications), with tests.
2. Outputs in `data/models/demand-forecast/`, no `card.json` until the page exists.
3. Storyline spec, critics and page as for Bought together (`design/demand-forecast/`), then
   `frontend/src/models/forecast/` with `copy.ts` under `lexicon.test.ts`.
4. A log entry in `DESIGN_REVIEW.md` per review round.

## 12. Decisions for Kilian

Each with my recommendation; "yes to all" works.

| # | Decision | Recommendation |
|---|---|---|
| 1 | Horizon and card | Up to 24 months ahead; card "Demand forecast"; years 6 to 10 left to O |
| 2 | Granularity | Monthly units; no weekly or daily |
| 3 | Hierarchy | Six levels, Region × SKU at the bottom |
| 4 | Bulk orders | Kept in demand |
| 5 | Methods | Seasonal naive, ETS, LightGBM, neural network (scikit-learn); add `lightgbm` |
| 6 | Uncertainty | Joint error bootstrap, 1,000 paths, width factor chosen on validation |
| 7 | Reconciliation | Bottom-up versus MinT, chosen on validation |
| 8 | Protocol | As in section 6, with the refit for the one-year test forecast |
| 9 | Selection metric | Scaled pinball loss, equal weight per level |
| 10 | Ground truth | 200 replications, check only, disclosed |
| 11 | Business check | Regional capacity two years ahead, game cost ratio, full range of ratios shown |
| 12 | Storyline | Eight screens as in section 9 |
| 13 | Vocabulary | Section 10 |
| 14 | Process | Same agent loop as Bought together, then your screen-by-screen review |
