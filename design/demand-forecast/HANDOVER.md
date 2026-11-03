# Demand forecast (model B), handover

For the local session that reviews the model and designs the page. Written 3 November 2026 at the end of
the cloud session on branch `claude/festive-johnson-bl41d7`. No frontend exists for this model and no
`card.json` is written; Kilian asked to stop before the page.

Read in this order. This file (what was built and how), `RESULTS.md` (results, disclosures, open
questions), `PROPOSAL.md` (the agreed design, sections 2 and 4a in particular), `ML_ROADMAP.md` section B
(decisions in one place).

---

## 1. The split, corrected

The brief for this handover speaks of training years 1 to 3, validation year 4 and test year 5. That is
the split of the clustering and Bought together pages. For this model Kilian agreed a different one
(PROPOSAL section 7, decision 8), because a five-year forecast needs five unseen years.

| Set | Months | Years | Used for |
|---|---|---|---|
| Training | 1 to 36 | 1 to 3 | Fitting every method for validation |
| Validation | 37 to 60 | 4 and 5 | Choosing every setting and the method, forecast from month 36 (1 to 24 months ahead) |
| Refit | 1 to 60 | 1 to 5 | Refitting the chosen settings before the test |
| Test | 61 to 120 | 6 to 10 | Scored once, forecast from month 60 (upper levels 60 months ahead, SKU levels 24) |

Years 6 to 10 exist only for this model (`scgame.world.future`); the other pages never read them.

## 2. Inputs

- **World v1 years 1 to 5** (`scgame.world.generator`, seed 7), unchanged byte for byte by this work
  (content hashes in `tests/test_world_future.py`). The generator now also writes `bulk_buyers.parquet`
  (who places the bulk orders), a new file; the six existing files are identical.
- **World v1 years 6 to 10** (`scgame.world.future`, random stream `[7, 6]`). Same customers, regions,
  prices, bundles and rules as years 1 to 5, plus three rules fixed before any model was fitted.
  - About 7.4 launches a year (Poisson; the average of years 1 to 5), drawn with the original category
    sizes, type mix and parameter ranges. Realised 7, 8, 4, 5, 4 (28 new SKUs, ids from 301).
  - At the start of each year the slowest sellers by expected units over the past year are delisted, as
    many as launch that year (Kilian's choice of 2 November 2026). On sale at year start 282 to 285.
  - A growing SKU's growth phase ends at an age drawn between 4 and 7 years (or its age at the end of
    year 5, if older); its trend is flat afterwards.
  - Bulk buyers of existing lumpy SKUs keep buying with their mean gaps; new lumpy SKUs draw buyers. A
    bundle partner is drawn only among SKUs on sale that day.
  - Files `skus_future.parquet` (all 328 SKUs with `stop_day` and `growth_stop_age`),
    `orders_future.parquet`, `order_lines_future.parquet`, `bulk_buyers_future.parquet`,
    `world_future.json` (launches, delistings, catalogue size).
- **Ground truth** (`scgame.world.replicate`). Years 4 to 10 re-run 200 times with fresh random streams
  `[7, 7, r]`, everything else fixed (catalogue with its future launches and delistings, customers,
  bundles, bulk buyers and their mean gaps). File `truth_replications.npz`, units per replication, region,
  SKU and month 37 to 120 (30 MB, not committed, like all of `data/world_*`).
- **Quantity forecast.** Units ordered per month (demand; the shop has no stock-outs). A month is one
  twelfth of a 365-day year (`month_of_day` in `replicate.py`).
- **Series** (`scgame.ml.demand_series`). Six levels, Total (1), Region (8), Category (8),
  Region × category (64), SKU (300) and Region × SKU (2,363 with a sale in months 1 to 36; 2,391 in months
  1 to 60). The upper hierarchy is the first four levels (81 series, bottom Region × category); SKU and
  Region × SKU are forecast two years ahead only. Bulk orders stay in the demand.

## 3. The method, step by step

Code in `backend/src/scgame/ml/`. `demand_series.py` (series, windows), `forecast_baselines.py`
(seasonal naive, ETS), `forecast_models.py` (LightGBM, network, recursion), `forecast_uncertainty.py`
(paths, reconciliation, scores), `demand_forecast.py` (the run).

1. **Series and windows.** For a forecast origin T (last observed month), a window holds the series' last
   12 months divided by its scale (mean of those months, from first sale if younger), growth over the last
   year (last 12 months over the 12 before, where the series is 24 months old), months since first sale
   (capped at 24), the share of empty months, the share of units in bulk lines (20 units or more), and the
   origin's calendar month. Static inputs are level, region, category, region population, income and urban
   share, and SKU log price and log unit volume. Fewer than 3 months since first sale, or a zero scale,
   falls back to seasonal naive.
2. **Seasonal naive.** Every future year repeats the last observed year; a series on sale for less than a
   year repeats the mean of its months on sale.
3. **ETS.** Holt-Winters on log(1 + y) from the first sale, additive damped trend and additive season,
   trend kept or dropped per series by AIC (statsmodels `ExponentialSmoothing`). Mean back-transformed
   with half the residual variance. Fewer than 24 months on sale, or a failed fit, falls back to seasonal
   naive. Forecasts from earlier origins reuse the fitted states (checked against statsmodels).
4. **LightGBM.** One global model over all series of all six levels, one row per series, origin and month
   ahead (1 to 12). Target is the month's units over the scale. Training windows at origins 12 to
   fit_end − 12. Weights proportional to the series' scale within its level and origin, each level summing
   to the same total. Category, region and level as native categorical features. Learning rate 0.05.
5. **Neural network.** scikit-learn `MLPRegressor`, two hidden layers, ReLU, Adam, learning rate 0.001,
   batch 256, the same windows with twelve outputs at once. Inputs standardised, categories one-hot,
   missing values as 0 plus a flag. Heavily weighted windows are repeated round(w) times and weighted
   w / copies (same objective, stable mini-batches). Five networks from seeds 0 to 4, averaged, trained in
   parallel processes, with snapshots after 25, 50, 100 and 200 epochs.
6. **Recursion.** The learned models forecast one year from the year before and are applied five times,
   feeding their own forecasts back in. Months since first sale advance (still capped), the empty and bulk
   shares stay at their origin values, nothing after the origin is visible (tested).
7. **Sample paths.** For each method, one-year-ahead errors at rolling origins 12 to fit_end − 12 (13
   origins for validation, 37 for the test) as log((y + 1) / (forecast + 1)). Each origin's block splits
   into a level error per series (the block mean) and month noise stored by calendar month. Both are
   centred per series over the origins (and per calendar month for the noise). A path draws one origin
   per forecast year for all series together; level errors add up over the years, month noise does not.
   1,000 paths. A width factor k multiplies the log errors, one for the upper levels and one for the SKU
   levels. For the learned models these errors are in-sample.
8. **Reconciliation.** Upper levels, bottom-up from Region × category or MinT with a shrunk covariance of
   the in-sample unit errors (Schäfer and Strimmer 2005), applied to every path. SKU levels always
   middle-out: Region × SKU paths scaled to the reconciled Region × category paths, SKU paths summed from
   them. Every level adds up.
9. **Quantiles and scores.** P10, P50 and P90 per series and month from the paths. Weighted quantile loss
   (mean over the three quantiles of 2 × summed pinball loss over summed actual units; equals WAPE at
   P50), WAPE of P50, bias of P50 and coverage of P10 to P90, per level and forecast year.
10. **Ground truth.** The replications aggregated to every series give the true mean and quantiles. Scored
    against the realised years, the true mean is the noise floor; the share of replications inside our
    band measures calibration.
11. **Capacity check.** At the end of year 5 each region builds capacity (units per month) for years 7 to
    10. Cost is 4 c K + m Σ (D − K)+ over 48 months, with c / m = ρ. The best K under a set of paths is the
    (1 − ρ / 12) quantile of all their path-months (sample average approximation, exact). Plans compared,
    each method's paths, our P50 only (the busiest month of the median forecast), last year's peak (year 5)
    and the ground truth (replications 0 to 99). Each is costed on the realised years and on replications
    100 to 199. ρ runs from 0.1 to 10 (41 values); the game's ratio is 0.2267 × price / margin per unit,
    0.76 to 0.83 by region.
12. **Explanations.** Mean absolute SHAP value per input of the test-fit LightGBM at month 60.

## 4. Baselines

- **Seasonal naive** and **ETS** are forecasting baselines run through the same paths, reconciliation and
  scores as the learned models.
- **Ground-truth mean** is the reference for model quality (what a perfect model of the process scores).
- **Capacity plans** have their own baselines, last year's peak (do nothing clever), our P50 only (ignore
  uncertainty) and the ground-truth plan (the best any forecast could do).

## 5. How every setting was chosen, and on which data

Everything below was chosen on the validation set (months 37 to 60, forecast from month 36, models fitted
on months 1 to 36), never on the test set or the ground truth.

| Setting | Candidates | Chosen by | Result |
|---|---|---|---|
| LightGBM hyperparameters | Leaves 15 or 63, rows per leaf 50 or 500, sampling 0.8 or 1.0, squared error or Tweedie (16), trees 250, 500, 1,000 or 1,500 (64 trials) | Mean WAPE of the unreconciled mean forecast over 6 levels × 2 years | 63 leaves, 500 rows, 0.8, Tweedie, 500 trees |
| Network hyperparameters | Width 32 or 64, penalty 0.0001, 0.001 or 0.01, squared error or Poisson (12), epochs 25, 50, 100 or 200 (48 trials) | Same | 64, 0.0001, Poisson, 50 epochs |
| ETS trend | Damped trend or none, per series | AIC on the fitting window | Per series |
| Upper width and reconciliation | k 0.5 to 6 (8 values) × bottom-up or MinT | Mean weighted quantile loss over the 4 upper levels × 2 years | Per method, see `validation.<method>.best` |
| SKU width | k 0.25 to 3 (8 values), with the chosen upper paths | Mean weighted quantile loss over the 2 SKU levels × 2 years | Per method |
| Method | Four methods at their best settings | Mean weighted quantile loss over 6 levels × 2 years | Neural network |
| Months-since-first-sale cap (24) | Uncapped or capped | Validation WAPE of the network (18% at Total uncapped, about 2% capped) | Capped |
| Network epochs instead of early stopping | Early stopping on the latest origin, or epochs as a setting | Early stopping stopped after 1 to 5 epochs on the training windows | Epochs as a setting |

Fixed in advance, not chosen. Learning rates (LightGBM 0.05, network 0.001), 1,000 paths, five network
seeds, the error offset of 1 unit, 200 replications, the rules for years 6 to 10, the capacity cost
formula.

## 6. Headline results and their output fields

From `data/models/demand-forecast/overview.json` (written `O`) and `series.json` (written `S`). Levels are
keyed by name ("Total", "Region", "Category", "Region × category", "SKU", "Region × SKU"); lists run over
forecast years.

| Result | Value | Field |
|---|---|---|
| Method chosen on validation | Neural network | `O.chosen` |
| Validation score per method | Network 0.117, LightGBM 0.124, ETS 0.170, seasonal naive 0.185 | `O.validation.<method>.best.score` |
| Chosen settings and widths | See section 5 | `O.validation.<method>.setting`, `.best.width_upper`, `.best.width_lower`, `.best.reconciliation` |
| Every trial and grid point | | `O.validation.<method>.trials`, `.grid_upper`, `.grid_lower` |
| Validation scores per level and year | e.g. network WAPE at Total 2.1%, 1.5% | `O.validation.<method>.best.table.<level>[j].wape` (also `wql`, `bias`, `coverage`) |
| Test WAPE at Total, network, years 6 to 10 | 2.4%, 4.1%, 7.6%, 15.0%, 24.5% | `O.test["Neural network"].table.Total[j].wape` |
| Test WAPE at Total, ETS | 2.7%, 2.3%, 3.4%, 2.7%, 4.0% | `O.test.ETS.table.Total[j].wape` |
| Test bias at Total, network | +1.9%, +3.8%, +7.6%, +15.0%, +24.5% | `O.test["Neural network"].table.Total[j].bias` |
| Test score over all cells | ETS 0.097, network 0.109, LightGBM 0.110, seasonal naive 0.113 | `O.test.<method>.score` |
| Test SKU level, network WAPE | 22.4%, 23.4% | `O.test["Neural network"].table.SKU[j].wape` |
| Noise floor at Total (ground-truth mean WAPE) | 1.9%, 1.3%, 1.6%, 1.4%, 1.4% | `O.truth.test.Total[j].wape_truth_mean` |
| Calibration at Total (replications inside our P10 to P90) | 70%, 50%, 22%, 2%, 0% | `O.truth.test.Total[j].truth_inside_our_p10_p90` |
| True band width over ours | 0.76 to 0.46 at Total | `O.truth.test.<level>[j].true_p10_p90_width_vs_ours` |
| Test coverage, network, years 6 and 7, Region to Region × category | 83% to 92% | `O.test["Neural network"].table.<level>[j].coverage` |
| Capacity cost over the ground-truth plan at ρ = 0.79 | ETS +3%, seasonal naive +5%, last year's peak +7%, LightGBM +12%, our forecast +15%, our P50 only +26% | `O.capacity.curves.<plan>.expected` against `O.capacity.curves["Ground truth"].expected`, index of `O.capacity.rho` nearest 0.79 |
| Game ratio per region | 0.76 to 0.83 | `O.capacity.rho_game` |
| Capacities and costs per region at the game ratio | | `O.capacity.per_region` |
| Strongest LightGBM inputs | Target month, growth, category, last month | `O.shap` |
| Total actual and network P50, years 6 to 10 | 362,338 … 405,455; 369,242 … 504,852 | `S.regions[0].actual[60:]`, `S.regions[0].test[1]` (rows P10, P50, P90) |
| Presets for the page (median-volume series per level) | | `S.presets[]` with `actual` (120 months), `validation.<method>` (24 months), `test` (60 or 24), `truth_test` (60) |

## 7. Chosen or changed after seeing test results or ground truth

- **The first full run was scored on the test years before review** (`run-1.log`). The independent review
  then found faults in the uncertainty layer, fixed in runs 2 and 3, each justified by validation evidence
  or a code error.
  - Shrinkage intensity used (n − 1)² instead of (n − 1)³, so MinT had been diagonal.
  - Level errors were not centred.
  - One width factor for all levels became two, with the upper grid extended to 6.
  - Month noise was drawn from the wrong calendar month; then each calendar month's noise was centred.
  - SKU levels became always middle-out.
- **What did not change.** The point forecasts, the hyperparameters, the chosen method and the world are
  identical in all three runs.
- **Test scores seen during development.** A dry run with tiny grids printed test scores; nothing was
  changed because of them.
- **The ground truth was used only to check and to explain** (the reviewer used it to show that the
  growth ceiling explains the turn after year 2). The sanity check that replicated years 4 and 5 match
  the realised totals (650k against 652k units) used validation years only.
- **Proposal stage.** Yearly totals of all five years and the generator's expected units were printed
  while profiling. The launch and end-of-life counts informed the rules for years 6 to 10, as the
  proposal discloses.

## 8. Known limits and doubts

1. **The test result after year 2 is decided by an assumption.** The growth ceiling of years 6 to 10 has
   no trace in years 1 to 5. The learned models extrapolate growth; ETS damps it and wins. One future,
   one set of rules.
2. **The validation set checks only two years ahead.** Years 3 to 5 ahead are first seen on the test.
3. **In-sample error samples for the learned models.** Their Total and Region errors are memorised (level
   error sd 0.005 against 0.024 out of sample), so bands rely on the width factor. SKU-level bands stay
   too narrow (54 to 62% coverage).
4. **The months-since-first-sale cap hides SKU age**, the only signal that could foresee the end of growth.
5. **The growth input is learned from one origin per series in the validation fit** (only origin 24 has
   two years of history).
6. **Middle-out spreads new SKUs' demand over existing SKUs** (1.1% of units in year 6, 2.4% in year 7).
7. **Path means sit slightly above the forecast** (the median is centred). Matters for the capacity check.
8. **Hyperparameters were chosen by the WAPE of the mean forecast,** which scores each SKU series' own
   forecast; the published SKU level is the middle-out sum.
9. **Departures from the agreed design** (all listed in `RESULTS.md` section 9). Epochs instead of early
   stopping (C5); loss in the network grid and the learning rate fixed (C7); repeated rows instead of
   weights; ETS on log(1 + y) with an additive season for every series; "3 months since first sale" for
   the fallback (A3 said 3 months of sales).
10. **Years 6 to 10 details.** New fading SKUs never end (only delisting removes them), new SKUs join no
    bundle, delisting assumes two bulk buyers per lumpy SKU, and some random draws for new SKUs' launch
    days are made and overwritten.
11. **World v1 quirk, kept.** 0.13% of units in years 1 to 5 are sold outside a SKU's life through bundles.
12. **Capacity check simplifications.** Each region is served by its own FC; the cost ratio comes from the
    game's factory capacity price; P50 only ignores the cost ratio by construction.
13. **Fallbacks.** Seasonal naive replaces ETS for 194 series in validation and 26 in the test, and the
    learned models for 86 and 133 series.

## 9. Dependencies

- Backend `lightgbm>=4.7.0` (4.7.0 locked in `uv.lock`), the only new dependency. The network uses
  scikit-learn; statsmodels was already there.
- No frontend change. `npm install` was run only for the checks.

## 10. Rebuild

From a fresh clone.

```bash
cd backend && uv sync
uv run python -m scgame.world.generator ../data/world_v1 7      # years 1 to 5, about 10 s
uv run python -m scgame.ml.demand_forecast ../data/world_v1 ../data/models/demand-forecast
```

The model module builds years 6 to 10 (`scgame.world.future`, about 10 s) and the 200 replications
(`scgame.world.replicate`, about 13 minutes on 4 cores) when they are missing, then runs about 47 minutes
(2,835 s in run 3, `O.runtime_s`), so about an hour in all on 4 cores. It pins linear algebra to one thread
per process, as in the committed run.

**Not yet verified.** A rebuild from a fresh clone was started (generator 11 s, years 6 to 10 built) and
stopped at Kilian's request during the replications; he checks it locally. That the rebuilt outputs equal
the committed ones is therefore unconfirmed. LightGBM is deterministic (identical SHAP values across the
three runs); the network depends on fixed seeds and single-threaded linear algebra. Steps can also be run
alone.

```bash
uv run python -m scgame.world.future ../data/world_v1
uv run python -m scgame.world.replicate ../data/world_v1 200 4
```

Checks.

```bash
cd backend && uv run pytest && uv run ruff check src
```

`tests/test_world_future.py` regenerates years 1 to 5 and compares content hashes (about 10 s);
`tests/test_demand_forecast.py` covers leakage, coherence, baselines, paths, the capacity optimum and the
committed outputs.

## 11. For the page

- The storyline draft is PROPOSAL section 10; vocabulary to approve is section 11. Kilian's open
  questions are RESULTS section 10 (how to present the turn after year 2, which forecast feeds O, the
  SKU bands, the capacity wording).
- Do not write `card.json` before the page exists.
