# Demand forecast, results of phases 1 to 3

Report for Kilian's review, 2 November 2026. Design in `PROPOSAL.md` (agreed). Every number comes from
`data/models/demand-forecast/overview.json` and `series.json` of run 3 (`run-3.log`). No page exists yet.

---

## 1. Summary

- **What was built.** Years 6 to 10 of the world by the agreed rules, 200 ground-truth replications of
  years 4 to 10, four forecasting methods with sample paths and reconciliation, validation on years 4 and
  5, one test on years 6 to 10, the comparison with the ground truth and the capacity check. An
  independent reviewer checked the work twice (section 8).
- **Chosen on validation.** The neural network, weighted quantile loss 0.117, ahead of LightGBM (0.124),
  ETS (0.170) and seasonal naive (0.185).
- **Test, one year ahead.** The two learned models are the best methods at every level. At Total the
  network misses by 2.4% (WAPE), against 1.9% for the ground-truth mean, the noise floor.
- **Test, three to five years ahead.** The network over-forecasts by 8%, 15% and 25% at every upper level.
  It carries forward the growth of years 1 to 5 (4 to 8% a year), while demand in years 6 to 10 flattens
  (362k, 384k, 401k, 406k, 405k units) because of the growth ceiling we wrote into years 6 to 10. ETS,
  whose damped trend flattens by design, misses by 2 to 4% at Total in every year and is the best method
  at Total, Region and Category from year 3 and at Region × category from year 4. Over all five years ETS has the best test score (0.097 against
  0.109 for the network).
- **Capacity check (years 7 to 10, cost ratio 0.79).** The plan from our forecast costs 15% more than the
  plan from the ground truth, more than building to last year's peak (7%). The ETS plan costs 3% more. The
  plan that ignores uncertainty (P50 only) costs 26% more.
- **Uncertainty.** After the review fixes the bands are close to calibrated in the first two test years at
  Region, Category and Region × category (83 to 92% of months inside P10 to P90, 80% when calibrated),
  too narrow at Total (67% and 50%, of 12 months each) and at the SKU levels (54 to 62%).

The honest reading is that the forecast is good for the horizon we could validate and wrong from year 3,
and that the wrong part comes from an assumption about the future (growth stops) that no history could
show.

## 2. Validation (years 4 and 5, forecast from month 36)

Weighted quantile loss, mean over six levels and two years; lower is better.

| Method | Score | Setting chosen on validation | Width factor (upper, SKU levels) | Reconciliation |
|---|---|---|---|---|
| Neural network | **0.117** | 64 units per layer, penalty 0.0001, Poisson loss, 50 epochs | 2.0, 0.75 | MinT |
| LightGBM | 0.124 | 63 leaves, 500 rows per leaf, sampling 0.8, Tweedie, 500 trees | 1.5, 1.0 | MinT |
| ETS | 0.170 | Trend kept or dropped per series by AIC | 2.0, 1.0 | MinT |
| Seasonal naive | 0.185 | None | 1.5, 0.75 | Bottom-up |

The network on validation, WAPE against the ground-truth mean, and coverage of P10 to P90 (80% when
calibrated).

| Level | WAPE year 4 | Year 5 | Ground-truth mean year 4 | Year 5 | Coverage year 4 | Year 5 |
|---|---|---|---|---|---|---|
| Total | 2.1% | 1.5% | 1.5% | 1.2% | 25% | 83% |
| Region | 6.3% | 6.6% | 5.7% | 5.4% | 58% | 64% |
| Category | 7.5% | 9.9% | 4.1% | 3.6% | 42% | 43% |
| Region × category | 14.1% | 16.6% | 10.7% | 10.3% | 60% | 60% |
| SKU | 22.6% | 29.4% | 15.1% | 14.8% | 39% | 41% |
| Region × SKU | 38.2% | 42.5% | 29.7% | 28.2% | 38% | 45% |

ETS and seasonal naive fall back to seasonal naive for 194 and 0 series; the two learned models for 86
series with fewer than 3 months on sale.

## 3. Test (years 6 to 10, forecast once from month 60)

WAPE by forecast year.

| Level | Method | Year 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| Total | Neural network (chosen) | 2.4% | 4.1% | 7.6% | 15.0% | 24.5% |
| Total | LightGBM | 2.2% | 3.1% | 6.3% | 13.4% | 22.7% |
| Total | ETS | 2.7% | 2.3% | 3.4% | 2.7% | 4.0% |
| Total | Ground-truth mean | 1.9% | 1.3% | 1.6% | 1.4% | 1.4% |
| Region | Neural network (chosen) | 5.1% | 6.2% | 9.3% | 14.9% | 24.0% |
| Region | ETS | 6.7% | 7.3% | 7.5% | 7.4% | 9.1% |
| Region | Ground-truth mean | 4.2% | 4.2% | 4.1% | 4.5% | 4.3% |

SKU levels (two years). The network misses by 22.4% and 23.4% at SKU and 34.9% and 35.2% at Region × SKU,
against 13.7% and 13.2%, 26.7% and 25.9% for the ground-truth mean. ETS misses by 38.1% and 37.0% at SKU.
At the SKU levels the learned models are clearly the best methods.

Weighted quantile loss over all test cells. ETS 0.097, network 0.109, LightGBM 0.110, seasonal naive 0.113.

Total units, actual and network P50 (years 6 to 10).

| | Year 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|
| Actual | 362,338 | 384,009 | 400,620 | 405,763 | 405,455 |
| Ground-truth median | 363,711 | 383,578 | 397,985 | 404,414 | 405,972 |
| Network P50 | 369,242 | 398,681 | 431,243 | 466,651 | 504,852 |

## 4. Why the ranking turns after year 2

- Years 1 to 5 grew 4 to 8% a year, driven by growing SKUs. No growing SKU stopped growing before year 5.
- Years 6 to 10 follow the agreed rule that a growing SKU's growth phase ends at an age between 4 and 7
  years, so total growth falls to about 1% in year 9 and 0% in year 10. The realised years are a typical
  draw of that process (each within about one standard deviation of the ground-truth mean).
- The two learned models learn that a series which grew x% last year grows about x% next year, and apply
  it five times. The cap on months since first sale (24) also hides how old a SKU is, the only signal that
  could predict the end of growth.
- ETS damps its trend by construction, which happens to match the ceiling. On validation (years 4 and 5,
  still growing) the damping cost it; on the test it wins.
- The validation set only checks one and two years ahead, so nothing before the test could show this. The
  proposal named this caveat (section 7).

## 5. Ground truth

- **Noise floor.** The ground-truth mean misses the realised year by 1.3 to 1.9% at Total, 4.1 to 4.5% at
  Region, 8.5 to 9.8% at Region × category, 13% at SKU and 26% at Region × SKU. In year 6 the network is
  within about 1 point of that floor at Total and Region.
- **Calibration against the truth.** Share of the 200 replications inside our P10 to P90 band, years 6 to
  10. Total 70%, 50%, 22%, 2%, 0%; Region 86%, 85%, 75%, 56%, 34%; Region × category 89%, 93%, 91%, 85%,
  78%; SKU 57%, 60%; Region × SKU 54%, 62%. Our upper bands are wider than the true noise alone (the true
  P10 to P90 width is 0.33 to 0.76 of ours), as they should be, since they also carry model error; from
  year 3 on they are centred in the wrong place.

## 6. Capacity check (years 7 to 10)

Expected cost over the 100 held-out replications, relative to the plan built from the ground truth, at a
cost ratio of 0.79 (the game's ratio is 0.76 to 0.83 by region).

| Plan | Extra cost over the ground-truth plan |
|---|---|
| ETS | 3% |
| Seasonal naive | 5% |
| Last year's peak | 7% |
| LightGBM | 12% |
| Our forecast (network) | 15% |
| Our P50 only | 26% |

Across the whole range of cost ratios (0.1 to 10), our plan costs 0.4 to 27% more than the ground-truth
plan, and more than the ETS plan at every ratio below 10. Building to last year's peak is far worse when capacity
is cheap (139% at a ratio of 0.1), close to the best between ratios of 1 and 3, and worse again when
capacity is dear. P50 only builds for the median forecast's busiest month whatever capacity costs, so it
gets worse as capacity gets dearer.

## 7. What the setting rewards and punishes

- **Rewards every method.** The same seasonal shape every year, constant regional shares within a
  category, no prices, promotions, events or market growth, Poisson noise.
- **Punishes the learned models.** Growth that stops in years 6 to 10, a rule we fixed before fitting but
  absent from the history. The test is one future; other future rules would give another ranking after
  year 2.
- **Cannot be forecast by any method.** New SKUs (28 in years 6 to 10) and the delisting of the slowest
  sellers. Middle-out spreads the new SKUs' demand (1.1% of units in year 6, 2.4% in year 7) over existing
  SKUs, because new SKUs have no series of their own.

## 8. Changes made after the first test run (disclosure)

The first full run (`run-1.log`) was scored on the test years before the independent review. The review
found no leakage and no bug in the point forecasts, but faults in the uncertainty layer. They were fixed
in two steps (run 2 after the first review, run 3 after the reviewer checked the fixes).

| Fault | Fix | Evidence that did not come from the test |
|---|---|---|
| Shrinkage intensity used (n − 1)² instead of (n − 1)³, so MinT was diagonal | Formula corrected | Code against Schäfer and Strimmer (2005) and the R package hts |
| Level errors were not centred, so paths carried past drift and "seasonal naive" was not seasonal naive | Centred per series | Validation P50 differed from the mean forecast by 3 to 6% at Total |
| Each calendar month's noise was not centred, so the width factor still moved the P50 | Centred per series and calendar month | Seasonal naive's validation bias at Total moved from −9% to +18% as the width grew |
| One width factor for all levels | Separate factors for the upper and the SKU levels | Validation coverage 8 to 45%; wider settings made SKU paths explode on validation |
| Month noise drawn from the wrong calendar month | Stored and drawn by calendar month | Code |
| SKU levels could be left unreconciled | Always middle-out, as PROPOSAL section 6 promised | Proposal |

The point forecasts, the hyperparameters, the chosen method and the world did not change between the
runs. A dry run with tiny grids also printed test scores during development; nothing was changed because
of them.

## 9. Decisions taken while working alone (to revisit)

1. **Months since first sale capped at 24.** Without it the network extrapolated wildly (validation WAPE at
   Total 18% instead of about 2%). It also hides SKU age, which matters for section 4.
2. **Network epochs chosen on validation (25, 50, 100, 200) instead of early stopping on the latest
   origin** (agreed choice C5), which stopped training after 1 to 5 epochs.
3. **Network loss as a setting (squared error or Poisson), learning rate fixed at 0.001** (C7 listed the
   learning rate). Poisson was chosen.
4. **Heavily weighted windows repeated instead of weighted.** Same objective, stable training.
5. **Hyperparameters chosen by the WAPE of the mean forecast; widths, reconciliation and the method by
   weighted quantile loss.**
6. **Error samples are in-sample for the learned models.** The reviewer recommends out-of-sample errors;
   not done (section 10).
7. **Years 6 to 10 details.** Bulk buyers of existing lumpy SKUs keep buying; new lumpy SKUs draw new
   buyers; bundle partners only among SKUs on sale; new SKUs join no bundle and new fading SKUs never end
   (only delisting removes them); delisting assumes two bulk buyers per lumpy SKU.
8. **Generator refactor.** World v1 years 1 to 5 are unchanged (content hashes tested); a new file
   `bulk_buyers.parquet` records the bulk buyers. A quirk of world v1 was found and kept, 0.13% of units
   are sold outside a SKU's life through bundles.
9. **Path means sit slightly above the mean forecast** (the median is centred), which matters only where
   paths serve as scenarios (the capacity check).

## 10. Questions for Kilian

1. **How to present the reversal.** My recommendation is to make it the point of the page. The forecast is
   validated for two years and good there; beyond that the assumption about growth decides, and the page
   shows the learned forecast and the damped one against the ground truth. Changing the world rules now
   would be tuning the test to the model.
2. **Which forecast feeds the network plan (O).** I recommend the chosen network for the first two
   years and say plainly that beyond them it rests on an assumption; or, if you prefer one rule, choose the
   method on a longer backtest once more history exists. Do not switch to ETS because it won the test.
3. **SKU-level bands.** Still too narrow (54 to 62%). I recommend out-of-sample error samples, and,
   because the test years have now been seen, score the change on a second, untouched future (years 6 to
   10 drawn with another seed under the same rules), disclosed on the page.
4. **Capacity check wording.** Our chosen plan loses to last year's peak at the game's ratio.
   I recommend showing it as it is, with the ETS plan beside it, and keep model quality (sections 3 and 5)
   apart from business value (section 6).
5. **Storyline.** Section 10 of the proposal still fits, with screen 7 showing the fan against the actual
   and the ground truth, and the results screen led by the capacity cost across cost ratios.
