# Demand forecast, results of phases 1 to 3

Report for Kilian's review, 2 November 2026. Design in `PROPOSAL.md` (agreed). Numbers from
`data/models/demand-forecast/overview.json` and `series.json` (run 2, `run-2.log`). No page exists yet.

---

## 1. Summary

- **What was built.** Years 6 to 10 of the world by the agreed rules, 200 ground-truth replications of
  years 4 to 10, four forecasting methods with sample paths and reconciliation, validation on years 4 and
  5, one test on years 6 to 10, the comparison with the ground truth and the capacity check.
- **Chosen on validation.** The neural network (weighted quantile loss 0.138), just ahead of LightGBM
  (0.141), with ETS (0.183) and seasonal naive (0.209) behind.
- **Test, one and two years ahead.** The network and LightGBM are the best methods at every level. At
  Total the network misses by 2.1% and 3.6% (WAPE), against 1.9% and 1.3% for the ground-truth mean (the
  noise floor).
- **Test, three to five years ahead.** The network over-forecasts by 7%, 14% and 23% at every upper level.
  It carries forward the growth of years 1 to 5 (about 8% a year), while demand in years 6 to 10 flattens
  (362k, 384k, 401k, 406k, 405k units) because of the growth ceiling we wrote into years 6 to 10. ETS,
  whose damped trend flattens by design, misses by 3% at Total in every year and is the best method from
  year 3 on.
- **Capacity check (years 7 to 10, game cost ratio).** The plan from our forecast costs 12% more than the
  plan from the ground truth, more than simply building to last year's peak (7%). The ETS plan costs 2%
  more. The plan that ignores uncertainty (P50 only) costs 26% more.
- **Uncertainty.** The P10 to P90 bands are too narrow. On validation they hold 17 to 35% of actual months
  instead of 80%.

The honest reading is that the forecast is good for the two years we could validate and wrong in years 3
to 5, and that the wrong part comes from an assumption about the future (growth stops) that no history
could show.

## 2. Validation (years 4 and 5, forecast from month 36)

Weighted quantile loss, mean over six levels and two years; lower is better.

| Method | Score | Setting chosen on validation | Width factor (upper, SKU levels) | Reconciliation |
|---|---|---|---|---|
| Neural network | **0.138** | 64 units per layer, penalty 0.0001, Poisson loss, 50 epochs | 1.0, 0.35 | MinT |
| LightGBM | 0.141 | 63 leaves, 500 rows per leaf, sampling 0.8, Tweedie, 500 trees | 0.75, 0.5 | Bottom-up |
| ETS | 0.183 | Trend kept or dropped per series by AIC | 1.0, 0.5 | MinT |
| Seasonal naive | 0.209 | None | 0.75, 0.25 | Bottom-up |

WAPE of the chosen network on validation, against the ground-truth mean.

| Level | Year 4 | Year 5 | Ground-truth mean, year 4 | Year 5 |
|---|---|---|---|---|
| Total | see overview | | 1.5% | 1.2% |

(Table filled from `overview.json` in the final version of this report.)

## 3. Test (years 6 to 10, forecast once from month 60)

WAPE by forecast year at Total and Region, chosen method against ETS and the ground-truth mean.

| Level | Method | Year 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| Total | Neural network (chosen) | 2.1% | 3.6% | 6.6% | 13.8% | 23.1% |
| Total | ETS | 2.0% | 3.0% | 3.3% | 2.8% | 3.1% |
| Total | Ground-truth mean | 1.9% | 1.3% | 1.6% | 1.4% | 1.4% |
| Region | Neural network (chosen) | 4.9% | 6.6% | 9.6% | 14.9% | 23.5% |
| Region | ETS | 5.4% | 6.6% | 6.7% | 7.3% | 8.7% |
| Region | Ground-truth mean | 4.2% | 4.2% | 4.1% | 4.5% | 4.3% |

SKU levels (two years). Network WAPE 19.2% and 21.9% at SKU, 32.4% and 34.0% at Region × SKU; ground-truth
mean 13.7% and 13.2%, 26.7% and 25.9%. LightGBM is slightly better than the network here (18.2%, 20.1%).

Total units, actual and network P50 (years 6 to 10).

| | Year 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|
| Actual | 362,338 | 384,009 | 400,620 | 405,763 | 405,455 |
| Ground-truth median | 363,711 | 383,578 | 397,985 | 404,414 | 405,972 |
| Network P50 | 366,601 | 395,333 | 427,087 | 461,629 | 499,154 |

## 4. Why the ranking reverses after year 2

- Years 1 to 5 grew 4 to 8% a year, driven by growing SKUs (no growing SKU stopped growing before year 5).
- Years 6 to 10 follow the agreed rule that a growing SKU's growth phase ends at an age between 4 and 7
  years, so total growth falls to about 0% by year 9. The realised years are a typical draw of that process
  (each within about one standard deviation of the ground-truth mean).
- The two global models learn "a series that grew x% last year grows about x% next year" and apply it
  five times. The cap on months since first sale (24) also hides how old a SKU is, which is the only
  signal that could predict the end of growth.
- ETS damps its trend by construction, which happens to match the ceiling. On validation (years 4 and 5,
  still growing) the damping cost it; on the test it wins.
- The validation set only checks one and two years ahead, so nothing before the test could reveal this.
  This was a known caveat (PROPOSAL section 7).

## 5. Ground truth

- **Noise floor.** The ground-truth mean misses the realised year by 1.3 to 1.9% at Total, 4.1 to 4.5% at
  Region, 8.5 to 9.8% at Region × category, 13% at SKU and 26% at Region × SKU. In years 6 and 7 the network
  is within about 1 to 2 points of that floor at the upper levels.
- **Calibration.** Share of the 200 replications inside our P10 to P90 band (80% if calibrated), Total
  44%, 33%, 14%, 0%, 0% for years 6 to 10; Region × category 61%, 64%, 60%, 51%, 42%; SKU 34%, 35%. In
  years 6 and 7 the bands are too narrow; later they are wide enough but centred in the wrong place.

## 6. Capacity check (years 7 to 10)

Expected cost over the 100 held-out replications, relative to the plan built from the ground truth, at
each region's game cost ratio (0.76 to 0.83).

| Plan | Extra cost over the ground-truth plan |
|---|---|
| ETS | 2% |
| Last year's peak | 7% |
| Seasonal naive | 6% |
| LightGBM | 11% |
| Our forecast (network) | 12% |
| Our P50 only | 26% |

Across the whole range of cost ratios (0.1 to 10), our plan costs 1 to 21% more than the ground-truth plan,
always more than ETS. Building to last year's peak is far worse when capacity is cheap (139% at a ratio of
0.1), close to the best near the game's ratio and worse again when capacity is dear. P50 only builds for
the median forecast's busiest month whatever capacity costs, so it gets worse as capacity gets dearer.

## 7. What the setting rewards and punishes

- Rewards every method. The same seasonal shape every year, constant regional shares within a category,
  no prices, promotions, events or market growth, Poisson noise.
- Punishes the learned models. Growth that stops in years 6 to 10, a rule fixed by us before fitting but
  absent from the history. The test is a single future; another set of future rules would give another
  ranking after year 2.
- Cannot be forecast by any method. New SKUs (28 in years 6 to 10) and the delisting of the slowest
  sellers.

## 8. Changes made after the first test run (disclosure)

The first full run (`run-1.log`) was scored on the test years before the independent review. The review
found no leakage and no bug in the forecasts, but four faults in the uncertainty layer, fixed before the
second run.

| Fault | Fix | Evidence that did not come from the test |
|---|---|---|
| Shrinkage intensity used (n − 1)² instead of (n − 1)³, so MinT was diagonal | Formula corrected | Code against Schäfer and Strimmer (2005) and the R package hts |
| Level errors were not centred, so paths carried past drift and "seasonal naive" was not seasonal naive | Centred per series | Validation P50 differed from the mean forecast by 3 to 5% at Total |
| One width factor for all levels | Separate factors for the upper and the SKU levels | Validation coverage 8 to 45%; wider settings made SKU paths explode on validation |
| Month noise drawn from the wrong calendar month | Stored and drawn by calendar month | Code |
| SKU levels could be left unreconciled ("Bottom-up") | Always middle-out, as PROPOSAL section 6 promised | Proposal |

The point forecasts, the hyperparameters and the world did not change between the runs. The method chosen
on validation is the same. A dry run with tiny grids also printed test scores during development; nothing
was changed because of them.

## 9. Decisions taken while working alone (to revisit)

1. **Months since first sale capped at 24.** Without it the network extrapolated wildly (validation WAPE at
   Total 18% instead of 2.6%). It also hides SKU age, which matters for the reversal in section 4.
2. **Network epochs chosen on validation (25, 50, 100, 200) instead of early stopping on the latest
   origin** (agreed choice C5), which stopped after 1 to 5 epochs.
3. **Network loss as a setting (squared error or Poisson), learning rate fixed at 0.001** (C7 listed the
   learning rate). Poisson was chosen.
4. **Heavily weighted windows repeated instead of weighted** (same objective, stable training).
5. **Hyperparameters chosen by the WAPE of the mean forecast; widths, reconciliation and method by
   weighted quantile loss.**
6. **Error blocks are in-sample for the learned models.** The review suggests out-of-sample errors (MAJOR
   2); not done, see section 10.
7. **Years 6 to 10 details.** Bulk buyers of existing lumpy SKUs keep buying; new lumpy SKUs draw new
   buyers; bundle partners only among SKUs on sale; new SKUs join no bundle and new fading SKUs never end
   (only delisting removes them); delisting assumes two bulk buyers per lumpy SKU.
8. **Generator refactor.** World v1 years 1 to 5 are unchanged (content hashes tested); a new file
   `bulk_buyers.parquet` records the bulk buyers. A quirk of world v1 was found and kept, 0.13% of units
   are sold outside a SKU's life through bundles.

## 10. Questions for Kilian

1. **How to present the reversal.** Recommendation: make it the point of the page. The forecast is
   validated for two years and good there; beyond that, the assumption about growth decides, and the page
   shows both the learned forecast and the damped one against the ground truth. Changing the world rules
   now would be tuning the test to the model.
2. **Uncertainty bands.** They are too narrow on validation. Recommendation: build the error samples
   out-of-sample (refits at earlier origins), and, because the test years have now been seen, score the
   changed version on a second, untouched future (years 6 to 10 drawn with another seed under the same
   rules), disclosed on the page.
3. **Capacity check wording.** Our chosen plan loses to last year's peak at the game ratio. Recommendation:
   show it as it is, with the ETS plan beside it, and keep "model quality" (sections 3 and 5) apart from
   "business value" (section 6).
4. **Storyline.** Section 10 of the proposal still fits, with screen 7 showing the fan against the actual
   and the ground truth, and the results screen led by the capacity cost across cost ratios.
