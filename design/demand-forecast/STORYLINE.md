# Demand five years out, storyline (round 2)

Spec for `#/models/demand-forecast`, lean loop. Round 2 applies the combined critic's review
(`review-storyline-r1.md`). Data from `data/models/demand-forecast/`, written `O` (overview.json), `S`
(series.json) and `L` (labels.json, names). Point accuracy is always scored on the mean forecast
(`wape_mean`), never the P50. "Simulation 1" and "Simulation 2" are the two draws of years 6 to 10; the
forecast is the same in both (refit on years 1 to 5), only the realised years and their ground truth differ.
The 200 runs per simulation are "ground-truth replications".

Departures from the approved nine screens, for Kilian. The SKU heatmap (screen 2) is dropped for now
(its data is in the world files, not in the outputs; it could return). Reconciliation before and after
(screen 5) has no data and moves to a drawer. Screen 3 shows one fan with method and level controls
instead of a 2 by 2 grid. Screen 1 shows months 1 to 60. The reversal and the ground truth get a screen
each.

## Models home card

Title "Demand five years out". Question "How much will each region and category sell over five years,
and how sure are we?"

## Opening

Title "Demand five years out". Question "How far ahead can five years of sales see?" Boxes "Series"
(2,744, `O.facts.series_validation` summed), "Levels" (6), "Years of history" (5).

## 1. Monthly sales.

- Visual. Six small line charts, three by two, one per level, each the level's median-volume series
  (`S.presets[i].actual`, months 1 to 60). Each chart titled by its level, the series named under it except
  for Total ("Entworpe", "Toys", "Tyran, Beauty", "ELE-0021", "Nordhal, PET-0139", from `L`). Y axis
  "Units per month".
- Caption "Median-volume series of each level."
- Drawer. What a level is and the series per level (`O.facts.series_validation`; Region × SKU counts
  series with a sale in years 1 to 3, 2,363, the test has 2,391). Units ordered per month; the shop has no
  stock-outs, so orders are demand; bulk orders stay in.

## 2. Data split.

- Visual. The year strip for ten years, years 1 to 3 "Training set", 4 and 5 "Validation set", 6 to 10
  "Test set".
- Caption none.
- Drawer, before any number.
  - Validation forecasts years 4 and 5 from the end of year 3 with models fitted on years 1 to 3. Every
    setting and the method are chosen there. The chosen settings are refitted on years 1 to 5 and
    forecast years 6 to 10. Upper levels five years ahead, SKU levels two.
  - Years 6 to 10 are simulated with the same customers and rules plus three rules fixed before any model
    was fitted (about 7.4 launches a year, as many delistings of the slowest sellers, growth ending at an
    age of 4 to 7 years, `O.second_future.world.rules`). Simulation 2 draws them again with another seed.
  - Changed after test scores had been seen, all disclosed. The test years were scored in six runs. After
    the first, a review found faults in the bands and six fixes followed. Then the bands switched to
    out-of-sample errors from rolling refits, and simulation 2 was added after simulation 1 had been
    scored. Two last corrections made the paths average the mean forecast and scored point accuracy on
    the mean (`design/demand-forecast/HANDOVER.md` section 0). The point forecasts, the chosen method and
    the world never changed.

## 3. Forecasting methods.

- Visual. Left, a table "Method" and "Rule" (seasonal naive repeats last year; ETS follows level, trend
  and season and damps the trend; LightGBM and the neural network learn from every series at once and
  forecast a year from the year before, applied five times). Right, one fan, the validation forecast for
  months 37 to 60 (P10 to P90 band, P50 line, `S.presets[i].validation[method]`) against the actual.
- Controls "Method" (four) and "Level" (six).
- Caption "One series per level. Methods are compared on every series on the next screen."
- Drawer. ETS rules (log(1 + y), damped trend or none by AIC, additive season), learned models' settings
  (`O.validation[m].setting`), fallbacks to seasonal naive (ETS 194 series on validation, 26 on test;
  learned 86 and 133), sample paths from out-of-sample errors of rolling refits, reconciliation (bottom-up
  or MinT, chosen per method) so every level adds up.

## 4. Validation.

- Visual. Table, methods as rows, the six levels as columns, cells the pinball loss in percent averaged
  over years 4 and 5 (`O.validation[m].best.table[level][j].wql`), last column the mean
  (`O.validation[m].best.score`). Each cell coloured by its excess over the best method in its column (no
  fill within 10% of the best), the chosen method's row in bold.
- Caption "Pinball loss (%), validation set, lower is better."
- Drawer. Pinball loss averaged over P10, P50 and P90, summed over series and months, over actual units
  (at the P50 it equals WAPE). Grids searched and settings chosen. Only two years ahead are seen here.

## 5. Five-year forecast.

- Visual. Fan from the end of year 5 to the end of year 10 (P10 to P90, P50, `S.regions[i].test`) against
  the actual (`S.regions[i].actual`, history grey).
- Controls "Series" (Total and the eight regions) and "Test years" (Simulation 1, Simulation 2; actual
  from `actual` or `actual_second_future`).
- Caption none.
- Drawer. How to read it (the band holds the middle 80% of 1,000 joint sample paths; the line is the P50,
  and errors on the next screen use the mean forecast, slightly below it). The forecast carries the growth
  of years 1 to 5 forward; growth slows in years 6 to 10, and in simulation 1 it stops.

## 6. WAPE by forecast year.

- Visual. Line chart, x "Forecast year" (1 to 5), y "WAPE (%)" (`wape_mean`), one line per method and the
  "Ground-truth mean" as reference (`O.truth.test` or `O.second_future.truth`, `wape_truth_mean`), end
  labels. Neural network in indigo.
- Controls "Level" (Total, Region, Category, Region × category) and "Test years".
- Caption none.
- Drawer. The learned models lead for two forecast years in simulation 1 and for three to four in
  simulation 2; beyond that ETS, which damps growth, is closer, because the end of growth is not in years
  1 to 5. Validation saw two years ahead, so it could not show this; the method was not switched. Over all
  levels and years the test score still favours the learned models (`O.test[m].score`, LightGBM 0.103,
  neural network 0.106, ETS 0.128), and LightGBM edges the chosen network. SKU levels two years ahead in a
  small table.

## 7. Forecast versus ground truth.

- Visual. Two bands from the end of year 5, ours (P10 to P90, `S.regions[i].test`) and the ground truth's
  (P10 to P90 of the 200 replications, `truth_test` or `truth_second_future`), with the actual.
- Controls "Series" and "Test years".
- Caption none.
- Drawer. The replications rerun years 6 to 10 with the simulation's launches, delistings and growth-end
  ages held fixed, so the ground-truth band shows only the remaining randomness, narrower than what a
  planner faces; the two simulations' bands for year 10 do not even overlap. Shares of replications inside
  our band and the true band's width over ours (`truth_inside_our_p10_p90`, `true_p10_p90_width_vs_ours`),
  both simulations. Used only to check.

## 8. Results.

- Visual. Bars, capacity cost for years 7 to 10 of each plan over the ground-truth plan (%), at the game's
  cost ratio (index of `rho` nearest the mean of `rho_game`, `capacity` or `second_future.capacity`,
  `curves[plan].expected`). Plans Last year's peak, Seasonal naive, ETS, LightGBM, Neural network, Neural
  network P50. Neural network in indigo.
- Controls "Test years".
- Caption none.
- Drawer. The decision (each region builds capacity for years 7 to 10 at the end of year 5; shortfall
  costs margin, spare capacity costs money) and the plans in plain words. The ranking depends on the test
  years. In simulation 1, where growth stops, last year's peak is cheapest and the neural network costs
  more because it runs high and its band is wide; in simulation 2 LightGBM is cheapest and last year's peak
  falls behind. Neural network P50 costs most because it ignores the cost ratio. What flatters the model,
  a stable simulator (about 1.5% noise at Total) and a growth rule fixed before fitting that years 1 to 5
  do not show. Cost across the cost ratio as a small table, per-region table for the five plans it holds.
