# Network plan (model O), handover

For the session that reviews the model and designs the page. Written 5 November 2026 on branch
`claude/model-o` (from `integration` at `28cac5c`). No frontend exists for this model and no `card.json` is
written.

Read in this order. This file (what was built, how, and what came out), `PROPOSAL.md` (the decision, the
formulation in section 5, Kilian's decisions in section 14), `docs/ML_ROADMAP.md` section O.

The question. Where should the shop open or expand FCs over the next five years, and by how much?

---

## 1. Time and data

O learns nothing from labels. The plan is made at the end of year 5 from forecasts, so years 1 to 5 are its
history and years 6 to 10 its test.

| Use | Data | Years |
|---|---|---|
| Costs per unit, today's capacity | World v1 order lines | Year 5 |
| Demand forecast, existing regions | Model B's test paths (fit on months 1 to 60) | 6 to 10 |
| Demand forecast, new markets | Model C's log-linear index refitted on years 1 to 5, scaled by B's category paths | 6 to 10 |
| Choosing the number of scenarios | The forecast paths only (paths 0 to S − 1 against 200 to 999) | 6 to 10, forecast |
| Ground-truth plan (reference) | Replications 0 to 99 of the first future | 6 to 10 |
| Scoring every plan | Replications 100 to 199 of both futures, and both realised futures | 6 to 10 |

The ground truth is never used to choose anything.

## 2. Inputs

- **World v1 years 1 to 5** (`scgame.world.generator`, seed 7) and **years 6 to 10 of both futures** with
  their 200 replications each, built by model B's `ensure_world` (imported, unchanged): first future
  (stream 6, replications `[7, 7, r]`), second future in `data/world_v1/future2` (stream 16, replications
  `[7, 17, r]`).
- **Model B's test paths** (`b_paths.npz`, written by `network_paths.write_b_paths` from B's code at
  `6af2ad8`). B's chosen method (the neural network) and ETS, each refitted on months 1 to 60 with the
  setting B chose on its validation set (read from B's committed `overview.json`), then B's
  `Probabilistic.upper_paths` with B's chosen width and reconciliation and B's seed for that method (200 +
  its index), so these are the paths B scores. 1,000 paths × (Total, 8 regions, 8 categories) × 60 months,
  and the mean forecasts. B's code is not changed.
- **Model C as committed.** The chosen setting (log-linear regression, consumer label with the network's bulk
  rate, income, urban share, island and log population, penalty 0.01) and its error model (σ_region 0.097,
  σ_cell 0.184, month dispersion 19.6, 8 regions), read from C's `overview.json`.
- **New markets** (PROPOSAL section 2). C's test regions 109 and 119 (continent) and 102 (island), chosen by
  a rule fixed before any plan was made. Their futures (`markets.npz`) are simulated by
  `network_paths.simulate_markets` with C's `region_sim.simulate_year` (imported, unchanged) on the catalogue
  of years 6 to 10 of each future; 200 replications (streams `[7, 40, r]` and `[7, 42, r]`) and one realised
  draw (`[7, 41, 0]`, `[7, 43, 0]`) per future. C's simulator draws a region's customers and bulk buyers
  afresh each year.
- **Costs** (PROPOSAL section 4) from year 5 and the game. Section 3, step 2.

## 3. The method, step by step

Code in `backend/src/scgame/ml/`. `network_paths.py` (B's paths, the new markets' futures),
`network_plan.py` (network, costs, demand paths, the programme, plans, evaluation, ranges, the rolling plan).

1. **Network.** 11 sites, the 8 regions (Calopeia first, the central FC) and the 3 new markets, each a
   candidate FC site and a demand region. Distance multiplier 1.0 within a region, 1.4 on the same landmass,
   2.2 across water (each new island is its own landmass); share of demand kept 1, 0.97, 0.90 (`policy_sim`).
2. **Costs.** Local delivery cost per unit by region from year 5's orders with the shop's parcel formula
   ($1.50 per parcel plus $0.35 per kg); margin per unit by region; new markets take the network's
   averages. Capacity c = 0.2267 × the network's price per unit of monthly capacity per year (B's
   `GAME_CAPACITY_SHARE`); opening F = 68.97 × price per FC per year (the game's $500,000 factory cost over
   five years, scaled by price). Today's capacity is year 5's busiest network month.
3. **Demand paths.** Existing regions, B's Region paths. New markets, for each path n and category c,
   units = population × (exp(index + region error + cell error) × consumer rate + bulk rate), with the
   network's consumer and bulk rate per 1,000 inhabitants in each month taken from B's Category path n
   (split by year 5's bulk share of the category), C's region error (t with 7 degrees of freedom, widened by
   √(1 + 1/8)) and centred cell errors fixed over the five years, and month noise with C's dispersion. So
   the new markets move with B's network in each path and keep their own cold-start error.
4. **The programme** (`solve_years`). First stage per site and year 7 to 10, open (binary) and capacity
   (continuous), never closed or reduced; the central FC starts at today's capacity; year 6 runs on today's
   network. Second stage per scenario and month, units served from each FC to each region (cost delivery ×
   multiplier plus the margin of demand lost to slower delivery) and units lost (margin). Expected cost
   over the scenarios. HiGHS through `scipy.optimize.milp`, relative gap 0.1%, a site counts as open when it
   has capacity.
5. **Number of scenarios.** Plans from the first S paths (S = 25, 50, 100, 200), each costed on paths 200
   to 999; the smallest S within 0.5% of S = 200's out-of-sample cost.
6. **Plans.** Do nothing (today's network). Expand the busiest FC (the central FC only, each year at the
   busiest month of the network's P50 forecast, never smaller). P50 plan and P90 plan (the programme on one
   scenario, every region and month at its P50 or P90). Our plan (the programme on S paths). ETS plan (our
   plan on ETS paths, sensitivity only). Ground-truth plan (the programme on replications 0 to 99,
   reference). Wait and see (the programme on each scoring replication alone, a bound).
7. **Evaluation** (`Evaluator`). For a plan and a future, all 60 months served at the lowest cost given the
   plan's capacity (one linear programme per future). Costs over the five years are capacity (including
   today's), opening (new FCs), delivery, slower delivery and lost sales; fill rate is units served over
   demand.
8. **Ranges.** The P50, P90, our and ground-truth plans re-solved at 9 capacity cost ratios (B's ratios
   0.1 to 10, every fifth) and at 6 opening costs (0, 1, 3, 10, 30 and 50 times the game's), each scored on
   the scoring replications of the first future. Do nothing and expand the busiest FC are re-priced.

## 4. Baselines

Do nothing, expand the busiest FC, the P50 plan and the P90 plan are the baselines (Kilian's decision 10).
The ground-truth plan and the wait-and-see bound are references, not rivals. The ETS plan shows how much
B's growth assumption decides.

## 5. How every setting was chosen, and on which data

| Setting | Candidates | Chosen by | Result |
|---|---|---|---|
| Number of scenarios S | 25, 50, 100, 200 | Out-of-sample cost on forecast paths 200 to 999 | S = 25. Out-of-sample $13.413M, $13.413M, $13.412M, $13.412M (all within 0.02%); solve 25 s against 550 s for S = 200 |

Fixed in advance, not chosen. The three new markets (rule), the costs (game and year 5), the 0.1% gap, the
1,000 paths, replications 0 to 99 for the reference and 100 to 199 for scoring, the ranges, seed 29.

## 6. Headline results and their output fields

From `data/models/network-plan/overview.json` (written `O`, run 1, `run-1.log`) and `rolling.json` (written
`R`). Costs are five-year totals (years 6 to 10) at the game's costs (capacity $9.04 per unit of monthly
capacity per year, opening $2,749 per FC per year, ratio 0.78), mean over replications 100 to 199.

| Result | Value | Field |
|---|---|---|
| Today's capacity | 40,422 units per month (year 5's busiest month) | `O.costs.k0` |
| Our plan | Every one of the 10 candidate sites opens in year 7; capacity 80,140 units per month in year 7 rising to 96,628 in year 10 (Calopeia stays at 40,422) | `O.plans["Our plan"]` (K per site and year, open) |
| Ground-truth plan | The same 11 FCs, 76,714 rising to 79,146 | `O.plans["Ground-truth plan"]` |
| P50, P90, ETS plans in year 10 | 90,445; 111,269; 85,190 units per month, all 11 FCs | `O.plans.<plan>` |
| Expand the busiest FC | Calopeia alone, 72,155 rising to 91,232 | `O.plans["Expand the busiest FC"]` |
| Expected cost, first future | Ground-truth plan $11.82M; ETS plan $11.87M (+0.5%); P50 plan $11.92M (+0.8%); our plan $11.99M (+1.4%); P90 plan $12.32M (+4.2%); expand the busiest FC $14.75M (+24.8%); do nothing $15.78M (+33.5%) | `O.evaluation.first.<plan>.replications.expected` |
| Expected cost, second future | Ground-truth plan (built on the first future) $12.30M; ETS +0.2%; P50 +0.4%; ours +0.9%; P90 +3.3%; busiest +24.3%; do nothing +39.4% | `O.evaluation.second.<plan>.replications.expected` |
| Realised futures | Within 0.1% of each plan's expected cost | `O.evaluation.<future>.<plan>.realised.expected` |
| Our plan against do nothing | $3.79M less over five years on the first future (24%), $4.73M on the second | same |
| Fill rate | Our plan 97.3%; do nothing 81.9%; busiest FC 94.5% (first future) | `.replications.fill_rate` |
| Lost sales | $0.71M in every plan that builds, all of it in year 6, before any new capacity exists | `.replications.parts.lost_sales` |
| Wait and see (each future known in advance) | $11.82M, 0.03% below the ground-truth plan | `O.wait_and_see.expected` |
| Value of the stochastic solution | Negative. The P50 plan beats our plan by $73k (0.6%) on the first future and $61k on the second | P50 against our plan |
| Rolling plan (extension) | $11.90M on the first future (+0.65% over the ground-truth plan) and $12.35M on the second (+0.44%), below our fixed plan on both (−$92k, −$53k); year-10 capacity 88,369 on average against 96,628 | `R.first.replications`, `R.second.replications` |
| Capacity cost ratio range | At ratios 0.1 to 0.56, our plan within 0.5 to 0.8% of the ground-truth plan, P90 1.2 to 2.5%, do nothing 39 to 60%. At 10 every plan keeps the central FC alone. Rows at 1.0, 1.78 and 3.16 are unreliable (below) | `O.ranges.capacity_cost_ratio[]` |
| Opening cost range | Up to 10 times the game's ($27,493 per FC per year) every plan opens 10 or 11 FCs and our plan stays within 0.9 to 1.4% of the reference; at 30 and 50 times plans open fewer sites, but those rows are unreliable (below) | `O.ranges.opening_cost[]` |
| Solver gaps in the ranges | The ground-truth plan stopped at the 120 s cap with gaps of 16% (ratio 1.0), 5% (3.16), 2% (1.78), 10 to 25% (opening 10 to 50 times); our plan 5 to 18% at opening 10 to 50 times. Where the reference has a large gap, plans can look better than it (−11% at ratio 1.0), so those rows say nothing about plan quality | `O.ranges.*[].plans.*.gap` |
| B reproduction check | Bias of the network's mean forecast +1.0, +2.8, +6.5, +13.7, +23.2% (B committed +1.4 … +25.8%) | `O.b_check` |

What it means. At the game's costs every regional FC pays: serving a region from its own FC saves more in
delivery and slower-delivery losses than the opening cost, which is small next to capacity (as Kilian
noted). So the decision is about how much capacity, not where. Planning over many futures does not beat
planning on the median here, because both forecasts behind the plan run high (B's late years, and the new
markets by 11 to 35%, section 8); hedging against more demand then builds even more capacity that is not
needed. The ETS plan, built on the damped forecast, comes closest, which says the plan's quality is decided
by the growth assumption, not by the optimisation. Revising the plan each year as sales arrive (the rolling
plan) corrects most of the over-forecast and is the best forecast-based plan. Compared with doing nothing,
every plan that builds saves about a quarter of the five-year cost, almost all from lost sales avoided.

## 7. Chosen or changed after seeing test results or ground truth

- **No setting was chosen on years 6 to 10.** The number of scenarios is chosen on forecast paths only; the
  new markets by a rule fixed in the proposal; costs from year 5 and the game.
- **Ground truth seen before the run, for checks only.** A smoke draw of the new markets (a separate stream,
  `[7, 99, 0]`) printed their yearly units to time the simulator. The realised draws of the new markets were
  compared with their replications' means (they agree within about 2%). Neither feeds anything.
- **B's reproduction gap, found before any plan was scored.** These paths are B's method refitted here
  (`b_check` in `overview.json`). The network's mean forecast is 0.4 to 2.6 points lower each year than B's
  committed one (bias of the mean +1.0, +2.8, +6.5, +13.7, +23.2% against B's +1.4, +3.6, +7.7, +15.5,
  +25.8% on the first future); ETS is within 0.4 points. The world is identical to B's month by month, so
  the gap comes from refitting on another platform (B's last run was local, on a 10-core Mac). B's seeds
  and thread settings are reused; exact equality is not possible from here.
- **ETS paths clipped at zero.** B's MinT-reconciled ETS paths dip below zero in 1,489 of 480,000
  region-months (`negative_path_cells_clipped`); negative demand made the ETS programme infeasible.
  Clipped before any plan was scored.
- **Restarts, none after scoring was read.** The first attempt stopped on the ETS infeasibility; the second
  and third hung in their worker pools (forked workers after HiGHS had started threads, then an
  out-of-memory kill that left the pool waiting). Fixes changed how programmes are run (spawned workers,
  two at a time, range solves capped at 120 s with the gap reported), not what they solve. Those runs had
  scored the plans in memory but wrote nothing; no result of theirs was read.

## 8. Known limits and doubts

The independent review (`review-truth-r1.md`, written while the run was in progress) found no leakage and
no fault in the programme, the evaluation or the cost accounting. Its findings were not fixed in the time
available; they are listed here, with what each means for the results.

1. **The new markets are over-forecast from year 6** (review, Major 1). Their forecast means exceed the
   ground truth by 11 to 14% in year 6, rising to about 35% by year 10, and they are about 31% of demand in
   years 7 to 10. The combination of C with B is faithful (C's formula at year 5's network rate lands within
   3% of the year-6 truth); most of the gap is B's 8.9% network growth from year 5 to year 6 carried onto
   markets whose simulated demand grows less in their first years, plus about 2.5% from C's median-preserving
   errors. Every forecast-based plan therefore over-builds for the new markets. The rolling plan's level
   corrections show it directly (factors 0.88 to 0.91 for the three markets after year 6). Not tuned away.
2. **The ground-truth plan is built on the first future only** (review, Major 2) but is also used as the
   reference on the second future. Comparisons with it on the second future are against a plan built for
   another future of the same process, not that future's own best plan.
3. **B's late-year over-forecast** (PROPOSAL section 3), slightly smaller in these refitted paths (section 7).
4. **"Expand the busiest FC" uses the sum of regional P50s** per month, not the P50 of the network total
   (review, Minor 3); the sum of medians is a little below the median of the sum for skewed paths.
5. **C's errors are applied to consumer units only** (review, Minor 4); the bulk part follows B's path
   without C's error, so the new markets' spread is slightly narrower than C's.
6. **Range solves stop at 120 s** (review, Minor 5); their gaps are stored beside each cost
   (`ranges.*.plans.*.gap`), and a plan's cost there mixes plan quality with where the solver stopped.
7. **ETS clipping** removed values down to −16,135 units a month (review, Minor 6), in 1,489 region-months.
8. **Wait-and-see** is solved to a 0.1% gap on the first future only (review, Note 7).
9. **Fill rate counts demand lost to slower delivery as unfilled** (review, Note 8).
10. **Every total includes about $1.83 million for today's capacity** (review, Note 9), the same in every
    plan, which makes percentage differences look smaller than the differences in money.
11. **No inventory between months.** Capacity must cover each month's peak, as in B's and C's checks.
12. **One plan fixed for five years**, except in the rolling extension (section 11).
13. **Costs from a game.** The opening cost especially; the results show its whole range.
14. **New markets add demand and take none from existing regions**, with no ramp-up, as in C.

## 9. Dependencies

None new. HiGHS through `scipy.optimize.milp` (scipy was already locked).

## 10. Rebuild

From a fresh clone.

```bash
cd backend && uv sync
uv run python -m scgame.world.generator ../data/world_v1 7
uv run python -c "from scgame.ml.demand_forecast import ensure_world; ensure_world('../data/world_v1')"
uv run python -m scgame.ml.network_paths b ../data/world_v1 ../data/models/network-plan
uv run python -m scgame.ml.network_paths markets ../data/world_v1 ../data/models/network-plan
uv run python -m scgame.ml.network_plan ../data/world_v1 ../data/models ../data/models/network-plan
uv run python -m scgame.ml.network_plan rolling ../data/world_v1 ../data/models ../data/models/network-plan 25
```

Runtimes on this 4-core container. B's world extension and both sets of replications about 30 minutes;
B's paths 36 minutes (network 16, ETS 21); the new markets about 13 minutes; the model about 35 minutes
(the scenario check with S = 200 alone 9 minutes); the rolling plan see section 11. `b_paths.npz` and
`markets.npz` are committed, so the last two commands are enough once the world and B's replications exist.

Checks.

```bash
cd backend && uv run pytest tests/test_network_plan.py && uv run ruff check src
```

`tests/test_network_plan.py` covers a hand-made network whose optimum is known (an island FC opens when
opening is free and capacity cheap, none opens when opening is dear, the central FC grows only when capacity
pays), that evaluating a plan reproduces the programme's own second stage, that every unit of demand is
served or lost, that fixed plans never shrink, that the rolling plan keeps what is committed and never
shrinks, and the committed outputs.

## 11. Extension, the rolling plan

Added after the review, at Kilian's request (PROPOSAL limit 2). `network_plan.rolling` re-solves our plan at
the end of years 6, 7 and 8 for each scoring replication of both futures and for both realised futures. A
year's build is committed a year ahead, so each revision fixes capacity up to the next year and re-plans the
rest. The 25 scenarios are B's and C's paths with each region's level corrected by what it sold in the year
just observed (observed over forecast, applied to all later months); B is not refitted. Results in
`rolling.json` (`R`). The first-year factors on the realised first future are 0.99 to 1.04 for the existing
regions and 0.88 to 0.91 for the new markets (`R.first.realised.revisions`). The rolling plan costs $11.90M
and $12.35M on the two futures, 0.65% and 0.44% above the ground-truth plan, the lowest of every
forecast-based plan; it ends year 10 with 88,369 units per month on average. Runtime 516 s. The paths are not
clipped there (B's network paths have no negative values). The review did not see these results.
