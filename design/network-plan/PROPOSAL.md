# Network plan, implementation proposal

Model O from `ML_ROADMAP.md`. Written 5 November 2026 for Kilian's decisions, corrected the same day against
model B at `6af2ad8` (section 3). Kilian decided the 15 points the same day (section 14); the build follows
them.

This proposal sits on `claude/model-o`, branched from `integration` at `28cac5c`, which holds models B (at
`6af2ad8`), C and H.

Figures come from world v1, seed 7, years 1 to 5. For this model all five years are history, because the plan
is made at the end of year 5 and tested on years 6 to 10. I printed year 5's units, peak month, delivery cost
and margin per unit by region, and the features of model C's test regions. To correct section 3 I then read
model B's committed test scores for the network total (its bias by year on both futures), which B's handover
already reports. Nothing else from years 6 to 10 or the ground-truth replications was looked at, and no O
setting exists yet that those figures could inform.

---

## 1. The decision

At the end of year 5 the shop commits a five-year build plan. It says in which regions to open an FC, in which
year, and how much capacity (units per month) every FC has in each of years 6 to 10. This is the strategic question
an operations research team answers before a lease is signed, and the one models B (demand of the
existing regions) and C (demand of regions the shop has never served) were built to feed.

| Item | Proposal |
|---|---|
| Today's network | One central FC at Calopeia with capacity equal to year 5's busiest month for the whole network (40,422 units). Every region is served from it |
| Candidate sites | The other seven regions, and three new markets from model C (section 2) |
| Decisions | Open a site in a year (binary, never closed), capacity per FC per year (units per month, continuous, never reduced, as in the game) |
| Timing | Yearly decisions. A new FC or new capacity takes a year to build, so year 6 runs on today's network and the plan shapes years 7 to 10. The whole schedule is fixed at the end of year 5 |
| Monthly operations | Each month, demand is served from open FCs within their capacity, cheapest first; what cannot be served is lost |
| Not decided here | Which SKUs each FC lists (models H and K), stock levels (model J), inventory carried between months |

Why yearly and fixed now. A planner commits a lease and a fit-out once, and the game's capacity cannot be
reduced. Revising the plan each year as demand arrives (a rolling or multi-stage plan) is closer to practice
but needs B refitted every year; section 12 lists it as an extension.

**Recommendation.** This decision, with today's network as the central FC sized to year 5's peak.

## 2. Candidate new markets

Model C forecasts the demand of a region the shop has never served, for 30 simulated regions. O needs only a
few. Three are chosen by a rule fixed now, before any plan is made. They are the two most populous test regions
on the continent inside C's training range, and the most populous island inside it.

| Region | Landmass | Population (M) | Income index | Urban share |
|---|---|---|---|---|
| New region 9 (id 109) | Continent | 5.80 | 0.84 | 0.54 |
| New region 19 (id 119) | Continent | 5.09 | 1.00 | 0.72 |
| New region 2 (id 102) | Island | 2.73 | 0.99 | 0.60 |

Inside the range, because C's forecasts are tested there; outside it, C's own handover warns of larger errors.
The three open as markets at the start of year 6, as new markets do in the game, so their demand exists
whatever the plan. The plan decides only how they are served (from an existing FC across a lane, or from an FC
of their own). Each island is its own landmass. New regions add demand and take none from the others, as in C.

**Recommendation.** These three, chosen by this rule.

## 3. Demand inputs from models B and C

| Demand of | Comes from | Form |
|---|---|---|
| The 8 existing regions, years 6 to 10 | Model B's test fit (origin month 60), its chosen method, Region level after reconciliation | 1,000 sample paths per region and month (60 months) |
| The 3 new markets, years 6 to 10 | Model C's chosen log-linear model refitted on years 1 to 5, scaled by B's network growth | 1,000 draws per region and month |

How C and B combine. C forecasts a region's units per 1,000 inhabitants relative to the network's (its index)
and turns it into units with the network's rate in the level year. For years 6 to 10 that network rate comes
from B instead of last year repeated. Each draw takes one B path (the network's total units per 1,000
inhabitants in each month) and one C draw (index error per region and category, month noise), so new markets
grow and shrink with the network in the same path and keep their own cold-start error. C's handover names
this combination as O's job (its limit 5).

Two facts about B that decide much of O.

1. **B's forecast over-forecasts the late years.** Its chosen method (the neural network) is close for about
   three years, then over-forecasts because growth stops in years 6 to 10 while it extrapolates growth. Bias
   of its mean forecast for the network total in years 6 to 10 (B at `6af2ad8`, `O.test["Neural
   network"].table.Total[j].bias_mean` and the same under `O.second_future.test`) is as follows.

   | Future | Year 6 | Year 7 | Year 8 | Year 9 | Year 10 |
   |---|---|---|---|---|---|
   | First (the test) | +1.4% | +3.6% | +7.7% | +15.5% | +25.8% |
   | Second | −0.6% | +1.7% | +4.4% | +9.2% | +14.9% |
   | ETS, first | −3.2% | −4.7% | −5.0% | −2.6% | +1.0% |
   | ETS, second | −5.1% | −6.5% | −7.8% | −7.9% | −7.7% |

   So years 8 to 10 are over-forecast by 8 to 26% on the first future and 4 to 15% on the second. Its bands
   now hold the replications at the network total (100, 100, 100, 84 and 30% inside P10 to P90 on the first
   future, 95 to 100% on the second), so the over-forecast lies mainly in the late years of the first future.
   A plan built on it will over-build towards year 10. B's `RESULTS.md` (open question 2) recommends using the
   chosen method and saying plainly that beyond two or three years it rests on an assumption. Switching to
   ETS because it won B's first test would tune O on B's test, and ETS under-forecasts the second future.
2. **B's paths are not saved.** B's committed outputs are current. Code and outputs last changed together in
   `6af2ad8` (mean-preserving sample paths, scores on the mean forecast), the outputs hold the out-of-sample
   error samples and the second future, and B's `test_outputs` passes. But only `overview.json` and
   `series.json` are written, so O still needs the Region × month sample paths of B's chosen method, which
   no file holds.

**Recommendation.** B's chosen method for the existing regions, with the ETS-based plan shown as a sensitivity
in a drawer and never chosen. C combined with B as above. Ask the session that owns B to write its test paths
(method × path × region × month, years 6 to 10) to `data/models/demand-forecast/` in its next full run; until
then O imports B's functions and rebuilds the paths itself (B's run 6 took 32 minutes on 10 cores, so more than
an hour on this 4-core container).

## 4. Costs

Every coefficient is per unit and comes from the game or the shop, scaled the way B and C already scale the
game. Year 5 values below, by region.

| Cost | Value | Source |
|---|---|---|
| Capacity | 0.2267 × price per unit of monthly capacity per year, about $9.00 (price $38 to $41 by region) | The game's $50,000 per drum per day over a five-year life, as in B and C |
| Opening an FC | 69 × price per year while open, about $2,700, shown across 0 to 50 times that | The game's factory fixed cost of $500,000 over five years, scaled by price as for capacity |
| Delivery | Local cost per unit × distance multiplier (1.0 same region, 1.4 same landmass, 2.2 across water); local cost $2.29 to $2.59 per unit | The shop's parcel formula ($1.50 per parcel plus $0.35 per kg) on year 5's orders, as on the Bought together and split orders pages |
| Slower delivery | 3% of demand lost when served from another region on the same landmass, 10% across water | `policy_sim`, model J's page |
| Lost sale | Margin per unit, $11.10 to $11.93 | Year 5's margin by region |

The capacity cost ratio (capacity cost over margin per unit, about 0.78) is the main assumption, as in B and C,
and the opening cost is the second. Results are shown across both, with the game's values marked. Costs are
annualised and not discounted, as in B and C. With these numbers a regional FC saves about $1.30 per unit on
the continent and $4 per unit on an island (delivery plus slower-delivery losses), so the plan weighs that
saving against the opening cost and the loss of pooling, since separate FCs each need room for their own peak.

**Recommendation.** These costs, both ratios shown over their full range.

## 5. The model, from first principles

**In plain words.** Choose where and when to build and how big, so that the cost of capacity and opening plus
the expected cost of serving demand (delivery, slower-delivery losses, lost sales) is as low as possible over
many possible futures. Build decisions are made once and hold in every future; monthly serving adapts to the
future that happens.

This is a **two-stage stochastic mixed-integer programme**. The first stage holds the decisions made now
(opening and capacity), the second the decisions made later in each scenario (monthly flows, lost sales).
Solving it over a sample of scenarios is sample average approximation.

**Sets.** Sites i (11: the 8 existing regions and the 3 new markets), regions j (the same 11), years t (6 to
10), months m (12 per year), scenarios s (S sample paths).

**First-stage variables.**
- o(i, t) ∈ {0, 1}, FC at site i open in year t.
- K(i, t) ≥ 0, capacity of the FC at site i in year t (units per month).

**Second-stage variables (per scenario s, year t, month m).**
- x(i, j, t, m, s) ≥ 0, units of region j's demand served from the FC at site i.
- u(j, t, m, s) ≥ 0, region j's demand lost for lack of capacity.

**Constraints.**
- Today's network. o(Calopeia, t) = 1; K(Calopeia, 6) = 40,422; o(i, 6) = 0 and K(i, 6) = 0 elsewhere.
- Open for good and never smaller. o(i, t) ≥ o(i, t − 1); K(i, t) ≥ K(i, t − 1).
- Capacity only where open. K(i, t) ≤ M o(i, t), with M the largest forecast peak.
- Capacity per month. Σ_j x(i, j, t, m, s) ≤ K(i, t).
- Demand. Σ_i x(i, j, t, m, s) / k(i, j) + u(j, t, m, s) = D(j, t, m, s), with k the share kept (1, 0.97 or
  0.90), so serving a unit from afar uses up a little more demand.

**Objective.** Minimise

```
Σ_t Σ_i [ F o(i,t) + c K(i,t) ]
  + (1/S) Σ_s Σ_t Σ_m [ Σ_i Σ_j ( d(j) μ(i,j) + g(j) (1/k(i,j) − 1) ) x(i,j,t,m,s) + Σ_j g(j) u(j,t,m,s) ]
```

with F the opening cost per year, c the capacity cost per unit per year, d the local delivery cost, μ the
distance multiplier and g the margin per unit.

**Size.** 40 binaries and 44 capacities (10 candidate sites and the central FC over years 7 to 10), and
121 × 60 + 11 × 60 = 7,920 second-stage variables per scenario, about 400,000 at S = 50. A month's
second stage is a small transportation problem, so the programme is large but easy for a modern solver.

**Risk.** The objective is the expected cost. The 90th percentile of cost over the scenarios is reported beside
it. A risk-averse version (minimising conditional value at risk) is a small change, left as an option.

**Solver.** HiGHS through `scipy.optimize.milp`, which is already installed (scipy is locked as a dependency
of scikit-learn), so there is no new dependency. `highspy` (HiGHS's own Python interface, a new dependency)
or OR-Tools only if building the sparse matrices by hand proves unwieldy; I would ask first.

**Recommendation.** This formulation, expected cost, HiGHS through scipy.

## 6. Settings, and on which data they are chosen

O learns nothing from labels, so it has no training set in the usual sense. Its settings are chosen on B's and
C's own sample paths, never on the realised years 6 to 10 or the ground-truth replications.

| Setting | Candidates | Chosen by |
|---|---|---|
| Scenarios in the programme (S) | 25, 50, 100, 200 | Solve on S paths, evaluate the plan on the 1,000 − S other paths; the smallest S whose out-of-sample cost is within 0.5% of the largest |
| Optimality gap | 0.1% | Fixed |
| Capacity cost ratio, opening cost | The game's values | Fixed for the headline; full range shown |

This is the standard check that sample average approximation has enough scenarios (in-sample cost against
out-of-sample cost), and it uses only the forecasts.

**Recommendation.** This scheme.

## 7. Baselines

| Plan | In plain words | Technical rule |
|---|---|---|
| Do nothing | Keep today's network | Central FC only, capacity 40,422 units per month in every year |
| Expand the busiest FC | Grow the one FC that exists to the forecast | Central FC only, capacity in each of years 7 to 10 equal to the busiest month of the P50 forecast for the whole network that year |
| P50 plan | Plan as if the median forecast were certain | The same programme with one scenario, the P50 of every region and month |
| P90 plan | Plan for a high forecast | The same programme with one scenario, the P90 of every region and month |
| Our plan | Plan over many futures | The two-stage programme over S paths |
| ETS plan (drawer) | Our plan on B's ETS paths | Sensitivity for B's late-year over-forecast, never chosen |
| Ground-truth plan (reference) | The best plan any forecast could give | The two-stage programme over 100 ground-truth replications |

The P50 plan against our plan is the value of planning with uncertainty (the value of the stochastic
solution). The ground-truth plan against our plan is what a perfect forecast of the distribution would still
save. The best plan for each replication on its own (wait and see) bounds both and goes in a drawer.

**Recommendation.** These plans.

## 8. The check in the simulator

Every plan is costed on futures the plans never saw. Given a plan, each month of each future is served at the
lowest cost (the second stage solved for that month), so the comparison is about the plan and not about
routing.

| Futures | Existing regions | New markets |
|---|---|---|
| Ground-truth replications | B's 200 replications of years 6 to 10 (`truth_replications.npz`, stream `[7, 7, r]`), summed to regions | 200 replications simulated by O's own module, which calls C's `region_sim.simulate_year` with the catalogue of years 6 to 10 (imported, unchanged) |
| Realised future | The world's years 6 to 10 (`orders_future.parquet`) | One further draw on its own stream |
| Second future (if B's outputs are rebuilt) | B's second draw of years 6 to 10 and its replications | Not available; reported for the existing regions only |

Replications 0 to 99 build the ground-truth plan; replications 100 to 199 score every plan (as in B), so the
reference is not scored on the futures it was built from. Reported per plan are expected and 90th-percentile
cost over the five years, its parts (capacity, opening, delivery, slower delivery, lost sales), fill rate,
capacity built per site and year, utilisation, and cost over the ground-truth plan. The ground truth is used
only to check, never to choose.

Two caveats for the page. The replications share the future's rules with the realised future, so they test
the plan against the noise of a known process, not against a wrong process. And B's late-year over-forecast
(section 3) will make every forecast-based plan look worse than the ground-truth plan after year 7. That is a
finding about B's assumption, to be shown as such.

**Recommendation.** This check.

## 9. Business value

The headline is the expected cost of each plan over years 6 to 10 on the scoring replications, against do
nothing and against the ground-truth plan, at the game's cost ratios, with the whole range of the capacity
cost ratio (0.1 to 10, B's 41 values, solved at 9 of them) and of the opening cost (6 values) behind it. Model
quality (how close the forecasts are) stays on B's and C's pages; this page reports only what the plan costs.

**Recommendation.** This framing.

## 10. Draft storyline

Titles are plain noun phrases, captions only where a visual cannot be read without one, the plan shown before
its cost, results last, no recap.

**Models home card.** Network plan. "Where should the shop open or expand FCs over the next five years, and
by how much?"

**Opening.** Title "Network plan". Question "Where should the shop build capacity for the next five years?"
Boxes "FC sites" (11), "Years" (5) (Kilian removed "Scenarios", since S is chosen on screen 4).

1. **Demand forecast.** Small multiples (3 × 4) of the P10 to P90 band and P50 of every region and new market
   for years 6 to 10, plus the network, from B and C. Drawer on how C and B combine and on B's late-year
   assumption.
2. **Today's network.** A map of Pangea with the central FC, its capacity against year 5's peak, the
   candidate sites and the three lane classes. Drawer with the cost table (section 4) and its sources.
3. **Planning model.** The model in plain words beside its technical form (section 5), as a two-column
   table. Drawer with the full formulation.
4. **Scenarios.** In-sample against out-of-sample cost for 25 to 200 scenarios, the chosen number marked.
5. **Plans.** A heatmap of capacity (units per month) by site and year for each plan, unbuilt sites greyed
   out, with presets for the five plans.
6. **Capacity and demand.** For one site at a time (presets per site), the capacity steps of each plan
   against the demand band, with the capacity drawn as a limit line.
7. **Simulated years 6 to 10.** The distribution of five-year cost of each plan over the scoring
   replications, split into its parts. Drawer on the replications and the realised future.
8. **Results.** Cost of each plan over the ground-truth plan across the capacity cost ratio, the game's
   value marked; boxes for our plan at the game's ratios (cost against do nothing, fill rate, capacity
   built). Drawer with the opening-cost range, the value of the stochastic solution, the wait-and-see
   bound, the ETS plan, and what flatters the setting.

## 11. Vocabulary

New surface terms to approve. Network plan, candidate site, mixed-integer programme, decision variable,
constraint, objective, two-stage, scenario, recourse, capacity (approved for B), utilisation, value of the
stochastic solution.

Drawer only (to add to `GLOSSARY.md`). Sample average approximation (proposed for B), HiGHS, branch and
bound, big-M constraint, conditional value at risk, wait-and-see bound, expected value of perfect
information.

## 12. Known limits, before building

1. **No inventory between months.** Capacity is monthly throughput and cannot be smoothed by building stock
   ahead of November, as in B's and C's capacity checks. It overstates the capacity needed for the peak.
2. **One plan fixed for five years.** A rolling plan revised each year would adapt to the turn in growth B
   misses. Extension, if wanted, after the fixed plan.
3. **The plan follows B's assumption.** Section 3. The page shows it and the ETS sensitivity.
4. **Region totals only.** One capacity per FC in units, no categories or SKUs; split orders and listings are
   models H and K.
5. **New markets add demand.** No cannibalisation, no ramp-up, as in C.
6. **Costs from a game.** The opening cost in particular is a scaled game figure; the page shows its range.
7. **Ground truth of new markets is O's own simulation**, built from C's simulator with the catalogue of
   years 6 to 10. The delisting of lumpy SKUs follows that catalogue's stop days (an adapter in O's module).

## 13. Build plan after approval

1. `backend/src/scgame/ml/network_paths.py` (B's and C's paths and their combination, the new markets'
   replications, costs from year 5) with tests.
2. `backend/src/scgame/ml/network_plan.py` (the programme, the baselines, the scenario check, the evaluation
   on the replications, the cost ranges) with tests, including a toy network whose optimum is known by hand and
   a check that the evaluation of a plan reproduces the programme's own second stage.
3. Outputs in `data/models/network-plan/`, no `card.json`. One independent review (leakage, fair baselines,
   choices on forecasts only), then `design/network-plan/HANDOVER.md`.

Estimated runtime. B's paths more than an hour on 4 cores unless B saves them; B's replications about 13 minutes if
missing; the new markets' replications about 5 minutes; the programmes about 30 to 60 minutes (scenario check,
5 plans, 9 capacity cost ratios, 6 opening costs).

## 14. Decisions

Kilian's decisions of 5 November 2026. Accepted as recommended, 1, 2, 3, 5, 6, 7, 9, 10, 12, 14 and 15. For 7,
the figures of the recheck against B at `6af2ad8`. For 9, S in 25, 50, 100 and 200, and all 9 capacity cost
ratios. Changed or extended:

- **4.** Accepted; the opening-cost range is shown in the results, not only in a drawer, since at the game's
  value it is small next to capacity.
- **8.** Changed. B's session is not asked; O writes B's test paths itself from B's code on the integration
  branch, into `data/models/network-plan/`.
- **11.** Accepted, including B's second future (available on the integration branch).
- **13.** Accepted with two changes. No "Scenarios" box in the opening; screen 1 is called "Demand forecast".
- **Extension.** After the review and the handover, if time allows, the rolling plan of limit 2 (revised
  each year as demand arrives), evaluated on the same scoring replications.

| # | Decision | Recommendation |
|---|---|---|
| 1 | Decision and starting point | Open sites and set capacity per FC per year for years 6 to 10, fixed at the end of year 5; one-year build time; today's network is the central FC sized to year 5's peak |
| 2 | New markets | Three of C's test regions by the rule in section 2 (109, 119, 102), open as markets from year 6 |
| 3 | Granularity | Region totals per month, one capacity per FC; categories only in the cost coefficients |
| 4 | Costs | Section 4, game-derived ratios, capacity cost ratio and opening cost shown over their full range |
| 5 | Formulation | Two-stage stochastic mixed-integer programme by sample average approximation, expected cost, 90th percentile reported |
| 6 | Solver | HiGHS through `scipy.optimize.milp`, no new dependency |
| 7 | Forecast for the existing regions | B's chosen method, ETS plan as a sensitivity only |
| 8 | B's paths | Ask B's session to save its test paths in its next run; O rebuilds them meanwhile |
| 9 | Settings | Number of scenarios by in-sample against out-of-sample cost on the forecasts' own paths; nothing chosen on years 6 to 10 |
| 10 | Baselines | Do nothing, expand the busiest FC, P50 plan, P90 plan; ground-truth plan as reference |
| 11 | Check | B's 200 replications plus O's own for the new markets; 0 to 99 build the reference, 100 to 199 score; realised future and B's second future as extra futures |
| 12 | Card and title | "Network plan", with the question in section 10 |
| 13 | Storyline | Eight screens as in section 10 |
| 14 | Vocabulary | Section 11 |
| 15 | Process | Build, one independent review, handover, stop before the page |
