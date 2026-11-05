# Network plan, truth review round 1

Independent review of model O (network plan) at `07088a3` on `claude/model-o`, by a critic that did not write it.
Read CLAUDE.md, PROPOSAL.md sections 1 to 9 and 14, the HANDOVER draft, `network_plan.py`, `network_paths.py`
and `test_network_plan.py`. Small experiments only (scripts in the session scratchpad, each under a minute).
`overview.json` had not been written when this review closed, so no result was spot-checked.

## Summary

The programme, the evaluator and the cost accounting are correct as far as I could check. Variable indexing,
big-M, monotonicity, the one-year build, capacity rows, demand rows with the share kept, the 1/S scaling and
the year 6 handling all match PROPOSAL section 5, and the evaluator's row order matches `b_ub` and `b_eq`.
No plan or setting reads years 6 to 10 realised data or replications 100 to 199; S is chosen on forecast
paths only. Two findings matter for the page. The new markets' forecast sits 11 to 14% above their ground
truth already in year 6 (B's own year 6 bias is about 1.7%), and the ground-truth plan built on the first
future is also used as the reference on the second future, where it is not that future's ground truth.
There is no blocker.

## Findings

### 1. Major. The new markets are over-forecast by 11 to 14% in year 6, before B's late-year drift starts

Evidence. `market_paths` (`network_plan.py:112-152`) with B's neural-network paths gives mean yearly units
for regions 109, 119 and 102 in year 6 of 74,877, 84,954 and 41,023. The mean of their first-future
replications (`markets.npz`) is 65,488, 75,466 and 36,927, so +14.3%, +12.6% and +11.1%. By year 10 the gap
is +37%, +35% and +33% (102,028 against 74,375 for region 109). The composition itself is faithful. C's own
formula (`region_demand.predict_units`) with year 5's network rate gives 67,537, 76,805 and 36,822, within
3% of the year 6 truth. B's mean category paths grow the network rate 8.9% from year 5 to year 6, while the
existing regions' realised growth is 7.1%. C's lognormal region and cell errors add about 2.5% to the mean
(E[exp(a)] about 1.008, E[exp(ε)] about 1.017). So the gap decomposes into B's growth carried onto markets
whose simulated ground truth does not grow with the network in their first year (their truth for year 6 is
close to year 5's network rate, unscaled), plus B's small year 6 bias, plus C's median-preserving errors.

Why it matters. The three new markets hold about 31% of years 7 to 10's demand (their truth against B's existing-region forecast), and the island (102) is
where a regional FC pays most (about $4 per unit). An over-forecast there pushes the forecast-based plans
towards opening or oversizing the island FC, and the page will blame "B's late-year over-forecast" (PROPOSAL
section 8, caveat 2) for what is partly a different mechanism, from year 6 on. Rule 6 asks to say what about
the setting flatters or hurts the model.

Suggested fix. Do not tune anything on it (that would use the ground truth). Report in the handover and on
the page the forecast against the truth by year for existing regions and new markets separately, and explain
the year 6 gap (ground-truth ramp of a new market against the network rate B extrapolates, and C's
median-preserving error model). Consider making the market errors mean-preserving (divide by
E[exp(a + ε)]) as B's paths are, decided before scoring and disclosed.

### 2. Major. The ground-truth plan of the first future is the reference on the second future too

Evidence. `truth_build` takes replications 0 to 99 of the first future only (`network_plan.py:426`). The same
plan is scored on the second future (`network_plan.py:436-442`). HANDOVER section 1 lists "Replications 0
to 99 of the first future", but sections 3 and 4 call it the reference without qualification.

Why it matters. On the second future that plan was built on a different process, so "cost over the
ground-truth plan" there is not "what a perfect forecast of the distribution would still save" (PROPOSAL
section 7). It can even lose to a forecast plan, which would read as a forecast beating perfect knowledge.

Suggested fix. Build a second ground-truth plan from the second future's replications 0 to 99 (a 100-scenario
solve, as for the first) and use each future's own reference, or label the second-future comparison as "the
first future's ground-truth plan" and never compute a gap against it there.

### 3. Minor. "Expand the busiest FC" uses the sum of regional medians, not the network's P50

Evidence. `net_p50 = q[0.5].sum(axis=0)`, a sum of per-region medians (`network_plan.py:423-425`). PROPOSAL
section 7 says "the busiest month of the P50 forecast for the whole network".

Why it matters. For right-skewed regional paths the sum of medians is below the median of the network total,
so this baseline is slightly undersized and slightly flattered or penalised depending on cost ratio. Small,
but the label claims a quantity the code does not compute.

Suggested fix. `np.quantile(paths["nn"].sum(axis=1), 0.5, axis=0)`.

### 4. Minor. C's error model is applied to consumer units only, while C applies it to the total

Evidence. `network_plan.py:148`, `popk * (exp(p + a + ε) * cons + bulk)`. C's `draws` (`region_demand.py:270-283`)
multiplies the whole point forecast, bulk included, by exp(a + ε), and C's σ were estimated on that total.

Why it matters. The new markets' spread is narrower than C's by the bulk share, so the plan sees slightly less
uncertainty than C reports, and the HANDOVER (step 3) presents it as C's error model.

Suggested fix. Apply exp(a + ε) to `cons + bulk`, or disclose the difference in the handover.

### 5. Minor. Range solves are capped at 120 s, and the ground-truth plan has the most scenarios

Evidence. `RANGE_LIMIT = 120.0` (`network_plan.py:60`, used at `:456`). The ground-truth plan solves 100
scenarios against S_chosen for our plan (possibly 25). The gap reached is stored per point (`:465`).

Why it matters. Where the reference stops at a larger gap than our plan, "cost over the ground-truth plan" in
the ranges mixes plan quality with solver stopping. The same holds for the main solves if the 1,200 s limit
binds (status 1 instead of 0).

Suggested fix. Report the status and gap of every plan beside its cost on the page's drawer, and flag any range
point whose gap exceeds 0.1%.

### 6. Minor. ETS region paths reach −16,135 units before clipping

Evidence. `b_paths.npz` `ets_region` minimum −16,135, 1,489 negative cells; `ets_category` has 1,019 negative
cells (checked here). Clipped at zero for regions only (`network_plan.py:402`); the new markets built from ETS
category paths stay positive (minimum 418), so nothing is infeasible.

Why it matters. Clipping raises the mean of the ETS forecast where its paths are wild, so the ETS plan is not
exactly "our plan on B's ETS paths". The handover discloses the count but not the size.

Suggested fix. Add the clipped mass (units added, share of demand) to the handover and the ETS drawer.

### 7. Note. Wait and see is not strictly a bound

Evidence. `ws_plans` use the 0.1% gap (`network_plan.py:446`); the incumbent can exceed the true optimum by up
to 0.1%. It covers the first future only. HANDOVER step 6 calls it "a bound".

Suggested fix. Use `mip_dual_bound` for a true lower bound, or say "within 0.1% of the bound", first future.

### 8. Note. Fill rate counts slower-delivery losses as unfilled, and `lost_share` repeats it

Evidence. `fill_rate = served / demand` with `served = x.sum()` (`network_plan.py:313-317`); `lost_share =
1 − served/demand` (`:350`) is 1 − fill rate.

Why it matters. Units lost to slower delivery and units lost for lack of capacity are merged in "lost". The page
should define fill rate this way in the drawer, or report the two losses separately. Drop the duplicate field.

### 9. Note. Today's capacity is in every plan's cost, so percentages are diluted

Evidence. `first_stage_cost` charges `c × K.sum()` over five years including today's 40,422 units
(`network_plan.py:320-324`); the central FC's opening cost is excluded. This is consistent across plans and
in `recost`, so differences are right. A percentage saving over the total includes a sunk part identical for
every plan (about $1.83 M at the game's ratio), and it moves with the capacity cost ratio.

Suggested fix. State differences in dollars, or say in a caption that totals include today's capacity.

### 10. Note. Rolling plan does not clip its paths

Evidence. `rolling` builds `P` from unclipped `nn_region` (`network_plan.py:567`) while `main` clips. B's
neural-network paths have no negative cells (minimum 428), so the start plan equals our plan today; it would
diverge for ETS.

## What I checked and found correct

- **Formulation (A).** `O(i, y)` covers sites 1 to 10 only, so the central FC has no opening variable or cost,
  matching "opening cost of the central FC excluded". `K(i, y)` covers all 11 sites for years 7 to 10.
  Big-M rows `K − big·o ≤ 0` for i ≥ 1, with `big` the largest monthly network demand over scenarios
  (or today's capacity) × 1.05, which is valid because Σ_j x(i, j) ≤ Σ_j D(j). Monotonicity rows for o and K.
  `lb[K(i, 0)] = k_prev` makes year 7 at least year 6, which is the one-year build. Capacity rows
  Σ_j x(i, j) − K(i, y) ≤ 0 with the K column `no + i·Y + y_` of the block's year. Demand rows Σ_i x/k + u = D,
  with `keep[i, j]` broadcast over (block, i, j). `D[s_, :, y_·M + m_]` returns (blocks, 11) because the two
  advanced indices are split by a slice, which matches `dem_row = row + blocks·I + b·J + j`. Second-stage costs
  divided by S, first-stage not. Year 6 is outside the programme and identical for every plan. The tests
  reconcile the programme's objective with the evaluator to 0.2%.
- **Evaluator (B).** Columns month-major (`b·nb + i·J + j`, then u at `b·nb + I·J + j`). `b_ub =
  repeat(K, 12).T.ravel()` and `b_eq = D.T.ravel()` are month-major, index `m·11 + i` and `m·11 + j`, as the
  rows. Delivery, slower delivery and lost sales are split correctly.
- **Leakage (C).** Costs, today's capacity and the bulk shares come from year 5 (order lines span days 1 to
  1,825). C's index is refitted on years 1 to 5 and its error model is C's validation one. S is chosen from
  paths 0 to S − 1 against paths 200 to 999 of the forecast. P50 and P90 come from forecast paths. The
  ground-truth plan uses replications 0 to 99 only; every score uses 100 to 199 or the realised futures.
  The rolling plan reads only years already past at each revision.
- **Baselines (D).** Do nothing and expand the busiest FC are fixed central-FC plans, never shrinking. P50 and
  P90 plans use the same programme on one scenario as described. All plans are scored by the same evaluator on
  the same replications, and fixed plans are re-priced correctly in the ranges.
- **Composition of C with B (E).** Units per 1,000 inhabitants from B's category paths over the 8 regions'
  population, split by year 5's bulk share (C's "network average" setting), the index from C's chosen
  log-linear setting, region error t with 7 degrees of freedom widened by √(1 + 1/8), centred cell errors
  fixed over the five years, month noise with C's dispersion. Reproduces C's point within 3% of the truth when
  given year 5's rate (finding 1 explains the remaining gap).
- **Cost accounting (F).** Capacity per year including today's, opening per year while open, central FC never
  charged an opening, consistent in `summarise`, `first_stage_cost`, `recost` and the wait-and-see total. Costs
  printed here match the proposal ($9.04 capacity, $2,749 opening, ratio 0.781, delivery $2.29 to $2.59,
  margin $11.10 to $11.93, today's capacity 40,422).
- **Handover against code (G).** The data table, steps 1 to 8, the S rule, the B reproduction gap and the ETS
  clipping all match the code, apart from findings 2, 4 and 7 (reference on the second future, C's error on the
  total, "a bound").
