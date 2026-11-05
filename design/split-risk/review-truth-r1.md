# Split orders (model H), truth review, round 1

Independent review of the build described in `HANDOVER.md`, written 3 November 2026 by a reviewer that did
not write the model. Every finding below was checked by reading the code or by a small experiment. The
scripts are in the session scratchpad (`review/e1` to `e9`) and are not part of the repository. No
existing file was changed.

## Summary

No leakage was found. Plans, listings, stock levels, SKU statistics, pair lifts, the replay, the
calibration maps and the business-check estimates use only information available at the end of the year
before the scored year, the test site sets are unseen, and the code is unchanged since the outputs were
written. The order-level simulation reproduces exactly under an independent pure-Python reference, and
13 features match a brute-force recomputation on 400 orders.

The main problem is the replay simulation baseline. Its in-stock rates pool first-choice lines with
fallback lines that arrive at an FC only after another FC ran out, so they understate the home FC's
in-stock rate and the replay overpredicts splits by about 8 points. The handover attributes this to
correlated stock-outs, which is mostly wrong. With rates kept apart by route position the replay's
cross-fitted validation log loss falls from 0.2798 to 0.2612, ahead of logistic regression (0.2655) and
still behind gradient boosting (0.2428). Every replay comparison in section 6 is affected.

The second problem is that the business-check claim "estimates a network's split rate better than the
replay" is not supported once uncertainty is shown (the 95% interval of the difference includes zero).
The rest are smaller truth and disclosure issues.

| # | Severity | Finding |
|---|---|---|
| 1 | Major | The replay's in-stock rates mix first choices and fallbacks; the replay is much weaker than it should be and the stated cause is wrong |
| 2 | Major | The business-check advantage of gradient boosting over the replay is within noise |
| 3 | Minor | The "-3.8%" SKU total is mostly the model's overall underprediction, not basket size |
| 4 | Minor | Noise floor overstated by about two, and the conclusion drawn from it is not supported |
| 5 | Minor | "Best on every order-level score" is false, and two rounded numbers are wrong |
| 6 | Minor | The 0.5% calibration error hides clear miscalibration between 0.5 and 0.8 |
| 7 | Minor | Feature set deviates from the agreed proposal without disclosure; the top feature means two things |
| 8 | Minor | Plans ignore overflow between FCs, a driver of the stock-out splits, and its scale is not disclosed |
| 9 | Minor | Tests miss the wiring of years, the replay, the business check and the inventory dynamics |
| 10 | Note | Decision check cannot discriminate between methods |
| 11 | Note | Dollars use one average extra cost although per-network costs vary from $3 to $8 |
| 12 | Note | Chosen LightGBM setting sits in a corner of the grid |
| 13 | Note | SHAP group totals add per-feature absolute means, which overstates correlated groups |
| 14 | Note | All six page examples are January orders |

---

## Findings

### 1. The replay's in-stock rates mix first choices and fallbacks (Major)

**What is wrong.** `split_sim.simulate` counts an attempt at every FC a line reaches with the SKU listed,
in every routing round (`split_sim.py` lines 277 to 295), and `replay_rates` divides fills by these pooled
attempts (`split_risk.py` lines 229 to 239). Lines reach an FC in round two or later only after their
first choice ran out that day, and they are served after every first choice there. Their fill rate is
therefore far lower than that of the FC's own first-choice lines. `replay_probability` then applies the
pooled rate to the home FC of every line (`split_risk.py` lines 253 to 257), so the home FC looks much
less reliable than it is, and the order looks much more likely to split.

**Evidence.** Experiment `e6_replay_rates.py` on the 20 validation networks (year 4, replay of year 3 with
levels from year 2, exactly as built). The as-built numbers reproduce the run (raw 0.3659, isotonic
0.2798), so the harness is faithful. Keeping the rates apart for first choices and later rounds, nothing
else changed, gives the following.

| Replay, validation | Raw log loss | Cross-fitted Platt | Cross-fitted isotonic | Bias (points) | Network split rate error, raw (points) |
|---|---|---|---|---|---|
| As built (rates pooled) | 0.3659 | 0.3000 | 0.2798 | +7.9 | 7.9 |
| Rates by route position | 0.3261 | 0.2906 | 0.2612 | +0.7 | 1.3 |

At the regional FCs the pooled fill rate is 0.41 to 0.70, against 0.85 to 0.92 for first-choice lines
(same experiment). On the test networks 17% to 53% of the line attempts at regional FCs come from other
regions' customers (`e4b_cascade.py`).

**Why it matters.** The proposal makes the replay "the second, stronger competitor" that the classifier
must beat. As built it is handicapped by a bookkeeping choice, not by its independence assumption. With
the fix the ranking on validation becomes gradient boosting 0.2428, replay 0.2612, logistic 0.2655, rule
0.2970, so the replay passes logistic regression and the gap to gradient boosting halves (0.037 to 0.018).
These rows of `HANDOVER.md` section 6 change or may change. Test log loss, the interval of gradient
boosting minus the replay, Brier score, average precision, precision at the rule's recall (replay 73.2%),
replay calibration (2.1%, +2.1), split rate per network (replay 2.8 points) and SKUs at risk (replay MAE
24.2, total +14%). The explanation in section 7 ("overpredicts in-sample because stock-outs at one FC come
together") and limit 7 ("assumes independent stock-outs, which overpredicts splits") state a cause the
evidence does not support. Most of the overprediction is the pooling.

**Suggested fix.** Count attempts and fills by route position (at least first choice against later
rounds, as in the experiment), or define the in-stock rate as the share of days the FC holds the SKU at
the start of the day, and use the matching rate at each step of `replay_probability`. Rerun, choose the
calibration map on validation as before, and disclose in section 7 that the change was made after the test
set had been scored, prompted by this review, and that it strengthens a baseline. Rewrite limit 7 and the
section 7 note on why the replay overpredicts.

### 2. The business-check advantage over the replay is within noise (Major)

**What is wrong.** Section 6 and "What it means" say gradient boosting "estimates a network's split rate
better than the replay simulation" (0.68 against 0.92 points, about $600 a year), and `ML_ROADMAP.md`
repeats "within 0.7 points against 0.9 for the replay". No interval is given, the 35 candidates are not
independent (15 of them share three site sets), and the difference is not resolved.

**Evidence.** `e5_business.py`, from `O.business.candidates`, bootstrapping 23 clusters (each test
network on its own, each decision site set as one cluster), 4,000 draws.

| Difference in mean absolute error | Point estimate | 95% interval |
|---|---|---|
| Gradient boosting minus replay | -0.23 points | -0.50 to +0.07 |
| Gradient boosting minus logistic | -0.26 points | -0.45 to -0.00 |

Gradient boosting is closer than the replay on 20 of 35 candidates. On the test networks alone it is 0.78
against 0.91 points. The direct replay is also much better on the validation year analogue (simulated year
3 against year 4, mean absolute error 0.45 points, bias -0.01 points, `e3_replay_bias.py`), so its +0.9
point bias on year 5 is specific to that year rather than a property of the method.

**Why it matters.** Design rule 6 asks to keep model quality and business value apart and to say what the
model is worth. As written, a reader takes away a measured business advantage that the data cannot
distinguish from zero.

**Suggested fix.** Report the cluster bootstrap interval beside the three mean absolute errors and word
the result as "no clear difference from the replay in estimating a network's split rate; both are far
better than the rule". Keep the $ figures only with their interval. Update `ML_ROADMAP.md` section H.

### 3. The "-3.8%" SKU total is mostly the overall underprediction (Minor)

**What is wrong.** Limit 10 explains the 3.8% shortfall of gradient boosting's SKU totals as "Orders with
many SKUs count once per SKU, and the model underpredicts them slightly". Most of the shortfall is the
model's overall underprediction on the test year.

**Evidence.** A SKU total is the sum over orders of probability times SKUs in the order. The test
prediction bias is -0.497 points (`O.test.methods.gbm.prediction_bias`), year 5 has 58,850 multi-SKU orders
with 3.217 SKUs on average per network (`e1_basic.py`), so a bias that did not depend on basket size would
already give -0.00497 x 20 x 58,850 x 3.217 = -18,800 SKU split orders. The observed difference is 706,126
- 734,294 = -28,168. Two thirds (-2.6% of -3.8%) is the general bias and one third (-1.3%) the basket-size
effect. For the rule it is the other way round (bias +0.1 points, total -8.8%, almost all basket size).

**Why it matters.** The page will need to explain this number, and the stated reason is the smaller part.

**Suggested fix.** "SKU totals are 3.8% low, mostly because the model underpredicts split orders on the
test year by 0.5 points (3% of split orders); orders with many SKUs, which count once per SKU, add about one
point."

### 4. Noise floor overstated, and its conclusion not supported (Minor)

**What is wrong.** `O.test.noise.orders_flipping` is the share of orders whose label differs in at least
one of five re-simulations (`split_risk.py` line 543). Section 6 and limit 4 read it as "7.8% of orders
change label with another order of arrival". Limit 4 then concludes "so no model can score far better at
the order level".

**Evidence.** `e7_noise.py` on the 20 test networks. Any of five re-runs 7.8% (matches the output), one
other order of arrival 3.6%. The label variance due to arrival order, an estimate of the Brier score a
perfect model could reach, is about 0.015 (0.018 after the small-sample correction), against a test Brier
score of 0.065 for gradient boosting.

**Why it matters.** Arrival-order noise leaves room for a much better order-level score, so the stated
bound is not what the evidence shows. Other irreducible uncertainty (the year's demand) may well cap the
score, but it is not measured.

**Suggested fix.** Write "3.6% of orders change label with one other order of arrival (7.8% in at least
one of five)" and drop or reword the bound, for example "arrival order alone would allow a Brier score near
0.02; the rest of the gap is uncertainty about the year's demand and stock, which no feature can remove,
and limits of the model, which this check does not separate".

### 5. "Best on every order-level score" is false, and two rounded numbers are wrong (Minor)

**What is wrong.** "What it means" says the classifier is "the best of the four on every order-level
score". It is not best on expected calibration error (logistic 0.491% against 0.498%) or on prediction bias
(rule +0.10 points, logistic -0.46, gradient boosting -0.50). Two numbers are mis-rounded. The logistic
test log loss is 0.24549, which is 0.245, not 0.246 (also in `ML_ROADMAP.md`), and the rule's average
precision is 0.59248, which is 0.592, not 0.593.

**Evidence.** `O.test.methods.<m>` values printed in full during the review.

**Why it matters.** Small, but these are the headline sentences the page will copy.

**Suggested fix.** "Best on log loss, Brier score and average precision; calibration error is tied with
logistic regression." Correct the two numbers.

### 6. The 0.5% calibration error hides miscalibration between 0.5 and 0.8 (Minor)

**What is wrong.** The expected calibration error is weighted by bin count, and 83% of test orders sit
below 0.2. In the middle bins both learned models are clearly under-confident.

**Evidence.** `O.test.methods.<m>.reliability`.

| Bin | Orders (logistic) | Predicted | Observed | Orders (gradient boosting) | Predicted | Observed |
|---|---|---|---|---|---|---|
| 0.5 to 0.6 | 2,046 | 0.544 | 0.687 | 3,238 | 0.551 | 0.609 |
| 0.6 to 0.7 | 2,601 | 0.639 | 0.859 | 4,072 | 0.646 | 0.710 |
| 0.7 to 0.8 | 6,025 | 0.751 | 0.896 | 6,282 | 0.749 | 0.794 |

**Why it matters.** A planner summing probabilities over a network is fine, but anyone reading a single
order's probability around 0.6 to 0.7 is misled by 6 to 22 points. "Calibration error 0.5%" invites the
wrong reading.

**Suggested fix.** On the page, show the reliability diagram with bin counts and say in its drawer where
calibration is good (below 0.3 and above 0.8, where most orders are) and where it is not.

### 7. Feature set deviates from the agreed proposal without disclosure (Minor)

**What is wrong.** Decision 4 agreed the features of `PROPOSAL.md` section 5. The build adds
`n_listed_elsewhere` (the strongest input by SHAP, 0.78) and `some_not_all` (the rule's flag, second at
0.67), and reduces the stock-risk features for unlisted SKUs from the full set ("the same for the unlisted
SKUs at the FC they would come from") to `cover_min_central` and `growth_max_central`. `ML_ROADMAP.md`
"Decided while working alone" does not list these changes. `CLAUDE.md` asks that each model's exact
implementation is agreed before it is built.

`n_listed_elsewhere` also means two different things (`split_risk.py` line 179). For a customer served by
the central FC it counts basket SKUs that some regional FC lists, the route by which a central stock-out
becomes a split. For a customer with a regional home FC every line listed at home counts, because the
central FC lists everything, so the feature is close to the number of SKUs listed at home plus unlisted
SKUs that another regional FC lists.

**Evidence.** `split_risk.py` lines 67 to 80 and 179; brute-force check `e8_features.py` confirms the
code does what it says, so this is a meaning and disclosure issue, not a bug.

**Why it matters.** The page will name the strongest driver "SKUs listed at another FC", which a reader
takes as "another source exists"; for half the orders it is mostly a basket-size count.

**Suggested fix.** List the added and removed features in `HANDOVER.md` section 7 and in the roadmap's
"Decided while working alone". Either split the feature into "SKUs a regional FC could ship if the central
FC runs out" (central home) and "unlisted SKUs listed at another regional FC" (regional home), or name and
explain it precisely in the drawer.

### 8. Plans ignore overflow between FCs, and the scale is not disclosed (Minor)

**What is wrong.** Order-up-to levels come from the demand that would reach each FC if every line shipped
from its first listing FC (`split_sim.py` lines 181 to 192), but in the simulation lines out of stock at
the central FC raid regional FCs and lines out of stock at a regional FC raid the central FC. Neither plan
expects this flow, so stock-outs cascade.

**Evidence.** On ten test networks (year 5, `e4_central.py`, `e4b_cascade.py`) 17% to 53% of the line
attempts at regional FCs come from other regions' customers; lines arriving at the central FC exceed its
planned lines by 1% to 11%; 48% to 83% of the splits the rule misses belong to customers served by the
central FC. The handover's 53% (ten training networks) reproduces (0.531, `e9_misses.py`).

**Why it matters.** Together with the season-blind levels (limit 1) this is what creates the stock-out
splits the classifier learns and the rule cannot see. Limit 2 names the routing choice but not that the
plans ignore the resulting flow or how large it is. A planner whose levels include the expected overflow
would see fewer such splits, which flatters the classifier.

**Suggested fix.** Add to limits 1 and 2 that levels ignore the overflow the routing creates, with the
measured range (a sixth to a half of regional FC attempts come from other regions).

### 9. Tests miss the wiring of years, the replay, the business check and the inventory dynamics (Minor)

**What is wrong.** `test_the_plan_and_the_features_read_nothing_from_the_scored_year` multiplies the
quantities of other year-5 orders by seven and checks that features do not move. It cannot catch leakage
through other orders' SKUs or days, and it calls `features` with the statistics it chooses, so the wiring
in `Runner.labelled` (`self.stats[year - 1]`), in `scored_set` (replay plan with levels from `year - 2`,
simulated on `year - 1`) and in the business check (`R.stats[TEST_YEAR - 1]`, plan for year 5 on year-4
baskets) is untested. Nothing checks the inventory dynamics (review day, pipeline slot, arrival day), the
pooled attempt counts behind finding 1, `scores` (precision at the rule's recall), `bootstrap`, or `cost` and
`cost_one`.

**Evidence.** `backend/tests/test_split_risk.py`. All ten tests pass (4 s). The review's reference
simulation (`e2_reference.py`) matches the vectorised one line for line on two networks, so the dynamics
are right today, but nothing protects them.

**Suggested fix.** Add a small reference-simulation test (one SKU, known demand, check arrival on day t + 7
and the order-up-to arithmetic), a test that the replay and business check read no year after the one
they claim (perturb year 5 entirely, including SKUs and days, and check the business-check estimates do
not move), and a hand-made test of `scores` at the rule's recall.

### 10. The decision check cannot discriminate between methods (Note)

All four methods, including the plain rule, choose regional best sellers in all three site sets, and best
sellers lead by 3.2, 5.9 and 7.4 points (`O.business.decisions`, `O.business.candidates`). Limit 6
discloses that the decision is easy. On the page, say plainly that this check shows no difference between
methods rather than "regret 0 for all", or add candidates that trade listing share against service level,
where the choice is closer.

### 11. Dollars use one average extra parcel cost (Note)

The extra parcel cost per split order varies from $2.97 to $7.94 across the 35 candidates (from
`extra_parcel_cost / split_orders` in `O.business.candidates`), against the single $4.40 used for every
candidate. Limit 6 discloses one average. Converting each candidate's error with its own simulated extra
cost per split would be more accurate and costs nothing.

### 12. Chosen LightGBM setting sits in a corner of the grid (Note)

The chosen setting has the most leaves (63), the fewest rows per leaf (100) and the fewest trees (250).
With 63 leaves the validation loss rises with the tree count, so fewer trees might be better. The whole
grid lies within 0.0012, so it does not matter for the results, but the corner should be stated with the
choice in section 5.

### 13. SHAP group totals add per-feature absolute means (Note)

`O.shap.groups` sums each feature's mean absolute SHAP value (`split_risk.py` line 553). For correlated
features that push in opposite directions (the listing group holds four overlapping counts and the flag)
this overstates the group. Agreed choice M7 says "summed per feature group", which is more naturally the
mean absolute value of the group's summed contributions per order. Compute that, or state on the page how
the group figure is formed.

### 14. All six page examples are January orders (Note)

`examples.json` picks the first order of each kind (`split_risk.py` line 643), so every example has month
0. Splits the rule misses gather in November, so January examples are the least typical. Pick by a seeded
random draw, or from the month the screen discusses.

---

## Checked and found correct

- **No leakage.** Plans rank listings on year y - 1 and set levels from year y - 1; SKU statistics come
  from y - 1; pair lifts from years 1 to 3 (in-sample for training rows, disclosed); the replay plans with
  levels from y - 2 and simulates y - 1 with the same listing as the real plan (same seed and year);
  calibration maps are cross-fitted over halves of the validation networks and refitted on all of them;
  the business check uses year-4 baskets, the year-5 plan, year-4 statistics and maps fitted on year-4
  labels, all known at the end of year 4 (year-5 order counts only convert errors for reporting).
- **Unseen site sets.** Test, validation, training and decision site sets are disjoint (code, and the test
  passes); 123 sets of size 1 to 6 after removing the three decision sets; settings balanced as stated.
- **Code unchanged since the run.** `git diff 773d68a HEAD -- backend/` is empty.
- **Inventory logic.** Review on days 0, 7, 14 and so on after the day's demand, arrival at the start of day
  t + 7, order-up-to formula and pipeline slot identical to `policy_sim.py`. An independent pure-Python
  reference reproduces every line's shipping FC for two networks (`e2_reference.py`). The within-day
  cumulative-sum fill is correct given quantities of at least 1 and no repeated SKU within an order (both
  checked); it implements first come, first served with later lines of the round blocked once an FC is
  short, as documented.
- **Routing.** "Home, then central, then by distance" equals the proposal's "nearest, central first among
  equals" in this world, because the central FC is never farther than another FC from any region.
- **Labels and rule.** Parcels count distinct shipping FCs, lost lines are not splits, the rule is exact
  with unlimited stock (test passes), `cost` uses the right multiplier orientation, `cost_one` prices one
  parcel of the shipped weight from the home FC.
- **Features.** Thirteen features recomputed by brute force on 400 random orders of a year match exactly
  (`e8_features.py`), including cover at home and at the central FC, `n_listed_elsewhere`, largest lift,
  communities and the community cut. Reductions align with orders.
- **Replay probability.** The formula is the correct split probability under independence, and with all
  rates 1 it equals the rule.
- **Scores.** Log loss, Brier score, average precision, reliability bins and calibration error, precision
  at the rule's recall (highest threshold still reaching the rule's recall), paired bootstrap by network.
- **Numbers.** Every figure in `HANDOVER.md` sections 5 and 6 matches `overview.json` apart from the two
  roundings in finding 5, including the grid spreads (0.0012 and 0.00008), the rule's 0.875 and 0.071, the
  month shares (2.6% and 13.0%), the rule's flags and errors on the test set, the SHAP order, the SKU risk
  errors and totals, the business errors in points, orders and dollars, and the $600 difference.
- **The 53% of misses** served by the central FC with a line shipped from a regional FC reproduces
  (0.531, ten training networks, year 3).
- **Like for like in the business check.** The replay estimate is its raw simulated rate while the
  classifiers carry a map fitted on year 4. A validation-based bias correction would not help the replay
  (its validation-year bias is -0.01 points), so this asymmetry does not favour the classifiers.
