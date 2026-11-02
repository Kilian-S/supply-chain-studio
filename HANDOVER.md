# Handover

State on 2 November 2026, for a new session (local or cloud). Read `CLAUDE.md` first.

## Done

- **Game clone and scenario designer.** Engine verified against the live Run 3 data
  (`reference/GAME_MECHANICS.md`).
- **Product clustering** (`#/models/sku-segmentation`, `backend/src/scgame/ml/sku_segmentation.py`).
  Reviewed screen by screen with Kilian; train, validate, test split; stocking cost versus purity
  trade-off.
- **Bought together** (`#/models/product-affinity`, `backend/src/scgame/ml/product_affinity.py`).
  Built by an agent loop (writer, rules, truth and slop critics, builder, visual reviewer; specs and
  reviews in `design/bought-together/`), then one review round with Kilian (`DESIGN_REVIEW.md`, last
  entries). Communities match the 45 planted bundles; the split-catalogue check cuts split orders by
  3 to 7% against storing by category.

## Open on Bought together (small, ask Kilian before doing them)

1. Screen 6 (assignment): draw each FC's SKU limit as a faint outline, so the shared capacity and the
   overshoot of "best sellers together" are visible. Kilian asked why capacity seemed to change
   between rules; it does not.
2. Screen 4: possible title "Communities of linked SKUs." to show the step from linked pairs to
   communities.
3. Keyboard access to the matrix and scatter hover read-outs.

## Next

The remaining models, in storyline order, are B (long-range demand forecasting), C (new-region demand),
A (what-if surrogate), H (risk classification), O (network design MILP). See `ML_ROADMAP.md`,
section 6. Each needs its implementation discussed with Kilian before building. Prepare a proposal
(inputs, method, validation on the year 4 and year 5 split, baselines, business check, draft
storyline under the design rules) and wait for his decisions.

## How Kilian works

- He reviews page by page with screenshots and short comments; change exactly what he names, then
  report briefly. Keep a log entry per round in `DESIGN_REVIEW.md` (comment, change, habit).
- He prefers recommendations over option lists, short answers when asked ("in one sentence",
  "yes/no"), and honest caveats (what flatters a result).
- The models home card appears when a model writes `card.json`; do not write it before the page
  exists.
