# Bought together, storyline

Screen-by-screen spec for the product affinity page. No code. Every visible string goes into
`frontend/src/models/affinity/copy.ts` and must pass `lexicon.test.ts`.

Data sources, all in `data/models/product-affinity/`. `overview.json` is written `O`, `graph.json` is
written `G`, `matrix.json` (300 x 300 matrices of counts, lift and orders with both by chance,
training set) is written `M`. Figures are cited by field, with the value of the run of 1 November
2026 in brackets. The page reads every figure from its field, never from this file.

Conventions carried over from the clustering page. Titles are plain noun phrases with no label above
and nothing below. Box labels are bare, capitalised subject nouns with units in the label. Captions
appear only where the visual cannot be read without one, in one line. Depth lives in the drawer
behind the filled indigo plus. The check against the hidden bundles comes second to last and the
business result last, as on the clustering page. There is no recap.

One word per concept on every surface and in every drawer. "Order" (never "basket"), "linked pair"
(a pair that passes the test), "community", "lift", "unlinked SKU" (a SKU with no linked pair),
"hidden bundles", "best sellers first" (for the ordering by `G.nodes[].baskets`), "orders with both
by chance" (for the expected count). Rule names are capitalised only where they are bar or control
labels. Indigo has one job, marking our result (linked pairs on screen 3, our communities in bars
and lines). Magnitude scales use the blue `sequential` scale from `charts/colors.ts`. Communities use
the `cat` palette without the violet #6250d6, which is too close to the indigo accent. Everything
else is neutral grey.

---

## Models home card

- Title: Bought together
- Question: Which SKUs are bought together, so they should be stored in the same FC?

(The current `PLANNED` entry in `ModelsHome.tsx`. Once the page is built, the backend writes
`card.json` with the same title and question, which `listModels` then serves. Nothing else on the card.)

## Page opening

- Title: Bought together (`O.model.title`)
- Question: Can storing SKUs that are bought together in one FC reduce split orders?
- Boxes (`Numbers`, value plus label only, no notes)
  - SKUs, count of `G.nodes` [300]
  - Categories, distinct `G.nodes[].category` [8]
  - SKU pairs, `O.facts.pairs_possible` [44,850]

---

## Screen 1 (`data`)

- Title: Orders from the simulated shop.
- Visual
  - `Numbers` with two boxes. "Orders", sum of `O.facts.orders_by_year` [449,068]. "SKUs per order",
    `O.facts.skus_per_order_all` [2.3], one decimal.
  - The year strip, moved from `SkuSegmentationPage.tsx` into `ui.tsx` as a shared `Years` part so
    both pages use it (years 1 to 3 Training set, year 4 Validation set, year 5 Test set).
  - Vertical bars of orders by SKUs per order over all five years (`O.facts.basket_size_counts_all`),
    sizes 1 to 7 as bars and 8 or more pooled into one bar labelled "8+". Neutral grey fill. Axis
    titles "SKUs per order" and "Orders (thousands)".
- Caption: none
- Drawer
  - An order counts each SKU once, whatever the quantity.
  - Share of orders with two or more SKUs, computed from `O.facts.basket_size_counts_all` [56%]. Only
    these can be split.
  - Pair statistics use the training set (years 1 to 3). Settings are chosen on the validation set
    (year 4). The results come from the test set (year 5).
- Controls: none
- States. Page-level loading text and error text as on the clustering page.
- Why this screen. Over half of all orders hold two or more SKUs, and only these can be split.

## Screen 2 (`counts`)

- Title: Orders per SKU pair.
- Visual
  - A 300 x 300 symmetric matrix. Rows and columns are SKUs, best sellers first (`G.nodes[].baskets`).
    Cell colour = `M.count` on the blue `sequential` scale, log, from 1 to the maximum [1,942]; cells
    with 0 and the diagonal stay white. Stepped legend drawn in the same canvas, titled "Orders with
    both".
  - Hover read-out under the matrix, as in `Heatmap`, giving the two SKU names and "Orders with both".
  - New component `PairMatrix` (canvas). `Heatmap` draws SVG cells with labels and values and is built
    for about 20 x 20 cells; 90,000 cells need a canvas to stay fast in Safari.
- Caption: Each cell is a pair of SKUs, best sellers first.
- Drawer
  - Orders with both SKUs, as a formula, training set.
  - Pairs bought together at least once (`O.facts.pairs_seen`) [27,264].
  - Two best sellers meet in many orders by chance alone. Under independence, orders with both by
    chance are about the product of their order counts divided by the number of orders, so the most
    frequent pairs are mostly pairs of best sellers.
  - The 20 most frequent pairs, with each SKU's orders (from `M.count` and `G.nodes[].baskets`).
- Controls: none
- States. `matrix.json` is loaded when the screen comes near. Loading text "Loading..." in the
  matrix frame; error text "The matrix could not be loaded."
- Why this screen. Counts of orders with both SKUs mostly rank how popular the two SKUs are.

## Screen 3 (`chance`)

- Title: Orders per SKU pair versus chance.
- Visual
  - Scatter of every pair with `M.count` at least 1 [27,264 dots]. Across "Orders with both by
    chance" (`M.expected`), log scale. Up "Orders with both" (`M.count`), log scale.
  - Linked pairs (`G.edges`) [317] in indigo, all other pairs as small light grey dots.
  - One dashed diagonal where the two are equal, labelled "As often as chance".
  - Direct labels inside the plot, "Linked pairs" in indigo beside the indigo dots and "Other pairs"
    in grey beside the grey mass. No legend box.
  - Hover only on linked pairs, giving the two SKU names, "Orders with both" and "Lift"
    (`G.edges[].count`, `G.edges[].lift`).
  - Reuses `Scatter` (log axes, dashed lines, hover), extended to draw the grey pairs on one canvas
    layer underneath. 27,264 SVG circles with pointer handlers are slow in Safari.
- Caption: Each dot is a pair of SKUs.
- Drawer
  - Chance is computed within orders of the same region (regions sell different SKUs), the same size
    (2, 3, 4, 5+, since large orders hold many) and the same quarter (91-day block, since growing or
    seasonal SKUs sell in the same months). Orders with one SKU hold no pair and are left out.
  - Orders with both by chance as a sum over these strata, as a formula. The quarter was added after
    the first run of the model.
  - Lift = orders with both / orders with both by chance.
  - Poisson tail p-value for each pair, as a formula. Its variance is at least the hypergeometric
    one, so the test is slightly conservative.
  - Benjamini-Hochberg over all pairs is designed to keep the false discovery rate among the pairs
    that pass it at the chosen level. A linked pair passes this test and has lift above 1.
  - Small counts give extreme lifts by luck. Pairs with lift above 3 that appear in fewer than five
    orders, computed from `M` [775 of 1,100]. The highest lift among pairs that are not linked [103,
    from one order].
  - The level is chosen later, by delivery cost on the validation set.
- Controls: none
- States. Shares the `matrix.json` load of screen 2.
- Why this screen. Comparing with chance removes popularity, and the significance test removes pairs that look strong by luck.

## Screen 4 (`network`)

- Title: Network of linked pairs.
- Visual
  - The 3D `Projector`, reused. Points are the 300 SKUs placed by a seeded 3D force layout
    (`G.nodes[].xyz`; unlinked SKUs sit on a loose outer shell). Point size by `G.nodes[].baskets`.
  - Every linked pair drawn as a faint line, opacity by log lift. Needs a new `edges` prop on
    `Projector` (today it draws lines only from a selected point to its neighbours).
  - Colour by community (`G.nodes[].community`), assigned so that communities placed near each other
    get different colours. Unlinked SKUs in light grey.
  - Shared-component change. `Projector` starts its rotation only when `prefers-reduced-motion` is
    not set to reduce, and then leaves "Turn slowly" unticked. This touches only viewers who ask for
    reduced motion; nothing else on the clustering page changes.
  - Side panel, default state, with no box borders inside the map frame.
    - Two plain lines with a hairline between them, "{n} communities" (length of `G.communities`)
      [45] and, after a light grey dot, "{n} unlinked SKUs" (count of `G.nodes[].alone`) [115].
    - The community list as on the clustering map (colour dot, name, size, click to show one on its
      own), from `G.communities[].name` and `.size`, largest first. Each name is the community's two
      best-selling SKUs ("ELE-0005 and ELE-0004"), so names are unique. The panel keeps the map's
      height and the list scrolls inside it.
  - Side panel, SKU selected. Name, category, "Orders" (`G.nodes[].baskets`), and one list headed
    "Strongest partners". It gives every member of the SKU's community with its category and, where
    linked to the selected SKU, the lift (from `G.edges`), sorted by lift; linked partners outside the
    community follow.
- Caption: Each dot is a SKU and each line a linked pair. Drag to turn, scroll to zoom, click a dot for details.
- Drawer
  - Graph weighted by log lift. Louvain maximises modularity, formula with the resolution γ.
  - Resolutions tried (`O.settings.resolutions`). A sentence, shown only when the backend flag
    `O.resolution_invariant` (see Data to add) is true, that every resolution gives the same
    communities, because each connected set of SKUs is already tightly knit. Ties go to the first
    resolution tried.
  - Community sizes, from `G.communities[].size` [2 to 6 SKUs].
  - Linked pairs that join two categories, from `G.edges` and `G.nodes[].category` [106 of 317].
  - `PairMatrix` again, SKUs ordered by community with unlinked SKUs last, colour = lift on the blue
    `sequential` scale (log, white at lift 1 or below and where no order holds both), legend titled
    "Lift". The communities show as blocks on the diagonal.
- Controls
  - Colour by: Community, Category
  - Turn slowly (checkbox; ticked by default unless reduced motion is set)
- States. WebGL failure shows the existing `Projector` message.
- Why this screen. The linked pairs fall into small, separate communities, many of them joining two categories.

## Screen 5 (`truth`)

- Title: Communities versus the hidden bundles.
- Visual
  - `Bars`, scale 0 to 1, two decimals.
    - Categories (`O.truth.check.categories.ari`) [0.06]
    - Communities from raw counts (`O.truth.check.count_communities.ari`) [0.34]
    - Our communities (`O.truth.check.communities.ari`) [1.00], in indigo
- Caption: Agreement with the bundles the simulator plants (ARI, 1 is a perfect match).
- Drawer
  - The simulator plants `O.truth.bundles` bundles [45] covering `O.truth.skus_in_bundles` SKUs [185].
    They are the ground truth.
  - Table of precision, recall and pairs put together for the three groupings
    (`O.truth.check.*.precision`, `.recall`, `.pairs_together`, and `.pairs_true` [317]).
  - Communities from raw counts use the same number of pairs, taken as the most frequent ones. Their
    Louvain resolution is chosen on the validation set like ours (`O.chosen.count_baseline_resolution`)
    [1.5]. They reproduce some bundles exactly (`O.truth.check.count_communities.bundles_exact`, see
    Data to add) [18 of 45]; categories reproduce none.
  - Of the 200 most frequent pairs, `O.facts.top200_count_true_share` [39%] lie inside a bundle. Of
    the 200 with the highest lift among pairs in at least five orders,
    `O.facts.top200_lift_true_share` [100%]. The five-order cut was set in the first version of the
    model, and the share is the same for any cut from 3 to 10 orders (computed from `M` and
    `G.nodes[].bundle`).
  - The first run computed chance within region and order size only. This check then showed linked
    pairs outside any bundle, between SKUs launched or growing at the same time or peaking in the
    same season, so the quarter was added. On validation cost alone, the version without the quarter
    would have been chosen (`O.quarter_note.without_quarter.val_cost` [$1,091,228] against
    `O.quarter_note.with_quarter.val_cost` [$1,093,648]), and its agreement was
    `O.quarter_note.without_quarter.truth.ari` [0.90]. The hidden bundles therefore helped choose the
    chance baseline behind our communities.
- Controls: none
- Why this screen. The communities recover every hidden bundle, raw counts recover fewer than half of them exactly, and categories none.

## Screen 6 (`assign`)

- Title: Assignment of SKUs to FCs.
- Visual
  - One column per FC, side by side, each headed only by its name ("FC 1", "FC 2" ...). Squares have
    a fixed size and each column a fixed grid width at every FC count. Assignment from
    `O.assignments[rule][fcs]`.
  - Within a column, squares sit in contiguous runs by community with a 2 px white gap between runs.
    Square colour = community, as on screen 4; unlinked SKUs light grey.
  - Hover and keyboard focus on a square give SKU name, community and FC. Selecting a square outlines
    every square of its community in all columns.
  - New component `FcColumns`. No existing part draws individual items placed into bins.
  - No split counts or costs on this screen.
- Caption: Each square is a SKU, placed in the FC that stores it.
- Drawer
  - An order with SKUs in two FCs ships as two parcels and counts as a split order.
  - All FCs sit at the central site (`O.settings.central_region`). Each SKU is stored in exactly one
    FC. Each FC may hold an equal share of the SKUs and of the workload (order lines in the training
    set), plus `O.settings.slack` [5%].
  - Table with the header "Rule" and "How SKUs are placed", one row per rule, built on the training
    set only.

    | Rule | How SKUs are placed |
    |---|---|
    | Random | In random order, each to the FC with the most room. |
    | Category | Whole categories, busiest first, each to the FC with the most room where it fits; a category too large for the room left is placed one SKU at a time. |
    | Best sellers spread | Best sellers first, each to the FC with the most room, so best sellers spread evenly. |
    | Best sellers together | Best sellers first, each to the first FC with room, filling FC 1, then FC 2, and so on. |
    | Our communities | Whole communities, busiest first, then unlinked SKUs, best sellers first, each to the FC with the most room. |
    | Hidden bundles | The same with the hidden bundles. Reference only, since a real shop does not know them. |

  - Every rule keeps within both limits except Best sellers together. Filling FCs in order leaves its
    last SKUs with no room anywhere, and each then goes where it overflows least, so it ends slightly
    over both limits (computed from `O.assignments.best_sellers` and `G.nodes[].baskets`) [up to 6
    SKUs and 6% of the workload limit over].
  - The workload limit and the Best sellers together rule were added after an earlier run had
    compared the rules on the test set. Without the limit, Best sellers together splits far fewer
    orders on the validation set by putting most order lines in FC 1
    (`O.workload_note.validation_without_workload_limit.best_sellers[].split_share` and `.load_share`)
    [10% split orders at 2 FCs with 94% of order lines in FC 1, against 34% for our communities]. A
    real FC cannot take that share of the work, so the limit applies to every rule.
  - Delivery cost per parcel `O.settings.parcel_fee` [$1.50] plus `O.settings.per_kg` [$0.35] per kg,
    times the distance multiplier from the central site.
  - Settings chosen by mean delivery cost of our communities over 2 to 5 FCs on the validation set
    (year 4). Table, one row per level in `O.settings.fdr_levels`, giving linked pairs
    (`O.grid[].edges`) and validation cost (`O.grid[].val_cost`). Chosen `O.chosen.fdr` [0.1%]. A
    generated sentence states which levels give the same pairs and tie on cost [1% and 5%, 318 pairs],
    that ties go to the first level tried, and the largest cost gap in percent [under 0.1%]. The
    quarter, described with the hidden bundles, and the workload limit above are the two choices not
    made on the validation set alone.
- Controls
  - Rule: Random, Category, Best sellers spread, Best sellers together, Our communities, Hidden bundles (default Random)
  - FCs: 2, 3, 4, 5 (default 2). One shared state with the screen 7 control; changing either changes both.
- States. Drawn from `O`, already loaded; nothing extra.
- Why this screen. Storing SKUs means choosing which FC holds each one, and only some rules keep each community in one FC.

## Screen 7 (`results`)

- Title: Results
- Visual
  - `Numbers`, value plus label only, our communities against Category on the test set at the
    selected number of FCs, from `O.results[rule][i]` matched on `fcs`.
    - Split orders, relative change in `split` [−3% at 2 FCs], whole percent
    - Parcels, relative change in `parcels` [−1%], whole percent
    - Delivery cost, relative change in `cost` [−0.2%], one decimal
  - `Bars` of split orders (% of orders, `split_share`) at the selected number of FCs, in this order.
    Random, Best sellers spread, Best sellers together, Category, Our communities (indigo), Hidden
    bundles (note "not known in a real shop") [39%, 40%, 39%, 36%, 35%, 35% at 2 FCs].
- Caption: none
- Drawer
  - Changes are relative to the Category rule. Of the plain rules it split the fewest orders on the
    validation set, apart from Best sellers together without the workload limit
    (`O.workload_note.validation_without_workload_limit`).
  - Orders in the test set (`O.results.*[].orders`) [104,573], of which `O.results.*[].multi`
    [58,850] hold two or more SKUs.
  - The test set was priced in more than one run while the method was being fixed. The quarter and
    the workload limit were added between runs, as the drawers above describe.
  - Full table, rule x FCs, with split orders, share of multi-SKU orders split (`multi_split_share`),
    parcels, delivery cost and a column "Workload share per FC, test set" (`load_share`). One line
    under it says the limit is set on the training set, and growth shifts workload by year 5, so
    shares on the test set can exceed it.
  - Split orders against Category at each FC count, as a range [3% to 7% fewer at 2 to 5 FCs].
  - `LineChart` (`frontend/src/components/LineChart.tsx`) of split orders (%) against FCs (2 to 5),
    one line per rule, with flags to hide its title, its Chart/Data toggle and its swatch legend. Each
    line labelled at its right end with the rule name, our communities in indigo and the others in
    neutral greys with distinct dash patterns. Lines that coincide share one stacked label ("Our
    communities, Hidden bundles").
  - Delivery cost moves little because most of it is the per-kg charge, which stays the same when an
    order ships as two parcels.
  - A sentence generated from `O.results`, shown only when true, that our communities and the hidden
    bundles give the same result [true at every FC count].
- Controls
  - FCs: 2, 3, 4, 5 (default 2; the shared state of screen 6; drives the boxes and the bars)
- Why this screen. Storing each community in one FC gives fewer split orders than every rule a shop could use, and the same as the hidden-bundle rule, at every number of FCs tried (2 to 5).

---

## Data to add (backend)

- `O.resolution_invariant`, true when the community labels (not only the grid summaries) are
  identical at every resolution tried.
- `O.truth.check.count_communities.bundles_exact`, the number of hidden bundles that communities from
  raw counts reproduce exactly [18].

Already in the outputs and used above are `O.facts.orders_by_year`, `O.facts.skus_per_order_all`,
`O.facts.basket_size_counts_all`, `O.workload_note`, `M.expected` (three significant figures),
`G.nodes[].xyz` and `G.communities`. The chosen row of `O.grid` is found by matching `O.chosen.fdr`
and `O.chosen.resolution`.

---

## Changes in round 1

High findings and their fixes.

- Rules 4 (context in the screen 1 axis label). Screen 1 shows all five years; axis "Orders
  (thousands)", no qualifier.
- Rules 7 (editorial "How often ..." title). Screen 2 title "Orders per SKU pair."
- Rules 16 and truth 5 ("Bought alone" is untrue). Renamed (now "unlinked SKUs", round 2).
- Rules 26 (bar note restates the row). Hidden bundles note "not known in a real shop".
- Truth 1 (chance strata out of date). Screen 3 drawer gives region, size (2, 3, 4, 5+) and quarter.
- Truth 2 and slop 2, 3, 6 (known limit now false, hard-coded counts). Paragraph removed.
- Truth 3 (communities edge out the bundles). Replaced by a generated sentence that they give the
  same result.
- Truth 4 (quarter added after the truth check). Full disclosure with `O.quarter_note` in the screen
  5 drawer, a short line in the screen 3 and screen 6 drawers.
- Truth 6 (raw-count baseline at its worst resolution). Resolution chosen on validation, stated.
- Truth 7 (no workload limit, best sellers together missing). Limit and rule added, `load_share` in
  the results table.
- Slop 1, 4, 5 (forced contrasts). Rewritten or removed.
- Slop 12, 13 (legends). Direct labels and end labels.
- Slop 14 (reduced motion). Honoured in the shared `Projector`.
- Slop 15 (community by colour alone). Hover, focus, outline on selection, runs separated by gaps.
- Slop 16 (boxes inside the map panel). Plain lines with a hairline.

## Changes in round 2

Settled decisions applied.

- One shared FCs control for screens 6 and 7, 2 to 5, default 2, no mean. Box and bar brackets now
  at 2 FCs (rules 1, 2; truth 6, 8).
- Two disclosed exceptions to choosing on the validation set. The quarter (full account on screen 5)
  and the workload limit with Best sellers together (full account on screen 6, from
  `O.workload_note`). "Priced once" replaced by an accurate account of the runs on screen 7
  (truth 1, the round 2 high).
- Best sellers together described as slightly over both limits, with the size computed from the
  data (truth 2).
- `load_share` labelled as measured on the test set, with a line on why it can exceed the limit
  (truth 3).
- No claim that our communities beat every rule; they match the hidden-bundle rule (truth 4).
- Screen 5 says raw counts reproduce fewer than half the bundles exactly, with the count from a new
  field (truth 5).
- "Unlinked SKUs" everywhere (rules 3).
- Community names from `G.communities[].name`, the two best sellers; category naming and its test
  dropped (rules 4).
- Screen 4 caption restored to the approved clustering wording (rules 5).
- Reduced motion stated as a change to the shared `Projector` only (rules 6).
- "Baseline" note removed from the Category bar; the drawer says why Category is the comparison
  (rules 7, 8).
- Screen 1 share of multi-SKU orders computed from `basket_size_counts_all` (rules 9, truth 9).
- Open decisions section removed; data fields that now exist moved out of "Data to add" (rules 10,
  truth 10).

Other medium and low findings taken. Slop 1 and rules 13 (label lead-in in the quarter paragraph),
slop 2 (negative close rewritten), slop 3 (one triad on screen 3), slop 4 (rules as a real table),
slop 5 ("best sellers first" throughout), slop 6 ("orders with both by chance" throughout), rules 11
("pairs that pass it"), rules 12 (no screen numbers in the drawer), rules 14 (rule names lower case in
prose), rules 15 ("priced once" removed), rules 16 (one "Strongest partners" list), rules 17 (list
order and scrolling), rules 18 (violet left out of the community palette), rules 19 (`LineChart`
flags and stacked labels), rules 20 (legend "Lift"), truth 11 (launched, growing or same-season SKUs),
truth 12 (the five-order cut), truth 13 (resolution sentence driven by a backend flag).

Not taken. Truth 7 (four significant figures for `M.expected`); the orchestrator settled on three,
which keeps the error under 0.5%. Truth 14 (reference in the box labels); Kilian's Gate 0 choice of
bare labels stands, and the drawer states the reference.
