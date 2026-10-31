# Design review log

A running record of Kilian's comments on the model pages, screen by screen: what he asked for, what
changed, and the drafting habit it corrects. After the last screen, the habits are distilled into
general lessons (design principles, design language, systems).

## Models home (2026-10-31)

- **Comment.** Subtitles and labels around titles where none are needed; messy.
- **Change.** Removed the "Machine learning and optimisation" label above the title; each card is now
  title plus question only (no "01 · Clustering" label, no Ready/Coming tag, no results line).
  Unbuilt models are greyed out instead of tagged.
- **Habit.** Decorating titles with labels, numbering and status tags.

## Clustering page, opening (2026-10-31)

- **Comment.** Label above and notes below every element; most of it obvious. Results given away in
  the first screen. Then: the bare title looked empty; bring the boxes back with introductory content.
- **Change.** Title plus question ("Can SKUs that sell alike share one stocking policy?"), then four
  plain boxes on the shop (300 SKUs, 8 categories, 8 regions, 30,000 customers). Results only in
  their own sections near the end. Customers and regions removed from "The data" so nothing repeats.
- **Habit.** Opening with the answer instead of building a storyline; a presentation introduces,
  then reveals.

## Clustering page, screen 1: SKUs sell differently (2026-10-31)

- **Comment.** "The question" label is wrong (the screen is not a question). Drop the sentence about
  the hidden type and the ground truth. "+ How it works" should be a bigger blue plus icon (white
  plus in a filled blue circle), no text.
- **Change.** Label removed (labels are now optional per screen). Caption cut to "Weekly sales of six
  typical SKUs over the training set." Every drawer now opens with a 30 px filled accent circle with
  a white plus (a minus when open); the text lives on as the accessible name and hover tooltip.
- **Habit.** Labels added by template rather than meaning; captions that pre-empt a later screen;
  text where an icon is enough.

## Clustering page, screen 2: the data (2026-10-31)

- **Comment.** "order lines" and "orders" should be capitalised. The caption explaining the training
  and test years is completely unnecessary.
- **Change.** Box labels in sentence case ("Order lines", "Orders"), applied to the opening boxes too
  ("Categories", "Regions", "Customers"). Caption removed; the year strip already says it.
- **Habit.** Lower-case labels that read as sentence fragments; captions that repeat what the visual
  already shows.

## Clustering page, screen 3: feature engineering (2026-10-31)

- **Comment.** Why is everything a percentage (ADI cannot be one)? "Pick a SKU" when nothing looks
  pickable. Seven cards, unbalanced. Card titles far too editorialised ("How much it sells"); plain
  ones like "Price and margin" are right.
- **Cause.** The bars showed one SKU's percentile rank among 300, not the feature values, and nothing
  said so. The buttons picked a SKU but were labelled with its type.
- **Change.** Replaced cards and buttons with a feature-vector heatmap: rows are the 16 features in
  their seven groups, columns the six SKUs from screen 1, cells show real values with units,
  colour shows the z-score the model sees. Plain group names (Sales volume, Variability,
  Intermittency, Seasonality and trend, Size and weight, Price and margin, Basket) and plain feature
  names; the subtitles under group names are gone; the caption is gone. Seven groups kept, because
  the grouping is part of the model (equal weight per group).
- **Habit.** Showing a derived statistic instead of the quantity itself, and not saying so;
  cute, editorial names instead of explicit ones; controls whose labels hide what they select.

## Clustering page, screen 3 again: cleaning the feature table (2026-10-31)

- **Comment.** Likes the table, but it looks messy: too many colours, too many units, too many symbols
  (+, %, wk, L, ...).
- **Change.** Units moved into the row labels ("Unit volume (L)", "Revenue ($k)"), cells hold bare
  numbers with fixed decimals per row. Colour only where a value is more than one standard deviation
  from the average SKU (two light tints each way, everything else white); dark text throughout.
  Outer border, column gaps and per-row lines removed; group names became header rows with a hairline
  above. Feature names grey, values black. Header dots kept.
- **Habit.** Encoding everything at once (fill on every cell, a unit on every number); letting
  decoration (lines, borders, symbols) compete with the data. Show signal, not everything.

## Clustering page, screen 3: bolder and tighter (2026-10-31)

- **Comment.** Make the heatmap more aggressive; the table could be more compact.
- **Change.** Colour still starts at one standard deviation, but now deepens continuously up to three,
  strong enough for white text at the extremes; gradient legend with a white middle band. Table no
  longer stretches to full width: equal 122 px value columns, tighter rows (3 px padding, 13.5 px
  text), smaller group gaps; the table is now 688 px tall.
- **Habit.** Over-correcting into timidity: when cutting noise, keep the remaining signal strong.
  Stretching tables to fill the page instead of sizing them to their content.

## Clustering page, screen 3: colour scale round trip (2026-10-31)

- **Comment.** Asked to bring back the original scale (every cell coloured), then to undo it again.
- **Change.** Tried, then reverted: the bolder scale stays (white within one standard deviation,
  deepening to three).

## Clustering page, screen 4: the embedding (2026-10-31)

- **Comment.** The headline ("SKUs that sell alike end up close together.") must change. "purity"
  must be capitalised. Asked how removing features raises purity, and whether that breaks good model
  verification.
- **Change.** "Purity" capitalised. The live score stays, with a warning under it that choosing
  features or k to raise it is selection bias. Headline now "Visualisation of SKUs in feature
  space." ("feature space" added to the approved vocabulary; Kilian's own wording, "vector space",
  sharpened to the standard ML term).
- **Follow-up.** "The embedding" label removed.
- **Habit.** Headlines that assert a finding instead of naming what is shown; an interactive score
  that invites cherry-picking without saying so; labels added by template.

## Method change: train, validate, test (2026-10-31)

- **Comment.** Was all features and k = 8 chosen on the training or the test set? Shouldn't there be
  a validation set (train, validate, test)?
- **Finding.** Nothing was chosen on the test set or the ground truth (k by BIC, features and method
  fixed in advance), but year 4 was both inside the feature window and the year policies were chosen
  on; not a clean split.
- **Change.** Training set years 1 to 3 (features, clustering, policy per cluster chosen on year 3),
  validation set year 4 (all 2,286 settings priced; cheapest chosen), test set year 5 (only the
  chosen setting and the baselines priced, once). The single-policy baseline is now also chosen on
  the training set instead of fixed. Year strip shows 3 + 1 + 1.
- **Lesson.** Say how every choice was made and on which data, before showing any number.

## Clustering page, screen 6: the trade-off (2026-10-31)

- **Comment.** After the validation split, purity fell from 84% to 61%: why (less data?). Then: make
  the trade-off the point.
- **Finding.** Not the data: the old setting scores 84% on three years as on four. Validation chooses
  by cost, and the cheapest setting keeps size and price and drops variability, intermittency and
  basket, the features that define the demand types.
- **Change.** The ground truth screen now shows one chart: every one of the 2,286 settings as a dot,
  stocking cost on the validation set across, purity up. Marked: our clusters (cheapest), the best
  match to the demand types, the demand types themselves, ABC, ABC/XYZ, demand classes and one
  policy for every SKU. Headline "Stocking cost and the demand types pull in different directions."
  The purity bars moved into the drawer.
- **Lesson.** When two goals conflict, show the whole space of options rather than a single winner.

## Clustering page, screens 4 and 5: setting presets, 3x3 clusters (2026-10-31)

- **Comment.** The visualisation needs buttons to switch between the lowest stocking cost and the
  highest purity or ARI. The clusters screen should be a 3x3 grid. Asked what the card labels mean and
  how central or regional stocking is decided.
- **Change.** New "Setting" control on the map: Lowest stocking cost (k-means, 9, four groups, 61%),
  Highest purity (98%), Highest ARI (0.82); each sets features, method and k at once. Clusters screen
  now three columns, so nine clusters form a 3x3 grid.
- **Habit.** Making the reader reconstruct a setting by hand when the interesting ones can be one
  click away; grids sized by a default rather than by the number of items.

## Clustering page, screen 5: the clusters (2026-10-31)

- **Comment.** Remove "The clusters"; make the title more direct. Still unclear what the card figures
  and the graphs show, and how central or regional stocking is decided.
- **Change.** Label removed; headline "The clusters and their stocking policies." Caption now says
  what the graphs and figures are, in Kilian's wording: "Graphs and numbers are per-cluster median
  weekly values (training set)." (Shorter than my two-sentence version.) Card labels follow the feature table system: "Units per week", "Weeks with a sale
  (%)", "Price ($)", "Unit volume (L)", bare numbers. The drawer explains the naming rules and how
  the stocking policy is chosen (cheapest of 12 options on the training set, and why small, cheap
  SKUs go central).
- **Habit.** Numbers and charts without saying what they are; one system (units in labels) applied
  on one screen but not the next. An explanation given in chat belongs on the page.

## Clustering page: cluster names, trade-off title, caption (2026-10-31)

- **Comment.** Remove "GROUND TRUTH" and make the title more concise. Caption: "per-cluster medians
  (training set)". (I misread "implement the fix" as the naming fix and went ahead with it; Kilian
  only meant the caption. Kept at his request to finish it.)
- **Change.** Trade-off screen: label removed, headline "Stocking cost versus purity." Caption fixed.
  Names: each SKU gets a pattern word from fixed cut-offs (bulk orders, then growing and fading, then
  intermittent, then seasonal if its own yearly pattern beats 19 reshuffles after removing the
  shop-wide calendar); a cluster takes a pattern word only when 60% of its SKUs share it, otherwise
  the two traits that set it apart, from the features it used ("Fast, bulky"). No ground truth used;
  a test checks the three presets against it. The life-cycle-before-intermittency order was changed
  after that test failed once, and the drawer says how names are made.
- **Habit.** Acting on an ambiguous instruction without asking which thing was meant; widening scope
  mid-review. Names that claim a pattern the cluster does not have.

## Clustering page, screen 7: results; "What we learned" removed (2026-10-31)

- **Comment.** Title "Results", no label. Remove the second line in the number boxes, the caption,
  and the policy counts under the first three bars. Capitalise the box labels. Remove the "What we
  learned" section.
- **Change.** All done: headline "Results"; boxes show value plus a capitalised label only ("Stocking
  cost on the test set", "Fill rate", "Stocking policies to manage"); bars keep only the per-SKU note;
  caption gone; the closing section and its drawer removed, so the page ends on the results.
- **Habit.** A note under every figure by default; summing up what the screens already showed.
- **Follow-up.** Box labels cut to the subject noun: "Stocking cost", "Fill rate", "Stocking
  policies". "(9 classes)" and "(300)" removed from the bar names (in the lexicon, so the drawer
  tables match). Lesson: labels are bare subject nouns; qualifiers like "on the test set" belong to
  the context, not the label.

## Distilled rules (2026-10-31)

The habits above, distilled into ten rules and saved in Kilian's global instructions
(`~/.claude/CLAUDE.md`, section "Design"), so they apply to every future project. Each rule and the
entries that led to it.

1. **Titles**, a plain noun phrase and nothing around it. Models home, opening, screens 1, 4, 5, 6, 7.
2. **Labels**, bare subject nouns, units in the label, context elsewhere, one system. Screens 2, 3, 7.
3. **Captions**, only to say what an unreadable visual is, in one line. Screens 1, 2, 5, 7.
4. **Depth** in drawers behind the plus icon. Screens 1, 6, and the explanations asked for in chat.
5. **Storyline**, introduce then reveal, no early results, no repeats, no recap. Opening, screens 2, 7.
6. **Truth**, real quantities, honest names, how each choice was made. Screens 3, 4, method change,
   screen 6, cluster names.
7. **Encoding**, one dense direct encoding, no ink without data, strong colour when used. Screen 3.
8. **Layout and controls**, sized to content, presets for interesting states. Screens 3, 4, 5.
9. **Restraint without timidity**, keep what remains strong. Opening, screen 3, LaTeX trial.
10. **Working with me**, smallest change asked, his wording kept, answer length as asked. Names,
    captions, the one-sentence and yes/no answers.

Checked by an independent review against this log before saving; its corrections (titles may keep a
question in the opening, his wording is kept but factually corrected, removed detail moves to drawers,
context goes in captions rather than labels) are included.
