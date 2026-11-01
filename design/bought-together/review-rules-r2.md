# Rules review, round 2

The revised STORYLINE.md was checked with the round 1 method against the house rules (Writing style, Design rules 1 to 10), the design review log, the approved vocabulary and Gate 0 (where a rule clashes with a choice Kilian made on the clustering page, his choice wins). Figures are not judged. The orchestrator's decisions are taken as settled. Screen 1 shows all five years. Screen 7 follows an FCs control (2 to 5) with a default of 2 FCs and no mean. The full quarter disclosure stays in the screen 5 drawer with one neutral line on screen 3. Kilian approved "community" and "lift" for the surface and replaced "accepted pair" with "linked pair".

## Round 1 high and medium findings

| Round 1 # | Finding | Status |
|---|---|---|
| 4 (high) | Context in the screen 1 axis label | Fixed. All five years, axis "Orders (thousands)". |
| 5 | Two names for SKUs per order | Fixed. |
| 6 | "Orders" and "baskets" mixed | Fixed on every surface and drawer (field names are not visible). |
| 7 (high) | Editorial screen 2 title | Fixed, "Orders per SKU pair." |
| 11 | Screen 3 caption restated the line label | Fixed. |
| 13 | Settings table before FCs were introduced | Fixed. Moved to the screen 6 drawer, and the hidden bundles are not named on screen 3. |
| 14 | Chance strata out of date | Fixed. |
| 15 | Lift under two names, "accepted pair" unapproved | Fixed. "Lift" and "linked pair" throughout, with one leftover verb (row 11). |
| 16 (high) | "Bought alone" was untrue | Fixed, but the replacement "unpaired SKUs" now clashes with the approved "linked pair" (row 3). |
| 17 | Community colours repeat in the unit chart | Fixed. Runs with gaps, colours assigned by proximity, and selection outlines the community. |
| 19 | Community names uncapped and untested | Partly fixed. Capped at two categories by a test, but names repeat across communities and a data change breaks the build (row 4). |
| 21 | Stale prose with hard-coded counts | Fixed. "Known limit" gone, and the sentences are generated from fields. |
| 23 | Screen 6 caption anticipated the results | Fixed. |
| 26 (high) | Bar note restated the row name | Fixed, "not known in a real shop". |
| 29 | Unconfirmed explanation on screen 7 | Fixed. Replaced by a generated sentence. |

## Findings, round 2

| # | Where (screen and element) | Finding | Rule or precedent | Severity | Concrete fix |
|---|---|---|---|---|---|
| 1 | Screen 7, controls, boxes and bars | The spec still makes Mean the default and offers it as an option ("at the selected number of FCs or the mean over 2 to 5", bar values "[mean ...]"). This contradicts the settled decision. | Orchestrator decision (settled); rule 6 (one figure, one stated basis) | Medium | The FCs control offers 2, 3, 4 and 5, with 2 as the default, and drives the boxes and the bars. Boxes read "at the selected number of FCs". Drop Mean everywhere on the screen, and bracket the example values at 2 FCs. |
| 2 | Screen 6, FCs control | It defaults to 3 FCs while screen 7 defaults to 2, so the assignment the reader studies is not the one the results describe. | Rule 2 (a convention set on one screen holds on every screen) | Medium | Default 2 on screen 6 as well. Link the two controls so a change on one screen shows on the other. |
| 3 | Screens 4 and 6, "unpaired SKUs" (side panel line, grey squares, rule descriptions, header word list) | Every SKU belongs to 299 SKU pairs (the opening box counts them), so "unpaired" is untrue. These SKUs only lack a linked pair, and with "linked pair" approved the honest opposite is ready to use. This was my own round 1 fix, so I am correcting it. | Rule 6 (a name may claim only what its members share; one vocabulary) | Medium | "{n} unlinked SKUs" in the side panel, "unlinked SKUs" in every other place, and "unlinked SKU" (a SKU with no linked pair) in the header word list. |
| 4 | Screen 4, side panel, community names | Category names will repeat across about 45 communities (several single-category communities in the same category get the same name), and colours repeat too, so the list cannot tell communities apart. The test also turns a data change (a third category) into a failed build rather than a naming rule. | Rule 6 (names give the distinguishing traits, claimed names get a test); log, cluster names | Medium | Name a community by its two best-selling SKUs joined by "and" ("Rose fertiliser and Garden hose"). Each SKU sits in one community, so names are unique, and `lexicon.test.ts` checks uniqueness. Categories stay visible through Colour by Category and the SKU panel. |
| 5 | Screen 4, caption | "Drag to turn it and click a dot for details." drops "scroll to zoom" from the clustering caption Kilian approved, although the same `Projector` zooms on scroll (OrbitControls), and "it" has no clear referent. The checklist's rule of three does not override his choice. | Gate 0; rule 9 (once a choice is approved, keep it); rule 2 | Medium | "Each dot is a SKU and each line a linked pair. Drag to turn, scroll to zoom, click a dot for details." |
| 6 | Screen 4, visual, reduced motion | "The same fix goes into the clustering page" changes a page Kilian has not asked to change. | Rule 10 (change only what I asked); log, clustering cluster names ("widening scope mid-review") | Medium | Delete the sentence. Ask Kilian in one line whether the clustering map should honour reduced motion too. |
| 7 | Screen 7, bars, note "baseline" on Category | A role tag on a bar. On the clustering results Kilian chose bare boxes and marked no baseline bar, and the drawer's first line already names Category. | Gate 0; rule 7 (show state by styling, not tags); log, models home (tags removed) | Medium | Drop the note. Keep the drawer's opening sentence. |
| 8 | Screen 7, drawer, first item | "the best of the other rules tried" does not say on which data Category was found best. If that was the test set, the baseline was chosen on what is reported. | Rule 6 (say how every choice was made and on which data; never tune on what is reported) | Medium | "Changes are relative to the Category rule, the best of the other rules on the validation set, measured once on the test set (year 5)." If the pick was made on the test set, redo it on the validation set. |
| 9 | Screen 1, drawer, share of orders with two or more SKUs | `O.facts.multi_share` is the training-set field, while the spec now says screen 1 shows one data set, all five years. | Rule 6 (on which data); rule 2 (one system per screen) | Medium | Compute the share in the page from `O.facts.basket_size_counts_all` (orders with two or more SKUs over all orders), so it covers the same five years. |
| 10 | Open decisions section | All three points are now settled, and point 2 was settled the other way (default 2 FCs, no mean). | Spec hygiene; rule 10 (record his decisions as made) | Low | Replace the section with three lines under "Settled in round 2" giving the decisions above. |
| 11 | Screen 3, drawer, Benjamini-Hochberg item | "among the pairs it accepts" brings back the word Kilian rejected. | Rule 6 (one vocabulary); Kilian's vocabulary decision | Low | "is designed to keep the false discovery rate among the pairs that pass it at the chosen level." |
| 12 | Screen 5, drawer, top-200 item | "the small-count cut of screen 3" points to a screen number the page never shows. | Rule 1 (no numbering) | Low | "(the five-order cut, fixed before this check)". |
| 13 | Screen 5, drawer, quarter item | It opens with the label-style lead "How the quarter entered the chance baseline.", the pattern removed from the screen 6 drawer in round 1. | Slop checklist, bold label lists (no clash with Gate 0) | Low | Start the item with "The first run computed chance within region and order size only." |
| 14 | Screens 6 and 7, drawers | "mean delivery cost of Our communities" and "that Our communities and the hidden bundles" capitalise a rule name mid-sentence. | Writing style (sentence case in prose); rule 2 | Low | Lower case in running text ("our communities"). Keep the capital only where it is a bar or control label. |
| 15 | Screens 1, 6 and 7, drawers | "The test set (year 5) is priced once" is stated three times. | Rule 5 (no repeats) | Low | Delete the sentence from the screen 6 drawer. |
| 16 | Screen 4, side panel, SKU selected | "its community's members" and "Strongest partners" mostly name the same SKUs, because communities hold 2 to 6 SKUs and linked pairs sit mostly inside them. | Rule 7 (each element says something nothing else says) | Low | One list, headed "Strongest partners", sorted by lift, giving each member's category and the lift for those linked to the selected SKU. Linked partners outside the community follow. |
| 17 | Screen 4, side panel, community list | About 45 rows inside the map frame, with no order or height given. | Rule 8 (size to content) | Low | Largest first. The panel keeps the map's height and the list scrolls inside it. |
| 18 | Screens 4 and 6, community palette | `cat` includes the violet #6250d6, close to the indigo accent #4338ca, so one community may look like our result, against the header's "Indigo has one job". | Rule 7 (colour only for meaning) | Low | Leave #6250d6 out of the community palette on this page. |
| 19 | Screen 7, drawer, `LineChart` | The game's `LineChart` brings an h3 title, a Chart/Data toggle and game button classes, not only a legend. The toggle repeats the full table above it, and the end labels of Our communities and Hidden bundles will overlap where their lines coincide. | Rule 9 (keep the established look); rule 7; rule 8 | Low | Hide the title, toggle and legend (or draw the lines with the extended `Scatter`), and stack end labels that would collide. |
| 20 | Screen 4, drawer, second `PairMatrix` | Its legend has no title, although the first matrix's legend has one. | Rule 2 (one system on every screen) | Low | Legend title "Lift". |

**Verdict.** PASS, with 0 high, 9 medium and 11 low.

The medium findings are still worth taking before the build, rows 1 to 3 in particular. Row 1 is the settled screen 7 decision, which is not yet in the spec. Rows 2 and 3 follow from it and from the approved vocabulary.
