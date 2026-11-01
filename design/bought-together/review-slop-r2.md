# Bought together, AI-slop review, round 2

Spec reviewed is the revised `STORYLINE.md` (lines 1 to 296, the change log excluded) against `design/SLOP_CHECKLIST.md`, with every exemption in its conflicts section applied, including Kilian's Gate 0 decision (equal number boxes, bare-label boxes, 12 to 14 px text and series colour dots are allowed). Numbers and method are out of scope. The "Why this screen" and "States" lines are treated as internal notes.

Mechanical checks with no hits are the em dash and en dash characters, the AI vocabulary list (zero on every screen), marketing words, copula avoidance, forced contrast patterns in visible text, hollow "-ing" tails, signposting, wrap-ups, vague attribution, chatbot residue, emoji, kickers above titles, Title Case and colons in prose. The word "basket" survives only inside the field name `G.nodes[].baskets`.

## Round 1 findings

| # (r1) | Sign | Status |
|---|---|---|
| 1 | Forced contrast, "never by the hidden bundles" | Fixed. Now "The level is chosen later, by delivery cost on the validation set." |
| 2 | Forced contrast, "timing rather than affinity" | Fixed. Paragraph removed. |
| 3 | Rounded, hard-coded counts | Fixed. Paragraph removed; new figures cite fields. |
| 4 | Forced contrast, regional FCs | Fixed. Line removed. |
| 5 | Forced contrast, delivery cost | Fixed. Positive statement taken word for word. |
| 6 | Label fragment "Known limit." | Fixed, but the same pattern returns in the screen 5 drawer (row 1 below). |
| 7 | Label fragment "Why not regional FCs." | Fixed. Line removed. |
| 8 | Rule of three, screen 4 caption | Fixed. Two instructions. A new pair of triads appears on screen 3 (row 3 below). |
| 9 | Elegant variation, order and basket | Fixed. "Order" everywhere. |
| 10 | Elegant variation, bundles and rule names | Fixed. "Hidden bundles" and "Our communities" throughout. |
| 11 | Elegant variation, "connected group" | Fixed. "connected set of SKUs". |
| 12 | Scatter legend | Fixed. Direct labels inside the plot, no legend box. |
| 13 | Line chart legend | Fixed. End labels, ours indigo, others grey with dashes, legend hidden. |
| 14 | Reduced motion | Fixed, here and on the clustering page. |
| 15 | Communities by colour alone on screen 6 | Fixed. Hover and focus read-out, outline on selection, runs split by gaps. |
| 16 | Boxes inside the map panel | Fixed. Plain lines with a hairline. |
| 17 | Unlabelled community colours | Fixed. Community list in the side panel, colours assigned by layout proximity. |
| 18 | Indigo histogram | Fixed. Neutral grey. |
| 19 | Indigo magnitude scale | Fixed. Blue `sequential` scale, stepped legend drawn in the canvas. |
| 20 | Happy path only | Fixed. Loading and error states given per screen. |

## New findings

| # | Where (screen and element) | Text or design element | Checklist sign | Severity | Concrete fix |
|---|---|---|---|---|---|
| 1 | Screen 5, drawer, quarter paragraph | "How the quarter entered the chance baseline." as a lead-in fragment | Bold label list (label then explanation, a colon in disguise; house colon rule) | Medium | Delete the fragment and open with "The first run computed chance within region and order size only." |
| 2 | Screen 5, drawer, quarter paragraph | "So the agreement of our communities is not a fully independent check." | Hedge stacks and corrective framing (a "does not establish" style negative close) | Medium | "The hidden bundles therefore helped choose the chance baseline behind our communities." |
| 3 | Screen 3, drawer, strata line | "the same region, the same size (2, 3, 4, 5+) and the same quarter" followed by "because regions sell different SKUs, large orders hold many, and growing or seasonal SKUs sell in the same months" | Rule of three (two triads in one sentence; the page allows one) | Medium | Fold the reasons into the list, so one triad remains. "Chance is computed within orders of the same region (regions sell different SKUs), the same size (2, 3, 4, 5+, since large orders hold many) and the same quarter (91-day block, since growing or seasonal SKUs sell in the same months)." |
| 4 | Screen 6, drawer, the six rules | Bullets of the form "Random. SKUs in random order, ...", "Category. Whole categories, ...", and so on | Bold label list (label then explanation, six times) | Medium | Make it a real two-column table with the header "Rule" and "How SKUs are placed", one row per rule, and the descriptions as they stand without the leading name. |
| 5 | Screens 2 and 6, caption and drawer | "best sellers first" (screen 2 caption, rule names) beside "most bought first" (screen 6 drawer, three times) for the same ordering | Elegant variation | Medium | Use "best sellers first" everywhere. Screen 6 drawer "Best sellers first, each to the FC with the most room, ..." for Best sellers spread, "Best sellers first, filling FC 1, then FC 2, and so on." for Best sellers together, and "then unpaired SKUs, best sellers first, each to the FC with the most room." for Our communities. |
| 6 | Screens 2 and 3, drawers | "the expected number" (screen 2) and "Expected count" (screen 3) beside "Orders with both by chance" (screen 3 axis and lift formula) | Elegant variation | Medium | Use the axis term in both drawers. Screen 2 "Under independence, orders with both by chance are about the product of their order counts divided by the number of orders, so the most frequent pairs are mostly pairs of best sellers." Screen 3 "Orders with both by chance as a sum over these strata, as a formula." |

## Notes outside the table

- No design finding is new. Indigo now has the one job the spec states (our result), magnitudes use the blue scale, community colours carry a text label in the side panel, every colour-coded view has a second channel (position, hover or outline), and the only number boxes are the exempt rows.
- The screen 7 "Why this screen" line says Our communities give fewer split orders "than every other rule tried", while the planned bars show Hidden bundles at the same 41%. It is internal and a numbers matter, so it is passed to the other critics and not flagged here.

PASS, 0 high, 6 medium, 0 low.
