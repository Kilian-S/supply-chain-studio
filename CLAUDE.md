# Supply Chain Studio, project instructions

Portfolio project in operations research and machine learning.
It has three pillars, a discrete-event simulator of a multi-region supply network with a scenario
designer, the G4 autonomous bots, and the core, ML and OR models for long-term (strategic)
supply chain decisions in a simulated multi-product shop. Read `docs/HANDOVER.md` first for the
current state and open items, `docs/ML_ROADMAP.md` for the models, and `docs/DESIGN_REVIEW.md` for
every design comment Kilian has made and the rules distilled from them.

## Setting up (also in a fresh cloud clone)

```bash
cd backend && uv sync && uv run python -m scgame.world.generator ../data/world_v1 7   # simulated shop, about 3 s, identical every time
cd ../frontend && npm install
./dev.sh   # from the repo root: API on :8000, web app on http://localhost:5173
```

Model outputs are committed in `data/models/<id>/`. Rebuild one with
`cd backend && uv run python -m scgame.ml.<module> ../data/world_v1 ../data/models/<id>`
(`sku_segmentation` writes `sku-segmentation`, `product_affinity` writes `product-affinity`).

Run these checks before any commit.
`cd backend && uv run pytest && uv run ruff check src` and
`cd frontend && npx tsc --noEmit -p tsconfig.app.json && npx eslint src/models && npx vitest run`
(`tsc -p .` checks nothing, the root config lists no files).

## Working agreements

- Each model's exact implementation is discussed with Kilian before it is built. External data
  sources are shown and agreed before use.
- Models are prepared (trained beforehand), never trained live on the page.
- Training set years 1 to 3, validation set year 4, test set year 5. Every setting is chosen on the
  validation set. The ground truth (hidden types, planted bundles) is never used to choose; any choice
  it informed is disclosed on the page.
- Never show internal model letters (A, B, J...) in the UI.
- Commit only when Kilian asks or when he has asked for autonomous work; never push to `main` without
  his go-ahead. Do not reuse any game credentials.

## Model pages, look and vocabulary

- Editorial look for the models area only, with a white ground, hairline borders, square corners, no
  shadows, one indigo accent, SF Pro only (no SF Mono). The game keeps its playful look.
- Every visible string lives in the page's `copy.ts`, covered by `lexicon.test.ts`.
- Approved surface terms are SKU, feature, feature vector, feature space, feature engineering,
  unsupervised learning, cluster(ing), k-means, k, centroid, Euclidean distance, kNN,
  standardisation/z-score, log transform, Pearson correlation, dimensionality reduction, embedding,
  PCA, ground truth, purity, ARI, hyperparameter, ABC/XYZ analysis, ADI, CV², intermittent demand,
  STL decomposition, training/validation/test set, selection bias, overfitting, stocking policy, FC,
  service level, fill rate, order-up-to level, lead time, safety stock, risk pooling, holding cost,
  lost sales, calibration, Pareto principle, community, lift, linked pair, co-occurrence.
- Drawer-only (see `docs/GLOSSARY.md`): GMM, soft assignment, EM, UMAP, NMI, silhouette score, BIC,
  bootstrap stability, Syntetos-Boylan. Ask Kilian before promoting a new term.
- The AI-slop checklist in `docs/design/SLOP_CHECKLIST.md` applies, with Kilian's choices winning
  where it conflicts (see its last section).

## Writing style (Kilian's global rules, copied here for cloud sessions)

- British English spelling and grammar ("-ise/-isation", colour, behaviour, organise, optimise,
  centre, analyse, favour, licence as noun, programme).
- Never use the em dash character. Rewrite with commas, parentheses or separate sentences. Hyphens in
  numeric ranges are fine.
- Never use colons in prose. In particular never write a sentence where a colon follows part of a
  sentence and then expands or restates it. Rewrite as separate sentences. This applies to everything
  written for Kilian, including replies.

## Design rules (Kilian's global rules, copied here for cloud sessions)

Design like an editor preparing a talk for a smart stranger. Every element must say something nothing
else on the screen says. Cut template decoration, keep the substance strong, and put depth one click
away.

1. **Titles.** One per unit, a plain noun phrase naming the subject ("Results", "Stocking cost versus purity", "Price and margin"). Nothing around it, so no kicker or label above, no subtitle below, no status tags, numbering or counts. Section titles never carry a claim or an editorial phrase ("How much it sells"). An opening may pose the question the piece answers.
2. **Labels.** Bare subject nouns in sentence case ("Stocking cost", "Fill rate"). Units go in brackets in the label ("Unit volume (L)") and values stay bare with fixed decimals. Context such as "on the test set" belongs in a caption or drawer, never in the label. Use the precise technical term the audience expects ("Expected co-occurrences", "Lift = 1", "Ground truth"), never an everyday paraphrase or a coined name. A convention set on one screen holds on every screen.
3. **Captions.** Only where a number or chart cannot be read without one, and then a single line saying what it is ("Graphs and numbers are per-cluster medians (training set)"). Never caption the encoding (what a cell, dot or square stands for), never restate the visual, never anticipate a later section. An explanation Kilian asks for in chat belongs on the page, usually in a drawer.
4. **Depth.** Methods, definitions and secondary figures go in a closed drawer behind an icon (a filled accent circle with a white plus, its text kept as accessible name and tooltip). The surface keeps one idea per screen.
5. **Storyline.** Introduce, then reveal. Open with context and the question, give each step its own screen, end on the results. Never show a result early (not in the opening, on overview cards or in an earlier widget), never repeat a figure across screens, never end with a recap.
6. **Truth.** Show the real quantity, not a derived one (rank, index) unless the label says so. A name may claim only what most of its members share, otherwise it names the distinguishing traits, and claimed names get a test. A group's name reads as one group, not as a list of members ("ELE-0005 +5"). State how every choice was made and on which data before the first number, never tune on what is reported, show the whole trade-off when goals conflict, and warn where interaction invites cherry-picking. Keep model quality and business value apart. Say what the model is good at, what it is worth, and what about the setting flatters it.
7. **Vocabulary.** One term per concept throughout. When the story moves to a new concept (linked pairs to communities), the screen shows the step and the new term grows out of the old one.
8. **Self-explaining screens.** Every visual answers "how do I read this and what does it tell me", at least in its drawer. Draw any constraint or reference a reader would otherwise guess wrongly (a shared capacity as a limit line, a baseline as a reference mark). Explain any counter-intuitive result where it appears (a dip, a tie, a reversal), and describe competing strategies in plain words next to their technical rules.
9. **Encoding.** One dense, direct encoding (a heatmap of real values) beats several sparse ones. Remove ink that carries no data (outer borders, column gaps, per-row lines, units repeated in cells). Colour only for meaning, and strong when used (no fill within a stated threshold, deepening beyond it until white text is needed). Show state by styling, not tags (unbuilt items greyed out).
10. **Layout and controls.** Size to content, not to the page (compact tables, equal columns, grids balanced to the item count, so nine items as three by three). Icons over words where the meaning is obvious, controls labelled with what they select, one-click presets for the interesting states.
11. **Restraint without timidity.** When cutting, keep what remains strong and substantial. An emptied screen gets real content back, never decoration. Keep the established typeface and look (SF Pro unless told otherwise), and once a choice is approved, keep it.
12. **Working with Kilian.** Change only what he asked. A short or ambiguous instruction means the smallest thing it could mean, or a one-line question. Keep his wording, correcting only wrong terms or facts and saying so. Answer at the length he asks for.
13. **Working alone.** When asked to work autonomously, decide by these rules and his past review comments, not by asking. Before showing work, have it checked by critics that did not write it (these rules, the truth of every number, AI-slop signs) and look at every screen. Disclose on the page any choice informed by the test set or the ground truth. Stop when the work is ready for his review, and list the decisions taken that he may want to revisit.
