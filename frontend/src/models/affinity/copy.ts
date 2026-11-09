// Visible wording of the Bought together page (docs/design/bought-together/STORYLINE.md). Every string the
// page shows lives here. Templates take values already formatted as strings. Inline formulas are
// written \( ... \) and block formulas \[ ... \] inside the text; the page renders them with KaTeX.
// Strings under a `drawer` key appear only inside the "How it works" drawers; lexicon.test.ts checks
// all of them, and keeps reading-list terms out of the rest.

export type RuleKey = 'random' | 'category' | 'popularity' | 'best_sellers' | 'communities' | 'hidden_bundles'

// Rule names as bar, table and control labels.
export const RULES: Record<RuleKey, string> = {
  random: 'Random',
  category: 'Category',
  popularity: 'Best sellers spread',
  best_sellers: 'Best sellers together',
  communities: 'Our communities',
  hidden_bundles: 'Ground-truth bundles',
}

// Rule names inside sentences, lower case.
export const RULES_IN_PROSE: Record<RuleKey, string> = {
  random: 'random',
  category: 'category',
  popularity: 'best sellers spread',
  best_sellers: 'best sellers together',
  communities: 'our communities',
  hidden_bundles: 'ground-truth bundles',
}

// How each rule places SKUs, for the table in the assignment drawer.
export const RULE_HOW: Record<RuleKey, string> = {
  random: 'In random order, each to the FC with the most room.',
  category: 'Whole categories, busiest first, each to the FC with the most room where it fits; a category too large for the room left is placed one SKU at a time.',
  popularity: 'Best sellers first, each to the FC with the most room, so best sellers spread evenly.',
  best_sellers: 'Best sellers first, each to the first FC with room, filling FC 1, then FC 2, and so on.',
  communities: 'Whole communities, busiest first, then unlinked SKUs, best sellers first, each to the FC with the most room.',
  hidden_bundles: 'The same with the ground-truth bundles. Reference only, since a real shop does not know them.',
}

// Each rule in simple terms, for the results drawer.
export const RULE_SIMPLE: Record<RuleKey, string> = {
  random: 'SKUs are shuffled and dealt out. A baseline with no logic at all.',
  category: 'Each category is kept in one FC, a common default. Products bought together across categories end up apart.',
  popularity: 'Best sellers are dealt out evenly, so every FC shares the busy SKUs.',
  best_sellers: 'Best sellers are packed into the first FC, then the second, so many orders could be filled from one FC.',
  communities: 'SKUs bought together on the training set are stored together, one community per FC where it fits.',
  hidden_bundles: 'The same with the bundles the simulator planted. It shows the best this approach could reach.',
}

export const COPY = {
  back: 'All models',
  title: 'Bought together',
  question: 'Can storing SKUs that are bought together in one FC reduce split orders?',
  loading: 'Loading...',
  failed: (message: string) => `The results could not be loaded (${message}).`,
  list: (items: string, last: string) => `${items} and ${last}`,
  pair: (a: string, b: string) => `${a} and ${b}`,
  opening: { skus: 'SKUs', categories: 'Categories', pairs: 'SKU pairs' },

  data: {
    headline: 'Orders from the simulated shop.',
    orders: 'Orders',
    skusPerOrder: 'SKUs per order',
    chartTitle: 'Orders by number of SKUs per order, all five years',
    xTitle: 'SKUs per order',
    yTitle: 'Orders (thousands)',
    pooled: (from: string) => `${from}+`,
    barOne: (orders: string) => `${orders} orders with one SKU`,
    barMany: (orders: string, size: string) => `${orders} orders with ${size} SKUs`,
    barPooled: (orders: string, from: string) => `${orders} orders with ${from} or more SKUs`,
    drawer: {
      once: 'An order counts each SKU once, whatever the quantity.',
      multi: (share: string) => `${share} of orders hold two or more SKUs, the dark bars. Only these can be split.`,
      sets: 'Pair statistics use the training set (years 1 to 3). Settings are chosen on the validation set (year 4). The results come from the test set (year 5).',
    },
  },

  counts: {
    headline: 'Orders per SKU pair.',
    matrixTitle: 'Orders with both SKUs for every pair of SKUs, best sellers first',
    legend: 'Orders with both',
    loading: 'Loading...',
    failed: 'The matrix could not be loaded.',
    drawer: {
      formula: String.raw`Orders with both SKUs \(i\) and \(j\) are counted on the training set, \(n_{ij} = \#\{\, o \mid i \in o \text{ and } j \in o \,\}\) over all orders \(o\).`,
      seen: (seen: string, possible: string) => `${seen} of the ${possible} pairs are bought together at least once.`,
      chance: 'Two best sellers meet in many orders by chance alone. Under independence, the expected co-occurrences of two SKUs are about the product of their order counts divided by the number of orders, so the most frequent pairs are mostly pairs of best sellers.',
      topTitle: 'The 20 most frequent pairs',
      colPair: 'Pair',
      colBoth: 'Orders with both',
      colEach: 'Orders of each SKU',
    },
  },

  chance: {
    headline: 'Orders per SKU pair versus chance.',
    chartTitle: 'Observed against expected co-occurrences, for every pair bought together at least once',
    x: 'Expected co-occurrences',
    y: 'Observed co-occurrences',
    diagonal: 'Lift = 1',
    linked: 'Linked pairs',
    other: 'Other pairs',
    both: 'Observed co-occurrences',
    lift: 'Lift',
    drawer: {
      strata: 'Chance is computed within orders of the same region (regions sell different SKUs), the same size (2, 3, 4, 5+, since large orders hold many) and the same quarter (91-day block, since growing or seasonal SKUs sell in the same months). Orders with one SKU hold no pair and are left out.',
      expected: String.raw`Expected co-occurrences are summed over these strata, \(E_{ij} = \sum_s n_{i,s}\, n_{j,s} / N_s\), where \(N_s\) is the number of orders in stratum \(s\) and \(n_{i,s}\) the number of them holding SKU \(i\). The quarter was added after the first run of the model.`,
      lift: String.raw`Lift is observed divided by expected co-occurrences, \(\text{lift}_{ij} = n_{ij} / E_{ij}\).`,
      pvalue: String.raw`Each pair gets a Poisson tail p-value, \(p_{ij} = P(X \ge n_{ij})\) with \(X \sim \text{Poisson}(E_{ij})\). Its variance is at least the hypergeometric one, so the test is slightly conservative.`,
      bh: 'Benjamini-Hochberg over all pairs is designed to keep the false discovery rate among the pairs that pass it at the chosen level. A linked pair passes this test and has lift above 1.',
      luck: (big: string, small: string, top: string, orders: string) =>
        `Small counts give extreme lifts by luck. Of the ${big} pairs with lift above 3, ${small} appear in fewer than five orders. The highest lift among pairs that are not linked is ${top}, from ${orders}.`,
      oneOrder: 'one order',
      orders: (n: string) => `${n} orders`,
      level: 'The level is chosen later, by delivery cost on the validation set.',
    },
  },

  network: {
    headline: 'Network of linked pairs.',
    caption: 'Each dot is a SKU and each line a linked pair. Drag to turn, scroll to zoom, click a dot for details.',
    colourBy: 'Colour by',
    community: 'Community',
    category: 'Category',
    turn: 'Turn slowly',
    communities: (n: string) => `${n} communities`,
    unlinked: (n: string) => `${n} unlinked SKUs`,
    tip: (name: string, category: string) => `${name} · ${category}`,
    close: 'Close',
    orders: 'Orders',
    partners: 'Strongest partners',
    lift: 'Lift',
    noPartners: 'This SKU has no linked pair.',
    drawer: {
      louvain: String.raw`The graph of linked pairs is weighted by log lift. Louvain maximises modularity, \[Q = \frac{1}{2m} \sum_{ij} \Big[ w_{ij} - \gamma \, \frac{k_i k_j}{2m} \Big] \, \delta(c_i, c_j),\] with the resolution \(\gamma\). Here \(w_{ij}\) is the log lift of a linked pair, \(k_i\) the sum of the weights at SKU \(i\), \(m\) the sum of all weights and \(c_i\) the community of SKU \(i\).`,
      resolutions: (list: string) => `The resolutions tried are ${list}.`,
      invariant: 'Every resolution gives the same communities, because each connected set of SKUs is already tightly knit.',
      ties: 'Ties go to the first resolution tried.',
      sizes: (lo: string, hi: string) => `Communities hold ${lo} to ${hi} SKUs.`,
      cross: (total: string, n: string) => `Of the ${total} linked pairs, ${n} join two categories.`,
      matrix: 'Lift for every pair, with SKUs ordered by community and unlinked SKUs last. The communities show as blocks on the diagonal.',
      matrixTitle: 'Lift for every pair of SKUs, ordered by community',
      legend: 'Lift',
    },
  },

  truth: {
    headline: 'Communities versus ground truth.',
    caption: 'Agreement with the bundles the simulator plants (ARI, 1 is a perfect match).',
    categories: 'Categories',
    counts: 'Communities from raw counts',
    ours: 'Our communities',
    drawer: {
      plants: (bundles: string, skus: string) => `The simulator plants ${bundles} bundles covering ${skus} SKUs. They are the ground truth.`,
      measures: (pairs: string) => `Precision is the share of pairs put together that lie in one bundle. Recall is the share of the ${pairs} pairs inside a bundle that are put together.`,
      colGrouping: 'Grouping',
      colPrecision: 'Precision',
      colRecall: 'Recall',
      colTogether: 'Pairs put together',
      counts: (resolution: string, exact: string, bundles: string, categories: string) =>
        `Communities from raw counts use the same number of pairs, taken as the most frequent ones. Their Louvain resolution is chosen on the validation set like ours (${resolution}). They reproduce ${exact} of the ${bundles} bundles exactly, and categories reproduce ${categories}.`,
      none: 'none',
      top: (countShare: string, liftShare: string) =>
        `Of the 200 most frequent pairs, ${countShare} lie inside a bundle. Of the 200 with the highest lift among pairs in at least five orders, ${liftShare}.`,
      cutSame: 'The five-order cut was set in the first version of the model, and the share is the same for any cut from 3 to 10 orders.',
      cutRange: (lo: string, hi: string) => `The five-order cut was set in the first version of the model. For cuts from 3 to 10 orders the share lies between ${lo} and ${hi}.`,
      quarter: (without: string, withQ: string, ari: string) =>
        `The first run computed chance within region and order size only. This check then showed linked pairs outside any bundle, between SKUs launched or growing at the same time or peaking in the same season, so the quarter was added. On validation cost alone, the version without the quarter would have been chosen (${without} against ${withQ}), and its agreement was ${ari}. The ground-truth bundles therefore helped choose the chance baseline behind our communities.`,
      quarterCheaper: (withQ: string, without: string, ari: string) =>
        `The first run computed chance within region and order size only. This check then showed linked pairs outside any bundle, between SKUs launched or growing at the same time or peaking in the same season, so the quarter was added. The version with the quarter also costs less on the validation set (${withQ} against ${without}). Without the quarter, agreement was ${ari}.`,
    },
  },

  assign: {
    headline: 'Assignment of SKUs to FCs.',
    caption: 'Each square is a SKU. Pale squares belong to a community split across FCs.',
    columnsTitle: 'SKUs in each FC, grouped by community',
    rule: 'Rule',
    fcs: 'FCs',
    fc: (n: string) => `FC ${n}`,
    sku: 'SKU',
    community: 'Community',
    unlinked: 'Unlinked SKU',
    square: (name: string, community: string, fc: string) => `${name}, ${community}, ${fc}`,
    drawer: {
      reading: 'Each rule decides which FC stores each SKU. A community whose squares all sit in one FC is kept whole, so an order of its SKUs ships as one parcel. Pale squares mark communities split across FCs, whose orders can need more than one parcel.',
      split: 'An order with SKUs in two FCs ships as two parcels and counts as a split order.',
      site: (region: string, slack: string) =>
        `All FCs sit at the central site (region ${region}). Each SKU is stored in exactly one FC. Each FC may hold an equal share of the SKUs and of the workload (order lines in the training set), plus ${slack}.`,
      colRule: 'Rule',
      colHow: 'How SKUs are placed',
      within: 'Every rule keeps within both limits.',
      except: (rules: string) => `Every rule keeps within both limits except ${rules}.`,
      overBoth: (skus: string, load: string) =>
        `Filling FCs in order leaves its last SKUs with no room anywhere, and each then goes where it overflows least, so it ends slightly over both limits, by up to ${skus} SKUs and ${load} of the workload limit.`,
      overSkus: (skus: string) =>
        `Filling FCs in order leaves its last SKUs with no room anywhere, and each then goes where it overflows least, so it ends slightly over the SKU limit, by up to ${skus} SKUs.`,
      overLoad: (load: string) =>
        `Filling FCs in order leaves its last SKUs with no room anywhere, and each then goes where it overflows least, so it ends slightly over the workload limit, by up to ${load}.`,
      workload: (split: string, fcs: string, load: string, ours: string) =>
        `The workload limit and the best sellers together rule were added after an earlier run had compared the rules on the test set. Without the limit, best sellers together splits far fewer orders on the validation set by putting most order lines in FC 1 (${split} split orders at ${fcs} FCs with ${load} of order lines in FC 1, against ${ours} for our communities). A real FC cannot take that share of the work, so the limit applies to every rule.`,
      fee: (parcel: string, kg: string) => `Delivery cost per parcel is ${parcel} plus ${kg} per kg, times the distance multiplier from the central site.`,
      choose: (lo: string, hi: string) => `Settings are chosen by the mean delivery cost of our communities over ${lo} to ${hi} FCs on the validation set (year 4).`,
      colLevel: 'False discovery rate',
      colPairs: 'Linked pairs',
      colCost: 'Mean delivery cost ($)',
      chosen: (level: string) => `The chosen level is ${level}.`,
      ties: (levels: string, pairs: string) => `The levels ${levels} give the same ${pairs} linked pairs and tie on cost, and ties go to the first level tried.`,
      gapUnder: (limit: string) => `The largest cost gap between levels is under ${limit}.`,
      gap: (gap: string) => `The largest cost gap between levels is ${gap}.`,
      twoChoices: 'The quarter, described with the ground truth, and the workload limit above are the two choices not made on the validation set alone.',
    },
  },

  results: {
    headline: 'Results',
    split: 'Split orders',
    parcels: 'Parcels',
    cost: 'Delivery cost',
    fcs: 'FCs',
    hiddenNote: 'not known in a real shop',
    barsTitle: 'Split orders on the test set by rule',
    drawer: {
      relative: 'Changes are relative to the category rule.',
      colIdea: 'In simple terms',
      dip: 'Nearly all of the saving comes from orders made only of SKUs from one community, which our communities never split. The category rule splits more of these as FCs are added, so the saving grows up to 4 FCs. From there it already splits almost all of them, so the saving stops growing while split orders keep rising, and the percentage dips slightly at 5 FCs.',
      why: 'Of the plain rules it split the fewest orders on the validation set, apart from best sellers together without the workload limit.',
      test: (orders: string, multi: string) => `The test set holds ${orders} orders, of which ${multi} hold two or more SKUs.`,
      runs: 'The test set was priced in more than one run while the method was being fixed. The quarter and the workload limit were added between runs, as the drawers above describe.',
      colRule: 'Rule',
      colFcs: 'FCs',
      colSplit: 'Split orders (%)',
      colMulti: 'Multi-SKU orders split (%)',
      colParcels: 'Parcels',
      colCost: 'Delivery cost ($)',
      colLoad: 'Workload share per FC, test set (%)',
      limit: 'The workload limit is set on the training set, and growth shifts workload by year 5, so shares on the test set can exceed it.',
      range: (fcsLo: string, fcsHi: string, lo: string, hi: string) =>
        `At ${fcsLo} to ${fcsHi} FCs, our communities split ${lo} to ${hi} fewer orders than the category rule.`,
      rangeOne: (fcsLo: string, fcsHi: string, v: string) => `At ${fcsLo} to ${fcsHi} FCs, our communities split ${v} fewer orders than the category rule.`,
      chartTitle: 'Split orders (%) on the test set by number of FCs, one line per rule',
      chartX: 'FCs',
      chartY: 'Split orders (%)',
      cost: 'Delivery cost moves little because most of it is the per-kg charge, which stays the same when an order ships as two parcels.',
      same: 'Our communities and the ground-truth bundles give the same result at every number of FCs.',
    },
  },
}

// Every visible string, split into surface and drawer text, for the wording test. Templates are
// filled with a placeholder.
function collect(x: unknown, inDrawer: boolean, out: { surface: string[]; drawer: string[] }) {
  if (typeof x === 'string') (inDrawer ? out.drawer : out.surface).push(x)
  else if (typeof x === 'function') collect(x(...Array(x.length).fill('1')), inDrawer, out)
  else if (x && typeof x === 'object') for (const [k, v] of Object.entries(x)) collect(v, inDrawer || k === 'drawer', out)
}

export const WORDING = (() => {
  const out = { surface: [] as string[], drawer: [] as string[] }
  collect(COPY, false, out)
  collect(RULES, false, out)
  collect({ drawer: [RULES_IN_PROSE, RULE_HOW, RULE_SIMPLE] }, false, out)
  return out
})()
