// Visible wording of the Demand five years out page (design/demand-forecast/STORYLINE.md). Strings under
// a `drawer` key appear only inside the drawers; lexicon.test.ts checks all of them.

export type Method = 'Seasonal naive' | 'ETS' | 'LightGBM' | 'Neural network'
export const METHODS: Method[] = ['Seasonal naive', 'ETS', 'LightGBM', 'Neural network']

export const LEVELS = ['Total', 'Region', 'Category', 'Region × category', 'SKU', 'Region × SKU'] as const
export type Level = (typeof LEVELS)[number]
export const UPPER: Level[] = ['Total', 'Region', 'Category', 'Region × category']

// Each method in plain words, for the table on screen 3.
export const METHOD_RULE: Record<Method, string> = {
  'Seasonal naive': 'Every future year repeats the last observed year.',
  ETS: 'Follows each series\' level, trend and season, and damps the trend over time.',
  LightGBM: 'Gradient-boosted trees learn from every series at once and forecast a year from the year before, applied five times.',
  'Neural network': 'A small network learns from every series at once and forecasts a year from the year before, applied five times.',
}

// Capacity plans: output key, label, rule in plain words.
export const PLANS: { key: string; label: string; rule: string }[] = [
  { key: "Last year's peak", label: 'Last year\'s peak', rule: 'Build each region\'s busiest month of year 5 again.' },
  { key: 'Seasonal naive', label: 'Seasonal naive', rule: 'Size capacity on seasonal naive\'s sample paths.' },
  { key: 'ETS', label: 'ETS', rule: 'Size capacity on the ETS sample paths.' },
  { key: 'LightGBM', label: 'LightGBM', rule: 'Size capacity on the LightGBM sample paths.' },
  { key: 'Our forecast', label: 'Neural network', rule: 'Size capacity on the chosen neural network\'s sample paths.' },
  { key: 'Our P50 only', label: 'Neural network P50', rule: 'Build the busiest month of the neural network\'s P50, ignoring the band and the cost ratio.' },
]

export const COPY = {
  back: 'All models',
  title: 'Demand five years out',
  question: 'How far ahead can five years of sales see?',
  loading: 'Loading...',
  failed: (message: string) => `The results could not be loaded (${message}).`,
  opening: { series: 'Series', levels: 'Levels', years: 'Years of history' },
  simulation: (n: number) => `Simulation ${n}`,
  testYears: 'Test years',
  level: 'Level',
  series: 'Series',
  total: 'Total',
  key: { history: 'History', actual: 'Actual', p50: 'P50', band: 'P10 to P90' },

  sales: {
    headline: 'Monthly sales.',
    caption: 'Units per month of the median-volume series of each level, years 1 to 5.',
    yTitle: 'Units per month',
    chartTitle: (level: string) => `Units per month of the median-volume ${level} series, years 1 to 5`,
    drawer: {
      levels: 'Six levels are forecast. The total, each of the 8 regions, each of the 8 categories, every region and category pair, every SKU, and every SKU in every region.',
      colLevel: 'Level',
      colSeries: 'Series',
      pairs: (validation: string, test: string) => `Region × SKU counts series with a sale in years 1 to 3 (${validation}); the test has ${test}.`,
      units: 'Units are ordered units per month. The shop never runs out of stock, so orders are demand. Bulk orders stay in.',
    },
  },

  split: {
    headline: 'Data split.',
    drawer: {
      validation: 'Validation forecasts years 4 and 5 from the end of year 3, with models fitted on years 1 to 3. Every setting and the method are chosen there. The chosen settings are then refitted on years 1 to 5 and forecast years 6 to 10, the upper levels five years ahead and the SKU levels two.',
      future: (launches: string, lo: string, hi: string) =>
        `Years 6 to 10 are simulated with the same customers and rules, plus three rules fixed before any model was fitted. About ${launches} launches a year, as many delistings of the slowest sellers, and growth that ends at an age of ${lo} to ${hi} years. Simulation 2 draws the same years again with another seed.`,
      changes: 'Changed after test scores had been seen, all disclosed in the handover. The test years were scored in six runs. After the first, a review found faults in the bands and six fixes followed. The bands then switched to out-of-sample errors from rolling refits, and simulation 2 was added after simulation 1 had been scored. Two last corrections made every sample path average the mean forecast and scored point accuracy on the mean forecast. The point forecasts, the chosen method and the simulated world never changed.',
    },
  },

  methods: {
    headline: 'Forecasting methods.',
    caption: 'Median-volume series of each level, forecast from the end of year 3.',
    colMethod: 'Method',
    colRule: 'Rule',
    method: 'Method',
    chartTitle: (method: string, level: string) => `${method}, validation forecast of the median-volume ${level} series against the actual`,
    drawer: {
      read: 'The band holds the middle 80% of 1,000 joint sample paths, the indigo line is their P50, the black line the actual and the grey line the history. One series per level invites picking a winner, so methods are compared on every series on the next screen.',
      ets: 'ETS is Holt-Winters exponential smoothing on log(1 + y), with an additive season and a damped trend kept or dropped per series by AIC.',
      learned: (lgb: string, nn: string) => `LightGBM uses ${lgb}. The neural network uses ${nn}. Both read the last 12 months scaled by their mean, growth over the last year, months since first sale, the share of empty months and of bulk units, the calendar month, and the region's and SKU's fixed traits.`,
      fallback: 'Series too young for a method fall back to seasonal naive. On validation this happens for 194 series under ETS and 86 under the learned models, on the test for 26 and 133.',
      paths: 'Bands come from 1,000 joint sample paths. Each method is refitted at rolling origins inside its fitting window, and its one-year-ahead errors on the months it had not seen are resampled, a level error per year carried forward and noise per calendar month.',
      reconcile: 'Every path is reconciled so the levels add up, bottom-up from region × category or by MinT, chosen per method on validation. SKU paths are scaled to their region and category.',
    },
  },

  validation: {
    headline: 'Validation.',
    caption: 'Pinball loss (%), validation set, lower is better.',
    colMethod: 'Method',
    colMean: 'Mean',
    drawer: {
      loss: 'Pinball loss is averaged over P10, P50 and P90, summed over series and months, and divided by the actual units. At the P50 it equals WAPE.',
      colour: 'Each cell is coloured by how far it lies above the best method of its level. Cells within 10% of the best are not coloured.',
      chosen: (method: string) => `The method with the lowest mean is chosen, here the ${method.toLowerCase()}.`,
      horizon: 'Validation looks only two years ahead. Years 3 to 5 ahead are first seen on the test.',
      colSetting: 'Settings chosen',
      colWidth: 'Width factor (upper, SKU levels)',
      width: 'The width factor scales the resampled errors. It is chosen on validation, one for the upper levels and one for the SKU levels.',
      colRec: 'Reconciliation',
      none: 'none',
    },
  },

  forecast: {
    headline: 'Five-year forecast.',
    yTitle: 'Units per month',
    chartTitle: (series: string) => `${series}, five-year forecast from the end of year 5 against the actual`,
    drawer: {
      read: 'The band holds the middle 80% of 1,000 joint sample paths and the line is their P50. Errors on the next screen are scored on the mean forecast, which sits slightly below the line.',
      growth: 'The forecast carries the growth of years 1 to 5 forward. In the simulated years growth slows, and in simulation 1 it stops.',
    },
  },

  wape: {
    headline: 'WAPE by forecast year.',
    xTitle: 'Forecast year',
    yTitle: 'WAPE (%)',
    truth: 'Ground-truth mean',
    chartTitle: (level: string) => `WAPE (%) of the mean forecast by forecast year, ${level}, one line per method`,
    drawer: {
      lead: (first: string, second: string) =>
        `At the total the learned models lead for ${first} in simulation 1 and for ${second} in simulation 2. Beyond that ETS, which damps growth, is closer, because the end of growth is not in years 1 to 5.`,
      unseen: 'Validation saw two years ahead, so it could not show this, and the method was not switched after the test.',
      overall: (scores: string, best: string) => `Over all levels and years of simulation 1 the pinball loss still favours the learned models (${scores}), and ${best} edges the chosen network.`,
      overallSame: (scores: string) => `Over all levels and years of simulation 1 the pinball loss still favours the learned models (${scores}).`,
      truthLine: 'The ground-truth mean is the average of the 200 ground-truth replications. No forecast can beat it by more than luck.',
      skuTitle: 'SKU and region × SKU levels, WAPE (%) of the mean forecast, simulation 1',
      colMethod: 'Method',
      colYear: (n: number) => `Year ${n}`,
    },
  },

  truth: {
    headline: 'Forecast versus ground truth.',
    ours: 'Neural network band',
    true: 'Ground-truth band',
    chartTitle: (series: string) => `${series}, the neural network's P10 to P90 band against the ground truth's, years 6 to 10`,
    drawer: {
      reps: 'The 200 ground-truth replications rerun years 6 to 10 with the simulation\'s launches, delistings and growth-end ages held fixed. Their band shows only the randomness left once those are known, so it is narrower than what a planner faces, and the two simulations\' bands for year 10 do not even overlap.',
      colYear: 'Forecast year',
      tableTitle: (level: string) => `Ground-truth replications against the neural network band, ${level}`,
      colInside: (sim: number) => `Inside the band, simulation ${sim} (%)`,
      colWidth: (sim: number) => `Ground-truth band width over neural network band, simulation ${sim}`,
      regions: 'all regions',
      total: 'total',
      check: 'The replications are used only to check, never to choose.',
    },
  },

  results: {
    headline: 'Results',
    caption: 'Capacity cost for years 7 to 10 over the ground-truth plan (%), at the game\'s cost ratio.',
    drawer: {
      decision: 'At the end of year 5 each region builds monthly capacity for years 7 to 10. Demand above capacity loses its margin, and capacity costs money whether it is used or not. Bars show each plan\'s cost over the plan built from the ground truth, at the game\'s ratio of capacity cost to margin, on 100 held-out replications.',
      colPlan: 'Plan',
      colRule: 'Rule',
      ranking: (s1: string, s2: string) => `The ranking depends on the test years. In simulation 1, where growth stops, ${s1} is cheapest. In simulation 2, where growth only slows, ${s2} is cheapest.`,
      ours: 'The neural network runs high in years 8 to 10 of simulation 1, and its band adds spare capacity. Neural network P50 costs most because it ignores the cost ratio.',
      flatters: 'What flatters every forecast here is a stable simulator, with about 1.5% noise at the total, and what hurts the learned ones is a growth rule fixed before fitting that years 1 to 5 do not show.',
      ratioTitle: 'Cost over the ground-truth plan (%) across the ratio of capacity cost to margin, simulation 1',
      colRatio: 'Ratio',
      game: 'game',
      regionTitle: 'Capacity (units per month) by region at the game\'s ratio, simulation 1',
      colRegion: 'Region',
    },
  },
}

// Every visible string, for the wording test.
function collect(x: unknown, inDrawer: boolean, out: { surface: string[]; drawer: string[] }) {
  if (typeof x === 'string') (inDrawer ? out.drawer : out.surface).push(x)
  else if (typeof x === 'function') collect(x(...Array(x.length).fill('1')), inDrawer, out)
  else if (x && typeof x === 'object') for (const [k, v] of Object.entries(x)) collect(v, inDrawer || k === 'drawer', out)
}

export const WORDING = (() => {
  const out = { surface: [] as string[], drawer: [] as string[] }
  collect(COPY, false, out)
  collect({ METHOD_RULE, levels: [...LEVELS] }, false, out)
  collect({ drawer: PLANS.map((p) => p.rule) }, false, out)
  collect(PLANS.map((p) => p.label), false, out)
  return out
})()
