// Visible wording of the SKU clustering page. Short: a label, a headline, a caption per screen.

export const COPY = {
  title: 'Clustering SKUs by how they sell',
  intro: '300 SKUs, five years of orders. One stocking policy per cluster instead of one per SKU.',
  screens: {
    question: {
      label: 'The question',
      headline: 'SKUs sell in very different ways.',
      caption: 'Weekly sales of six typical SKUs over the training set. Each was generated with a hidden type: the ground truth the model never sees.',
    },
    data: {
      label: 'The data',
      headline: 'Five years of orders from our simulated shop.',
      caption: 'The model learns on the training set (years 1 to 4). The test set (year 5) is kept back to price the stocking policies.',
    },
    fingerprint: {
      label: 'Feature engineering',
      headline: 'Each SKU becomes a feature vector of 16 features.',
      caption: 'Pick a SKU to see where it ranks among all 300 on each feature.',
    },
    map: {
      label: 'The embedding',
      headline: 'SKUs that sell alike end up close together.',
      caption: 'Each dot is a SKU, placed by dimensionality reduction of its 16 features. Drag to turn, scroll to zoom, click a dot for details.',
    },
    groups: {
      label: 'The clusters',
      headline: 'Every cluster gets its own stocking policy.',
      caption: 'Median values per cluster. The stocking policy is the cheapest one on the training set.',
    },
    truth: {
      label: 'Ground truth',
      headline: 'The clusters match the ground truth better than ABC/XYZ analysis.',
      caption: 'Purity: share of SKUs sitting in a cluster dominated by their own true type. ARI: agreement corrected for chance (1 is perfect, 0 is chance).',
    },
    worth: {
      label: 'What it is worth',
      headline: 'Cluster policies cut stocking cost by a tenth.',
      caption: 'Policies chosen on year 4 of the training set, then priced on the test set, which no step had seen.',
    },
    learned: {
      label: 'What we learned',
      headline: 'Clustering pays, and feature choice decides what a cluster means.',
    },
  },
  learned: [
    'Cluster policies cost about a tenth less than one stocking policy for every SKU.',
    'Features without size and price recover the ground truth best; size and price give the cheapest policies.',
    'A new SKU gets a policy from its cluster on day one. One policy per SKU needs years of history and risks overfitting.',
  ],
  howItWorks: 'How it works',
  get surface(): string[] {
    const s = Object.values(this.screens).flatMap((x) => Object.values(x as Record<string, string>))
    return [this.title, this.intro, ...s, ...this.learned, this.howItWorks]
  },
}
