// Visible wording of the SKU clustering page. Short: a label, a headline, a caption per screen.

export const COPY = {
  title: 'Clustering SKUs by how they sell',
  intro: 'Can SKUs that sell alike share one stocking policy?',
  screens: {
    question: {
      headline: 'SKUs sell in very different ways.',
      caption: 'Weekly sales of six typical SKUs over the training set.',
    },
    data: {
      label: 'The data',
      headline: 'Five years of orders from our simulated shop.',
    },
    fingerprint: {
      label: 'Feature engineering',
      headline: 'Each SKU becomes a feature vector of 16 features.',
    },
    map: {
      headline: 'Visualisation of SKUs in feature space.',
      caption: 'Each dot is a SKU, placed by dimensionality reduction of its 16 features. Drag to turn, scroll to zoom, click a dot for details.',
    },
    groups: {
      headline: 'The clusters and their stocking policies.',
      caption: 'Graphs and numbers are per-cluster medians (training set).',
    },
    truth: {
      headline: 'Stocking cost versus purity.',
      caption: 'Each dot is one setting of features, method and k.',
    },
    worth: {
      headline: 'Results',
    },
  },
  selectionWarning: 'Choosing features or k to raise this score is selection bias. It makes the score look better than it is.',
  howItWorks: 'How it works',
  get surface(): string[] {
    const s = Object.values(this.screens).flatMap((x) => Object.values(x as Record<string, string>))
    return [this.title, this.intro, ...s, this.selectionWarning, this.howItWorks]
  },
}
