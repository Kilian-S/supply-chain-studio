// Visible wording of the product groups page. Short: a label, a headline, a caption per screen.

export const COPY = {
  title: 'Grouping products by how they sell',
  intro: 'Three hundred products, five years of orders. One stocking rule per group instead of one per product.',
  screens: {
    question: {
      label: 'The question',
      headline: 'Products sell in very different ways.',
      caption: 'Weekly sales of six typical products over four years. Each was built with a hidden type the model never sees.',
    },
    data: {
      label: 'The data',
      headline: 'Five years of orders from our simulated shop.',
      caption: 'The model learns from years 1 to 4. Year 5 is kept back to test the stocking rules.',
    },
    fingerprint: {
      label: 'What we measure',
      headline: 'Each product becomes a fingerprint of 16 numbers.',
      caption: 'Pick a product to see where it ranks among all 300 on each number.',
    },
    map: {
      label: 'The map',
      headline: 'Products that sell alike end up close together.',
      caption: 'Each dot is a product. Drag to turn, scroll to zoom, click a dot for details.',
    },
    groups: {
      label: 'The groups',
      headline: 'Every group gets its own stocking rule.',
      caption: 'Typical values per group. The stocking rule is the cheapest one on year 4.',
    },
    truth: {
      label: 'Did it find the truth?',
      headline: 'The groups match the hidden types better than the textbook rule.',
      caption: 'Share of products sitting in a group dominated by their own true type.',
    },
    worth: {
      label: 'What it is worth',
      headline: 'Group rules cut stocking cost by a tenth.',
      caption: 'Rules chosen on year 4, then tested on year 5, which no step had seen.',
    },
    learned: {
      label: 'What we learned',
      headline: 'Grouping pays, and the fingerprint decides what a group means.',
    },
  },
  learned: [
    'Grouping saves money: about a tenth less than one rule for every product.',
    'Fingerprints without size and price find the hidden types best; size and price make the cheapest rules.',
    'A new product gets a rule on day one from its group. One rule per product needs years of history.',
  ],
  howItWorks: 'How it works',
  get surface(): string[] {
    const s = Object.values(this.screens).flatMap((x) => Object.values(x as Record<string, string>))
    return [this.title, this.intro, ...s, ...this.learned, this.howItWorks]
  },
}
