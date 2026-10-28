# Calibration of the world generator

The world (`backend/src/scgame/world/generator.py`) is synthetic so that the true drivers are
known. Where real data helps, we calibrate shapes and ratios to it. Every external number used is
listed here with its source.

## Sources

| Source | Licence | File |
|---|---|---|
| Chen, D. (2019). *Online Retail II*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/502/online+retail+ii | CC BY 4.0 | Raw data in `reference/external/online_retail_ii/` (not in git); measurements in `online_retail_ii.json` (from `backend/scripts/calibrate_online_retail.py`) |

## What we adopt and what we do not

| Measure (UCI Online Retail II, Dec 2009 to Dec 2011) | Value | Used? | World v1 (seed 7) |
|---|---|---|---|
| Revenue share of the top 20% of SKUs | 79.4% | Yes, as a target | 75.5% |
| Monthly revenue index (Jan..Dec, peak November 1.74) | see JSON | Yes, damped by half, for overall retail seasonality | built in |
| Orders per customer per year: mean, median | 3.1, 1.5 | Yes, as a target | 3.0, 1.8 |
| Distinct products per order: median | 15 | **No**: mostly wholesale buyers; a general consumer retailer has small baskets | mean 2.25, 44% single-line |
| SKU demand classes (lumpy 71%, erratic 18%, intermittent 9%, smooth 2%) | see JSON | **No**: wholesale-driven; we use our own type mix | types: intermittent 130, steady 57, seasonal 36, declining 32, lumpy 26, trending 19 |
| Quantity per line: median 4, p99 108 | see JSON | **No**: wholesale quantities; consumer lines are 1 or a few units, bulk buyers are modelled separately | |

## Assumptions without an external source (to revisit)

- Region populations, income and urbanisation: plausible values, not yet taken from Eurostat.
- Category price, margin, size and density medians: plausible values for a general online retailer.
- Bundle structure (45 hidden bundles of 3 to 6 products, 45% follow probability).
