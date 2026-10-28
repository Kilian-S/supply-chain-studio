"""Feature engineering for SKU segmentation (model J).

One row per SKU, computed from order lines in the training window (years 1-4). The final year is
held back for the simulator check of the resulting policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

TRAIN_END = 4 * 365  # days 1..1460 train, 1461..1825 test


@dataclass(frozen=True)
class Feature:
    key: str
    group: str
    label: str
    formula: str
    description: str
    log: bool  # log-transform before scaling


FEATURES = [
    Feature("mean_weekly_units", "volume", "Mean weekly units", r"\bar{d} = \frac{1}{W}\sum_{w} d_w",
            "Average units per week over the weeks the SKU was on sale.", True),
    Feature("revenue", "volume", "Revenue", r"R = \sum_{\ell} q_\ell \, p",
            "Total revenue in the training window (drives the ABC class).", True),
    Feature("cv2", "variability", "CV squared", r"CV^2 = \left(\sigma_{d_w>0} / \mu_{d_w>0}\right)^2",
            "Squared coefficient of variation of non-zero weekly demand (Syntetos-Boylan).", True),
    Feature("adi", "intermittency", "ADI", r"ADI = W / \#\{w : d_w > 0\}",
            "Average demand interval: weeks on sale per week with any demand.", True),
    Feature("zero_share", "intermittency", "Zero-demand weeks", r"z = \#\{w : d_w = 0\} / W",
            "Share of weeks on sale without a single unit sold.", False),
    Feature("seasonality", "pattern", "Seasonal strength", r"F_S = \max\left(0, 1 - \frac{\mathrm{Var}(R_t)}{\mathrm{Var}(S_t + R_t)}\right)",
            "Strength of the yearly cycle from an STL decomposition of weekly demand (Hyndman's F_S).", False),
    Feature("trend", "pattern", "Trend", r"\beta \text{ in } \log(1 + d_m) = \alpha + \beta \, t_{\text{years}}",
            "Log-linear growth per year of monthly demand: positive grows, negative declines.", False),
    Feature("recent_share", "pattern", "Recent share", r"\sum_{\text{last 52 w}} d_w / \sum_w d_w",
            "Share of all units sold in the most recent year: life-cycle position.", False),
    Feature("volume_l", "physical", "Unit volume", r"v \;[\text{litres}]", "Volume of one unit.", True),
    Feature("weight_kg", "physical", "Unit weight", r"m \;[\text{kg}]", "Weight of one unit.", True),
    Feature("margin", "economic", "Gross margin", r"\mu = (p - c)/p", "Gross margin as a share of price.", False),
    Feature("price", "economic", "Price", r"p", "Selling price per unit.", True),
    Feature("holding_to_margin", "economic", "Holding vs margin", r"h / (p\mu), \; h = 0.25c + 2v",
            "Yearly cost of holding one unit relative to the margin it earns.", True),
    Feature("solo_share", "basket", "Bought alone", r"\#\{\ell : |o(\ell)| = 1\} / \#\{\ell\}",
            "Share of the SKU's order lines that are single-item orders.", False),
    Feature("basket_size", "basket", "Basket size", r"\mathbb{E}[\,|o| \mid \text{SKU} \in o\,]",
            "Average number of products in orders containing the SKU.", False),
    Feature("qty_per_line", "basket", "Units per line", r"\mathbb{E}[q_\ell]",
            "Average units per order line: high for bulk buyers.", True),
]
GROUPS = ["volume", "variability", "intermittency", "pattern", "physical", "economic", "basket"]
GROUP_LABELS = {"volume": "Volume", "variability": "Variability", "intermittency": "Intermittency",
                "pattern": "Pattern", "physical": "Physical", "economic": "Economic", "basket": "Basket role"}


def weekly_matrix(lines: pd.DataFrame, skus: pd.DataFrame, end_day: int) -> tuple[np.ndarray, np.ndarray]:
    """Units per SKU per week (rows follow `skus`), and a mask of weeks the SKU was on sale."""
    w = lines[lines["day"] <= end_day].assign(week=lambda d: (d["day"] - 1) // 7)
    n_weeks = (end_day - 1) // 7 + 1
    idx = {s: i for i, s in enumerate(skus["sku_id"])}
    mat = np.zeros((len(skus), n_weeks))
    agg = w.groupby(["sku_id", "week"])["quantity"].sum()
    for (s, wk), q in agg.items():
        mat[idx[s], wk] = q
    launch_week = ((skus["launch_day"].to_numpy() - 1) // 7)
    end_week = ((np.minimum(skus["end_day"].to_numpy(), end_day + 1) - 1) // 7)
    weeks = np.arange(n_weeks)
    on_sale = (weeks[None, :] >= launch_week[:, None]) & (weeks[None, :] < end_week[:, None])
    return mat, on_sale


def _seasonal_strength(series: np.ndarray) -> float:
    if len(series) < 2 * 52 + 2 or series.sum() == 0:
        return 0.0
    res = STL(np.log1p(series), period=52, robust=True).fit()
    var_r = np.var(res.resid)
    var_sr = np.var(res.seasonal + res.resid)
    return float(max(0.0, 1 - var_r / var_sr)) if var_sr > 0 else 0.0


def build_features(lines: pd.DataFrame, orders: pd.DataFrame, skus: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    train = lines[lines["day"] <= TRAIN_END]
    weekly, on_sale = weekly_matrix(lines, skus, TRAIN_END)
    rows = []
    order_size = train.groupby("order_id").size()
    t = train.assign(order_size=train["order_id"].map(order_size))
    by_sku = t.groupby("sku_id")
    solo = by_sku.apply(lambda g: (g["order_size"] == 1).mean(), include_groups=False)
    bsize = by_sku["order_size"].mean()
    qline = by_sku["quantity"].mean()
    for i, s in enumerate(skus.itertuples()):
        d = weekly[i][on_sale[i]]
        if len(d) == 0:
            d = np.zeros(1)
        nz = d[d > 0]
        mean_w = d.mean()
        cv2 = (nz.std(ddof=1) / nz.mean()) ** 2 if len(nz) > 1 else 0.0
        adi = len(d) / max(len(nz), 1)
        months = d[: len(d) // 4 * 4].reshape(-1, 4).sum(axis=1) if len(d) >= 8 else d
        tm = np.arange(len(months)) / 13.0  # 4-week months -> years
        trend = float(np.polyfit(tm, np.log1p(months), 1)[0]) if len(months) >= 3 else 0.0
        recent = d[-52:].sum() / d.sum() if d.sum() > 0 else 0.0
        holding = 0.25 * s.unit_cost + 2.0 * s.volume_l
        rows.append(dict(
            sku_id=s.sku_id, mean_weekly_units=mean_w, revenue=float(d.sum() * s.price), cv2=cv2, adi=adi,
            zero_share=float((d == 0).mean()), seasonality=_seasonal_strength(d), trend=trend, recent_share=recent,
            volume_l=s.volume_l, weight_kg=s.weight_kg, margin=s.margin, price=s.price,
            holding_to_margin=holding / max(s.price * s.margin, 0.01),
            solo_share=float(solo.get(s.sku_id, 0.0)), basket_size=float(bsize.get(s.sku_id, 1.0)),
            qty_per_line=float(qline.get(s.sku_id, 1.0)),
        ))
    return pd.DataFrame(rows), weekly


def transform(features: pd.DataFrame, groups: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Log where marked, standardise, then weight each feature by 1/sqrt(features in its group)
    so every group contributes equal variance whatever its size."""
    groups = groups or GROUPS
    chosen = [f for f in FEATURES if f.group in groups]
    cols = []
    for f in chosen:
        x = features[f.key].to_numpy(dtype=float)
        if f.log:
            x = np.log1p(np.maximum(x, 0))
        sd = x.std()
        z = (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)
        size = sum(1 for g in chosen if g.group == f.group)
        cols.append(z / np.sqrt(size))
    return np.column_stack(cols), [f.key for f in chosen]
