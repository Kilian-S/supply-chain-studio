"""Measure calibration targets for the world generator from UCI Online Retail II.

Source: Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/502/online+retail+ii (CC BY 4.0).

Usage: uv run python scripts/calibrate_online_retail.py <path to online_retail_II.xlsx> <output json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def quantiles(s: pd.Series) -> dict[str, float]:
    q = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    return {"mean": float(s.mean()), **{f"p{int(k * 100)}": float(v) for k, v in q.items()}}


def main(xlsx: str, out: str) -> None:
    sheets = pd.read_excel(xlsx, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = [c.strip().replace(" ", "") for c in df.columns]
    raw_rows = len(df)
    df["Invoice"] = df["Invoice"].astype(str)
    sales = df[~df["Invoice"].str.startswith("C") & (df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    sales["InvoiceDate"] = pd.to_datetime(sales["InvoiceDate"])
    sales["Revenue"] = sales["Quantity"] * sales["Price"]
    sales = sales.drop_duplicates()

    # Baskets: distinct products per invoice.
    lines = sales.groupby("Invoice")["StockCode"].nunique()

    # Customers: orders per customer per year and gaps between orders.
    known = sales.dropna(subset=["CustomerID"]) if "CustomerID" in sales else sales.dropna(subset=["Customer ID"])
    cust_col = "CustomerID" if "CustomerID" in known else "Customer ID"
    inv = known.groupby([cust_col, "Invoice"])["InvoiceDate"].min().reset_index()
    span_years = (sales["InvoiceDate"].max() - sales["InvoiceDate"].min()).days / 365.25
    orders_per_customer = inv.groupby(cust_col).size()
    gaps = inv.sort_values("InvoiceDate").groupby(cust_col)["InvoiceDate"].diff().dt.days.dropna()

    # SKU demand classes on weekly demand (Syntetos-Boylan thresholds), SKUs sold in >= 10 weeks.
    sales["Week"] = sales["InvoiceDate"].dt.to_period("W")
    weeks = pd.period_range(sales["Week"].min(), sales["Week"].max(), freq="W")
    weekly = sales.groupby(["StockCode", "Week"])["Quantity"].sum().unstack(fill_value=0).reindex(columns=weeks, fill_value=0)
    active = weekly[(weekly > 0).sum(axis=1) >= 10]
    nz = (active > 0).sum(axis=1)
    adi = len(weeks) / nz
    cv2 = active.apply(lambda r: (r[r > 0].std(ddof=1) / r[r > 0].mean()) ** 2 if (r > 0).sum() > 1 else 0, axis=1)
    cls = np.where(adi < 1.32, np.where(cv2 < 0.49, "smooth", "erratic"), np.where(cv2 < 0.49, "intermittent", "lumpy"))
    class_share = pd.Series(cls).value_counts(normalize=True).round(4).to_dict()

    # Revenue concentration (Pareto) and monthly seasonality.
    rev = sales.groupby("StockCode")["Revenue"].sum().sort_values(ascending=False)
    top20 = rev.iloc[: max(1, int(len(rev) * 0.2))].sum() / rev.sum()
    monthly = sales.groupby(sales["InvoiceDate"].dt.month)["Revenue"].sum()
    monthly_index = (monthly / monthly.mean()).round(3).to_dict()

    result = {
        "source": {
            "name": "UCI Online Retail II",
            "url": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
            "licence": "CC BY 4.0",
            "citation": "Chen, D. (2019). Online Retail II. UCI Machine Learning Repository.",
            "period": f"{sales['InvoiceDate'].min().date()} to {sales['InvoiceDate'].max().date()}",
            "rows_raw": raw_rows,
            "rows_sales_clean": int(len(sales)),
            "note": "UK online gift retailer; many customers are wholesalers, so quantities are large. "
                    "We use shapes and ratios, not absolute quantities.",
        },
        "basket_distinct_products": quantiles(lines),
        "share_single_item_baskets": float((lines == 1).mean()),
        "orders_per_customer_per_year": quantiles(orders_per_customer / span_years),
        "days_between_orders": quantiles(gaps),
        "sku_demand_class_share": class_share,
        "skus_classified": int(len(active)),
        "revenue_share_top20pct_skus": float(top20),
        "monthly_revenue_index": {int(k): float(v) for k, v in monthly_index.items()},
        "quantity_per_line": quantiles(sales["Quantity"]),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
