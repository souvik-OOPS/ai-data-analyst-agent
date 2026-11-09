"""Generate the reproducible synthetic e-commerce dataset in ``data/raw``.

The store is fictional ("DDL Labs") and launched on 2024-11-01. The generator
simulates daily orders with seasonality, growth, customer acquisition and churn,
then deliberately injects data-quality problems (duplicates, malformed values,
missing fields, orphaned references) so the ingestion pipeline has real work to do.

It also plants one business story the agent should be able to diagnose:
in October 2026 revenue drops because (1) three high-revenue Electronics
products go out of stock, (2) new-customer acquisition falls sharply and
(3) the cancellation rate roughly doubles.

Usage:
    python scripts/generate_data.py            # writes data/raw/*.csv
    python scripts/generate_data.py --out DIR  # custom output directory
"""

from __future__ import annotations

import argparse
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
LAUNCH = date(2024, 11, 1)
END = date(2026, 10, 31)
STOCKOUT_START = date(2026, 10, 4)
STOCKOUT_PRODUCTS = {"Nimbus 27in 4K Monitor", "Vector Noise-Cancelling Headphones", "Spark 10in Tablet"}

# (name, list price, popularity weight)
CATALOG: dict[str, list[tuple[str, float, float]]] = {
    "Electronics": [
        ("Aurora Wireless Earbuds", 79.0, 1.6),
        ("Nimbus 27in 4K Monitor", 329.0, 1.3),
        ("Volt Fast Charger 65W", 39.0, 1.2),
        ("Pulse Smartwatch S2", 199.0, 0.9),
        ("Echo Bluetooth Speaker", 89.0, 1.0),
        ("Orbit Mechanical Keyboard", 119.0, 0.8),
        ("Glide Wireless Mouse", 45.0, 1.1),
        ("Lumen Laptop Stand", 59.0, 0.6),
        ("Vector Noise-Cancelling Headphones", 249.0, 1.4),
        ("Spark 10in Tablet", 299.0, 1.2),
        ("Retro Film Camera", 159.0, 0.0),  # listed but never sold
    ],
    "Home & Kitchen": [
        ("Brew Master Coffee Maker", 129.0, 1.2),
        ("Chef Pro Knife Set", 149.0, 0.7),
        ("Aroma Essential Oil Diffuser", 35.0, 1.0),
        ("Cozy Knit Throw Blanket", 49.0, 1.1),
        ("Crisp Air Fryer 5L", 119.0, 1.4),
        ("Pure Water Filter Pitcher", 32.0, 0.9),
        ("Nest Bamboo Storage Set", 42.0, 0.6),
        ("Glow LED Desk Lamp", 38.0, 0.8),
        ("Steam Garment Steamer", 69.0, 0.5),
        ("Fresh Vacuum Food Sealer", 89.0, 0.4),
    ],
    "Clothing": [
        ("Everyday Cotton Tee", 19.0, 1.8),
        ("Trail Fleece Jacket", 89.0, 0.9),
        ("Classic Slim Jeans", 59.0, 1.2),
        ("Breeze Linen Shirt", 45.0, 0.8),
        ("Stride Running Shorts", 29.0, 1.0),
        ("Comfort Crew Socks 6-Pack", 18.0, 1.3),
        ("Urban Hooded Sweatshirt", 55.0, 1.0),
        ("Summit Rain Shell", 119.0, 0.5),
    ],
    "Beauty": [
        ("Radiance Vitamin C Serum", 34.0, 1.3),
        ("Hydra Daily Moisturizer", 28.0, 1.2),
        ("Silk Hair Repair Mask", 22.0, 0.8),
        ("Bloom Eau de Parfum", 75.0, 0.6),
        ("Clear Skin Cleanser", 16.0, 1.1),
        ("Velvet Matte Lipstick", 18.0, 0.7),
        ("Glow Sunscreen SPF 50", 21.0, 0.9),
    ],
    "Sports & Outdoors": [
        ("Peak Yoga Mat", 39.0, 1.2),
        ("Ridge Hiking Backpack 30L", 99.0, 0.8),
        ("Flex Resistance Bands Set", 25.0, 1.0),
        ("Hydro Steel Water Bottle", 29.0, 1.4),
        ("Apex Adjustable Dumbbells", 249.0, 0.5),
        ("Trailblazer Camping Tent", 189.0, 0.4),
        ("Swift Jump Rope", 15.0, 0.7),
    ],
    "Books": [
        ("The Data-Driven Retailer", 32.0, 0.9),
        ("Modern Python Recipes", 45.0, 0.8),
        ("Mindful Leadership", 24.0, 0.7),
        ("Kitchen Science Cookbook", 29.0, 0.8),
        ("Atlas of Hidden Places", 35.0, 0.6),
        ("Quiet Strength Novel", 18.0, 1.0),
    ],
    # A product whose category is missing in the source system.
    "": [("Gift Card Holder", 12.0, 0.3)],
}
CATEGORY_WEIGHTS = {
    "Electronics": 0.24, "Home & Kitchen": 0.21, "Clothing": 0.20, "Beauty": 0.14,
    "Sports & Outdoors": 0.12, "Books": 0.08, "": 0.01,
}
SEASONALITY = {1: 0.82, 2: 0.85, 3: 0.95, 4: 0.97, 5: 1.0, 6: 0.98, 7: 1.02,
               8: 1.04, 9: 1.0, 10: 1.02, 11: 1.30, 12: 1.45}
REGIONS = {
    "North": ["Boston", "Chicago", "Minneapolis", "Detroit"],
    "South": ["Atlanta", "Houston", "Miami", "Nashville"],
    "East": ["New York", "Philadelphia", "Baltimore", "Pittsburgh"],
    "West": ["Seattle", "San Francisco", "Los Angeles", "Portland"],
    "Central": ["Denver", "Kansas City", "Omaha", "St. Louis"],
}
REGION_WEIGHTS = [0.22, 0.2, 0.24, 0.22, 0.12]
CHANNELS = ["web", "mobile", "marketplace"]
CHANNEL_WEIGHTS = [0.5, 0.35, 0.15]
FIRST_NAMES = ["Ava", "Liam", "Mia", "Noah", "Zoe", "Ethan", "Isla", "Lucas", "Maya", "Owen", "Aria",
               "Leo", "Nora", "Kai", "Ella", "Ravi", "Priya", "Sofia", "Mateo", "Hana", "Omar",
               "Lena", "Arjun", "Chloe", "Diego", "Amara", "Felix", "Yuki", "Sara", "Jonah"]
LAST_NAMES = ["Smith", "Patel", "Garcia", "Kim", "Nguyen", "Johnson", "Brown", "Lopez", "Chen", "Singh",
              "Martin", "Rossi", "Khan", "Silva", "Novak", "Okafor", "Larsen", "Cohen", "Haddad", "Ito"]


def daily_rate(day: date) -> float:
    """Expected orders per day: ramp-up after launch, seasonality and weekend lift."""
    months = (day - LAUNCH).days / 30.4
    base = 8.0 + 9.0 * (1.0 - math.exp(-months / 8.0))
    weekend = 1.15 if day.weekday() >= 5 else 1.0
    return base * SEASONALITY[day.month] * weekend


def new_customer_probability(day: date) -> float:
    months = (day - LAUNCH).days / 30.4
    return 0.18 + 0.6 * math.exp(-months / 3.0)


def build_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    pid = 1
    for category, items in CATALOG.items():
        for name, price, popularity in items:
            launched = LAUNCH if rng.random() > 0.15 else LAUNCH + timedelta(days=int(rng.integers(30, 300)))
            rows.append({
                "product_id": f"P{pid:03d}", "product_name": name, "category": category,
                "list_price": price, "launch_date": launched.isoformat(), "_popularity": popularity,
            })
            pid += 1
    return pd.DataFrame(rows)


def simulate(rng: np.random.Generator, products: pd.DataFrame):
    customers: list[dict] = []
    churn_at: list[date] = []
    propensity: list[float] = []
    orders: list[dict] = []
    items: list[dict] = []

    categories = list(CATEGORY_WEIGHTS)
    cat_probs = np.array([CATEGORY_WEIGHTS[c] for c in categories])
    cat_probs = cat_probs / cat_probs.sum()
    by_category = {c: products[products["category"] == c] for c in categories}

    order_seq = 1
    item_seq = 1
    day = LAUNCH
    while day <= END:
        n_orders = rng.poisson(daily_rate(day))
        is_october_2026 = day.year == 2026 and day.month == 10
        for _ in range(n_orders):
            eligible = [i for i, c in enumerate(customers) if churn_at[i] >= day]
            wants_new = rng.random() < new_customer_probability(day) or not eligible
            if wants_new:
                # October 2026: paid acquisition was paused -> far fewer first-time buyers.
                if is_october_2026 and rng.random() < 0.55:
                    continue
                cid = len(customers)
                region = rng.choice(list(REGIONS), p=REGION_WEIGHTS)
                first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
                signup = max(LAUNCH, day - timedelta(days=int(rng.integers(0, 21))))
                customers.append({
                    "customer_id": f"C{cid + 1:05d}", "first_name": first, "last_name": last,
                    "email": f"{first.lower()}.{last.lower()}{cid + 1}@example.com",
                    "city": rng.choice(REGIONS[region]), "region": region,
                    "signup_date": signup.isoformat(),
                })
                churn_at.append(day + timedelta(days=int(rng.exponential(300))))
                propensity.append(float(rng.gamma(2.0, 1.0)))
                customer_idx = cid
            else:
                weights = np.array([propensity[i] for i in eligible])
                customer_idx = eligible[int(rng.choice(len(eligible), p=weights / weights.sum()))]

            cancel_rate = 0.11 if is_october_2026 else 0.05
            r = rng.random()
            status = "cancelled" if r < cancel_rate else ("returned" if r < cancel_rate + 0.035 else "completed")
            promo = day.month in (11, 12)
            order_id = f"O{order_seq:06d}"
            n_lines = int(rng.choice([1, 2, 3, 4], p=[0.55, 0.28, 0.12, 0.05]))
            chosen: set[str] = set()
            lines = []
            for _ in range(n_lines):
                category = categories[int(rng.choice(len(categories), p=cat_probs))]
                pool = by_category[category]
                pool = pool[pool["launch_date"] <= day.isoformat()]
                pool = pool[pool["_popularity"] > 0]
                if pool.empty:
                    continue
                w = pool["_popularity"].to_numpy()
                product = pool.iloc[int(rng.choice(len(pool), p=w / w.sum()))]
                if product["product_id"] in chosen:
                    continue
                if (is_october_2026 and day >= STOCKOUT_START
                        and product["product_name"] in STOCKOUT_PRODUCTS and rng.random() < 0.92):
                    continue  # out of stock
                chosen.add(product["product_id"])
                discount = 0.0
                if rng.random() < (0.35 if promo else 0.12):
                    discount = float(rng.choice([0.05, 0.10, 0.15, 0.20]))
                lines.append({
                    "order_item_id": f"OI{item_seq:07d}", "order_id": order_id,
                    "product_id": product["product_id"],
                    "quantity": int(rng.choice([1, 2, 3], p=[0.8, 0.15, 0.05])),
                    "unit_price": float(product["list_price"]), "discount": discount,
                })
                item_seq += 1
            if not lines:
                continue
            orders.append({
                "order_id": order_id, "customer_id": customers[customer_idx]["customer_id"],
                "order_date": day.isoformat(), "status": status,
                "channel": CHANNELS[int(rng.choice(3, p=CHANNEL_WEIGHTS))],
            })
            items.extend(lines)
            order_seq += 1
        day += timedelta(days=1)

    # Registered accounts that never purchased.
    for _ in range(int(len(customers) * 0.06)):
        cid = len(customers)
        region = rng.choice(list(REGIONS), p=REGION_WEIGHTS)
        first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        signup = LAUNCH + timedelta(days=int(rng.integers(0, (END - LAUNCH).days)))
        customers.append({
            "customer_id": f"C{cid + 1:05d}", "first_name": first, "last_name": last,
            "email": f"{first.lower()}.{last.lower()}{cid + 1}@example.com",
            "city": rng.choice(REGIONS[region]), "region": region, "signup_date": signup.isoformat(),
        })

    return pd.DataFrame(customers), pd.DataFrame(orders), pd.DataFrame(items)


def inject_quality_issues(rng: np.random.Generator, customers, products, orders, items):
    """Make the raw extracts look like real operational exports."""
    customers = customers.copy()
    products = products.drop(columns=["_popularity"]).copy()
    orders = orders.copy()
    items = items.copy()
    for df in (customers, products, orders, items):
        for col in df.columns:
            df[col] = df[col].astype(object)

    def pick(df, n):
        return rng.choice(df.index.to_numpy(), size=n, replace=False)

    # Customers: inconsistent casing/whitespace, missing emails/regions, duplicates.
    for idx in pick(customers, 40):
        customers.at[idx, "first_name"] = f"  {str(customers.at[idx, 'first_name']).lower()} "
        customers.at[idx, "city"] = str(customers.at[idx, "city"]).upper()
    for idx in pick(customers, 20):
        customers.at[idx, "email"] = str(customers.at[idx, "email"]).upper()
    customers.loc[pick(customers, 30), "email"] = None
    customers.loc[pick(customers, 12), "region"] = None
    customers = pd.concat([customers, customers.loc[pick(customers, 15)]], ignore_index=True)

    # Products: prices exported as formatted strings, stray whitespace.
    for idx in pick(products, 8):
        products.at[idx, "list_price"] = f"${float(products.at[idx, 'list_price']):,.2f}"
    for idx in pick(products, 3):
        products.at[idx, "product_name"] = f"{products.at[idx, 'product_name']}  "

    # Orders: status variants, invalid dates, missing/unknown customers, missing channel, duplicates.
    variants = ["Completed", "COMPLETED", " completed ", "Cancelled", "canceled", "RETURNED"]
    for idx in pick(orders, 150):
        s = str(orders.at[idx, "status"])
        if s == "completed":
            orders.at[idx, "status"] = rng.choice(variants[:3])
        elif s == "cancelled":
            orders.at[idx, "status"] = rng.choice(variants[3:5])
        else:
            orders.at[idx, "status"] = variants[5]
    for idx, bad in zip(pick(orders, 6), ["2025-02-30", "N/A", "", "2025/13/01", "yesterday", "0000-00-00"], strict=True):
        orders.at[idx, "order_date"] = bad
    orders.loc[pick(orders, 5), "customer_id"] = None
    for idx, ghost in zip(pick(orders, 3), ["C99991", "C99992", "C99993"], strict=True):
        orders.at[idx, "customer_id"] = ghost
    orders.loc[pick(orders, 25), "channel"] = None
    orders = pd.concat([orders, orders.loc[pick(orders, 40)]], ignore_index=True)

    # Order items: missing prices/discounts/quantities, negative quantities, unknown products, duplicates.
    items.loc[pick(items, 20), "unit_price"] = None
    items.loc[pick(items, 10), "discount"] = None
    items.loc[pick(items, 8), "quantity"] = None
    for idx in pick(items, 4):
        items.at[idx, "quantity"] = -int(items.at[idx, "quantity"])
    items.loc[pick(items, 3), "product_id"] = "P999"
    items = pd.concat([items, items.loc[pick(items, 60)]], ignore_index=True)

    # Shuffle so duplicates are not adjacent.
    customers = customers.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    orders = orders.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    items = items.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return customers, products, orders, items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    products = build_products(rng)
    customers, orders, items = simulate(rng, products)
    customers, products, orders, items = inject_quality_issues(rng, customers, products, orders, items)

    args.out.mkdir(parents=True, exist_ok=True)
    customers.to_csv(args.out / "customers.csv", index=False)
    products.to_csv(args.out / "products.csv", index=False)
    orders.to_csv(args.out / "orders.csv", index=False)
    items.to_csv(args.out / "order_items.csv", index=False)
    print(f"Wrote {len(customers)} customers, {len(products)} products, "
          f"{len(orders)} orders, {len(items)} order items to {args.out}")


if __name__ == "__main__":
    main()
