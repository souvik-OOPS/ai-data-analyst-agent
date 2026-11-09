"""Business analytics computed with SQL over the cleaned SQLite database."""

from app.analytics.customers import customer_metrics, customer_trend, inactive_customers, top_customers
from app.analytics.diagnostics import analyze_revenue_change
from app.analytics.insights import compute_insights
from app.analytics.periods import DatasetInfo, DateRange, load_dataset_info
from app.analytics.products import product_performance
from app.analytics.sales import monthly_sales, revenue_breakdown, sales_metrics

__all__ = [
    "DatasetInfo",
    "DateRange",
    "analyze_revenue_change",
    "compute_insights",
    "customer_metrics",
    "customer_trend",
    "inactive_customers",
    "load_dataset_info",
    "monthly_sales",
    "product_performance",
    "revenue_breakdown",
    "sales_metrics",
    "top_customers",
]
