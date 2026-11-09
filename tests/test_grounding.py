"""The numeric grounding check that stops invented figures from reaching the user."""

from __future__ import annotations

from app.agent.grounding import Evidence, check_grounding, extract_numbers

EVIDENCE = Evidence([{
    "revenue": 70567.7,
    "revenue_change": -19639.8,
    "revenue_change_pct": -27.8,
    "orders": 1833,
    "total": 1234567.0,
    "share": 0.472,
    "products": [{"product_name": "Nimbus 27in 4K Monitor", "revenue": 184930.9, "product_id": "P002"}],
    "findings": ["Revenue fell 27.8% to $50,928."],
}])


def test_number_parsing():
    values = [(n.text, n.value) for n in extract_numbers("Revenue was $1.2M, down -$19,640 or 27.8% (3.5K units)")]
    assert values == [("$1.2M", 1_200_000.0), ("-$19,640", -19640.0), ("27.8%", 27.8), ("3.5K", 3500.0)]


def test_exact_and_rounded_figures_are_verified():
    g = check_grounding("Revenue was $70,568 ($70.6K, about $1.2M overall). It fell 27.8% (-$19,640).", EVIDENCE)
    assert g.grounded and g.verified_numbers == 5


def test_trailing_zero_rounding_is_tolerated():
    assert check_grounding("Roughly 1,800 customers are inactive.", EVIDENCE).grounded


def test_invented_numbers_are_flagged():
    g = check_grounding("Revenue was $82,000 and grew 12.5%.", EVIDENCE)
    assert not g.grounded and g.unverified_numbers == ["$82,000", "12.5%"]


def test_wrong_precision_is_flagged():
    assert not check_grounding("Total revenue was $1.5M.", EVIDENCE).grounded


def test_absolute_values_and_ratios():
    g = check_grounding("Revenue dropped by $19,640; the category holds 47.2% of sales.", EVIDENCE)
    assert g.grounded


def test_digits_inside_names_and_dates_are_ignored():
    g = check_grounding("Nimbus 27in 4K Monitor (P002) led in 2026-10 and in October 2026 with $184,931.", EVIDENCE)
    assert g.grounded and g.checked_numbers == 1


def test_numbers_from_tool_prose_are_evidence():
    assert check_grounding("October revenue was $50,928.", EVIDENCE).grounded


def test_small_counts_and_question_numbers_are_not_checked():
    g = check_grounding("Here are the top 5 products for the last 37 days.", EVIDENCE,
                        question="top 5 products in the last 37 days")
    assert g.checked_numbers == 0 and g.grounded
