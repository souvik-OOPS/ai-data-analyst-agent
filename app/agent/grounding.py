"""Numeric grounding check: every number in an answer must be traceable to the data.

How it works
------------
1. Collect *evidence*: every number in the tool outputs (recursively), numbers
   embedded in tool-produced strings (e.g. diagnostic findings), and their
   absolute values (an answer may say "fell by $19,640" for a change of -19640).
2. Remove text that legitimately contains digits but is not a measurement:
   dates (2026-10), and any string copied from tool output (product names such
   as "Nimbus 27in 4K Monitor", customer IDs).
3. Extract numbers from the answer, honouring $, %, K/M/B suffixes and the
   precision the answer used: "$1.2M" may match 1,234,567, and "about 1,800"
   may match 1,833, but "$1.5M" may not.
4. Ignore numbers the user typed in the question, small counting words such as
   "top 5", and four-digit years.

Anything left that cannot be matched is reported as unverified.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterable
from typing import Any

from app.agent.models import Grounding

_NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>[-−])?\s?(?P<cur>\$)?\s?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?"
    r"(?:\s?(?P<suffix>%|[kKmMbB]\b))?"
)
_DATE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
_MULTIPLIER = {"k": 1e3, "m": 1e6, "b": 1e9}
SMALL_INT_LIMIT = 10


def _strings_and_numbers(node: Any, strings: list[str], numbers: list[float]) -> None:
    if isinstance(node, bool) or node is None:
        return
    if isinstance(node, (int, float)):
        numbers.append(float(node))
    elif isinstance(node, str):
        strings.append(node)
    elif isinstance(node, dict):
        for value in node.values():
            _strings_and_numbers(value, strings, numbers)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _strings_and_numbers(value, strings, numbers)


class Evidence:
    """Sorted pool of numbers an answer is allowed to quote, plus strings to mask."""

    def __init__(self, sources: Iterable[Any]):
        strings: list[str] = []
        numbers: list[float] = []
        for source in sources:
            _strings_and_numbers(source, strings, numbers)
        for text in strings:  # numbers embedded in tool-produced text (findings etc.)
            numbers.extend(n.value for n in extract_numbers(_DATE.sub(" ", text)))
        pool = set()
        for n in numbers:
            pool.update({n, abs(n)})
            if abs(n) <= 1:
                pool.add(abs(n) * 100)  # ratios quoted as percentages
        self.values = sorted(pool)
        # Mask identifiers that merely contain digits (product names, IDs), never measurement prose:
        # a finding such as "Revenue fell 27.8%" must still have its numbers checked.
        # Longest first so "Nimbus 27in 4K Monitor" is masked before "4K".
        self.mask_strings = sorted(
            {s.strip() for s in strings
             if 3 <= len(s.strip()) <= 80 and re.search(r"\d", s) and not re.search(r"[$%]", s)},
            key=len, reverse=True)

    def contains(self, target: float, tolerance: float) -> bool:
        i = bisect.bisect_left(self.values, target - tolerance)
        return i < len(self.values) and self.values[i] <= target + tolerance


class FoundNumber:
    __slots__ = ("text", "value", "tolerance", "is_plain_small_int", "is_year")

    def __init__(self, match: re.Match):
        self.text = match.group(0).strip()
        int_part = match.group("int").replace(",", "")
        dec = match.group("dec") or ""
        suffix = (match.group("suffix") or "").lower()
        mult = _MULTIPLIER.get(suffix, 1.0)
        value = float(int_part + dec) * mult
        if match.group("sign"):
            value = -value
        self.value = value
        if dec:
            precision = 10 ** -(len(dec) - 1)
        else:
            trailing_zeros = len(int_part) - len(int_part.rstrip("0")) if int_part != "0" else 0
            precision = 10 ** trailing_zeros
        self.tolerance = 0.5 * precision * mult + 1e-9
        plain = not match.group("cur") and not suffix and not dec
        self.is_plain_small_int = plain and abs(value) <= SMALL_INT_LIMIT
        self.is_year = plain and "," not in match.group("int") and 1900 <= abs(value) <= 2100


def extract_numbers(text: str) -> list[FoundNumber]:
    return [FoundNumber(m) for m in _NUMBER.finditer(text)]


def check_grounding(answer: str, evidence: Evidence, question: str = "") -> Grounding:
    text = answer
    for s in evidence.mask_strings:
        text = text.replace(s, " ")
    text = _DATE.sub(" ", text)
    question_values = {round(abs(n.value), 6) for n in extract_numbers(question)}

    checked, verified, unverified = 0, 0, []
    for n in extract_numbers(text):
        if n.is_plain_small_int or n.is_year or round(abs(n.value), 6) in question_values:
            continue
        checked += 1
        if evidence.contains(n.value, n.tolerance) or evidence.contains(abs(n.value), n.tolerance):
            verified += 1
        elif n.text not in unverified:
            unverified.append(n.text)
    return Grounding(checked_numbers=checked, verified_numbers=verified, unverified_numbers=unverified)
