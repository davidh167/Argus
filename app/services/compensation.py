"""Deterministic (regex-only, no LLM) compensation extraction from job
description text. Patterns and sanity bounds are calibrated against real
snippets pulled from this DB across Anthropic, Apple, Google, Notion,
Capital One, Ford, Together AI, and Zoom -- see tests/test_compensation.py
for the exact real-shaped fixtures. Deliberately narrow: only the
connector/format shapes actually observed. Not a general money-parser.
"""

import re
from dataclasses import dataclass

# USD/GBP/EUR/CAD are the only currencies observed in this DB's postings
# (Anthropic pays some non-US roles in GBP/EUR; one CAD posting seen). Not
# an attempt at full ISO-4217 coverage.
_CURRENCY_SYMBOL = r"[$£€]"
_CURRENCY_CODE = r"(?:USD|EUR|GBP|CAD)"
# A decimal/thousands group is only consumed when a digit follows the dot --
# without that guard, "$150,000." at the end of a sentence would greedily
# swallow the sentence-ending period as part of the number and then fail to
# parse (no digits after it). Handles both a genuine decimal ("151,800.00")
# and European period-grouped thousands ("155.000") since both have digits
# right after the dot; only a bare trailing "." is excluded.
_AMOUNT = r"\d[\d,]*(?:\.\d+)?"

_SYMBOL_TO_CURRENCY = {"$": "USD", "£": "GBP", "€": "EUR"}

# "$270,000 - $340,000", "$270,000-$340,000" (Notion/Ford: no spaces),
# "$147000 - $210000 (USD)" (Google: no thousands comma), "$150,000 to
# $160,000", "between $184,700 and $324,800" (Apple -- "between" itself
# isn't matched, just the $X ... and ... $Y it introduces), "$270,000 —
# $340,000 USD" / "€155.000 — €205.000 EUR" (Anthropic -- em-dash divider,
# trailing currency code, and European period-grouped thousands).
_INLINE_RANGE_RE = re.compile(
    rf"{_CURRENCY_SYMBOL}\s?({_AMOUNT})\s*(?:-|–|—|to|and)\s*"
    rf"{_CURRENCY_SYMBOL}?\s?({_AMOUNT})"
    rf"(?:\s*\(?\s*({_CURRENCY_CODE})\s*\)?)?",
    re.IGNORECASE,
)

# "Salary Range or On Target Earnings:\n\nMinimum:\n\n$151,800.00\n\n
# Maximum:\n\n$332,200.00" (Zoom -- labeled, not connected by a word/dash,
# and far enough apart that _INLINE_RANGE_RE's small gap won't span it).
_LABELED_MIN_MAX_RE = re.compile(
    rf"Minimum:?\s*{_CURRENCY_SYMBOL}\s?({_AMOUNT}).{{0,120}}?"
    rf"Maximum:?\s*{_CURRENCY_SYMBOL}\s?({_AMOUNT})",
    re.IGNORECASE | re.DOTALL,
)

# Hourly: no real example in this DB (all sampled postings are salaried),
# but explicitly asked for -- kept narrow and synthetic-tested.
_HOURLY_RANGE_RE = re.compile(
    rf"{_CURRENCY_SYMBOL}\s?({_AMOUNT})\s*(?:-|–|—|to)\s*"
    rf"{_CURRENCY_SYMBOL}?\s?({_AMOUNT})\s*"
    rf"(?:/\s?hr\b|/\s?hour\b|per\s+hour\b)",
    re.IGNORECASE,
)
_HOURLY_SINGLE_RE = re.compile(
    rf"{_CURRENCY_SYMBOL}\s?({_AMOUNT})\s*(?:/\s?hr\b|/\s?hour\b|per\s+hour\b)",
    re.IGNORECASE,
)

ANNUAL_BOUNDS = (20_000, 2_000_000)
HOURLY_BOUNDS = (10, 500)


@dataclass
class CompExtraction:
    comp_min: int
    comp_max: int
    comp_currency: str
    comp_period: str  # "year" | "hour"
    comp_raw: str  # every matched snippet, in document order, "; "-joined


def _parse_amount(num_str: str) -> int | None:
    """'270,000' -> 270000, '151,800.00' -> 151800 (cents dropped -- comp is
    stored as a whole-currency-unit Integer), '155.000' -> 155000 (European
    thousands grouping), '147000' -> 147000.

    Comma is always a thousands separator in this data (the only place a
    comma co-occurs with a decimal point is Zoom's "151,800.00" -- comma
    grouping, period decimal). A lone period is ambiguous between a
    thousands separator (Anthropic's EUR figures) and a decimal point --
    disambiguated by the observed rule that only the thousands case ever
    has exactly 3 digits after the period; a genuine decimal (e.g. hourly
    cents) has 1-2.
    """
    num_str = num_str.strip()
    if "," in num_str:
        num_str = num_str.replace(",", "")
        if re.fullmatch(r"\d+\.\d{2}", num_str):
            num_str = num_str.split(".")[0]
        return int(num_str) if num_str.isdigit() else None
    if "." in num_str:
        integer_part, _, frac = num_str.rpartition(".")
        if not integer_part.isdigit() or not frac.isdigit():
            return None
        if len(frac) == 3:
            return int(integer_part + frac)
        return int(integer_part)  # decimal point -- drop the fractional part
    return int(num_str) if num_str.isdigit() else None


def _currency_for(symbol: str, code: str | None) -> str:
    # An explicit trailing code wins over the symbol -- covers Anthropic's
    # "$420,000 — $495,000 CAD" (a $ posting priced in Canadian dollars).
    if code:
        return code.upper()
    return _SYMBOL_TO_CURRENCY.get(symbol, "USD")


def extract_compensation(text: str | None) -> CompExtraction | None:
    """Scans the whole text for every annual range this module recognizes,
    keeping only matches within ANNUAL_BOUNDS, and returns the overall
    min-of-mins/max-of-maxes across all of them -- the real shape of a
    posting listing one range per office location (seen verbatim in both
    Intuit's and Capital One's real postings). comp_raw keeps every
    matched snippet so the UI can show exactly what was matched, not just
    the derived numbers.

    Annual patterns are tried before hourly, and a match found by one
    pattern is not also considered by the other passes over the same text
    (each pattern's matches are independent, but annual ranges dominate:
    a posting is either annual-priced or hourly-priced, not both, in every
    sample seen). Returns None when nothing recognizable and in-bounds is
    found.
    """
    if not text:
        return None

    matches: list[tuple[int, int, str, str]] = []  # (min, max, currency, raw_snippet)

    for m in _INLINE_RANGE_RE.finditer(text):
        lo, hi, code = _parse_amount(m.group(1)), _parse_amount(m.group(2)), m.group(3)
        if lo is None or hi is None:
            continue
        lo, hi = min(lo, hi), max(lo, hi)
        if ANNUAL_BOUNDS[0] <= lo and hi <= ANNUAL_BOUNDS[1]:
            symbol = m.group(0)[0]
            matches.append((lo, hi, _currency_for(symbol, code), m.group(0).strip()))

    for m in _LABELED_MIN_MAX_RE.finditer(text):
        lo, hi = _parse_amount(m.group(1)), _parse_amount(m.group(2))
        if lo is None or hi is None:
            continue
        lo, hi = min(lo, hi), max(lo, hi)
        if ANNUAL_BOUNDS[0] <= lo and hi <= ANNUAL_BOUNDS[1]:
            symbol = re.search(_CURRENCY_SYMBOL, m.group(0))
            currency = _currency_for(symbol.group(0), None) if symbol else "USD"
            matches.append((lo, hi, currency, re.sub(r"\s+", " ", m.group(0).strip())))

    if matches:
        overall_min = min(lo for lo, _, _, _ in matches)
        overall_max = max(hi for _, hi, _, _ in matches)
        # Currency of whichever match produced the overall max -- postings
        # with multiple ranges in this data are always single-currency
        # (per-location bands within one US company), so this is just a
        # deterministic tie-breaker, not a real ambiguity in practice.
        currency = next(c for lo, hi, c, _ in matches if hi == overall_max)
        raw = "; ".join(dict.fromkeys(r for _, _, _, r in matches))  # de-dup, keep order
        return CompExtraction(overall_min, overall_max, currency, "year", raw)

    hourly_matches: list[tuple[int, int, str, str]] = []
    for m in _HOURLY_RANGE_RE.finditer(text):
        lo, hi = _parse_amount(m.group(1)), _parse_amount(m.group(2))
        if lo is None or hi is None:
            continue
        lo, hi = min(lo, hi), max(lo, hi)
        if HOURLY_BOUNDS[0] <= lo and hi <= HOURLY_BOUNDS[1]:
            hourly_matches.append((lo, hi, _currency_for(m.group(0)[0], None), m.group(0).strip()))
    if not hourly_matches:
        for m in _HOURLY_SINGLE_RE.finditer(text):
            amount = _parse_amount(m.group(1))
            if amount is None or not (HOURLY_BOUNDS[0] <= amount <= HOURLY_BOUNDS[1]):
                continue
            hourly_matches.append(
                (amount, amount, _currency_for(m.group(0)[0], None), m.group(0).strip())
            )

    if hourly_matches:
        overall_min = min(lo for lo, _, _, _ in hourly_matches)
        overall_max = max(hi for _, hi, _, _ in hourly_matches)
        currency = next(c for lo, hi, c, _ in hourly_matches if hi == overall_max)
        raw = "; ".join(dict.fromkeys(r for _, _, _, r in hourly_matches))
        return CompExtraction(overall_min, overall_max, currency, "hour", raw)

    return None
