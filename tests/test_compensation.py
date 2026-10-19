"""Compensation extraction. Positive fixtures are shaped after real pay-range phrasings seen in
public postings (see app/services/compensation.py's module docstring); the hourly cases are
synthetic."""

from app.services.compensation import extract_compensation


def test_em_dash_range_with_currency_code():
    text = "Annual Salary:\n$270,000 — $340,000 USD\nLogistics\nMinimum education: ..."
    result = extract_compensation(text)
    assert result.comp_min == 270_000
    assert result.comp_max == 340_000
    assert result.comp_currency == "USD"
    assert result.comp_period == "year"


def test_eur_range_with_period_thousands_grouping():
    text = "Annual Salary:\n€155.000 — €205.000 EUR\nLogistics"
    result = extract_compensation(text)
    assert result.comp_min == 155_000
    assert result.comp_max == 205_000
    assert result.comp_currency == "EUR"


def test_gbp_range():
    text = "Annual Salary:\n£165,000 — £190,000 GBP\nLogistics"
    result = extract_compensation(text)
    assert result.comp_min == 165_000
    assert result.comp_max == 190_000
    assert result.comp_currency == "GBP"


def test_currency_code_overrides_dollar_symbol():
    # A $-symbol posting priced in CAD -- the trailing code must win.
    text = "Annual Salary:\n$420,000 — $495,000 CAD\nLogistics"
    result = extract_compensation(text)
    assert result.comp_currency == "CAD"
    assert result.comp_min == 420_000


def test_between_and_phrasing():
    text = (
        "At Apple, base pay is one part of our total compensation package. "
        "The base pay range for this role is between $184,700 and $324,800, "
        "and your base pay will depend on your skills, qualifications, experience."
    )
    result = extract_compensation(text)
    assert result.comp_min == 184_700
    assert result.comp_max == 324_800
    assert result.comp_currency == "USD"


def test_range_without_thousands_comma():
    text = "US: $147000 - $210000 (USD) + 15% bonus target + equity + benefits"
    result = extract_compensation(text)
    assert result.comp_min == 147_000
    assert result.comp_max == 210_000


def test_range_with_thousands_comma():
    text = "salary range for this full-time position is $149,400 - $175,000 + 15% bonus target"
    result = extract_compensation(text)
    assert result.comp_min == 149_400
    assert result.comp_max == 175_000


def test_no_spaces_around_dash():
    text = "The estimated base salary range for this role is $130,000-$150,000."
    result = extract_compensation(text)
    assert result.comp_min == 130_000
    assert result.comp_max == 150_000


def test_salary_grade_sentence():
    text = "This position is a salary grade 8 and ranges from $138,800-$232,700. Final determination..."
    result = extract_compensation(text)
    assert result.comp_min == 138_800
    assert result.comp_max == 232_700


def test_range_in_a_sentence():
    text = (
        "The salary range for this full-time position is $150,000 - $160,000 + equity + benefits."
    )
    result = extract_compensation(text)
    assert result.comp_min == 150_000
    assert result.comp_max == 160_000


def test_labeled_minimum_maximum():
    text = (
        "Salary Range or On Target Earnings:\n\nMinimum:\n\n$151,800.00\n\n"
        "Maximum:\n\n$332,200.00\n\nIn addition to the base salary..."
    )
    result = extract_compensation(text)
    assert result.comp_min == 151_800
    assert result.comp_max == 332_200
    assert result.comp_period == "year"


def test_multiple_location_ranges_take_overall_min_max():
    text = (
        "Cambridge, MA: $218,700 - $249,600 for Applied Researcher 4\n\n\n\n\n"
        "McLean, VA: $218,700 - $249,600 for Applied Researcher 4\n\n\n\n\n"
        "New York, NY: $238,600 - $272,300 for Applied Researcher 4\n\n\n\n\n"
        "San Jose, CA: $238,600 - $272,300 for Applied Researcher 4"
    )
    result = extract_compensation(text)
    assert result.comp_min == 218_700
    assert result.comp_max == 272_300
    assert "218,700" in result.comp_raw
    assert "272,300" in result.comp_raw


def test_multiple_location_ranges_with_city_prefixes_take_overall_min_max():
    text = (
        "The base pay range for this position is:\n"
        "Mountain View $180,000 - $243,500\n"
        "New York $177,500- $240,000\n"
        "Oakland, CA $180,000- $243,500\n"
        "San Diego, CA $165,500- $223,500"
    )
    result = extract_compensation(text)
    assert result.comp_min == 165_500
    assert result.comp_max == 243_500


def test_to_connector():
    text = "Compensation for this role is $150,000 to $160,000 annually."
    result = extract_compensation(text)
    assert result.comp_min == 150_000
    assert result.comp_max == 160_000


# -- Hourly (synthetic -- no real hourly posting in this DB, but explicitly
# requested by the plan) --


def test_hourly_range_with_per_hour_phrase():
    text = "This role pays $28 - $35 per hour depending on experience."
    result = extract_compensation(text)
    assert result.comp_min == 28
    assert result.comp_max == 35
    assert result.comp_period == "hour"


def test_hourly_range_with_slash_hr():
    text = "Pay: $22.50/hr - $30/hr"
    result = extract_compensation(text)
    assert result.comp_min == 22
    assert result.comp_max == 30
    assert result.comp_period == "hour"


def test_single_hourly_rate():
    text = "This is a contract role paying $45/hour."
    result = extract_compensation(text)
    assert result.comp_min == 45
    assert result.comp_max == 45
    assert result.comp_period == "hour"


# -- Negatives --


def test_none_when_no_dollar_amounts_at_all():
    assert extract_compensation("We build search systems for millions of users.") is None


def test_none_for_empty_or_none_text():
    assert extract_compensation("") is None
    assert extract_compensation(None) is None


def test_rejects_below_annual_floor_signing_bonus():
    # $5,000-$10,000 reads like a range but is far below a plausible salary
    # -- almost certainly a bonus/relocation figure, not comp.
    text = "You'll also receive a $5,000 - $10,000 signing bonus paid over your first year."
    assert extract_compensation(text) is None


def test_rejects_above_annual_ceiling():
    text = "Total company valuation range: $3,000,000 - $4,000,000 (not a real salary)."
    assert extract_compensation(text) is None


def test_rejects_years_as_amounts():
    # No $ prefix on a bare year range -- must never be picked up as comp.
    text = "Founded in 2015, growing steadily from 2015 to 2024."
    assert extract_compensation(text) is None


def test_requires_currency_signal_not_bare_numbers():
    # A numeric range with no $ / currency signal at all must not match.
    text = "This role requires 5 to 10 years of experience."
    assert extract_compensation(text) is None
