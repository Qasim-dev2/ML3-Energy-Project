"""
tests/test_recommendations.py
------------------------------
Unit tests for the recommendation engine helper functions.
Run with: pytest tests/ -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.recommendations import estimate_bill, efficiency_score, peak_usage_hours


# ── estimate_bill ─────────────────────────────────────────────────────────────

class TestEstimateBill:
    def test_basic(self):
        monthly_kwh, bill = estimate_bill(10.0)
        assert monthly_kwh == pytest.approx(300.0)
        assert bill == pytest.approx(10500.0)

    def test_zero_consumption(self):
        monthly_kwh, bill = estimate_bill(0.0)
        assert monthly_kwh == 0.0
        assert bill == 0.0

    def test_custom_days(self):
        monthly_kwh, bill = estimate_bill(5.0, days_in_month=31)
        assert monthly_kwh == pytest.approx(155.0)

    def test_custom_rate(self):
        monthly_kwh, bill = estimate_bill(10.0, rate=25.0)
        assert bill == pytest.approx(7500.0)


# ── efficiency_score ──────────────────────────────────────────────────────────

class TestEfficiencyScore:
    def test_returns_float(self):
        score = efficiency_score(10.0)
        assert isinstance(score, float)

    def test_range(self):
        for kwh in [1.0, 5.0, 15.0, 30.0]:
            score = efficiency_score(kwh)
            assert 0.0 <= score <= 100.0

    def test_lower_consumption_higher_score(self):
        low_score  = efficiency_score(3.0)
        high_score = efficiency_score(20.0)
        assert low_score > high_score


# ── peak_usage_hours ──────────────────────────────────────────────────────────

class TestPeakUsageHours:
    def test_summer_high_ac(self):
        result = peak_usage_hours(ac_hours=8, fan_hours=6, season="Summer")
        assert "AC" in result or "PM" in result

    def test_winter(self):
        result = peak_usage_hours(ac_hours=0, fan_hours=2, season="Winter")
        assert "Winter" in result or "PM" in result

    def test_returns_string(self):
        result = peak_usage_hours(ac_hours=2, fan_hours=4, season="Spring")
        assert isinstance(result, str)
        assert len(result) > 0
