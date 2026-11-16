"""Dates in a plan: the model never does calendar arithmetic, it names a period and this module expands it.

A period is one of
    {"year": 2025}                     {"month": "2025-03"}               {"day": "2025-03-04"}
    {"token": "this_year" | "last_year" | "this_month" | "last_month" | "today" | "yesterday"}
    {"token": "last_n_days", "n": 30}  (the n days before today, and today)
    {"from": "2025-01-01", "to": "2025-03-31"}   (both days included)
and becomes a half-open range [start, end): `start <= date < end`.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

MAX_DAYS = 3650


class BadPeriod(ValueError):
    """The period is not one of the forms above, or is not a real calendar period."""


def _day(text) -> date:
    try:
        return datetime.strptime(str(text), "%Y-%m-%d").date()
    except ValueError as exc:
        raise BadPeriod(f"{text!r} is not a date (expected YYYY-MM-DD)") from exc


def _month_start(year: int, month: int) -> date:
    return date(year + (month - 1) // 12, (month - 1) % 12 + 1, 1)


def expand(period, today: date) -> tuple[date, date]:
    """The range [start, end) a period stands for. Raises BadPeriod."""
    if not isinstance(period, dict):
        raise BadPeriod("a date condition must be an object such as {\"year\": 2025}")
    try:
        if "year" in period:
            y = int(period["year"])
            if not 1900 <= y <= 2200:
                raise BadPeriod(f"{y} is not a plausible year")
            return date(y, 1, 1), date(y + 1, 1, 1)
        if "month" in period:
            y, m = str(period["month"]).split("-")
            start = _month_start(int(y), int(m)) if 1 <= int(m) <= 12 else None
            if start is None or not 1900 <= start.year <= 2200:
                raise BadPeriod(f"{period['month']!r} is not a month (expected YYYY-MM)")
            return start, _month_start(start.year, start.month + 1)
        if "day" in period:
            d = _day(period["day"])
            return d, d + timedelta(days=1)
        if "from" in period or "to" in period:
            if "from" not in period or "to" not in period:
                raise BadPeriod("a range needs both \"from\" and \"to\"")
            start, last = _day(period["from"]), _day(period["to"])
            if start > last:
                raise BadPeriod(f"the range starts ({start}) after it ends ({last})")
            return start, last + timedelta(days=1)
        token = period.get("token")
        if token is None and "last_n_days" in period:  # {"last_n_days": 30} means {"token": "last_n_days", "n": 30}
            token, period = "last_n_days", {"n": period["last_n_days"]}
        if token == "this_year":
            return date(today.year, 1, 1), date(today.year + 1, 1, 1)
        if token == "last_year":
            return date(today.year - 1, 1, 1), date(today.year, 1, 1)
        if token == "this_month":
            return _month_start(today.year, today.month), _month_start(today.year, today.month + 1)
        if token == "last_month":
            return _month_start(today.year, today.month - 1), _month_start(today.year, today.month)
        if token == "today":
            return today, today + timedelta(days=1)
        if token == "yesterday":
            return today - timedelta(days=1), today
        if token == "last_n_days":
            n = int(period.get("n"))
            if not 1 <= n <= MAX_DAYS:
                raise BadPeriod(f"'last {n} days' is not a usable period (1 to {MAX_DAYS})")
            return today - timedelta(days=n), today + timedelta(days=1)
    except BadPeriod:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise BadPeriod(f"could not read the date condition {period!r}") from exc
    raise BadPeriod(f"unknown date condition {period!r}")


def describe(start: date, end: date) -> str:
    """The range in words: '2025', 'March 2025', '2025-03-04' or '2025-01-01 to 2025-03-31'."""
    last = end - timedelta(days=1)
    if (start.month, start.day) == (1, 1) and (end.month, end.day) == (1, 1) and end.year == start.year + 1:
        return str(start.year)
    if start.day == 1 and end == _month_start(start.year, start.month + 1):
        return start.strftime("%B %Y")
    return str(start) if start == last else f"{start} to {last}"
