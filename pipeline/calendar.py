"""UTC date helpers. Months and days are calendar units in UTC."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterator


def iter_days(start: date, end: date) -> Iterator[date]:
    """Inclusive start and end."""
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def iter_months(start: date, end: date) -> Iterator[tuple[int, int]]:
    """Inclusive (year, month) pairs covering ``[start, end]``."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def month_first(year: int, month: int) -> date:
    return date(year, month, 1)


def month_last(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def clip_month(year: int, month: int, start: date, end: date) -> tuple[date, date]:
    """The overlap of this calendar month with ``[start, end]``."""
    first = max(start, month_first(year, month))
    last = min(end, month_last(year, month))
    if first > last:
        raise ValueError(f"{year}-{month:02d} does not overlap {start}..{end}")
    return first, last


def is_current_or_future_month(year: int, month: int, today: date | None = None) -> bool:
    today = today or date.today()
    return (year, month) >= (today.year, today.month)


def month_stamp(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"
