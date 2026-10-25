"""Due dates in plain code: "by Friday", "tomorrow", "EOD", "end of week", "next week", "in 3 days",
"on the 14th", "March 3", "3/14", "下周", "明天", "周五". Returns an ISO date (YYYY-MM-DD) or None.
The decider only PICKS; dates are never left to a model.
"""
from __future__ import annotations

import datetime as dt
import re

WEEKDAYS = {"monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2, "thursday": 3,
            "thu": 3, "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6}
ZH_WEEKDAYS = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9

_WEEKDAY = re.compile(r"\b(?:by|on|before|until|till|this|next|coming|due)?\s*(" + "|".join(sorted(WEEKDAYS, key=len, reverse=True)) + r")\b", re.I)
_NEXT_WEEKDAY = re.compile(r"\bnext\s+(" + "|".join(sorted(WEEKDAYS, key=len, reverse=True)) + r")\b", re.I)
_IN_DAYS = re.compile(r"\bin\s+(\d{1,2}|a|an|one|two|three|four|five|six|seven|ten)\s+(day|days|week|weeks)\b", re.I)
_MONTH_DAY = re.compile(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.I)
_DAY_MONTH = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b", re.I)
_NUMERIC = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_THE_NTH = re.compile(r"\b(?:on|by)\s+the\s+(\d{1,2})(?:st|nd|rd|th)\b", re.I)
_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "ten": 10}


def parse_due(text: str, today: dt.date | None = None) -> str | None:
    """ISO date for the first due phrase in `text`, relative to `today` (default: today)."""
    today = today or dt.date.today()
    t = (text or "").strip()
    if not t:
        return None
    low = t.lower()
    m = _ISO.search(t)
    if m:
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            pass
    if re.search(r"\b(today|eod|end of (the )?day|by tonight|tonight|cob|close of business)\b", low) or "今天" in t:
        return today.isoformat()
    if re.search(r"\btomorrow\b", low) or "明天" in t:
        return (today + dt.timedelta(days=1)).isoformat()
    if re.search(r"\bday after tomorrow\b", low) or "后天" in t:
        return (today + dt.timedelta(days=2)).isoformat()
    m = _NEXT_WEEKDAY.search(low)
    if m:
        return _weekday(today, WEEKDAYS[m.group(1)], next_week=True).isoformat()
    m = _MONTH_DAY.search(low) or None
    if m:
        d = _safe_date(today, MONTHS[m.group(1).rstrip(".")], int(m.group(2)))
        if d:
            return d.isoformat()
    m = _DAY_MONTH.search(low)
    if m:
        d = _safe_date(today, MONTHS[m.group(2)], int(m.group(1)))
        if d:
            return d.isoformat()
    m = _NUMERIC.search(t)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else today.year
        year = year + 2000 if year < 100 else year
        try:
            return dt.date(year, month, day).isoformat()
        except ValueError:
            pass
    m = _IN_DAYS.search(low)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _WORDS.get(m.group(1), 1)
        days = n * (7 if m.group(2).startswith("week") else 1)
        return (today + dt.timedelta(days=days)).isoformat()
    if re.search(r"\b(end of (the |this )?week|eow|by the weekend|this week)\b", low) or "本周" in t or "这周" in t:
        return _weekday(today, 4, next_week=False).isoformat()      # Friday of this week
    if re.search(r"\bnext week\b", low) or "下周" in t or "下星期" in t:
        zh = re.search(r"下(?:周|星期)([一二三四五六日天])", t)
        wd = ZH_WEEKDAYS[zh.group(1)] if zh else 0
        return _weekday(today, wd, next_week=True).isoformat()
    if re.search(r"\bend of (the |this )?month\b", low) or "月底" in t:
        first_next = (today.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        return (first_next - dt.timedelta(days=1)).isoformat()
    m = _WEEKDAY.search(low)
    if m:
        return _weekday(today, WEEKDAYS[m.group(1)], next_week=False).isoformat()
    zh = re.search(r"(?:周|星期|礼拜)([一二三四五六日天])", t)
    if zh:
        return _weekday(today, ZH_WEEKDAYS[zh.group(1)], next_week=False).isoformat()
    m = _THE_NTH.search(low)
    if m:
        d = _safe_date(today, today.month, int(m.group(1)))
        if d and d < today:
            d = _safe_date(today, today.month % 12 + 1, int(m.group(1)))
        if d:
            return d.isoformat()
    return None


def _weekday(today: dt.date, wd: int, *, next_week: bool) -> dt.date:
    """The coming weekday `wd` (0 = Monday). "Friday" said on a Friday means next Friday; "next Friday"
    means the one in the following calendar week."""
    days = (wd - today.weekday()) % 7
    if days == 0:
        days = 7
    if next_week and today.weekday() + days < 7:
        days += 7
    return today + dt.timedelta(days=days)


def _safe_date(today: dt.date, month: int, day: int) -> dt.date | None:
    for year in (today.year, today.year + 1):
        try:
            d = dt.date(year, month, day)
        except ValueError:
            return None
        if d >= today - dt.timedelta(days=30):
            return d
    return None


def humanize(iso: str | None) -> str:
    if not iso:
        return "no due date"
    try:
        d = dt.date.fromisoformat(iso)
    except ValueError:
        return iso
    return d.strftime("%a %b %d").replace(" 0", " ")
