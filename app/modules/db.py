"""Interview calendar, backed by SQLite.

The course supplied db_Tech.sql, which builds an MS SQL Server database on the
developer's own machine. Streamlit Community Cloud cannot reach that, so the
same table is shipped as data/schedule.db and built by scripts/build_schedule_db.py.

Two things this module is careful about, because both are easy to get wrong:

1. Slots are compared as a full timestamp, not as a date. A conversation that
   starts at 15:12 must not be offered a 09:00 slot on the same day.
2. The calendar has no Saturdays and no Mondays. Candidates ask for Mondays
   anyway (conversations 1, 3, 10), so the caller gets told that plainly
   instead of receiving an empty list.
"""

import datetime as dt
import sqlite3
from typing import List, Optional

from . import config

CLOSED_WEEKDAYS = {"Saturday", "Monday"}


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(config.SCHEDULE_DB)
    con.row_factory = sqlite3.Row
    return con


def _parse(moment: str) -> dt.datetime:
    """Accept '2024-04-03T15:12:00Z', '2024-04-03 15:12', or '2024-04-03'."""
    text = moment.strip().replace("Z", "").replace(" ", "T")
    if len(text) == 10:
        text += "T00:00:00"
    return dt.datetime.fromisoformat(text)


def find_available_slots(after: str,
                         position: str = config.POSITION,
                         limit: int = 3,
                         on_date: Optional[str] = None) -> List[dict]:
    """Free interview slots strictly after `after`.

    after    : ISO timestamp, normally the conversation's start_time_utc
    on_date  : optional 'YYYY-MM-DD' to restrict to one day
    returns  : [{'date','time','weekday'}, ...], soonest first
    """
    moment = _parse(after)
    sql = ("SELECT date, time FROM Schedule "
           "WHERE position = ? AND available = 1 "
           "AND (date || 'T' || time) > ? ")
    params: list = [position, moment.isoformat()]
    if on_date:
        sql += "AND date = ? "
        params.append(on_date)
    sql += "ORDER BY date, time LIMIT ?"
    params.append(limit)

    with _connect() as con:
        rows = con.execute(sql, params).fetchall()

    return [{"date": r["date"],
             "time": r["time"][:5],
             "weekday": dt.date.fromisoformat(r["date"]).strftime("%A")}
            for r in rows]


def resolve_relative_date(phrase: str, reference: str) -> Optional[str]:
    """Turn 'next Friday' / 'Monday' / 'tomorrow' into 'YYYY-MM-DD'.

    `reference` is the conversation's start_time_utc, NOT today. The
    conversations are from April 2024 and the calendar only covers 2024, so
    resolving against the real today returns nothing at all.
    """
    ref = _parse(reference).date()
    text = phrase.lower()

    if "tomorrow" in text:
        return (ref + dt.timedelta(days=1)).isoformat()
    if "today" in text:
        return ref.isoformat()

    weekdays = ["monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday"]
    for index, name in enumerate(weekdays):
        if name in text:
            ahead = (index - ref.weekday()) % 7
            if ahead == 0 or "next" in text:
                ahead = ahead or 7
            return (ref + dt.timedelta(days=ahead)).isoformat()
    return None


def day_is_closed(date_iso: str) -> bool:
    """True for a date the office does not interview on."""
    return dt.date.fromisoformat(date_iso).strftime("%A") in CLOSED_WEEKDAYS


def describe_slots(slots: List[dict]) -> str:
    """Render slots for a prompt: 'Wednesday 2024-04-10 at 09:00'."""
    if not slots:
        return "no free slots found"
    return "; ".join(f"{s['weekday']} {s['date']} at {s['time']}" for s in slots)
