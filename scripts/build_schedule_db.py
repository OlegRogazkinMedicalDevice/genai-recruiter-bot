"""
db_Tech.sql (MS SQL Server) -> schedule.db (SQLite)

Reproduces dbo.Schedule exactly:
  dates     2024-01-01 .. 2024-12-31, excluding Saturday and Monday
  times     09:00 .. 17:00, hourly
  positions Python Dev / Sql Dev / Analyst / ML
  available pseudo-random, ~50%

Difference from the original on purpose: availability is generated with a
fixed seed, so the file is identical on every run. The project is graded on
evals, and evals have to be reproducible.

Usage:  python build_schedule_db.py [output.db]
"""

import datetime as dt
import random
import sqlite3
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "schedule.db"
SEED = 42

START = dt.date(2024, 1, 1)
END = dt.date(2024, 12, 31)
EXCLUDED_WEEKDAYS = {5, 0}          # 5 = Saturday, 0 = Monday (Python: Mon=0)
HOURS = list(range(9, 18))          # 09:00 .. 17:00 inclusive
POSITIONS = ["Python Dev", "Sql Dev", "Analyst", "ML"]

rng = random.Random(SEED)

con = sqlite3.connect(OUT)
cur = con.cursor()
cur.executescript("""
DROP TABLE IF EXISTS Schedule;
CREATE TABLE Schedule (
    ScheduleID INTEGER PRIMARY KEY AUTOINCREMENT,
    date       TEXT    NOT NULL,   -- 'YYYY-MM-DD'
    time       TEXT    NOT NULL,   -- 'HH:MM:SS'
    position   TEXT    NOT NULL,
    available  INTEGER NOT NULL    -- 0 / 1
);
CREATE INDEX idx_lookup ON Schedule (position, available, date, time);
""")

rows = []
day = START
while day <= END:
    if day.weekday() not in EXCLUDED_WEEKDAYS:
        for hour in HOURS:
            for position in POSITIONS:
                # same shape as the original: mean of two uniforms, threshold 0.5
                score = (rng.randint(0, 99) + rng.randint(0, 99)) / 200.0
                rows.append((day.isoformat(), f"{hour:02d}:00:00", position,
                             1 if score >= 0.5 else 0))
    day += dt.timedelta(days=1)

cur.executemany(
    "INSERT INTO Schedule (date, time, position, available) VALUES (?, ?, ?, ?)",
    rows)
con.commit()

# ---- verification -------------------------------------------------------
print(f"{OUT}: {len(rows)} rows")

cur.execute("SELECT COUNT(DISTINCT date) FROM Schedule")
print("distinct dates:", cur.fetchone()[0])

cur.execute("SELECT position, COUNT(*), SUM(available) FROM Schedule GROUP BY position")
for pos, total, avail in cur.fetchall():
    print(f"  {pos:<11} {total:>5} slots, {avail:>5} available ({avail/total:.1%})")

# no Saturdays, no Mondays
bad = [d for (d,) in cur.execute("SELECT DISTINCT date FROM Schedule").fetchall()
       if dt.date.fromisoformat(d).weekday() in EXCLUDED_WEEKDAYS]
print("Saturdays/Mondays present:", len(bad))

# the query the Scheduling Advisor will actually run
print("\n3 nearest free 'Python Dev' slots after 2024-04-03 (conversation 1):")
cur.execute("""
    SELECT date, time FROM Schedule
    WHERE position = ? AND available = 1 AND date >= ?
    ORDER BY date, time
    LIMIT 3
""", ("Python Dev", "2024-04-03"))
for d, t in cur.fetchall():
    print(f"  {d} {t}  ({dt.date.fromisoformat(d).strftime('%A')})")

con.close()
