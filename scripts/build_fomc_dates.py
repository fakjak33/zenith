"""Build data/cas/fomc_dates.json — FOMC announcement dates 1994->2027.

Sources (compiled 2026-07-19):
  * 1994-2018: tobiasi/FOMCscrape FOMC_dates.csv (scraped from
    federalreserve.gov historical materials) — meeting start/end + scheduled
    flag; the END date is the announcement day (the Fed has announced the
    decision after the meeting since Feb 1994).
  * 2019-2020: federalreserve.gov/monetarypolicy/fomchistorical{2019,2020}.htm
    (hardcoded below; includes the March 2020 emergency actions, flagged
    unscheduled; the cancelled Mar 17-18 2020 meeting is excluded).
  * 2021-2027: federalreserve.gov/monetarypolicy/fomccalendars.htm
    (hardcoded below; all scheduled).

Cycle analysis (Cieslak-Morse-Vissing-Jorgensen) anchors on SCHEDULED
announcement days only. Refresh path: rerun this script after the Fed posts
the 2028 calendar.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zenith.config import CAS_DIR  # noqa: E402

CSV_URL = "https://raw.githubusercontent.com/tobiasi/FOMCscrape/master/FOMC_dates.csv"

# (end/announcement date, scheduled) — from the Fed's own pages, see docstring.
MANUAL = [
    # 2019
    ("2019-01-30", True), ("2019-03-20", True), ("2019-05-01", True),
    ("2019-06-19", True), ("2019-07-31", True), ("2019-09-18", True),
    ("2019-10-04", False), ("2019-10-30", True), ("2019-12-11", True),
    # 2020 (Mar 3 cut announced from the Mar 2 call; Mar 15 emergency meeting)
    ("2020-01-29", True), ("2020-03-03", False), ("2020-03-15", False),
    ("2020-04-29", True), ("2020-06-10", True), ("2020-07-29", True),
    ("2020-09-16", True), ("2020-11-05", True), ("2020-12-16", True),
    # 2021
    ("2021-01-27", True), ("2021-03-17", True), ("2021-04-28", True),
    ("2021-06-16", True), ("2021-07-28", True), ("2021-09-22", True),
    ("2021-11-03", True), ("2021-12-15", True),
    # 2022
    ("2022-01-26", True), ("2022-03-16", True), ("2022-05-04", True),
    ("2022-06-15", True), ("2022-07-27", True), ("2022-09-21", True),
    ("2022-11-02", True), ("2022-12-14", True),
    # 2023
    ("2023-02-01", True), ("2023-03-22", True), ("2023-05-03", True),
    ("2023-06-14", True), ("2023-07-26", True), ("2023-09-20", True),
    ("2023-11-01", True), ("2023-12-13", True),
    # 2024
    ("2024-01-31", True), ("2024-03-20", True), ("2024-05-01", True),
    ("2024-06-12", True), ("2024-07-31", True), ("2024-09-18", True),
    ("2024-11-07", True), ("2024-12-18", True),
    # 2025
    ("2025-01-29", True), ("2025-03-19", True), ("2025-05-07", True),
    ("2025-06-18", True), ("2025-07-30", True), ("2025-09-17", True),
    ("2025-10-29", True), ("2025-12-10", True),
    # 2026
    ("2026-01-28", True), ("2026-03-18", True), ("2026-04-29", True),
    ("2026-06-17", True), ("2026-07-29", True), ("2026-09-16", True),
    ("2026-10-28", True), ("2026-12-09", True),
    # 2027
    ("2027-01-27", True), ("2027-03-17", True), ("2027-04-28", True),
    ("2027-06-09", True), ("2027-07-28", True), ("2027-09-15", True),
    ("2027-10-27", True), ("2027-12-08", True),
]


def main() -> None:
    meetings: dict[str, bool] = {}
    r = requests.get(CSV_URL, timeout=30)
    r.raise_for_status()
    for row in csv.reader(io.StringIO(r.text)):
        if len(row) < 4 or not row[1].strip():
            continue
        try:
            d, m, y = row[2].strip().split("/")          # end date DD/MM/YYYY
            end = date(int(y), int(m), int(d))
            scheduled = row[3].strip() == "1"
        except (ValueError, IndexError):
            continue
        if end >= date(1994, 2, 1):                       # statements era
            meetings[end.isoformat()] = scheduled
    for iso, sched in MANUAL:
        meetings[iso] = sched
    # collapse duplicate rows for one meeting (e.g. the CSV lists 2003-09-15
    # AND 2003-09-16): consecutive scheduled dates -> keep the later (the
    # actual announcement day).
    keys = sorted(meetings)
    drop = {keys[i] for i in range(len(keys) - 1)
            if meetings[keys[i]] and meetings[keys[i + 1]]
            and (date.fromisoformat(keys[i + 1])
                 - date.fromisoformat(keys[i])).days == 1}
    rows = [{"date": k, "scheduled": v} for k, v in sorted(meetings.items())
            if k not in drop]
    out = {
        "as_of": date.today().isoformat(),
        "sources": ["tobiasi/FOMCscrape (1994-2018)",
                    "federalreserve.gov historical pages (2019-2020)",
                    "federalreserve.gov/monetarypolicy/fomccalendars.htm (2021-2027)"],
        "note": ("Announcement day = meeting end date. Scheduled meetings only "
                 "anchor CMVJ cycle time. Refresh when the 2028 calendar posts."),
        "meetings": rows,
    }
    path = CAS_DIR / "fomc_dates.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    n_sched = sum(1 for r_ in rows if r_["scheduled"])
    print(f"wrote {path} — {len(rows)} meetings ({n_sched} scheduled), "
          f"{rows[0]['date']} -> {rows[-1]['date']}")


if __name__ == "__main__":
    main()
