"""Calendar family. Dates are known in advance, so these are causal by nature.

Warning shown in EXPLORE: testing many calendar buckets is the classic way to
find an edge that is pure luck (Sullivan, Timmermann & White 2001).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from .catalog import ind

SNOOP = ("Sullivan, Timmermann & White (2001) J. Econometrics: once you account for how many calendar rules were "
         "tried, most calendar effects are not significant.")
SEAS_VID = "MemLabs · Quant Trading for Dummies (day-of-week seasonality)"


@ind("dow", "Day of week", "calendar", {}, signal="state",
     hover="Monday = 0 … Friday = 4. Use it to ask: are returns different on some weekdays?",
     how="The weekday of each bar's date.",
     use="Explore next-bar returns by weekday; MemLabs' first strategy was 'long every Wednesday' on bitcoin.",
     evidence="French (1980) weekend effect, which has largely disappeared since. " + SNOOP, video=SEAS_VID)
def _dow(df):
    return pd.Series(df.index.dayofweek, index=df.index, dtype=float)


@ind("month", "Month of year", "calendar", {}, signal="state",
     hover="January = 1 … December = 12. For seasonal patterns like 'sell in May'.",
     how="The calendar month of each bar.", use="Explore returns by month; beware small samples (one per year).",
     evidence="Bouman & Jacobsen (2002) Halloween effect; Heston & Sadka (2008) same-month seasonality. " + SNOOP)
def _month(df):
    return pd.Series(df.index.month, index=df.index, dtype=float)


@ind("tom_flag", "Turn of month window", "calendar",
     {"pre": (1, 0, 5, 1, "Trading days before month end."), "post": (3, 0, 10, 1, "Trading days after.")},
     signal="level",
     hover="1 in the last few and first few trading days of each month, 0 otherwise.",
     how="Counts trading days from each month boundary (month boundaries are known in advance).",
     use="Equities have earned most of their return around the turn of the month.",
     evidence="Ariel (1987); Lakonishok & Smidt (1988); McConnell & Xu (2008).")
def _tom(df, pre, post):
    idx = df.index
    m = pd.Series(idx.to_period("M"), index=idx)
    pos = m.groupby(m).cumcount()                          # 0 = first trading day of the month
    # bars to month end, from a calendar of business days (known in advance, not from future bars)
    day = pd.Timedelta(days=1)
    ends = idx + pd.offsets.BMonthEnd(0)
    bme = pd.Series(np.busday_count((idx + day).values.astype("datetime64[D]"),
                                    (ends + day).values.astype("datetime64[D]")), index=idx)
    return ((pos < int(post)) | (bme < int(pre))).astype(float)


@ind("pre_holiday", "Pre-holiday", "calendar", {}, signal="level",
     hover="1 on the trading day before a US market holiday, 0 otherwise.",
     how="From the US federal holiday calendar (published in advance).",
     use="Pre-holiday days have historically had unusually strong equity returns.",
     evidence="Ariel (1990); Lakonishok & Smidt (1988).")
def _preh(df):
    idx = df.index
    if len(idx) == 0:
        return pd.Series(dtype=float)
    hol = USFederalHolidayCalendar().holidays(idx[0], idx[-1] + pd.Timedelta(days=10))
    nxt = idx + pd.offsets.BDay(1)
    return pd.Series(np.isin(nxt.normalize(), hol).astype(float), index=idx)


@ind("overnight", "Overnight return", "calendar", {}, signal="level",
     hover="Return from yesterday's close to today's open — the part of the move that happened while the "
           "market was shut.",
     how="open_t ÷ close_t−1 − 1.",
     use="In US equities nearly all of the long-run return has come overnight.",
     evidence="Cliff, Cooper & Gulen (2008); Lou, Polk & Skouras (2019) J. Fin. Econ.")
def _overnight(df):
    return df["open"] / df["close"].shift(1) - 1


@ind("intraday", "Intraday return", "calendar", {}, signal="level",
     hover="Return from today's open to today's close — the trading-hours part of the move.",
     how="close_t ÷ open_t − 1.", use="Compare with the overnight return; the two often behave very differently.",
     evidence="Lou, Polk & Skouras (2019).")
def _intraday(df):
    return df["close"] / df["open"] - 1
