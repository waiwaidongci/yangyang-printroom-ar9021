"""北版房的合成晨检台账，包含已补测记录和未补测日期。"""

from datetime import date, timedelta
from math import sin, tau
from random import Random


FIRST_DAY = date(2026, 7, 20)
LAST_DAY = date(2026, 11, 24)
MISSING_DAYS = frozenset({
    date(2026, 8, 7), date(2026, 9, 13), date(2026, 9, 14),
    date(2026, 10, 28), date(2026, 10, 29), date(2026, 11, 12),
})
RECHECK_DAYS = frozenset({
    date(2026, 8, 8), date(2026, 9, 15), date(2026, 10, 30), date(2026, 11, 13),
})


def demo_rows() -> list[dict]:
    rng = Random(20261125)
    humidity = 54.0
    daily_change = 0.0
    rows = []

    for offset in range((LAST_DAY - FIRST_DAY).days + 1):
        day = FIRST_DAY + timedelta(days=offset)
        seasonal_level = 54 + 0.09 * offset + 4 * sin(tau * offset / 38)
        daily_change = (
            0.77 * daily_change + 0.09 * (seasonal_level - humidity)
            + 0.43 * sin(tau * offset / 17) + rng.gauss(0, 0.40)
        )
        humidity += daily_change
        if day in MISSING_DAYS:
            continue
        rows.append({
            "id": f"BF-{day:%Y%m%d}",
            "date": day.isoformat(),
            "humidity_pct": round(humidity, 2),
            "reading_type": "同口径补测" if day in RECHECK_DAYS else "晨检",
        })

    return rows
