"""校验台账并按自然日识别缺测区间。"""

import re
from dataclasses import dataclass
from datetime import date
from math import isfinite

from .spec import SPEC


MIN_ROWS = 20
READING_TYPES = frozenset({"晨检", "同口径补测"})
ISO_DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


@dataclass(frozen=True)
class MissingRange:
    first: int
    last: int

    @property
    def days(self) -> int:
        return self.last - self.first + 1


def normalize_rows(rows: list[dict]) -> list[dict]:
    expected = set(SPEC["columns"])
    seen_ids = set()
    seen_dates = set()
    result = []

    for line, row in enumerate(rows, start=1):
        prefix = f"第 {line} 条数据"
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError(f"{prefix}列名须与模板完全一致：{', '.join(SPEC['columns'])}。")
        identifier = row["id"].strip() if isinstance(row["id"], str) else ""
        if not identifier:
            raise ValueError(f"{prefix}缺少台账编号。")
        if identifier in seen_ids:
            raise ValueError(f"{prefix}台账编号重复：{identifier}。")

        day_text = row["date"].strip() if isinstance(row["date"], str) else ""
        try:
            if not ISO_DAY.fullmatch(day_text):
                raise ValueError
            day = date.fromisoformat(day_text)
        except ValueError:
            raise ValueError(f"{prefix}日期须为有效的 YYYY-MM-DD。") from None
        if day in seen_dates:
            raise ValueError(f"{prefix}晨检日期重复：{day_text}；请在台账中确认当日唯一读数。")

        humidity_raw = row["humidity_pct"]
        try:
            if type(humidity_raw) not in (str, int, float):
                raise ValueError
            humidity = float(humidity_raw)
        except (ValueError, OverflowError):
            raise ValueError(f"{prefix}相对湿度须为数值（%RH）。") from None
        if not isfinite(humidity) or not 0 <= humidity <= 100:
            raise ValueError(f"{prefix}相对湿度须为 0 至 100 之间的有限数值（%RH）。")

        reading_type = row["reading_type"].strip() if isinstance(row["reading_type"], str) else ""
        if reading_type not in READING_TYPES:
            raise ValueError(f"{prefix}记录口径须为「晨检」或「同口径补测」。")

        seen_ids.add(identifier)
        seen_dates.add(day)
        result.append({"id": identifier, "date": day.isoformat(), "humidity_pct": humidity, "reading_type": reading_type})

    if len(result) < MIN_ROWS:
        raise ValueError(f"至少需要 {MIN_ROWS} 个不同日期的有效晨检记录。")
    return sorted(result, key=lambda row: row["date"])


def readings_by_day(rows: list[dict]) -> dict[int, float]:
    return {date.fromisoformat(row["date"]).toordinal(): row["humidity_pct"] for row in rows}


def missing_ranges(readings: dict[int, float]) -> list[MissingRange]:
    days = sorted(readings)
    return [
        MissingRange(previous + 1, next_day - 1)
        for previous, next_day in zip(days, days[1:])
        if next_day - previous > 1
    ]
