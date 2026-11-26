"""北版房通风排班。

业务口径（与 BUSINESS.md「通风排班口径」一致）：

* 北版房每周（ISO 周一至周日）最多通风两次，已交班的通风安排先占次数。
* 用晨检湿度与通风记录，分别拟合「开窗日」与「不开窗日」的次晨湿度变化；
  通风回执未到的日子暂不入样。
* 逐日推算次晨湿度（沿用上一晨检日的实测锚值，套用对应分组的平均变化，
  并按 0–100%RH 限幅）。
* 在不超过每周次数上限的候选排班里，选择本周高湿（次晨湿度达到阈值）
  天数最少的一份；同分（高湿天数相同）时选择开窗次数更少的一份；
  再同分则把开窗排向本周更早的日期，保证排班确定。
* 已交班的开窗/不开窗决定锁定，重跑也不改换日期。
* 晚到的通风记录若证实当日没有开窗，先把对应实测归入「未通风」样本再重算；
  由此空出的次数按新预报只排给未交班的日子。
* 两班同时提交清单也不得超额：次数在提交时原子占用。
"""

from dataclasses import dataclass, field
from datetime import date
from itertools import combinations
from statistics import fmean
from threading import Lock
from typing import Iterable


# 北版房每个自然周（ISO 周一至周日）允许的最大通风次数。
WEEKLY_OPENING_QUOTA = 2

# 通风记录的三种状态。
OPENED = "已开窗"
CLOSED = "未开窗"
PENDING = "回执未到"

# 每个计划日的排班动作。
ACTION_OPEN = "开窗"
ACTION_CLOSED = "不开窗"

# 次晨湿度的物理边界，与差分模型保持一致。
HUMIDITY_MIN = 0.0
HUMIDITY_MAX = 100.0


def to_ordinal(day) -> int:
    """接受 datetime.date、公历序日整数或 ISO 字符串，返回公历序日。"""
    if isinstance(day, bool):
        raise TypeError("日期须为 datetime.date、序日整数或 YYYY-MM-DD 字符串。")
    if isinstance(day, int):
        return day
    if isinstance(day, date):
        return day.toordinal()
    if isinstance(day, str):
        return date.fromisoformat(day.strip()).toordinal()
    raise TypeError("日期须为 datetime.date、序日整数或 YYYY-MM-DD 字符串。")


def iso_of(ordinal: int) -> str:
    return date.fromordinal(ordinal).isoformat()


def monday_of_week(ordinal: int) -> int:
    """返回该序日所在 ISO 周（周一为一周起点）的周一序日。"""
    return ordinal - date.fromordinal(ordinal).weekday()


def clamp_humidity(value: float) -> tuple[float, str]:
    """把还原湿度限到 0–100%RH，返回限幅值与触发方向。"""
    if value < HUMIDITY_MIN:
        return HUMIDITY_MIN, "触发下界"
    if value > HUMIDITY_MAX:
        return HUMIDITY_MAX, "触发上界"
    return float(value), "未触发"


@dataclass(frozen=True)
class VentilationRecord:
    """单日通风记录。

    status 取 OPENED / CLOSED / PENDING：
    PENDING 表示通风回执未到，当日暂不入拟合样本，也暂占本周次数以防超额。
    locked 为 True 表示该日决定已交班，重跑不得改换日期与动作。
    """

    day: int
    status: str
    locked: bool = False

    def __post_init__(self):
        if self.status not in (OPENED, CLOSED, PENDING):
            raise ValueError("通风记录状态须为「已开窗」「未开窗」或「回执未到」。")


@dataclass(frozen=True)
class HumiditySample:
    """一条拟合样本：day 的通风动作对应次晨（day+1）的湿度变化。"""

    day: int
    opened: bool
    change: float


@dataclass(frozen=True)
class GroupFit:
    """一组（开窗 / 不开窗）历史样本的拟合结果。"""

    opened: bool
    sample_count: int
    mean_change: float | None

    @property
    def label(self) -> str:
        return "开窗" if self.opened else "不开窗"


def build_samples(
    readings: dict[int, float],
    records: dict[int, VentilationRecord],
) -> list[HumiditySample]:
    """从晨检湿度与通风记录构造拟合样本。

    样本定义：通风日 d 有次日晨检实测 h[d+1] 且当日有晨检锚值 h[d]，
    变化量 change = h[d+1] - h[d]。回执未到（PENDING）的日子暂不入样；
    缺次日或当日实测的日子无法构造样本，同样跳过。
    """
    samples: list[HumiditySample] = []
    for day in sorted(records):
        record = records[day]
        if record.status == PENDING:
            continue
        anchor = readings.get(day)
        next_reading = readings.get(day + 1)
        if anchor is None or next_reading is None:
            continue
        samples.append(HumiditySample(day, record.status == OPENED, next_reading - anchor))
    return samples


def fit_groups(samples: Iterable[HumiditySample]) -> dict[bool, GroupFit]:
    """分别拟合开窗组与不开窗组的平均次晨变化。

    分组没有任何样本时 mean_change 为 None，推算时回退为零变化并在告警中说明。
    """
    grouped: dict[bool, list[float]] = {True: [], False: []}
    for sample in samples:
        grouped[sample.opened].append(sample.change)
    return {
        opened: GroupFit(
            opened=opened,
            sample_count=len(changes),
            mean_change=(fmean(changes) if changes else None),
        )
        for opened, changes in grouped.items()
    }


def applied_mean_change(fit: GroupFit) -> tuple[float, bool]:
    """返回用于推算的平均变化；分组无样本时回退为零变化。"""
    if fit.mean_change is None:
        return 0.0, True
    return fit.mean_change, False


@dataclass(frozen=True)
class PlannedDay:
    """周计划中的一个排班日。"""

    day: int
    action: str
    locked: bool
    status: str
    anchor_humidity: float | None
    raw_humidity: float | None
    predicted_humidity: float | None
    boundary: str | None
    high_humidity: bool | None
    note: str = ""

    @property
    def opens(self) -> bool:
        return self.action == ACTION_OPEN


@dataclass(frozen=True)
class VentilationPlan:
    """一次周排班的完整结果。"""

    week_monday: int
    days: list[PlannedDay]
    fits: dict[bool, GroupFit]
    threshold: float
    quota: int
    openings: int
    occupied: int
    high_humidity_days: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _effective_status(record: VentilationRecord | None) -> str | None:
    """该日已经确定的通风状态；未交班且无记录的待排日返回 None。"""
    if record is None:
        return None
    if record.locked or record.status != PENDING:
        return record.status
    return None


def _used_openings(records: dict[int, VentilationRecord], week_days: set[int]) -> int:
    """本周已经占用的通风次数。

    已交班的开窗先占次数；回执未到的日子按可能开窗保守预占，
    待回执证实未开窗时再由 receive_record 释放。
    """
    used = 0
    for day in week_days:
        record = records.get(day)
        if record is None:
            continue
        if record.status == OPENED or record.status == PENDING:
            used += 1
    return used


def _simulate_week(
    choices: dict[int, str],
    week_ordinals: list[int],
    readings: dict[int, float],
    records: dict[int, VentilationRecord],
    fits: dict[bool, GroupFit],
    threshold: float,
) -> list[PlannedDay]:
    """按给定的逐日开窗/不开窗选择，推算本周每个计划日的次晨湿度。"""
    # 从本周首日之前最近的一次晨检实测取初始锚值（允许中间存在缺测）。
    anchor = next(
        (readings[past] for past in range(week_ordinals[0] - 1, min(readings) - 1, -1)
         if past in readings),
        None,
    ) if readings else None
    planned: list[PlannedDay] = []
    for day in week_ordinals:
        record = records.get(day)
        action = choices[day]
        locked = bool(record and record.locked)
        status = record.status if record is not None else "待排班"

        note = ""
        if readings.get(day) is not None:
            # 当日晨检实测已存在，直接作为最新锚值，避免递推误差累积。
            anchor = readings[day]
        if anchor is None:
            planned.append(PlannedDay(
                day, action, locked, status, None, None, None, None, None,
                "缺少可用晨检锚值，暂不推算次晨湿度。",
            ))
            continue

        fit = fits[action == ACTION_OPEN]
        mean_change, fell_back = applied_mean_change(fit)
        if fell_back:
            note = f"{fit.label}组暂无样本，按零变化推算。"
        raw = anchor + mean_change
        humidity, boundary = clamp_humidity(raw)
        planned.append(PlannedDay(
            day=day,
            action=action,
            locked=locked,
            status=status,
            anchor_humidity=anchor,
            raw_humidity=raw,
            predicted_humidity=humidity,
            boundary=boundary,
            high_humidity=humidity >= threshold,
            note=note,
        ))

        # 次日锚值：优先次日晨检实测，否则沿用本次限幅预报。
        next_reading = readings.get(day + 1)
        anchor = next_reading if next_reading is not None else humidity
    return planned


def plan_ventilation_week(
    readings: dict[int, float],
    records: dict[int, VentilationRecord],
    week: date | int | str,
    threshold: float,
    quota: int = WEEKLY_OPENING_QUOTA,
) -> VentilationPlan:
    """生成指定 ISO 周的通风排班清单。

    只规划本周内、且尚无次日晨检实测的日期；已有次日实测的日期仍参与
    次数占用与锚值递推，但其动作由通风记录/交班状态固定。

    选择顺序：高湿天数最少 → 开窗次数最少 → 开窗日期尽量靠前（确定性兜底）。
    """
    week_monday = monday_of_week(to_ordinal(week))
    week_ordinals = [week_monday + offset for offset in range(7)]
    week_set = set(week_ordinals)

    samples = build_samples(readings, records)
    fits = fit_groups(samples)

    # 每个工作日的可选动作：已确定（有记录，或已交班）的日子固定；其余待排。
    choices_per_day: dict[int, tuple[str, ...]] = {}
    for day in week_ordinals:
        record = records.get(day)
        effective = _effective_status(record)
        if effective is None:
            choices_per_day[day] = (ACTION_CLOSED, ACTION_OPEN)
        else:
            choices_per_day[day] = (ACTION_OPEN if effective == OPENED else ACTION_CLOSED,)

    occupied = _used_openings(records, week_set)
    if occupied > quota:
        raise ValueError(
            f"本周已占用通风 {occupied} 次，超过每周上限 {quota} 次，请先核对交班与回执。"
        )

    # 待排日里仍可开窗的名额（已交班开窗与未到回执先占次数）。
    freeable = quota - occupied
    openable_days = [day for day, options in choices_per_day.items() if ACTION_OPEN in options]

    best: list[PlannedDay] | None = None
    best_key = None
    # 枚举待排日的开窗子集（名额有限，组合数很小）。
    for open_count in range(min(freeable, len(openable_days)) + 1):
        for open_subset in _combinations(openable_days, open_count):
            chosen = set(open_subset)
            choices = {
                day: (ACTION_OPEN if day in chosen else options[0])
                if ACTION_OPEN in options else options[0]
                for day, options in choices_per_day.items()
            }
            simulated = _simulate_week(choices, week_ordinals, readings, records, fits, threshold)
            high_days = sum(1 for item in simulated if item.high_humidity)
            openings = sum(1 for item in simulated if item.opens)
            # 确定性兜底：开窗日期按升序组成元组参与比较。
            ordering = tuple(item.day for item in simulated if item.opens)
            key = (high_days, openings, ordering)
            if best_key is None or key < best_key:
                best_key = key
                best = simulated

    assert best is not None and best_key is not None  # 至少存在「全不开窗」一个候选
    warnings: list[str] = []
    for fit in fits.values():
        if fit.sample_count == 0:
            warnings.append(f"{fit.label}组暂无可用样本，相关日期按零变化推算。")
    if any(item.anchor_humidity is None for item in best):
        warnings.append("部分日期缺少晨检锚值，次晨湿度留空，请先补齐晨检记录。")
    if any(item.boundary != "未触发" for item in best if item.boundary is not None):
        warnings.append("有日期的次晨湿度触发 0–100%RH 限幅，请核对还原原值。")

    return VentilationPlan(
        week_monday=week_monday,
        days=best,
        fits=fits,
        threshold=threshold,
        quota=quota,
        openings=best_key[1],
        occupied=occupied,
        high_humidity_days=best_key[0],
        warnings=tuple(warnings),
    )


def _combinations(items: list[int], size: int):
    """按序返回 size 个元素的全部组合（确定性枚举）。"""
    return list(combinations(items, size))


def ventilation_report(plan: VentilationPlan) -> dict:
    """把周排班整理成可导出的晨报表格，口径与 reporting.report_table 一致。"""
    rows = [
        {
            "date": iso_of(item.day),
            "action": item.action,
            "locked": "是" if item.locked else "否",
            "record_status": item.status,
            "anchor_humidity": _rounded(item.anchor_humidity),
            "raw_humidity": _rounded(item.raw_humidity),
            "predicted_humidity": _rounded(item.predicted_humidity),
            "boundary": item.boundary,
            "high_humidity": ("是" if item.high_humidity else "否") if item.high_humidity is not None else "",
            "note": item.note,
        }
        for item in plan.days
    ]
    fit_rows = [
        {
            "group": fit.label,
            "sample_count": fit.sample_count,
            "mean_change": _rounded(fit.mean_change),
        }
        for fit in (plan.fits[True], plan.fits[False])
    ]
    schedule = {
        "key": "ventilation_schedule",
        "title": "本周通风排班清单",
        "columns": [
            "date", "action", "locked", "record_status", "anchor_humidity",
            "raw_humidity", "predicted_humidity", "boundary", "high_humidity", "note",
        ],
        "labels": {
            "date": "日期",
            "action": "通风安排",
            "locked": "已交班锁定",
            "record_status": "通风记录状态",
            "anchor_humidity": "晨检锚值（%RH）",
            "raw_humidity": "还原原值（%RH）",
            "predicted_humidity": "次晨限幅湿度（%RH）",
            "boundary": "物理边界",
            "high_humidity": "是否高湿日",
            "note": "备注",
        },
        "rows": rows,
    }
    fit_table = {
        "key": "ventilation_fit",
        "title": "开窗与不开窗次晨变化拟合",
        "columns": ["group", "sample_count", "mean_change"],
        "labels": {
            "group": "分组",
            "sample_count": "样本数",
            "mean_change": "平均次晨变化（百分点）",
        },
        "rows": fit_rows,
    }
    return {
        "week_monday": iso_of(plan.week_monday),
        "quota": plan.quota,
        "occupied": plan.occupied,
        "openings": plan.openings,
        "high_humidity_days": plan.high_humidity_days,
        "threshold": plan.threshold,
        "warnings": list(plan.warnings),
        "tables": [schedule, fit_table],
    }


def _rounded(value):
    return None if value is None else round(float(value), 3)


class VentilationPlanner:
    """带交班锁定与并发保护的通风排班状态机。

     readings 与 records 按序日保存；submit/handover/receive_record 都在
    同一把锁内串行处理，保证两班同时提交清单也不会超过每周次数上限。
    重跑（plan）只重新推算未交班的日子，已交班的决定保持不变。
    """

    def __init__(
        self,
        readings: dict[int, float] | None = None,
        records: dict[int, VentilationRecord] | None = None,
        threshold: float = 65.0,
        quota: int = WEEKLY_OPENING_QUOTA,
    ):
        self._lock = Lock()
        self.readings: dict[int, float] = dict(readings or {})
        self.records: dict[int, VentilationRecord] = dict(records or {})
        self.threshold = threshold
        self.quota = quota

    @classmethod
    def from_maps(
        cls,
        readings: dict,
        records: dict,
        threshold: float = 65.0,
        quota: int = WEEKLY_OPENING_QUOTA,
    ) -> "VentilationPlanner":
        """便于调用方直接传 date/字符串键的晨检与通风记录。"""
        normalized_readings = {to_ordinal(day): float(value) for day, value in readings.items()}
        normalized_records = {}
        for day, record in records.items():
            ordinal = to_ordinal(day)
            normalized_records[ordinal] = (
                record if isinstance(record, VentilationRecord)
                else VentilationRecord(ordinal, record[0], bool(record[1]))
            )
        return cls(normalized_readings, normalized_records, threshold, quota)

    def add_reading(self, day, humidity: float) -> None:
        with self._lock:
            self.readings[to_ordinal(day)] = float(humidity)

    def plan(self, week: date | int | str) -> VentilationPlan:
        """按当前记录重算周排班；已交班日期的动作不会被改换。"""
        with self._lock:
            return plan_ventilation_week(
                self.readings, self.records, week, self.threshold, self.quota,
            )

    def handover(self, day, action: str) -> VentilationRecord:
        """开门班交班：锁定某日的开窗/不开窗决定，先占本周次数。"""
        if action not in (ACTION_OPEN, ACTION_CLOSED):
            raise ValueError("交班动作须为「开窗」或「不开窗」。")
        ordinal = to_ordinal(day)
        with self._lock:
            week_set = {monday_of_week(ordinal) + offset for offset in range(7)}
            record = self.records.get(ordinal)
            already_open = record is not None and record.status == OPENED
            if action == ACTION_OPEN and not already_open:
                used = _used_openings(self.records, week_set)
                if used >= self.quota:
                    raise ValueError(
                        f"本周通风次数已达上限 {self.quota} 次，不能再安排开窗。"
                    )
            status = OPENED if action == ACTION_OPEN else CLOSED
            locked = VentilationRecord(ordinal, status, locked=True)
            self.records[ordinal] = locked
            return locked

    def submit_plan(self, week: date | int | str) -> VentilationPlan:
        """提交一份周清单：在同一把锁内排班并把未锁定的开窗决定交班占用。

        两个班次并发调用时由锁串行化，后提交者看到前者已占用的次数，
        因此不会超额。
        """
        target_week = monday_of_week(to_ordinal(week))
        with self._lock:
            plan = plan_ventilation_week(
                self.readings, self.records, target_week, self.threshold, self.quota,
            )
            for item in plan.days:
                if item.locked:
                    continue
                record = self.records.get(item.day)
                if record is not None and record.status != PENDING:
                    continue
                status = OPENED if item.opens else CLOSED
                self.records[item.day] = VentilationRecord(item.day, status, locked=True)
            return plan_ventilation_week(
                self.readings, self.records, target_week, self.threshold, self.quota,
            )

    def receive_record(self, day, status: str) -> VentilationRecord:
        """登记晚到的通风回执。

        若证实当日没有开窗（CLOSED），对应实测会在下一次 plan/重算时
        经 build_samples 归入未通风样本；若此前以 PENDING 预占了次数，
        登记 CLOSED 即释放该次数，空出的名额按新预报排给未交班的日子。
        """
        if status not in (OPENED, CLOSED, PENDING):
            raise ValueError("通风回执状态须为「已开窗」「未开窗」或「回执未到」。")
        ordinal = to_ordinal(day)
        with self._lock:
            previous = self.records.get(ordinal)
            if previous is not None and previous.locked and previous.status != status:
                raise ValueError(
                    f"{iso_of(ordinal)} 的通风决定已交班锁定，不能被晚到回执改写；"
                    "如需更正请先走交班复核流程。"
                )
            record = VentilationRecord(ordinal, status, locked=bool(previous and previous.locked))
            self.records[ordinal] = record
            return record
