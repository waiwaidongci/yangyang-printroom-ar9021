"""端到端演示：高湿季节一周的通风排班、交班、晚到回执与重排。"""

import sys
from datetime import date, timedelta

sys.path.insert(0, "/workspace/backend")

from src.domain.ventilation import (  # noqa: E402
    CLOSED,
    OPENED,
    PENDING,
    VentilationPlanner,
    VentilationRecord,
    iso_of,
    ventilation_report,
)

# ---- 1. 构造连续 21 天历史：开窗次晨 -8，不开窗次晨 +0.8 ----
base = date(2026, 10, 5)  # 周一，与目标周之间留出空档，避免覆盖目标周锚值
readings = {base.toordinal(): 70.0}
records = {}
humidity = 70.0
for i in range(21):
    day = base + timedelta(days=i)
    opened = (i % 2 == 0)
    records[day.toordinal()] = VentilationRecord(day.toordinal(), OPENED if opened else CLOSED)
    humidity = max(0.0, min(100.0, humidity + (-8.0 if opened else 0.8)))
    readings[(day + timedelta(days=1)).toordinal()] = round(humidity, 3)

# 目标周周一；上周日实测 72，制造高湿压力。
target = date(2026, 11, 23)
anchor_day = target - timedelta(days=1)
readings[anchor_day.toordinal()] = 72.0

planner = VentilationPlanner(readings, records, threshold=65.0)

# ---- 2. 周一回执未到（未交班），先保守预占一次 ----
planner.records[target.toordinal()] = VentilationRecord(target.toordinal(), PENDING)
first = planner.plan(target)
print("=== 初次排班（周一回执未到，先占 1 次）===")
for item in first.days:
    flag = "高湿" if item.high_humidity else ""
    print(f"{iso_of(item.day)}  {item.action:<3}  预报 {item.predicted_humidity:<7.3f} {flag}")
print("occupied / openings / 高湿天数:", first.occupied, first.openings, first.high_humidity_days)
assert first.occupied == 1 and first.openings <= 2
assert first.days[0].anchor_humidity == 72.0

# ---- 3. 晚到回执（在周一交班之前到达）：证实未开窗 ----
# 归入未通风样本、释放预占次数，空出名额按新预报排给未交班日。
planner.add_reading(target, 72.0)
planner.add_reading(target + timedelta(days=1), 72.8)
planner.receive_record(target, CLOSED)
second = planner.plan(target)
print("\n=== 回执证实周一未开窗后重排（次数释放，名额排给未交班日）===")
for item in second.days:
    flag = "高湿" if item.high_humidity else ""
    print(f"{iso_of(item.day)}  {item.action:<3}  预报 {item.predicted_humidity:<7.3f} {flag}")
print("occupied / openings / 高湿天数:", second.occupied, second.openings, second.high_humidity_days)
assert second.occupied == 0, "CLOSED 应释放预占次数"
assert second.openings <= 2
monday = next(item for item in second.days if item.day == target.toordinal())
assert monday.action == "不开窗"

# 未通风样本已把周一纳入。
from src.domain.ventilation import build_samples  # noqa: E402

monday_sample = next(s for s in build_samples(planner.readings, planner.records)
                     if s.day == target.toordinal())
assert monday_sample.opened is False and abs(monday_sample.change - 0.8) < 1e-9

# ---- 4. 开门班提交清单：未锁定日一次临界区排班并锁定，仍不超额 ----
submitted = planner.submit_plan(target)
print("\n=== 提交清单并交班锁定 ===")
for item in submitted.days:
    print(f"{iso_of(item.day)}  {item.action:<3}  locked={item.locked}")
# 周一此前已有 CLOSED 记录（晚到回执），submit 不重复加锁；其余 6 天应被锁定。
assert all(item.locked for item in submitted.days[1:])
assert submitted.days[0].action == "不开窗"
assert submitted.openings <= 2

# 已交班锁定后，晚到的冲突回执必须被拒绝。
blocked = False
locked_open_day = next(item.day for item in submitted.days if item.opens)
try:
    planner.receive_record(locked_open_day, CLOSED)
except ValueError:
    blocked = True
assert blocked
print("\n锁定日冲突回执被拒绝：是")

report = ventilation_report(submitted)
print("\n报告周一起日:", report["week_monday"], "| 表:", [t["key"] for t in report["tables"]])
print("拟合:", [(r["group"], r["sample_count"], r["mean_change"])
                for r in report["tables"][1]["rows"]])
print("\n端到端演示通过。")
