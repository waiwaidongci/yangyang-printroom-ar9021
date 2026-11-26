"""通风排班口径的核验脚本（与业务代码分目录存放）。

覆盖需求逐条规则：
1. 每周最多通风两次，已交班先占次数；
2. 回执未到的日子暂不入样；
3. 逐日推算次晨湿度，选高湿天数最少、同分少开窗的排班；
4. 已交班决定锁定，重跑不改日期；
5. 晚到回执证实未开窗 -> 归入未通风样本重算，空出次数排给未交班日；
6. 并发提交不超额；
7. 拟合分组均值、限幅与零样本回退。
"""

import sys
import threading
from datetime import date, timedelta

sys.path.insert(0, "/workspace/backend")

from src.domain.ventilation import (  # noqa: E402
    ACTION_CLOSED,
    ACTION_OPEN,
    CLOSED,
    OPENED,
    PENDING,
    VentilationPlanner,
    VentilationRecord,
    build_samples,
    fit_groups,
    iso_of,
    monday_of_week,
    plan_ventilation_week,
    ventilation_report,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -> {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def ordinal(d):
    return d.toordinal()


# ---------------------------------------------------------------------------
# 构造一段历史：开窗日次晨明显下降，不开窗日次晨基本持平。
# ---------------------------------------------------------------------------
def build_history():
    readings = {}
    records = {}
    day = date(2026, 11, 2)  # 周一
    humidity = 70.0
    readings[ordinal(day)] = humidity
    # 20 个历史日，交替开窗/不开窗。
    for i in range(20):
        current = day + timedelta(days=i)
        record_day = current
        opened = (i % 2 == 0)
        records[ordinal(record_day)] = VentilationRecord(
            ordinal(record_day), OPENED if opened else CLOSED, locked=False
        )
        next_day = current + timedelta(days=1)
        change = -3.0 if opened else 0.5
        humidity = max(0.0, min(100.0, readings[ordinal(current)] + change))
        readings[ordinal(next_day)] = round(humidity, 4)
    return readings, records, day + timedelta(days=20)


readings, records, history_end = build_history()
samples = build_samples(readings, records)
check("历史样本数（有次日实测的通风日）", len(samples) == 20, f"got {len(samples)}")
fits = fit_groups(samples)
check("开窗组平均变化约 -3.0", abs(fits[True].mean_change + 3.0) < 1e-9, str(fits[True].mean_change))
check("不开窗组平均变化约 +0.5", abs(fits[False].mean_change - 0.5) < 1e-9, str(fits[False].mean_change))
check("两组各 10 个样本", fits[True].sample_count == 10 and fits[False].sample_count == 10)

# ---------------------------------------------------------------------------
# 回执未到的日子暂不入样
# ---------------------------------------------------------------------------
pending_day = ordinal(date(2026, 11, 4))
records_pending = dict(records)
records_pending[pending_day] = VentilationRecord(pending_day, PENDING, locked=False)
samples_pending = build_samples(readings, records_pending)
check("回执未到的日子不入样", len(samples_pending) == 19, f"got {len(samples_pending)}")
check("入样日不含 PENDING 日", all(s.day != pending_day for s in samples_pending))

# ---------------------------------------------------------------------------
# 周排班：目标周在历史之后，湿度锚值高于阈值，开窗能把次晨压到阈值以下
# ---------------------------------------------------------------------------
target_monday = monday_of_week(ordinal(history_end) + 2)
# 保证目标周晨检锚值偏高：直接设定上周日实测。
anchor_day = target_monday - 1
readings_week = dict(readings)
readings_week[anchor_day] = 72.0
plan = plan_ventilation_week(readings_week, records_pending, target_monday, threshold=65.0)
report = ventilation_report(plan)
check("报告含两张表", [t["key"] for t in report["tables"]] == ["ventilation_schedule", "ventilation_fit"])
check("周一起日正确", report["week_monday"] == iso_of(target_monday))
check("每周开窗不超过 2 次", plan.openings <= 2, f"openings={plan.openings}")
# 高湿阈值 65：不开窗次晨 ~72.5（高湿），开窗次晨 ~69（仍 >=65，高湿）。
# 调整阈值为 70，使开窗能把次晨降到阈值以下，从而显著降低高湿天数。
plan70 = plan_ventilation_week(readings_week, records_pending, target_monday, threshold=70.0)
high_by_action = {}
for item in plan70.days:
    high_by_action.setdefault(item.action, []).append(bool(item.high_humidity))
# 开窗日次晨 69 -> 非高湿；不开窗日 72.5/递推 -> 高湿。
open_days = [i for i in plan70.days if i.opens]
check("最优排班用满 2 次开窗", plan70.openings == 2, f"openings={plan70.openings}")
check("开窗日次晨均非高湿", open_days and all(not i.high_humidity for i in open_days),
      str([(iso_of(i.day), i.predicted_humidity) for i in open_days]))
# 开窗把周一、周二压到阈值以下；其后不开窗日逐日上抬，第 4 天起越过 70。
closed_days = [i for i in plan70.days if not i.opens]
check("高湿天数等于高湿的不开窗天数", plan70.high_humidity_days == sum(
    1 for i in closed_days if i.high_humidity), str(plan70.high_humidity_days))
check("前两天开窗后次晨低于阈值", [i.predicted_humidity for i in open_days] == [69.0, 66.0])
check("开窗日期排在本周更早的两个可排日",
      [i.day for i in open_days] == sorted(i.day for i in plan70.days if not i.locked)[:2],
      str([iso_of(i.day) for i in open_days]))

# ---------------------------------------------------------------------------
# 同分少开窗：阈值极低时每天都高湿，所有候选高湿天数相同 -> 选不开窗
# ---------------------------------------------------------------------------
plan_low = plan_ventilation_week(readings_week, records_pending, target_monday, threshold=0.0)
check("全部高湿同分时少开窗（0 次）", plan_low.openings == 0, f"openings={plan_low.openings}")
check("全部高湿天数为 7", plan_low.high_humidity_days == 7)

# 阈值 100 时没有高湿日，同样应 0 开窗。
plan_high = plan_ventilation_week(readings_week, records_pending, target_monday, threshold=100.0)
check("无高湿日时不开窗", plan_high.openings == 0 and plan_high.high_humidity_days == 0)

# ---------------------------------------------------------------------------
# 已交班先占次数，重跑不改换日期
# ---------------------------------------------------------------------------
planner = VentilationPlanner(readings_week, records_pending, threshold=70.0)
first_plan = planner.plan(target_monday)
# 人为锁定周三开窗并交班。
wednesday = target_monday + 2
planner.handover(wednesday, ACTION_OPEN)
second_plan = planner.plan(target_monday)
wed_item = next(i for i in second_plan.days if i.day == wednesday)
check("已交班开窗保持锁定", wed_item.locked and wed_item.opens)
check("已交班先占一次（occupied=1）", second_plan.occupied == 1, str(second_plan.occupied))
check("重排后总开窗仍不超过 2", second_plan.openings <= 2)
check("重排不改换已交班日期", next(i for i in second_plan.days if i.day == wednesday).opens)
# 其余开窗日不能落在周三。
other_open = [i.day for i in second_plan.days if i.opens and i.day != wednesday]
check("剩余开窗避开已交班日", wednesday not in other_open and len(other_open) <= 1)

# 连续三天交班开窗 -> 第三次应被拒绝（每周上限 2）。
planner2 = VentilationPlanner(readings_week, {}, threshold=70.0)
planner2.handover(target_monday, ACTION_OPEN)
planner2.handover(target_monday + 1, ACTION_OPEN)
over_quota = False
try:
    planner2.handover(target_monday + 2, ACTION_OPEN)
except ValueError:
    over_quota = True
check("第三次开窗交班被拒（每周两次）", over_quota)

# 不开窗交班不占开窗次数。
planner2.handover(target_monday + 2, ACTION_CLOSED)
check("不开窗交班可正常登记", planner2.records[target_monday + 2].status == CLOSED)

# ---------------------------------------------------------------------------
# 晚到回执：PENDING 证实未开窗 -> 归入未通风样本，释放次数并重新排给未交班日
# ---------------------------------------------------------------------------
planner3 = VentilationPlanner(dict(readings_week), dict(records_pending), threshold=70.0)
# 用高锚值保证开窗（次晨 -3）相对不开窗（次晨 +0.5）更能压低高湿天数。
planner3.add_reading(target_monday - 1, 72.0)
# 周一登记 PENDING（保守预占一次），并交班锁定为 PENDING 语义的占位。
planner3.records[target_monday] = VentilationRecord(target_monday, PENDING, locked=False)
plan_before = planner3.plan(target_monday)
check("PENDING 预占一次次数", plan_before.occupied == 1 and plan_before.openings <= 2)
# 给周一补上当日与次日实测，使回执到达后能形成未通风样本。
planner3.add_reading(target_monday, 71.0)
planner3.add_reading(target_monday + 1, 71.6)
planner3.receive_record(target_monday, CLOSED)
plan_after = planner3.plan(target_monday)
check("CLOSED 后释放预占次数（occupied=0）", plan_after.occupied == 0, str(plan_after.occupied))
new_samples = build_samples(planner3.readings, planner3.records)
monday_sample = next((s for s in new_samples if s.day == target_monday), None)
check("晚到未开窗记录归入未通风样本",
      monday_sample is not None and monday_sample.opened is False
      and abs(monday_sample.change - 0.6) < 1e-9, str(monday_sample))
# 空出的次数按新预报排给未交班日：最优仍会用满 2 次开窗且不排周一（周一已定 CLOSED）。
check("重排后用满两次开窗且避开周一",
      plan_after.openings == 2 and all(i.day != target_monday for i in plan_after.days if i.opens),
      str([(iso_of(i.day), i.action) for i in plan_after.days]))
check("周一动作固定为不开窗",
      next(i for i in plan_after.days if i.day == target_monday).action == ACTION_CLOSED)

# 晚到回执证实已开窗：次数应被占用。
planner3.receive_record(target_monday, OPENED)
plan_opened = planner3.plan(target_monday)
check("回执证实开窗后占用一次", plan_opened.occupied == 1 and plan_opened.openings <= 2)

# 已交班锁定的决定不能被晚到回执改写。
locked_day = target_monday + 4
planner3.handover(locked_day, ACTION_OPEN)
rewrite_blocked = False
try:
    planner3.receive_record(locked_day, CLOSED)
except ValueError:
    rewrite_blocked = True
check("晚到回执不能改写已交班决定", rewrite_blocked)
check("已交班开窗状态保持", planner3.records[locked_day].status == OPENED and planner3.records[locked_day].locked)
# 一致的回执（仍是已开窗）允许登记。
planner3.receive_record(locked_day, OPENED)
check("与锁定决定一致的回执可登记", planner3.records[locked_day].locked)

# ---------------------------------------------------------------------------
# 并发提交清单不超额
# ---------------------------------------------------------------------------
planner4 = VentilationPlanner(dict(readings_week), {}, threshold=70.0)
errors = []
results = []


def worker():
    try:
        plan = planner4.submit_plan(target_monday)
        results.append(plan)
    except ValueError as exc:  # 后提交者若名额已满则应收到明确错误而非超额
        errors.append(str(exc))


threads = [threading.Thread(target=worker) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
week_set = {target_monday + i for i in range(7)}
total_open = sum(
    1 for day in week_set
    if (rec := planner4.records.get(day)) is not None and rec.status == OPENED
)
check("两班并发提交后开窗总数不超过 2", total_open <= 2, f"total_open={total_open}")
check("并发调用都得到明确结果（成功或受控拒绝）", len(results) + len(errors) == 2,
      f"results={len(results)} errors={errors}")

# ---------------------------------------------------------------------------
# 限幅与零样本回退
# ---------------------------------------------------------------------------
clamp_readings = {target_monday - 1: 99.5}
# 不开窗组给一个 +3 的平均变化，使还原值超过 100（102.5 -> 限幅 100）。
# 样本日彼此隔开，避免相邻样本互相覆盖读数。
up_records = {}
sample_days = [target_monday - 60 + 2 * i for i in range(12)]
for d in sample_days:
    up_records[d] = VentilationRecord(d, CLOSED, locked=False)
    clamp_readings[d] = 60.0
    clamp_readings[d + 1] = 63.0
clamp_plan = plan_ventilation_week(clamp_readings, up_records, target_monday, threshold=65.0)
first = clamp_plan.days[0]
check("越界还原值被限幅到 100", first.predicted_humidity == 100.0 and first.raw_humidity > 100,
      str((first.raw_humidity, first.predicted_humidity)))
check("限幅方向标记为触发上界", first.boundary == "触发上界")
check("限幅告警存在", any("限幅" in w for w in clamp_plan.warnings))

empty_plan = plan_ventilation_week({target_monday - 1: 50.0}, {}, target_monday, threshold=65.0)
check("零样本时回退零变化并告警",
      empty_plan.fits[True].mean_change is None
      and any("暂无可用样本" in w for w in empty_plan.warnings))
check("零样本次晨等于锚值", empty_plan.days[0].predicted_humidity == 50.0)

# ---------------------------------------------------------------------------
# 摘要
# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} 项未通过：{FAILURES}")
    sys.exit(1)
print("全部核验通过。")
