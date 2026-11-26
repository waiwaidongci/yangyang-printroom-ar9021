"""木版版房晨检湿度预报的领域入口。"""

from collections import Counter
from datetime import date

from .evaluation import forecast_future, rolling_backtest
from .fixtures import demo_rows
from .inputs import missing_ranges, normalize_rows, readings_by_day
from .model import fit_difference_ar, minimum_pairs, training_pairs
from .reporting import backtest_report, forecast_report, gaps_report, iso_day, model_report
from .spec import SPEC, normalize_settings


__all__ = ["SPEC", "demo_rows", "normalize_rows", "analyze"]


def analyze(rows: list[dict], settings: dict) -> dict:
    configuration = normalize_settings(settings)
    order = configuration["lag_order"]
    horizon = configuration["forecast_days"]
    lookback = configuration["backtest_days"]
    readings = readings_by_day(normalize_rows(rows))
    first, last = min(readings), max(readings)
    if last + horizon > date.max.toordinal():
        raise ValueError("最新晨检日期太接近可表示的日期上限，请缩短预报天数或核对台账日期。")

    pairs = training_pairs(readings, order)
    minimum = minimum_pairs(order)
    if len(pairs) < minimum:
        raise ValueError(
            f"差分 AR({order}) 至少需要 {minimum} 个连续日历训练样本，当前仅有 {len(pairs)} 个。"
            "请降低滞后阶数，或补齐可核实的晨检记录。"
        )

    first_backtest = max(first, last - lookback + 1)
    gaps = missing_ranges(readings)
    backtest = rolling_backtest(readings, pairs, order, first_backtest, last)
    model = fit_difference_ar(pairs, order)
    forecasts = forecast_future(readings, model, last, order, horizon)
    missing_days = sum(gap.days for gap in gaps)
    metrics, backtest_table = backtest_report(backtest, missing_days)

    scored = metrics["回测评分天数"]
    summary = (
        f"台账截至 {iso_day(last)}，共 {len(readings)} 条实测。"
        f"差分 AR({order}) 在 {iso_day(first_backtest)} 至 {iso_day(last)} 滚动回测，"
        f"与沿用前一日实测的基线在相同的 {scored} 天评分。"
    )
    warnings = []
    if first_backtest > last - lookback + 1:
        warnings.append(f"台账跨度短于设置，实际回测覆盖 {len(backtest)} 个日历日。")
    if missing_days:
        warnings.append(f"台账内有 {len(gaps)} 段缺测，合计 {missing_days} 天。缺测会截断差分滞后链。")
    skipped = Counter(result.status for result in backtest if result.prediction is None)
    if skipped:
        counts = '、'.join(f"{reason} {count} 天" for reason, count in skipped.items())
        warnings.append(f"回测跳过 {sum(skipped.values())} 天（{counts}），两种 MAE 均只使用已评分日期。")
    if not scored:
        warnings.append("当前窗口没有可比较的日期，请扩大回测窗口、降低阶数或补充可核实记录。")
    if forecasts[0].prediction is None:
        summary += f"末段缺少连续 {order + 1} 天实测，已列出待补测的未来晨检日期。"
        warnings.append(f"末次实测无法构造 {order} 阶连续差分，未来预报暂缺。")
    else:
        summary += f"从末次实测递推 {horizon} 天，达到或超过 {configuration['wet_threshold']}%RH 的日期优先晨检。"
    bounded_backtest = sum(
        result.prediction is not None and result.prediction.boundary != "未触发"
        for result in backtest
    )
    bounded_future = sum(
        result.prediction is not None and result.prediction.boundary != "未触发"
        for result in forecasts
    )
    if bounded_backtest or bounded_future:
        warnings.append(f"有 {bounded_backtest} 天回测、{bounded_future} 天预报触发 0–100%RH 限幅，请在表中核对还原原值。")

    return {
        "summary": summary,
        "metrics": metrics,
        "tables": [
            forecast_report(forecasts, configuration["wet_threshold"]),
            backtest_table,
            gaps_report(gaps, last, order, first_backtest),
            model_report(model, last, len(pairs)),
        ],
        "warnings": warnings,
    }
