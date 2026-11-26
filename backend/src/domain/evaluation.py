"""滚动评分与从最新实测读数开始的递推预报。"""

from bisect import bisect_left
from dataclasses import dataclass

from .model import (
    DifferenceAR,
    Prediction,
    TrainingPair,
    fit_difference_ar,
    minimum_pairs,
    predict_level,
    trailing_differences,
)


@dataclass(frozen=True)
class BacktestDay:
    day: int
    sample_count: int
    status: str
    actual: float | None
    persistence: float | None = None
    prediction: Prediction | None = None


@dataclass(frozen=True)
class ForecastDay:
    day: int
    anchor: float | None
    prediction: Prediction | None


def rolling_backtest(
    readings: dict[int, float], pairs: list[TrainingPair], order: int, first: int, last: int,
) -> list[BacktestDay]:
    pair_days = [pair.day for pair in pairs]
    eligible = {pair.day: pair for pair in pairs}
    minimum = minimum_pairs(order)
    report = []
    for day in range(first, last + 1):
        train_count = bisect_left(pair_days, day)
        actual = readings.get(day)
        pair = eligible.get(day)
        if actual is None:
            report.append(BacktestDay(day, train_count, "缺少实测", None))
        elif pair is None:
            report.append(BacktestDay(day, train_count, "滞后链中断", actual))
        elif train_count < minimum:
            report.append(BacktestDay(day, train_count, "训练样本不足", actual))
        else:
            model = fit_difference_ar(pairs[:train_count], order)
            anchor = readings[day - 1]
            prediction = predict_level(model, anchor, pair.lags)
            report.append(BacktestDay(day, train_count, "已评分", actual, anchor, prediction))
    return report


def forecast_future(
    readings: dict[int, float], model: DifferenceAR, last: int, order: int, days: int,
) -> list[ForecastDay]:
    lags = trailing_differences(readings, last, order)
    if lags is None:
        return [ForecastDay(last + step, None, None) for step in range(1, days + 1)]

    report = []
    anchor = readings[last]
    for step in range(1, days + 1):
        prediction = predict_level(model, anchor, lags)
        report.append(ForecastDay(last + step, anchor, prediction))
        anchor = prediction.humidity
        lags = (prediction.applied_change, *lags[:-1])
    return report
