"""一阶差分模型及相对湿度的物理约束。"""

from dataclasses import dataclass

import numpy as np


RIDGE_WEIGHT = 1.0


@dataclass(frozen=True)
class TrainingPair:
    day: int
    lags: tuple[float, ...]
    change: float


@dataclass(frozen=True)
class DifferenceAR:
    intercept: float
    weights: tuple[float, ...]

    def predict_change(self, lags: tuple[float, ...]) -> float:
        return self.intercept + sum(weight * lag for weight, lag in zip(self.weights, lags, strict=True))


@dataclass(frozen=True)
class Prediction:
    raw_humidity: float
    humidity: float
    applied_change: float
    boundary: str


def trailing_differences(readings: dict[int, float], anchor: int, order: int) -> tuple[float, ...] | None:
    window = [readings.get(anchor - offset) for offset in range(order + 1)]
    if any(value is None for value in window):
        return None
    return tuple(recent - older for recent, older in zip(window, window[1:]))


def training_pairs(readings: dict[int, float], order: int) -> list[TrainingPair]:
    pairs = []
    for day in sorted(readings):
        lags = trailing_differences(readings, day - 1, order)
        if lags is not None:
            pairs.append(TrainingPair(day, lags, readings[day] - readings[day - 1]))
    return pairs


def minimum_pairs(order: int) -> int:
    return max(12, 4 * (order + 1))


def fit_difference_ar(pairs: list[TrainingPair], order: int) -> DifferenceAR:
    design = np.asarray([(1.0, *pair.lags) for pair in pairs], dtype=float)
    target = np.asarray([pair.change for pair in pairs], dtype=float)
    penalty = np.eye(order + 1, dtype=float) * RIDGE_WEIGHT**0.5
    penalty[0, 0] = 0.0
    coefficients, *_ = np.linalg.lstsq(
        np.vstack((design, penalty)), np.concatenate((target, np.zeros(order + 1))), rcond=None,
    )
    return DifferenceAR(float(coefficients[0]), tuple(map(float, coefficients[1:])))


def predict_level(model: DifferenceAR, anchor: float, lags: tuple[float, ...]) -> Prediction:
    raw = anchor + model.predict_change(lags)
    humidity = max(0.0, min(100.0, raw))
    boundary = "触发下界" if raw < 0 else "触发上界" if raw > 100 else "未触发"
    return Prediction(raw, humidity, humidity - anchor, boundary)
