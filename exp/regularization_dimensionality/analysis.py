from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_root: Path
    results_root: Path
    run_name: str
    n_samples: int
    n_terms: int
    m1: float
    m2: float
    noise_std: float
    seed: int
    epochs: int
    learning_rate: float
    lasso_lambda: float
    ridge_lambda: float
    zero_threshold: float
    history_stride: int


@dataclass
class TrialResult:
    name: str
    theta_init: list[float]
    theta_final: list[float]
    curve: list[float]
    loss_history: list[tuple[int, float, float]]
    data_loss: float
    objective: float
    nonzero_count: int
    dimension_estimate: int
    aggregate_linear: float
    aggregate_quadratic: float


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count == 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + step * index for index in range(count)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def squared_norm(values: list[float]) -> float:
    return sum(value * value for value in values)


def l1_norm(values: list[float]) -> float:
    return sum(abs(value) for value in values)


def generate_dataset(cfg: Config, rng: random.Random) -> tuple[list[float], list[float], list[float]]:
    x_values = linspace(-2.0, 2.0, cfg.n_samples)
    y_clean = [cfg.m1 * x + cfg.m2 * x * x for x in x_values]
    y_values = [value + rng.gauss(0.0, cfg.noise_std) for value in y_clean]
    return x_values, y_values, y_clean


def build_random_theta(n_terms: int, rng: random.Random, scale: float = 0.9) -> list[float]:
    return [rng.gauss(0.0, scale) for _ in range(2 * n_terms)]


def aggregate_coefficients(theta: list[float], n_terms: int) -> tuple[float, float]:
    return sum(theta[:n_terms]), sum(theta[n_terms:])


def prediction_series(theta: list[float], x_values: list[float], n_terms: int) -> list[float]:
    a_total, b_total = aggregate_coefficients(theta, n_terms)
    return [a_total * x + b_total * x * x for x in x_values]


def residuals(theta: list[float], x_values: list[float], y_values: list[float], n_terms: int) -> list[float]:
    predictions = prediction_series(theta, x_values, n_terms)
    return [pred - truth for pred, truth in zip(predictions, y_values)]


def data_loss(theta: list[float], x_values: list[float], y_values: list[float], n_terms: int) -> float:
    errors = residuals(theta, x_values, y_values, n_terms)
    return 0.5 * mean([value * value for value in errors])


def full_objective(
    theta: list[float],
    x_values: list[float],
    y_values: list[float],
    n_terms: int,
    mode: str,
    penalty: float,
) -> float:
    base = data_loss(theta, x_values, y_values, n_terms)
    if mode == "lasso":
        return base + penalty * l1_norm(theta)
    if mode == "ridge":
        return base + 0.5 * penalty * squared_norm(theta)
    return base


def data_gradient(theta: list[float], x_values: list[float], y_values: list[float], n_terms: int) -> list[float]:
    errors = residuals(theta, x_values, y_values, n_terms)
    mean_err_x = mean([err * x for err, x in zip(errors, x_values)])
    mean_err_x2 = mean([err * x * x for err, x in zip(errors, x_values)])
    return [mean_err_x] * n_terms + [mean_err_x2] * n_terms


def soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def count_nonzero(theta: list[float], threshold: float) -> int:
    return sum(1 for value in theta if abs(value) > threshold)


def train_model(
    name: str,
    mode: str,
    penalty: float,
    theta_init: list[float],
    x_values: list[float],
    y_values: list[float],
    cfg: Config,
) -> TrialResult:
    theta = theta_init[:]
    history: list[tuple[int, float, float]] = []

    for epoch in range(cfg.epochs + 1):
        if epoch % cfg.history_stride == 0 or epoch == cfg.epochs:
            history.append(
                (
                    epoch,
                    data_loss(theta, x_values, y_values, cfg.n_terms),
                    full_objective(theta, x_values, y_values, cfg.n_terms, mode, penalty),
                )
            )

        if epoch == cfg.epochs:
            break

        grad = data_gradient(theta, x_values, y_values, cfg.n_terms)
        if mode == "none":
            theta = [value - cfg.learning_rate * delta for value, delta in zip(theta, grad)]
        elif mode == "ridge":
            theta = [
                value - cfg.learning_rate * (delta + penalty * value)
                for value, delta in zip(theta, grad)
            ]
        else:
            updated = [value - cfg.learning_rate * delta for value, delta in zip(theta, grad)]
            theta = [soft_threshold(value, cfg.learning_rate * penalty) for value in updated]

    a_total, b_total = aggregate_coefficients(theta, cfg.n_terms)
    return TrialResult(
        name=name,
        theta_init=theta_init[:],
        theta_final=theta,
        curve=prediction_series(theta, x_values, cfg.n_terms),
        loss_history=history,
        data_loss=data_loss(theta, x_values, y_values, cfg.n_terms),
        objective=full_objective(theta, x_values, y_values, cfg.n_terms, mode, penalty),
        nonzero_count=count_nonzero(theta, cfg.zero_threshold),
        dimension_estimate=count_nonzero(theta, cfg.zero_threshold),
        aggregate_linear=a_total,
        aggregate_quadratic=b_total,
    )


def data_range(values: list[float], padding_fraction: float = 0.05) -> tuple[float, float]:
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    if span == 0.0:
        span = 1.0
    padding = span * padding_fraction
    return minimum - padding, maximum + padding
