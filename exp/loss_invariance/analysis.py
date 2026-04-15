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
    direction_steps: int
    equivalent_samples: int


@dataclass
class Summary:
    a_fit: float
    b_fit: float
    base_loss: float
    active_rank: int
    nullity: int
    max_null_loss: float
    aggregate_drift: float


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count == 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + step * index for index in range(count)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def norm(values: list[float]) -> float:
    return math.sqrt(dot(values, values))


def normalize(values: list[float]) -> list[float]:
    length = norm(values)
    return values[:] if length == 0.0 else [value / length for value in values]


def solve_2x2(a: float, b: float, c: float, d: float, e: float, f: float) -> tuple[float, float]:
    determinant = a * d - b * c
    x = (e * d - b * f) / determinant
    y = (a * f - e * c) / determinant
    return x, y


def eigenpairs_symmetric_2x2(a: float, b: float, d: float) -> list[tuple[float, list[float]]]:
    trace = a + d
    delta = math.sqrt((a - d) ** 2 + 4.0 * b * b)
    eigenvalues = [(trace - delta) * 0.5, (trace + delta) * 0.5]
    eigenpairs = []
    for eigenvalue in eigenvalues:
        vector = [b, eigenvalue - a] if abs(b) > 1e-12 else [1.0, 0.0]
        eigenpairs.append((eigenvalue, normalize(vector)))
    return eigenpairs


def generate_dataset(cfg: Config, rng: random.Random) -> tuple[list[float], list[float], list[float]]:
    x_values = linspace(-2.0, 2.0, cfg.n_samples)
    y_clean = [cfg.m1 * x + cfg.m2 * x * x for x in x_values]
    y_values = [value + rng.gauss(0.0, cfg.noise_std) for value in y_clean]
    return x_values, y_values, y_clean


def fit_aggregate_coefficients(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
    sum_x2 = sum(x * x for x in x_values)
    sum_x3 = sum(x * x * x for x in x_values)
    sum_x4 = sum((x * x) ** 2 for x in x_values)
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    sum_x2y = sum((x * x) * y for x, y in zip(x_values, y_values))
    return solve_2x2(sum_x2, sum_x3, sum_x3, sum_x4, sum_xy, sum_x2y)


def build_theta(a_total: float, b_total: float, n_terms: int) -> list[float]:
    return [a_total / n_terms] * n_terms + [b_total / n_terms] * n_terms


def build_random_theta(n_terms: int, rng: random.Random, scale: float = 0.9) -> list[float]:
    return [rng.gauss(0.0, scale) for _ in range(2 * n_terms)]


def aggregate_coefficients(theta: list[float], n_terms: int) -> tuple[float, float]:
    return sum(theta[:n_terms]), sum(theta[n_terms:])


def model_prediction(x_value: float, a_total: float, b_total: float) -> float:
    return a_total * x_value + b_total * x_value * x_value


def prediction_series(theta: list[float], x_values: list[float], n_terms: int) -> list[float]:
    a_total, b_total = aggregate_coefficients(theta, n_terms)
    return [model_prediction(x, a_total, b_total) for x in x_values]


def loss_from_theta(theta: list[float], x_values: list[float], y_values: list[float], n_terms: int) -> float:
    a_total, b_total = aggregate_coefficients(theta, n_terms)
    residuals = [model_prediction(x, a_total, b_total) - y for x, y in zip(x_values, y_values)]
    return 0.5 * mean([value * value for value in residuals])


def shift_theta(theta: list[float], direction: list[float], step: float) -> list[float]:
    return [value + step * delta for value, delta in zip(theta, direction)]


def zero_sum_noise(count: int, rng: random.Random, scale: float) -> list[float]:
    values = [rng.gauss(0.0, scale) for _ in range(count)]
    offset = mean(values)
    return [value - offset for value in values]


def sample_equivalent_solutions(
    theta: list[float],
    n_terms: int,
    sample_count: int,
    rng: random.Random,
) -> list[list[float]]:
    samples = [theta[:]]
    for _ in range(sample_count - 1):
        a_noise = zero_sum_noise(n_terms, rng, 1.4)
        b_noise = zero_sum_noise(n_terms, rng, 1.4)
        sample = theta[:]
        for index in range(n_terms):
            sample[index] += a_noise[index]
            sample[n_terms + index] += b_noise[index]
        samples.append(sample)
    return samples


def active_hessian_eigenpairs(x_values: list[float], n_terms: int) -> list[tuple[float, list[float]]]:
    mean_x2 = mean([x * x for x in x_values])
    mean_x3 = mean([x * x * x for x in x_values])
    mean_x4 = mean([(x * x) ** 2 for x in x_values])
    return eigenpairs_symmetric_2x2(n_terms * mean_x2, n_terms * mean_x3, n_terms * mean_x4)


def expand_active_direction(compressed: list[float], n_terms: int) -> list[float]:
    return normalize([compressed[0]] * n_terms + [compressed[1]] * n_terms)


def build_null_direction(n_terms: int) -> list[float]:
    direction = [0.0] * (2 * n_terms)
    direction[0] = 1.0
    direction[1] = -1.0
    return normalize(direction)


def estimate_rank(eigenvalues: list[float]) -> int:
    scale = max(1.0, max(abs(value) for value in eigenvalues))
    threshold = scale * 1e-10
    return sum(1 for value in eigenvalues if value > threshold)


def data_range(values: list[float], padding_fraction: float = 0.05) -> tuple[float, float]:
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    if span == 0.0:
        span = 1.0
    padding = span * padding_fraction
    return minimum - padding, maximum + padding

