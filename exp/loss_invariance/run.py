from __future__ import annotations

import argparse
import csv
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

from analysis import (
    Config,
    Summary,
    active_hessian_eigenpairs,
    aggregate_coefficients,
    build_random_theta,
    build_null_direction,
    build_theta,
    estimate_rank,
    expand_active_direction,
    fit_aggregate_coefficients,
    generate_dataset,
    linspace,
    loss_from_theta,
    model_prediction,
    prediction_series,
    sample_equivalent_solutions,
    shift_theta,
)
from png_dashboard import render_process_figure, write_png


def info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


@contextmanager
def timed_step(step_name: str, timings: list[dict[str, float]]) -> None:
    info(f"Starting: {step_name}")
    started = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - started
        timings.append({"step": step_name, "seconds": duration})
        info(f"Finished: {step_name} in {format_seconds(duration)}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic quadratic regression problem, analyze the "
            "loss-invariant parameter directions, and write results under results/."
        )
    )
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--run_name", default="loss_invariance_analysis")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_terms", type=int, default=6)
    parser.add_argument("--m1", type=float, default=1.5)
    parser.add_argument("--m2", type=float, default=-0.8)
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--direction_steps", type=int, default=161)
    parser.add_argument("--equivalent_samples", type=int, default=60)
    args = parser.parse_args()
    data_root = args.data_root if args.data_root is not None else Path("data")
    return Config(
        data_root=data_root,
        results_root=args.results_root,
        run_name=args.run_name,
        n_samples=args.n_samples,
        n_terms=args.n_terms,
        m1=args.m1,
        m2=args.m2,
        noise_std=args.noise_std,
        seed=args.seed,
        direction_steps=args.direction_steps,
        equivalent_samples=args.equivalent_samples,
    )


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_run_readme(path: Path, cfg: Config, summary: Summary, timings: list[dict[str, float]]) -> None:
    lines = [
        f"# {cfg.run_name}",
        "",
        "## What Was Run",
        "",
        f"`uv run python exp/loss_invariance/run.py --run_name {cfg.run_name} --n_samples {cfg.n_samples} --n_terms {cfg.n_terms} --m1 {cfg.m1} --m2 {cfg.m2} --noise_std {cfg.noise_std} --seed {cfg.seed}`",
        "",
        "## Files Present",
        "",
        "- `loss_invariance_process.png`: step-by-step figure explaining the method.",
        "- `summary.json`: key scalar outputs and file locations.",
        "- `timings.csv`: wall-clock timing for each major stage.",
        "- `equivalent_solutions.csv`: parameter settings with the same loss.",
        "- `direction_scan.csv`: loss sampled along one active and one null direction.",
        "- `README.md`: this run summary.",
        "",
        "## Conclusions",
        "",
        f"- The fitted aggregate coefficients are `A={summary.a_fit:.6f}` and `B={summary.b_fit:.6f}`.",
        f"- The Hessian has `{summary.active_rank}` active dimensions and `{summary.nullity}` null dimensions.",
        f"- The maximum loss observed along sampled null-space solutions is `{summary.max_null_loss:.3e}`.",
        "- In this toy problem, the loss only depends on two aggregate coefficients, so the effective data dimensionality is 2.",
        "",
        "## Suggested Next Steps",
        "",
        "- Add observation noise and study how the smallest non-zero Hessian eigenvalue changes.",
        "- Replace the quadratic generator with higher-order or mixed terms and test whether the recovered active rank scales as expected.",
        "- Compare local Hessian null-space structure with explicit global symmetry transformations.",
        "",
        "## Processing Steps",
        "",
        "1. Generate synthetic data from `y = m1*x + m2*x^2`.",
        "2. Fit the overparameterized model `y = sum(a_n*x + b_n*x^2)` in aggregate form.",
        "3. Derive the active Hessian spectrum and identify the null-space dimension.",
        "4. Sample parameter perturbations that keep the aggregate coefficients fixed.",
        "5. Explain the method visually in a process figure.",
        "6. Write the PNG figure and tabular summaries under this results directory.",
        "",
        "## Timings",
        "",
    ]
    for entry in timings:
        lines.append(f"- `{entry['step']}`: {entry['seconds']:.3f}s")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    overall_start = time.perf_counter()
    cfg = parse_args()
    rng = random.Random(cfg.seed)
    timings: list[dict[str, float]] = []

    info(f"Resolved data root: {cfg.data_root}")
    info(f"Resolved results root: {cfg.results_root}")
    info("This analysis generates synthetic data in memory. The data root is accepted for repository-wide CLI consistency.")

    run_dir = cfg.results_root / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    info(f"Writing analysis artifacts under: {run_dir}")

    with timed_step("generate synthetic dataset", timings):
        x_values, y_values, y_clean = generate_dataset(cfg, rng)

    with timed_step("fit aggregate coefficients", timings):
        a_fit, b_fit = fit_aggregate_coefficients(x_values, y_values)
        theta = build_theta(a_fit, b_fit, cfg.n_terms)
        y_fit = [model_prediction(x, a_fit, b_fit) for x in x_values]
        random_theta = build_random_theta(cfg.n_terms, rng)
        y_random = prediction_series(random_theta, x_values, cfg.n_terms)
        random_loss = loss_from_theta(random_theta, x_values, y_values, cfg.n_terms)

    with timed_step("analyze hessian structure", timings):
        eigenpairs = active_hessian_eigenpairs(x_values, cfg.n_terms)
        active_eigenvalues = [pair[0] for pair in eigenpairs]
        active_rank = estimate_rank(active_eigenvalues)
        nullity = 2 * cfg.n_terms - active_rank
        all_eigenvalues = [0.0] * nullity + active_eigenvalues
        active_direction = expand_active_direction(eigenpairs[-1][1], cfg.n_terms)
        null_direction = build_null_direction(cfg.n_terms)

    with timed_step("sample equivalent solutions", timings):
        equivalent_solutions = sample_equivalent_solutions(theta, cfg.n_terms, cfg.equivalent_samples, rng)
        equivalent_losses = [loss_from_theta(sample, x_values, y_values, cfg.n_terms) for sample in equivalent_solutions]
        aggregate_pairs = [aggregate_coefficients(sample, cfg.n_terms) for sample in equivalent_solutions]
        aggregate_drift = max(max(abs(a - a_fit), abs(b - b_fit)) for a, b in aggregate_pairs)
        null_projection_points = [
            (sample[0] - sample[1], sample[cfg.n_terms] - sample[cfg.n_terms + 1])
            for sample in equivalent_solutions
        ]

    with timed_step("scan loss along active and null directions", timings):
        steps = linspace(-2.0, 2.0, cfg.direction_steps)
        active_losses = [loss_from_theta(shift_theta(theta, active_direction, step), x_values, y_values, cfg.n_terms) for step in steps]
        null_losses = [loss_from_theta(shift_theta(theta, null_direction, step), x_values, y_values, cfg.n_terms) for step in steps]
        base_loss = loss_from_theta(theta, x_values, y_values, cfg.n_terms)

    summary = Summary(
        a_fit=a_fit,
        b_fit=b_fit,
        base_loss=base_loss,
        active_rank=active_rank,
        nullity=nullity,
        max_null_loss=max(null_losses),
        aggregate_drift=aggregate_drift,
    )

    with timed_step("render process figure", timings):
        canvas = render_process_figure(
            cfg,
            summary,
            x_values,
            y_values,
            y_clean,
            y_random,
            y_fit,
            random_loss,
            all_eigenvalues,
            null_projection_points,
            aggregate_pairs,
        )
        png_path = run_dir / "loss_invariance_process.png"
        write_png(png_path, canvas)
        info(f"Wrote figure: {png_path}")

    with timed_step("write tabular outputs", timings):
        summary_path = run_dir / "summary.json"
        summary_payload = {
            "config": {
                "run_name": cfg.run_name,
                "data_root": str(cfg.data_root),
                "results_root": str(cfg.results_root),
                "n_samples": cfg.n_samples,
                "n_terms": cfg.n_terms,
                "m1": cfg.m1,
                "m2": cfg.m2,
                "noise_std": cfg.noise_std,
                "seed": cfg.seed,
            },
            "summary": {
                "a_fit": summary.a_fit,
                "b_fit": summary.b_fit,
                "base_loss": summary.base_loss,
                "active_rank": summary.active_rank,
                "nullity": summary.nullity,
                "max_null_loss": summary.max_null_loss,
                "aggregate_drift": summary.aggregate_drift,
                "random_loss": random_loss,
            },
            "artifacts": {
                "figure_png": str(png_path),
                "timings_csv": str(run_dir / "timings.csv"),
                "equivalent_solutions_csv": str(run_dir / "equivalent_solutions.csv"),
                "direction_scan_csv": str(run_dir / "direction_scan.csv"),
                "run_readme": str(run_dir / "README.md"),
            },
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
        info(f"Wrote summary: {summary_path}")

        write_csv(
            run_dir / "timings.csv",
            ["step", "seconds"],
            [[entry["step"], f"{entry['seconds']:.6f}"] for entry in timings],
        )
        info(f"Wrote timing table: {run_dir / 'timings.csv'}")

        write_csv(
            run_dir / "equivalent_solutions.csv",
            ["sample_id"] + [f"a_{index + 1}" for index in range(cfg.n_terms)] + [f"b_{index + 1}" for index in range(cfg.n_terms)] + ["loss"],
            [[sample_id] + sample + [equivalent_losses[sample_id]] for sample_id, sample in enumerate(equivalent_solutions)],
        )
        info(f"Wrote equivalent solutions: {run_dir / 'equivalent_solutions.csv'}")

        write_csv(
            run_dir / "direction_scan.csv",
            ["step", "active_loss", "null_loss"],
            [[step, active_loss, null_loss] for step, active_loss, null_loss in zip(steps, active_losses, null_losses)],
        )
        info(f"Wrote direction scan: {run_dir / 'direction_scan.csv'}")

        write_run_readme(run_dir / "README.md", cfg, summary, timings)
        info(f"Wrote run README: {run_dir / 'README.md'}")

    total_seconds = time.perf_counter() - overall_start
    info(f"Completed analysis in {format_seconds(total_seconds)}")
    info(f"Recovered aggregate coefficients: A={summary.a_fit:.6f}, B={summary.b_fit:.6f}")
    info(f"Recovered active dimensions: {summary.active_rank}")
    info(f"Recovered null dimensions: {summary.nullity}")
    info(f"Random-model loss before fitting: {random_loss:.6e}")
    info(f"Maximum null-direction loss: {summary.max_null_loss:.6e}")


if __name__ == "__main__":
    main()
