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
    TrialResult,
    build_random_theta,
    generate_dataset,
    prediction_series,
    train_model,
)
from png_report import render_regularization_figure, write_png


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
            "Train the same overparameterized model from the same random start "
            "with no regularization, lasso, and ridge, then compare the number "
            "of non-zero coefficients as a dimensionality estimate."
        )
    )
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--run_name", default="regularization_dimensionality")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_terms", type=int, default=6)
    parser.add_argument("--m1", type=float, default=1.5)
    parser.add_argument("--m2", type=float, default=-0.8)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--lasso_lambda", type=float, default=0.08)
    parser.add_argument("--ridge_lambda", type=float, default=0.08)
    parser.add_argument("--zero_threshold", type=float, default=0.03)
    parser.add_argument("--history_stride", type=int, default=50)
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
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lasso_lambda=args.lasso_lambda,
        ridge_lambda=args.ridge_lambda,
        zero_threshold=args.zero_threshold,
        history_stride=args.history_stride,
    )


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_run_readme(path: Path, cfg: Config, trials: list[TrialResult]) -> None:
    trial_lookup = {trial.name: trial for trial in trials}
    lines = [
        f"# {cfg.run_name}",
        "",
        "## What Was Run",
        "",
        (
            "`uv run python exp/regularization_dimensionality/run.py "
            f"--run_name {cfg.run_name} --n_terms {cfg.n_terms} --epochs {cfg.epochs} "
            f"--lasso_lambda {cfg.lasso_lambda} --ridge_lambda {cfg.ridge_lambda}`"
        ),
        "",
        "## Files Present",
        "",
        "- `regularization_dimensionality.png`: main comparison figure.",
        "- `summary.json`: scalar summary and artifact paths.",
        "- `training_history.csv`: data-loss and objective history for each method.",
        "- `final_coefficients.csv`: learned coefficients for each method.",
        "- `README.md`: this run summary.",
        "",
        "## Conclusions",
        "",
        "- The synthetic ground-truth data have dimensionality `2`.",
        f"- The shared overparameterized model has `{2 * cfg.n_terms}` parameters.",
        f"- Gaussian noise with standard deviation `{cfg.noise_std}` is added to the synthetic data by default.",
        "- The dimensionality estimate is the raw count of coefficients whose absolute value exceeds the configured threshold.",
        f"- No regularization keeps `{trial_lookup['none'].nonzero_count}` non-zero coefficients and gives dimension estimate `{trial_lookup['none'].dimension_estimate}`.",
        f"- Lasso keeps `{trial_lookup['lasso'].nonzero_count}` non-zero coefficients and gives dimension estimate `{trial_lookup['lasso'].dimension_estimate}`.",
        f"- Ridge keeps `{trial_lookup['ridge'].nonzero_count}` non-zero coefficients and gives dimension estimate `{trial_lookup['ridge'].dimension_estimate}`.",
        "",
        "## Interpretation",
        "",
        "- All three methods reduce the fit error from the same random starting point.",
        "- With this raw non-zero-count estimate, the methods no longer recover the true dimensionality 2 exactly.",
        "- Lasso still gives the sparsest parameterization and therefore the smallest dimensionality estimate among the three methods.",
        "",
        "## Suggested Next Steps",
        "",
        "- Sweep the lasso penalty and track when the estimated dimensionality first matches the true value 2.",
        "- Add noise and study whether lasso still recovers a sparse representation.",
        "- Compare the coefficient-count estimate here against the Hessian-rank estimate from `exp/loss_invariance`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    overall_start = time.perf_counter()
    cfg = parse_args()
    rng = random.Random(cfg.seed)
    timings: list[dict[str, float]] = []

    info(f"Resolved data root: {cfg.data_root}")
    info(f"Resolved results root: {cfg.results_root}")
    info("This experiment reuses the same synthetic data definition and the same random model initialization as the loss_invariance experiment.")

    run_dir = cfg.results_root / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    info(f"Writing analysis artifacts under: {run_dir}")

    with timed_step("generate shared synthetic dataset", timings):
        x_values, y_values, y_clean = generate_dataset(cfg, rng)

    with timed_step("build shared random model", timings):
        theta_init = build_random_theta(cfg.n_terms, rng)
        y_random = prediction_series(theta_init, x_values, cfg.n_terms)

    with timed_step("train without regularization", timings):
        trial_none = train_model("none", "none", 0.0, theta_init, x_values, y_values, cfg)
        info(
            f"No regularization finished with loss={trial_none.data_loss:.6e}, "
            f"nonzero_count={trial_none.nonzero_count}, "
            f"dimension_estimate={trial_none.dimension_estimate}"
        )

    with timed_step("train with lasso regularization", timings):
        trial_lasso = train_model("lasso", "lasso", cfg.lasso_lambda, theta_init, x_values, y_values, cfg)
        info(
            f"Lasso finished with loss={trial_lasso.data_loss:.6e}, "
            f"nonzero_count={trial_lasso.nonzero_count}, "
            f"dimension_estimate={trial_lasso.dimension_estimate}"
        )

    with timed_step("train with ridge regularization", timings):
        trial_ridge = train_model("ridge", "ridge", cfg.ridge_lambda, theta_init, x_values, y_values, cfg)
        info(
            f"Ridge finished with loss={trial_ridge.data_loss:.6e}, "
            f"nonzero_count={trial_ridge.nonzero_count}, "
            f"dimension_estimate={trial_ridge.dimension_estimate}"
        )

    trials = [trial_none, trial_lasso, trial_ridge]

    with timed_step("render comparison figure", timings):
        canvas = render_regularization_figure(
            cfg=cfg,
            x_values=x_values,
            y_values=y_values,
            y_clean=y_clean,
            y_random=y_random,
            trials=trials,
            random_loss=trial_none.loss_history[0][1],
        )
        figure_path = run_dir / "regularization_dimensionality.png"
        write_png(figure_path, canvas)
        info(f"Wrote figure: {figure_path}")

    with timed_step("write tabular outputs", timings):
        write_csv(
            run_dir / "training_history.csv",
            ["method", "epoch", "data_loss", "objective"],
            [
                [trial.name, epoch, data_loss, objective]
                for trial in trials
                for epoch, data_loss, objective in trial.loss_history
            ],
        )
        info(f"Wrote training history: {run_dir / 'training_history.csv'}")

        write_csv(
            run_dir / "final_coefficients.csv",
            ["method"] + [f"theta_{index + 1}" for index in range(2 * cfg.n_terms)] + ["nonzero_count", "dimension_estimate", "aggregate_linear", "aggregate_quadratic"],
            [
                [trial.name] + trial.theta_final + [trial.nonzero_count, trial.dimension_estimate, trial.aggregate_linear, trial.aggregate_quadratic]
                for trial in trials
            ],
        )
        info(f"Wrote final coefficients: {run_dir / 'final_coefficients.csv'}")

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
                "epochs": cfg.epochs,
                "learning_rate": cfg.learning_rate,
                "lasso_lambda": cfg.lasso_lambda,
                "ridge_lambda": cfg.ridge_lambda,
                "zero_threshold": cfg.zero_threshold,
            },
            "summary": {
                trial.name: {
                    "data_loss": trial.data_loss,
                    "objective": trial.objective,
                    "nonzero_count": trial.nonzero_count,
                    "dimension_estimate": trial.dimension_estimate,
                    "aggregate_linear": trial.aggregate_linear,
                    "aggregate_quadratic": trial.aggregate_quadratic,
                }
                for trial in trials
            },
            "artifacts": {
                "figure_png": str(run_dir / "regularization_dimensionality.png"),
                "training_history_csv": str(run_dir / "training_history.csv"),
                "final_coefficients_csv": str(run_dir / "final_coefficients.csv"),
                "run_readme": str(run_dir / "README.md"),
            },
        }
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
        info(f"Wrote summary: {summary_path}")

        write_run_readme(run_dir / "README.md", cfg, trials)
        info(f"Wrote run README: {run_dir / 'README.md'}")

    total_seconds = time.perf_counter() - overall_start
    info(f"Completed experiment in {format_seconds(total_seconds)}")
    for trial in trials:
        info(
            f"{trial.name} -> loss={trial.data_loss:.6e}, "
            f"nonzero_count={trial.nonzero_count}, "
            f"dimension_estimate={trial.dimension_estimate}, "
            f"aggregate=({trial.aggregate_linear:.4f}, {trial.aggregate_quadratic:.4f})"
        )


if __name__ == "__main__":
    main()
