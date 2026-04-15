# loss_invariance_analysis

## What Was Run

`uv run python exp/loss_invariance/run.py --run_name loss_invariance_analysis --n_samples 200 --n_terms 6 --m1 1.5 --m2 -0.8 --noise_std 0.1 --seed 7`

## Files Present

- `loss_invariance_process.png`: step-by-step figure explaining the method.
- `summary.json`: key scalar outputs and file locations.
- `timings.csv`: wall-clock timing for each major stage.
- `equivalent_solutions.csv`: parameter settings with the same loss.
- `direction_scan.csv`: loss sampled along one active and one null direction.
- `README.md`: this run summary.

## Conclusions

- The synthetic ground-truth data have dimensionality `2` because the target is controlled by the coefficients of `x` and `x^2`.
- Gaussian noise with standard deviation `0.1` is added to the synthetic data by default.
- The overparameterized model uses `12` parameters.
- The run generated `60` equal-loss parameter vectors.
- The fitted aggregate coefficients are `A=1.503264` and `B=-0.801376`.
- The Hessian has `2` active dimensions and `10` null dimensions.
- The maximum loss observed along sampled null-space solutions is `4.925e-03`.
- In this toy problem, the loss only depends on two aggregate coefficients, so the effective data dimensionality is 2.

## Suggested Next Steps

- Add observation noise and study how the smallest non-zero Hessian eigenvalue changes.
- Replace the quadratic generator with higher-order or mixed terms and test whether the recovered active rank scales as expected.
- Compare local Hessian null-space structure with explicit global symmetry transformations.

## Processing Steps

1. Generate synthetic data from `y = m1*x + m2*x^2`.
2. Fit the overparameterized model `y = sum(a_n*x + b_n*x^2)` in aggregate form.
3. Derive the active Hessian spectrum and identify the null-space dimension.
4. Sample parameter perturbations that keep the aggregate coefficients fixed.
5. Explain the method visually in a process figure.
6. Write the PNG figure and tabular summaries under this results directory.

## Timings

- `generate synthetic dataset`: 0.000s
- `fit aggregate coefficients`: 0.000s
- `analyze hessian structure`: 0.000s
- `sample equivalent solutions`: 0.001s
- `scan loss along active and null directions`: 0.008s
- `render process figure`: 0.453s
