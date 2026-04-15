# regularization_dimensionality

## What Was Run

`uv run python exp/regularization_dimensionality/run.py --run_name regularization_dimensionality --n_terms 6 --epochs 5000 --lasso_lambda 0.08 --ridge_lambda 0.08`

## Files Present

- `regularization_dimensionality.png`: main comparison figure.
- `summary.json`: scalar summary and artifact paths.
- `training_history.csv`: data-loss and objective history for each method.
- `final_coefficients.csv`: learned coefficients for each method.
- `README.md`: this run summary.

## Conclusions

- The synthetic ground-truth data have dimensionality `2`.
- The shared overparameterized model has `12` parameters.
- Gaussian noise with standard deviation `0.1` is added to the synthetic data by default.
- The dimensionality estimate is the raw count of coefficients whose absolute value exceeds the configured threshold.
- No regularization keeps `11` non-zero coefficients and gives dimension estimate `11`.
- Lasso keeps `5` non-zero coefficients and gives dimension estimate `5`.
- Ridge keeps `12` non-zero coefficients and gives dimension estimate `12`.

## Interpretation

- All three methods reduce the fit error from the same random starting point.
- With this raw non-zero-count estimate, the methods no longer recover the true dimensionality 2 exactly.
- Lasso still gives the sparsest parameterization and therefore the smallest dimensionality estimate among the three methods.

## Suggested Next Steps

- Sweep the lasso penalty and track when the estimated dimensionality first matches the true value 2.
- Add noise and study whether lasso still recovers a sparse representation.
- Compare the coefficient-count estimate here against the Hessian-rank estimate from `exp/loss_invariance`.
