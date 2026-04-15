# tune_lasso_012

## What Was Run

`uv run python exp/regularization_dimensionality/run.py --run_name tune_lasso_012 --n_terms 6 --epochs 5000 --lasso_lambda 0.12 --ridge_lambda 0.08`

## Files Present

- `regularization_dimensionality.png`: main comparison figure.
- `summary.json`: scalar summary and artifact paths.
- `training_history.csv`: data-loss and objective history for each method.
- `final_coefficients.csv`: learned coefficients for each method.
- `README.md`: this run summary.

## Conclusions

- The synthetic ground-truth data have dimensionality `2`.
- The shared overparameterized model has `12` parameters.
- The dimensionality estimate is the number of coefficients whose absolute value exceeds the configured threshold.
- No regularization keeps `11` coefficients above threshold.
- Lasso keeps `5` coefficients above threshold.
- Ridge keeps `12` coefficients above threshold.

## Interpretation

- All three methods reduce the fit error from the same random starting point.
- Lasso is the only method that strongly prefers sparse solutions in this duplicated-feature model.
- No regularization and ridge both keep most redundant coefficients active, so they overestimate the dimensionality if the estimate is based on non-zero parameter count.

## Suggested Next Steps

- Sweep the lasso penalty and track when the estimated dimensionality first matches the true value 2.
- Add noise and study whether lasso still recovers a sparse representation.
- Compare the coefficient-count estimate here against the Hessian-rank estimate from `exp/loss_invariance`.
