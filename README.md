# regression_test

This repository is a simple testbed for studying whether loss-invariant directions in parameter space can reveal the true dimensionality of a dataset.

The core idea is to examine transformations of the model parameters that leave the model output, and therefore the loss, unchanged. Locally, these directions are approximated by the null space of the Hessian. Globally, they correspond to symmetries of the model.

To explore this, we construct a 2 dimensional dataset of the form

`y = m1 * x + m2 * x^2`

and fit it with an overparameterized model of the form. In the default experiment `n = 6`, so the model has `12` parameters and the script samples `60` equal-loss models.

Both experiments now add Gaussian noise to the synthetic observations by default.

`y = sum(a_n * x + b_n * x^2)`.

We then study the set of parameter values that preserve the same loss and test whether that structure can be used to recover the true dimensionality of the data, which in this example is 2.




## Run

Run the analysis with:

`uv run python exp/loss_invariance/run.py`

The script prints INFO messages, reports timing for each major stage, and writes all artifacts under:

`results/loss_invariance_analysis/`

The main outputs are:

- `loss_invariance_process.png`
- `summary.json`
- `timings.csv`
- `equivalent_solutions.csv`
- `direction_scan.csv`
- `README.md`

## Code Layout

The CLI entry point is:

- `exp/loss_invariance/run.py`

The implementation lives under:

- `exp/loss_invariance/analysis.py` for dataset generation and loss-invariance calculations
- `exp/loss_invariance/png_dashboard.py` for figure rendering
- `exp/loss_invariance/run.py` for the command-line workflow and output writing
- `exp/loss_invariance/README.md` for the detailed experiment description and interpretation guide

Additional experiment:

- `exp/regularization_dimensionality/run.py` compares no regularization, lasso, and ridge on the same synthetic problem and estimates dimensionality from summed non-zero coefficient groups
