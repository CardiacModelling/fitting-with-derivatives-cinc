# Derivative-based Inference for Cell and Channel Electrophysiology Models

This repository contains the code for the CINC 2022 paper _"Derivative-based Inference for Cell and Channel Electrophysiology Models"_.

## Requirements

- Myokit (github version, 2022-09-12)
- PINTS (github version, 2022-09-12)

## Scripts

Models, protocols, shared code:

- Model and protocol files are stored in [`resources'](./resources).
- Python modules used in fitting and stored in [`methods`](./methods).
- Results are written to the [`results'](./results) directory.

Benchmarking simulations:

- `cost.py` runs simulations from randomly sampled starting points and measures run times for simulations with and without derivative calculation.

Fitting:

- `fit.py` performs a fit; see `python fit.py --help` for details.
- `count.py` shows the number of fits performed for each method and test case.
- `best.py` shows the lowest error returned for each method and test case.
- `time.py` shows the mean time taken for runs getting within 5% of the best result, for each method and test case.

## Figures

Rendered figures are stored in [`figures`](./figures).
All figures can be generated in poster format by adding the command line argument `poster`.

- `m1-cases.py` generates a figure showing the results of a simulation with each test case (not included in paper due to space constraints).
- `m2-eval-cost.py` generates a figure comparing the cost per evaluation with and without derivative calculations (requires `cost.py` to have been run).
- `m3-opt-ikr.py` generates a figure showing the fitting error as a function of the number of evaluations and as a function of the run time, for the IKr case.
- `m4-opt-ap.py` generates a figure showing the fitting error as a function of the number of evaluations and as a function of the run time, for the AP case.
- `m5-robustness.py` generates a figure comparing "robustness" of fitting methods, for both cases.
