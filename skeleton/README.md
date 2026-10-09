## Empirical AutoML on Tabular Data

For this assignment, we compared three approaches of hyperparameter optimization: Random Search, SMBO and Hyperband. 
We compared how well these algorithms performed on a Random Forest Classifier accross five different tabular datasets. 
an untuned Random forest wasused as a baseline and TabPFN was evaluatedd as a pre-trained tabular foundation model on the largest dataset


## Set up

Use Python 3.11 or newer. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Get started

Complete or replace the relevant TODOs before running. For an initial baseline
check, implement the scoring functions in `random_forest.py`, then run:

```bash
python experiment.py --dataset breast-w --profile smoke --methods default
```

Use `python experiment.py --help` for available options.

Example profiles and budgets are in `experiment.py`: `smoke` caps rows for small
checks, `course` uses all rows, and `full` uses all rows with more trees. These
names do not indicate grading levels. `N_JOBS` in `random_forest.py` controls
parallelism for forest fits.

**Results are not saved automatically.** Add result saving in `experiment.py`
or your own pipeline before the main study.

Replace this README with your installation, pipeline-check, experiment, and
figure/table-generation instructions.
