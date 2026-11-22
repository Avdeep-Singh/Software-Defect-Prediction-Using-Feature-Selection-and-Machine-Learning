# Software defect prediction with feature selection

Exploratory notebooks for evaluating software defect classifiers with and without feature selection across six datasets.

## Repository layout

- [Without feature selection](Without%20Feature%20Selection/) contains baseline classifier notebooks.
- [With feature selection](With%20Feature%20Selection/) contains mutual-information experiments.
- [Additional feature selection](new%20feature%20selection/) contains filter, embedded, and wrapper methods.
- The root directory also contains an exploratory notebook and the six CSV snapshots below.

## Data

The original project documentation attributes the datasets to NASA and the PROMISE repository. The table describes the copies in this repository. Exact source editions, transformations, and redistribution terms remain unverified. Use the [tera-PROMISE defect catalog](https://openscience.us/repo/defect/) as a starting point when tracing a source edition.

| Dataset | Local CSV | Data rows | Target column |
|---|---|---:|---|
| CM1 | `cm1_csv(1).csv` | 498 | `defects` |
| JM1 | `jm1_csv.csv` | 10,885 | `defects` |
| KC1 | `kc1_csv.csv` | 2,109 | `defects` |
| KC2 | `kc2_csv.csv` | 522 | `problems` |
| PC1 | `pc1_csv.csv` | 1,109 | `defects` |
| PC3 | `pc3_csv.csv` | 1,563 | `LABEL` |

## Inspect and rerun

Start with the [CM1 baseline](Without%20Feature%20Selection/cm1_without_fs.ipynb) and [CM1 mutual-information example](With%20Feature%20Selection/cm1_FS_information%20gain.ipynb). They use Python, NumPy, pandas, matplotlib, seaborn, and scikit-learn. Open a working copy in Jupyter or Colab and check the CSV path in the data-loading cell. These CM1 notebooks reference the Colab path `/content/cm1_csv(1).csv`. For a local run from the repository root, point that cell at `cm1_csv(1).csv`.

Both CM1 notebooks’ code cells ran on Python 3.12.10 after substituting the local CSV path. All 79 notebook files parsed as JSON. This check covers those two execution paths. The remaining notebooks and stored results were outside the execution check.

## Evaluation limits

The two inspected CM1 notebooks use different split settings. The baseline fits scaling before splitting, and the mutual-information example fits supervised feature selection before splitting. Those choices let information from the held-out data influence preprocessing. Treat the saved outputs as exploratory. A controlled comparison would fit preprocessing and feature selection on training data only, reuse the same splits, and report results under that protocol.
