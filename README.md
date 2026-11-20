# DSC Group 22: EPC label prediction

Predict a building's EPC label (8 ordered classes, A+ → G) from 116 property, location and building
features. The notebooks follow the course handbook (Ch. 3.3 data understanding, 3.4 data preparation,
3.5 modelling and evaluation), adapted to an 8-class target and 179k rows.

## Folder structure

```
DSC_Groupe22/
├── README.md
├── requirements.txt
├── .gitignore                 data, models and results are never pushed
├── data/                      LOCAL ONLY: put the Excel files + password.txt here (see data/README.md)
├── outputs/                   LOCAL ONLY: results table, saved models, predictions
├── src/
│   └── dsc_utils.py           shared code: paths, loading, clean(), evaluate()
└── notebooks/
    ├── 01_data_understanding.ipynb
    ├── 02_data_preparation.ipynb
    ├── 03_model_decision_tree.ipynb
    ├── 04_model_logistic_regression.ipynb
    ├── 05_model_random_forest.ipynb
    ├── 06_model_svm.ipynb
    ├── 07_model_knn.ipynb
    ├── 08_model_hist_gradient_boosting.ipynb
    ├── 09_comparison_and_submission.ipynb
    ├── 10_crisp_dm_preprocessing_experiments.ipynb
    ├── 11_visualisation_lift_cost.ipynb
    └── 12_model_understanding.ipynb
```

## Setup (once per laptop)

1. Put `DSC_training.xlsx`, `DSC_scoring.xlsx` and `password.txt` in `data/`.
2. In a terminal in this folder:
   ```bash
   python3 -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. In VS Code, open the folder and select the `venv` kernel for the notebooks.

## Run order

| Step | Notebook | Output | Rough time on a laptop |
|---|---|---|---|
| 1 | `01_data_understanding` | plots and tables only | 2–3 min (first run decrypts the Excel files) |
| 2 | `02_data_preparation` | `data/DSC_preprocessed_v2.joblib` | 1–2 min |
| 3 | `03`–`08`, any order | one row each in `outputs/DSC_results.csv` + `outputs/model_*.pkl` | minutes (DT, LR, linear SVM) to 15+ min (RF, HGB, RBF SVM, kNN grid) |
| 4 | `09_comparison_and_submission` | comparison table + `outputs/predictions_group22.csv` | < 1 min |
| optional, after 02 | `10_crisp_dm_preprocessing_experiments` | which preparation choices help (validation macro-F1), learning curve | 5–15 min |
| after 03–08 | `11_visualisation_lift_cost` | lift and cost curves for finding E/F/G buildings | 1–3 min |
| after 03, 04, 05, 08 | `12_model_understanding` | feature importance per model, effect of build year | 5–10 min |

Rerun `02` whenever `clean()` in `src/dsc_utils.py` changes. Delete `outputs/DSC_results.csv` for a fresh
comparison table.

## Methods in short

**Cleaning (`clean()` in `src/dsc_utils.py`, applied to training and scoring data):**
- Build year: 84% are periods ("1930-1969") → midpoint, plus flag `builtyear_is_range`; "<1930" → 1915.
- Distances ("150m", "1,5km") → metres. ZIP and NIS codes → text (categories).
- `region` from postcode: the flood columns only exist for Flanders, so they stay missing elsewhere.
- Structural missingness → 0 (pool, garden, apartment surface). Price ≤ 0 and 0 facades → missing.
- Flag `no_3d_data` (~8% of rows miss all 3D building data).
- Dropped: IDs, duplicate location columns, constant columns, patrimony code.

**Preparation:** 64/16/20 train/validation/test split (random_state=0), stratified on label × region
(the EPC scale differs per region). Median imputation and
scaling for numbers, most-frequent imputation and one-hot encoding (min_frequency=100) for categories,
all fitted on the training set only.

**Modelling:** each model is tuned on the validation set and evaluated once on the test set.

| Handbook (German Credit) | This project | Why |
|---|---|---|
| 2 classes | 8 ordered classes | multiclass target |
| AUC to tune | macro-F1 to tune; one-vs-rest AUC reported | AUC is binary; macro-F1 weighs rare A+ and G |
| cost matrix (FP = 1, FN = 5) | ordinal MAE + E/F/G recall | no cost matrix; D for C is a smaller error than G for C |
| SVC on all data | LinearSVC on all data; RBF SVC on a 15k sample | SVC is far too slow on 115k rows |
| kNN on all features | kNN after TruncatedSVD (50 components) | distances on thousands of one-hot columns are noisy and slow |
| — | HistGradientBoosting added | strongest model in the first draft |

If the organisers announce an official metric, tune on that metric instead of macro-F1.

**Lift and cost curves (11):** drawn for the yes/no question "is the building energy-poor (E/F/G)?", with the
score = predicted probability of E + F + G. The cost matrix (missed E/F/G = 5, false alarm = 1) mirrors the
tutorial and is an assumption, not an official DSC cost.

## Data rules

- The challenge data may not be uploaded to the cloud or shared. Only code goes to GitHub.
- Keep the repository private. Check `git status` before every commit: no `.xlsx`, `.csv`, `.pkl`,
  `.joblib` or `password.txt` may appear.
- Clear notebook outputs before committing (VS Code: **Clear All Outputs**), as outputs can show data.
