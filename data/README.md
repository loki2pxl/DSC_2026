# data/ (local only, never committed)

Put these three files here on your own laptop:

| File | What |
|---|---|
| `DSC_training.xlsx` | training data from the organisers (179,446 rows, with EPC label) |
| `DSC_scoring.xlsx` | scoring data from the organisers (76,906 rows, no label) |
| `password.txt` | one line: the password of the Excel files |

The notebooks also create `DSC_raw.pkl` (cached raw data) and `DSC_preprocessed_v2.joblib`
(the preprocessed data) here. Everything in this folder except this README is ignored by Git.
