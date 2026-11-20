"""Shared helpers for the DSC Group 22 notebooks.

Everything that more than one notebook needs lives here, so a fix is made once:
paths, loading the (encrypted) Excel files, the cleaning rules, and the
evaluation function used by every model.
"""
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

TRAIN_FILE = DATA / "DSC_training.xlsx"
SCORE_FILE = DATA / "DSC_scoring.xlsx"
PASSWORD_FILE = DATA / "password.txt"
RAW_CACHE = DATA / "DSC_raw.pkl"
PREP_FILE = DATA / "DSC_preprocessed_v2.joblib"
RESULTS_FILE = OUT / "DSC_results.csv"

# ---------------------------------------------------------------- constants
TARGET = "EPC label"
ORDER = ["A+", "A", "B", "C", "D", "E", "F", "G"]
TO_I = {label: i for i, label in enumerate(ORDER)}

# IDs, duplicates and constant columns (see data understanding notebook)
DROP = ["finc_id", "fed_address_id", "built/unbuilt", "REAL_ESTATE_TYP",
        "matched_postalcode", "cleaned_municipality", "patrimony code",
        "floodrisk building - ogoz"]


# ---------------------------------------------------------------- loading
def read_excel_maybe_encrypted(path):
    """Read an .xlsx file; decrypt it first if it is password-protected."""
    import msoffcrypto
    from io import BytesIO
    from msoffcrypto import olefile

    path = Path(path)
    if olefile.isOleFile(str(path)):                 # encrypted Office files are OLE containers
        if not PASSWORD_FILE.exists():
            raise FileNotFoundError(f"Put the file password in {PASSWORD_FILE}")
        password = PASSWORD_FILE.read_text().strip()
        if not password:
            raise ValueError(f"{PASSWORD_FILE} is empty: put the Excel password in it and save")
        buf = BytesIO()
        with open(path, "rb") as f:
            office = msoffcrypto.OfficeFile(f)
            office.load_key(password=password)
            office.decrypt(buf)
        buf.seek(0)
        return pd.read_excel(buf, engine="openpyxl")
    return pd.read_excel(path, engine="openpyxl")


def load_raw(refresh=False):
    """Return (raw_train, raw_score) exactly as in the Excel files.

    The first call decrypts both files (slow) and caches them in data/DSC_raw.pkl;
    later calls load the cache in seconds. refresh=True re-reads the Excel files.
    """
    if RAW_CACHE.exists() and not refresh:
        return pd.read_pickle(RAW_CACHE)
    raw_train = read_excel_maybe_encrypted(TRAIN_FILE)
    raw_score = read_excel_maybe_encrypted(SCORE_FILE)
    pd.to_pickle((raw_train, raw_score), RAW_CACHE)
    return raw_train, raw_score


# ---------------------------------------------------------------- cleaning
def parse_year(v):
    """'1930-1969' -> 1949.5, '<1930' -> 1915, '2023' -> 2023."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s.startswith("<"):
        return 1915.0
    years = [int(y) for y in re.findall(r"\d{4}", s)]
    return float(np.mean(years)) if years else np.nan


def to_metres(v):
    """'150m' -> 150, '1,5km' -> 1500, 'lessThan100m' -> 50, 'moreThan10km' -> 12000."""
    if pd.isna(v):
        return np.nan
    s = str(v).replace(",", ".")
    if s == "lessThan100m":
        return 50.0
    if s.startswith("moreThan"):
        return 12000.0
    try:
        if s.endswith("km"):
            return float(s[:-2]) * 1000
        if s.endswith("m"):
            return float(s[:-1])
    except ValueError:
        return np.nan
    return np.nan


def region_from_zip(zip_series):
    """Belgian postcodes -> Brussels / Wallonia / Flanders."""
    z = pd.to_numeric(zip_series, errors="coerce")
    return np.where(z.between(1000, 1299), "Brussels",
           np.where(z.between(1300, 1499) | z.between(4000, 7999), "Wallonia", "Flanders"))


def fix_types(df):
    """Step 0: make every column the right type, without filling or dropping anything.

    Used by the data understanding notebook, so missing values stay visible.
    """
    df = df.copy()
    raw_year = df["builtyear"].astype(str)
    df["builtyear_is_range"] = (~raw_year.str.fullmatch(r"\d{4}(\.0)?")).astype(int)
    df["builtyear"] = df["builtyear"].apply(parse_year)
    df["habitable surface (apartment)"] = pd.to_numeric(df["habitable surface (apartment)"], errors="coerce")
    # text distance bins ("150m", "1,5km"); numeric "... flood distance" columns are left alone
    for col in [c for c in df.columns if c.endswith("distance") and not pd.api.types.is_numeric_dtype(df[c])]:
        df[col] = df[col].apply(to_metres)
    df["region"] = region_from_zip(df["ZIP_COD"])            # before ZIP becomes text
    for col in ["ZIP_COD", "matched_municipality_nis_code"]:
        df[col] = df[col].astype(str)
    return df


def clean(df):
    """fix_types + the data preparation rules. Apply to training AND scoring data."""
    df = fix_types(df.drop(columns=[TARGET], errors="ignore"))
    # structural missingness: the feature doesn't exist -> 0
    for col in ["swimming pool surface", "garden surface", "habitable surface (apartment)"]:
        df[col] = df[col].fillna(0)
    # impossible values -> missing (imputed later with the training median)
    df.loc[df["Price"] <= 0, "Price"] = np.nan
    df.loc[df["building facades"] == 0, "building facades"] = np.nan
    # ~8% of rows miss all 3D building data together
    df["no_3d_data"] = df["volume building"].isna().astype(int)
        # --- engineered features: proxies for renovation state and heat loss ---
    surface = df["habitable surface (apartment)"].where(df["habitable surface (apartment)"] > 0,
                                                        df["total floor surface (excl. terraces)"])
    df["price_per_m2"] = df["Price"] / surface
    df["heat_loss_per_m2"] = df["heat loss surface"] / df["total floor surface (excl. terraces)"]
    df["heat_loss_per_volume"] = df["heat loss surface"] / df["volume building"]
    df["exposed_facade_share"] = df["building facades"] / 4
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    # mixed text/number columns -> text, keep NaN for the imputer
    cat = df.select_dtypes(exclude="number").columns
    df[cat] = df[cat].apply(lambda s: s.where(s.isna(), s.astype(str)))
    return df


# ---------------------------------------------------------------- modelling
def load_prepared():
    """The dictionary saved by 02_data_preparation.ipynb."""
    import joblib
    return joblib.load(PREP_FILE)


def stratified_sample(X, y, n, seed=0):
    """Stratified subsample of n rows (all rows if n >= len(y)). For slow models."""
    from sklearn.model_selection import train_test_split
    y = np.asarray(y)
    if n >= len(y):
        return X, y
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return (X.iloc[idx] if hasattr(X, "iloc") else X[idx]), y[idx]


def macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro")


def evaluate(name, model, X, y, save=True, plot=True):
    """Handbook metrics adapted to 8 ordered classes.

    accuracy, macro-F1, ordinal MAE (replaces the handbook's cost), within-one
    accuracy, E/F/G recall, and one-vs-rest AUC when the model gives probabilities.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    y = np.asarray(y)
    pred = model.predict(X)
    yt = pd.Series(y).map(TO_I).to_numpy()
    yp = pd.Series(pred).map(TO_I).to_numpy()
    err = np.abs(yt - yp)
    poor = np.isin(yt, [5, 6, 7])
    res = {"model": name,
           "accuracy": float(accuracy_score(y, pred)),
           "macro_F1": float(macro_f1(y, pred)),
           "ordinal_MAE": float(err.mean()),
           "within_one": float((err <= 1).mean()),
           "EFG_recall": float(np.isin(yp[poor], [5, 6, 7]).mean())}
    if hasattr(model, "predict_proba"):
        classes = np.asarray(model.classes_)
        order = np.argsort(classes)                       # roc_auc_score needs labels in sorted order
        res["AUC_ovr"] = float(roc_auc_score(y, model.predict_proba(X)[:, order], multi_class="ovr",
                                             average="macro", labels=classes[order]))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
    if plot:
        ConfusionMatrixDisplay.from_predictions(y, pred, labels=ORDER, normalize="true",
                                                values_format=".2f", cmap="Blues")
        plt.title(f"{name}: confusion matrix (share of each true label)")
        plt.show()
    if save:
        pd.DataFrame([res]).to_csv(RESULTS_FILE, mode="a", header=not RESULTS_FILE.exists(), index=False)
    return res


def save_model(model, name):
    with open(OUT / f"model_{name}.pkl", "wb") as f:
        pickle.dump(model, f)


def load_model(name):
    with open(OUT / f"model_{name}.pkl", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- visualisation (lift and cost curves)
POOR = ["E", "F", "G"]          # "energy-poor" buildings: the group a renovation programme would target
GOOD = ["A+", "A", "B"]         # energy-efficient buildings: the group a green-mortgage offer would target
MODEL_NAMES = ["DT", "LR", "RF", "SVM_linear", "SVM_rbf", "KNN", "HGB"]


def class_scores(model, X):
    """Score per class (columns = model.classes_). Probabilities when the model has them;
    for SVMs a softmax of the decision values (a ranking score, not a calibrated probability)."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
    else:
        z = model.decision_function(X)
        z = np.exp(z - z.max(axis=1, keepdims=True))
        p = z / z.sum(axis=1, keepdims=True)
    return pd.DataFrame(p, columns=model.classes_)


def group_score(model, X, group=POOR):
    """Score for 'belongs to this group of labels' (e.g. E/F/G) = sum of the class scores."""
    return class_scores(model, X)[group].sum(axis=1).to_numpy()


def liftatp(scores, y_binary, p):
    """Lift at fraction p: share of positives in the top p (by score) / share of positives overall."""
    y_binary = np.asarray(y_binary)
    n = max(1, int(round(p * len(scores))))
    top = y_binary[np.argsort(-np.asarray(scores), kind="stable")[:n]]
    return top.mean() / y_binary.mean()


def costatp(scores, y_binary, p, cost_matrix=(0, 1, 5, 0)):
    """Total cost when the top p (by score) is targeted as positive.
    cost_matrix = (cost TN, cost FP, cost FN, cost TP), same order as the tutorial's [0, 1, 5, 0]."""
    y = np.asarray(y_binary).astype(int)
    pred = np.zeros(len(y), dtype=int)
    pred[np.argsort(-np.asarray(scores), kind="stable")[:int(round(p * len(y)))]] = 1
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tp = int(((pred == 1) & (y == 1)).sum())
    c_tn, c_fp, c_fn, c_tp = cost_matrix
    return tn * c_tn + fp * c_fp + fn * c_fn + tp * c_tp


# ---------------------------------------------------------------- model understanding
def original_feature(feature_name, original_columns):
    """'cat__ZIP_COD_2000' -> 'ZIP_COD', 'num__builtyear' -> 'builtyear'."""
    f = feature_name.split("__", 1)[-1]
    for c in sorted(original_columns, key=len, reverse=True):
        if f == c or f.startswith(c + "_"):
            return c
    return f


def group_by_original(values, original_columns):
    """Sum per original column: one-hot columns of the same variable are added together."""
    values = pd.Series(values)
    keys = values.index.map(lambda f: original_feature(f, original_columns))
    return values.groupby(keys).sum().sort_values(ascending=False)


# ---------------------------------------------------------------- DSC metric: 1-notch accuracy
def one_notch_accuracy(y_true, y_pred):
    """Official DSC metric: share of predictions at most one label away from the truth."""
    yt = pd.Series(np.asarray(y_true)).map(TO_I).to_numpy()
    yp = pd.Series(np.asarray(y_pred)).map(TO_I).to_numpy()
    return float((np.abs(yt - yp) <= 1).mean())


def predict_one_notch(proba):
    """Pick the label that maximises the chance of being within one notch:
    P(label-1) + P(label) + P(label+1), instead of just the most likely label."""
    p = pd.DataFrame(proba).reindex(columns=ORDER, fill_value=0).to_numpy()
    padded = np.pad(p, ((0, 0), (1, 1)))
    window = padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]
    return np.array(ORDER)[window.argmax(axis=1)]


class OneNotchClassifier:
    """Wraps a fitted model so .predict() uses the 1-notch decision rule. Probabilities are unchanged."""
    def __init__(self, model):
        self.model = model
        self.classes_ = getattr(model, "classes_", np.array(ORDER))

    def predict_proba(self, X):
        return class_scores(self.model, X).reindex(columns=self.classes_).to_numpy()

    def predict(self, X):
        return predict_one_notch(class_scores(self.model, X))


def _one_notch_from_proba(y_true, y_proba):
    classes = np.unique(np.asarray(y_true))          # sklearn orders predict_proba columns like this
    return one_notch_accuracy(y_true, predict_one_notch(pd.DataFrame(y_proba, columns=classes)))


def one_notch_scorer(rule="argmax"):
    """Scorer for GridSearchCV / RandomizedSearchCV.
    rule='argmax': normal .predict();  rule='window': the 1-notch decision rule on the probabilities."""
    from sklearn.metrics import make_scorer
    if rule == "window":
        return make_scorer(_one_notch_from_proba, response_method="predict_proba")
    return make_scorer(one_notch_accuracy)


# ---------------------------------------------------------------- configurable preprocessing (missing values + encoding)
def build_preprocessor(num_cols, cat_cols, num_impute="median", add_indicator=False,
                       cat_impute="most_frequent", encoding="onehot", min_frequency=100):
    """Preprocessing pipeline with switchable missing-value and encoding choices.

    num_impute:   'median' | 'mean' (SimpleImputer) | 'knn' (KNNImputer: values of the 5 most similar buildings)
                  | 'iterative' (IterativeImputer: predicts each missing value from the other columns)
                  | 'none' (keep NaN: only for models that handle it, e.g. HistGradientBoosting)
    add_indicator: also add a 0/1 column per numeric feature that had missing values
    cat_impute:   'most_frequent' | 'constant' (missing becomes its own category "missing")
    encoding:     'onehot' | 'ordinal' | 'target' (multiclass target encoding; fit with y)
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder

    if num_impute in ("median", "mean"):
        num_steps = [("imputer", SimpleImputer(strategy=num_impute, add_indicator=add_indicator)),
                     ("scaler", StandardScaler())]
    elif num_impute == "knn":           # scale first so distances are not dominated by e.g. Price
        num_steps = [("scaler", StandardScaler()),
                     ("imputer", KNNImputer(n_neighbors=5, add_indicator=add_indicator))]
    elif num_impute == "iterative":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        num_steps = [("scaler", StandardScaler()),
                     ("imputer", IterativeImputer(max_iter=10, random_state=0, add_indicator=add_indicator))]
    elif num_impute == "none":
        num_steps = [("scaler", StandardScaler())]          # StandardScaler keeps NaN as NaN
    else:
        raise ValueError(num_impute)

    cat_steps = [("imputer", SimpleImputer(strategy=cat_impute, fill_value="missing"))]
    if encoding == "onehot":
        cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency)))
    elif encoding == "ordinal":
        cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
    elif encoding == "target":
        cat_steps.append(("encoder", TargetEncoder(target_type="multiclass")))
    else:
        raise ValueError(encoding)

    return ColumnTransformer([("num", Pipeline(num_steps), num_cols),
                              ("cat", Pipeline(cat_steps), cat_cols)])
