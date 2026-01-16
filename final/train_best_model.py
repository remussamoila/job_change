import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42

TRAIN_PATH = Path("data/job_change_train.csv")
TEST_PATH = Path("data/job_change_test.csv")
BEST_CFG_PATH = Path("artifacts/best_model.json")
OUT_PATH = Path("final/predictions.csv")
OUT_PATH.parent.mkdir(exist_ok=True)

TARGET_COL = "willing_to_change_job"
ID_COL = "id"

EXPECTED_NUMERIC = [
    "age",
    "relative_wage",
    "years_since_job_change",
    "years_of_experience",
    "hours_of_training",
]


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _map_target(y: pd.Series) -> pd.Series:
    if y.dtype == "O":
        y2 = y.astype(str).str.strip().str.lower()
        return y2.map({"yes": 1, "no": 0}).astype("int64")
    return y.astype("int64")


def build_ohe_preprocessor(X: pd.DataFrame):
    cat_cols = [
        c for c in X.columns
        if c != ID_COL and (X[c].dtype == "object" or str(X[c].dtype) == "category" or X[c].dtype == "bool")
    ]
    num_cols = [
        c for c in X.columns
        if c != ID_COL and c not in cat_cols
    ]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre, cat_cols


def main():
    if not BEST_CFG_PATH.exists():
        raise FileNotFoundError(
            f"Missing {BEST_CFG_PATH}. Run: python src/01_model_search.py first."
        )

    cfg = json.loads(BEST_CFG_PATH.read_text())
    best_model = cfg["best_model"]
    best_params = cfg["best_params"]

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train = _coerce_numeric(train, EXPECTED_NUMERIC)
    test = _coerce_numeric(test, EXPECTED_NUMERIC)

    X_train = train.drop(columns=[TARGET_COL])
    y_train = _map_target(train[TARGET_COL])

    # Build preprocessors
    ohe_pre, cb_cat_cols = build_ohe_preprocessor(X_train)
    cb_cat_idx = [X_train.columns.get_loc(c) for c in cb_cat_cols]

    # ----- Instantiate ONLY the chosen model -----
    if best_model == "CatBoost":
        # Minimal imputation for CatBoost
        X_cb = X_train.copy()
        test_cb = test.copy()

        for c in EXPECTED_NUMERIC:
            if c in X_cb.columns:
                med = X_cb[c].median()
                X_cb[c] = X_cb[c].fillna(med)
                test_cb[c] = test_cb[c].fillna(med)

        for c in cb_cat_cols:
            mode = X_cb[c].mode(dropna=True)
            fill_val = mode.iloc[0] if len(mode) else "missing"
            X_cb[c] = X_cb[c].fillna(fill_val).astype(str)
            test_cb[c] = test_cb[c].fillna(fill_val).astype(str)

        model = CatBoostClassifier(
            **best_params,
            loss_function="Logloss",
            eval_metric="BalancedAccuracy",
            random_seed=RANDOM_STATE,
            verbose=False
        )
        model.fit(X_cb, y_train, cat_features=cb_cat_idx)
        pred = model.predict(test_cb).astype(int).reshape(-1)

    else:
        # sklearn pipeline models with OHE
        if best_model == "LogReg":
            model = LogisticRegression(max_iter=5000, solver="saga", random_state=RANDOM_STATE)
        elif best_model == "LinearSVC":
            model = LinearSVC(random_state=RANDOM_STATE)
        elif best_model == "RandomForest":
            model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        elif best_model == "XGBoost":
            model = XGBClassifier(
                random_state=RANDOM_STATE,
                n_estimators=800,
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=-1
            )
        elif best_model == "LightGBM":
            model = LGBMClassifier(random_state=RANDOM_STATE, n_estimators=2000, n_jobs=-1)
        else:
            raise ValueError(f"Unknown best_model: {best_model}")

        pipe = Pipeline(steps=[
            ("preprocess", ohe_pre),
            ("model", model),
        ])

        # Apply best params (they are already in correct key format for sklearn models)
        pipe.set_params(**best_params)

        pipe.fit(X_train, y_train)
        pred = pipe.predict(test).astype(int)

    # Save exactly as required: id + predicted value
    out = pd.DataFrame({
        ID_COL: test[ID_COL],
        TARGET_COL: pred
    })
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()

