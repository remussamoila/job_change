"""
Unified model search script:
- Uses GridSearchCV for small/fast models (LogReg, LinearSVC)
- Uses RandomizedSearchCV for larger spaces (RandomForest, RandomForest+SMOTE, XGBoost, LightGBM)
- Uses manual randomized CV for CatBoost (native categorical)
- Saves:
  - artifacts/leaderboard.csv
  - artifacts/best_model.json
  - artifacts/predictions_from_best_search.csv

Models evaluated

Baselines / linear:
1) Logistic Regression (OHE) GridSearchCV
2) Linear SVM (LinearSVC, OHE) GridSearchCV

Tree ensembles:
3) Random Forest (OHE, class_weight="balanced")
4) Random Forest + SMOTE (OHE + SMOTE inside CV, no leakage)

Boosting:
5) XGBoost (OHE)
6) LightGBM (OHE)
7) CatBoost (native categorical handling; no OHE)

Metric: Balanced Accuracy (5-fold Stratified CV)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Rebalancing
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

TRAIN_PATH = Path("data/job_change_train.csv")
TEST_PATH = Path("data/job_change_test.csv")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

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
    num_cols = [c for c in X.columns if c != ID_COL and c not in cat_cols]

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
    return pre, num_cols, cat_cols


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train = _coerce_numeric(train, EXPECTED_NUMERIC)
    test = _coerce_numeric(test, EXPECTED_NUMERIC)

    X = train.drop(columns=[TARGET_COL])
    y = _map_target(train[TARGET_COL])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scorer = make_scorer(balanced_accuracy_score)

    ohe_pre, num_cols, cat_cols = build_ohe_preprocessor(X)

    results = []
    best_overall = None

    # Each candidate is either:
    # - ("name", pipeline, "grid", param_grid)
    # - ("name", pipeline, "rand", param_distributions, n_iter)
    candidates = []

    # -------------------------
    # GRIDSEARCH (small spaces)
    # -------------------------
    candidates.append((
        "LogReg",
        Pipeline([
            ("preprocess", ohe_pre),
            ("model", LogisticRegression(max_iter=5000, solver="saga", random_state=RANDOM_STATE))
        ]),
        "grid",
        {
            "model__C": np.logspace(-3, 2, 10),
            "model__penalty": ["l2"],
        }
    ))

    candidates.append((
        "LinearSVC",
        Pipeline([
            ("preprocess", ohe_pre),
            ("model", LinearSVC(random_state=RANDOM_STATE))
        ]),
        "grid",
        {
            "model__C": np.logspace(-3, 2, 10),
        }
    ))

    candidates.append((
        "KNN",
        Pipeline([
            ("preprocess", ohe_pre),
            ("model", KNeighborsClassifier())
        ]),
        "grid",
        {
            "model__n_neighbors": [5, 11, 21, 31],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],  # Manhattan vs Euclidean
        }
    ))

    
    # -----------------------------
    # RANDOMIZEDSEARCH (larger)
    # -----------------------------
    candidates.append((
        "RandomForest",
        Pipeline([
            ("preprocess", ohe_pre),
            ("model", RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            ))
        ]),
        "rand",
        {
            "model__n_estimators": [600, 1000],
            "model__max_depth": [6, 8, 10, None],
            "model__min_samples_leaf": [2, 5, 10],
            "model__max_features": ["sqrt", "log2", None],
        },
        12
    ))

    candidates.append((
        "RandomForest_SMOTE",
        ImbPipeline([
            ("preprocess", ohe_pre),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        "rand",
        {
            "model__n_estimators": [600, 1000],
            "model__max_depth": [6, 8, 10, None],
            "model__min_samples_leaf": [2, 5, 10],
            "model__max_features": ["sqrt", "log2", None],
            "model__class_weight": [None],
        },
        12
    ))

    candidates.append((
        "XGBoost",
        Pipeline([
            ("preprocess", ohe_pre),
            ("model", XGBClassifier(
                random_state=RANDOM_STATE,
                n_estimators=300,
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=1
            ))
        ]),
        "rand",
        {
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__min_child_weight": [1, 5, 10],
            "model__reg_lambda": [1.0, 3.0],
        },
        8
    ))

    candidates.append((
        "LightGBM",
        Pipeline([
            ("preprocess", ohe_pre),
            ("model", LGBMClassifier(
                random_state=RANDOM_STATE,
                n_estimators=600,
                n_jobs=-1,
                verbose=-1
            ))
        ]),
        "rand",
        {
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__num_leaves": [15, 31, 63, 127],
            "model__max_depth": [-1, 4, 6, 8, 10],
            "model__min_child_samples": [10, 20, 50, 100],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_lambda": [0.0, 0.5, 1.0, 2.0],
        },
        8
    ))

    # -------------------------
    # Run sklearn searches
    # -------------------------
    for entry in candidates:
        name, pipe, mode = entry[0], entry[1], entry[2]
        print(f"\n=== Tuning: {name} ===")

        if mode == "grid":
            param_grid = entry[3]
            search = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring=scorer,
                cv=cv,
                n_jobs=1,
                refit=True,
                verbose=2
            )
        elif mode == "rand":
            param_dist, n_iter = entry[3], entry[4]
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring=scorer,
                cv=cv,
                random_state=RANDOM_STATE,
                n_jobs=1,
                refit=True,
                verbose=2
            )
        else:
            raise ValueError(f"Unknown mode={mode}")

        search.fit(X, y)

        mean_score = float(search.best_score_)
        std_score = float(search.cv_results_["std_test_score"][search.best_index_])
        best_params = search.best_params_

        print(f"Best CV balanced accuracy: {mean_score:.4f} ± {std_score:.4f}")

        results.append({
            "model": name,
            "cv_balanced_acc_mean": mean_score,
            "cv_balanced_acc_std": std_score,
            "best_params": best_params
        })

        if best_overall is None or mean_score > best_overall["cv_balanced_acc_mean"]:
            best_overall = {
                "model": name,
                "cv_balanced_acc_mean": mean_score,
                "cv_balanced_acc_std": std_score,
                "best_params": best_params,
                "best_estimator": search.best_estimator_
            }

    # -------------------------
    # CatBoost (native categorical)
    # -------------------------
    print("\n=== Tuning: CatBoost (native categorical) ===")

    cb_cat_cols = [
        c for c in X.columns
        if c != ID_COL and (X[c].dtype == "object" or str(X[c].dtype) == "category" or X[c].dtype == "bool")
    ]
    cb_cat_idx = [X.columns.get_loc(c) for c in cb_cat_cols]

    X_cb = X.copy()
    test_cb = test.copy()

    for c in EXPECTED_NUMERIC:
        if c in X_cb.columns:
            med = X_cb[c].median()
            X_cb[c] = X_cb[c].fillna(med)
            test_cb[c] = test_cb[c].fillna(med)

    for c in cb_cat_cols:
        mode_val = X_cb[c].mode(dropna=True)
        fill_val = mode_val.iloc[0] if len(mode_val) else "missing"
        X_cb[c] = X_cb[c].fillna(fill_val).astype(str)
        test_cb[c] = test_cb[c].fillna(fill_val).astype(str)

    rng = np.random.default_rng(RANDOM_STATE)
    cb_param_samples = []
    for _ in range(20):
        cb_param_samples.append({
            "depth": int(rng.choice([4, 5, 6, 7, 8, 10])),
            "learning_rate": float(rng.choice([0.02, 0.05, 0.1, 0.15])),
            "l2_leaf_reg": float(rng.choice([1.0, 3.0, 5.0, 10.0])),
            "iterations": int(rng.choice([800, 1200, 1600])),
        })

    best_cb = None

    for i, p in enumerate(cb_param_samples, start=1):
        fold_scores = []
        for tr_idx, va_idx in cv.split(X_cb, y):
            X_tr, X_va = X_cb.iloc[tr_idx], X_cb.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            model = CatBoostClassifier(
                **p,
                loss_function="Logloss",
                eval_metric="BalancedAccuracy",
                random_seed=RANDOM_STATE,
                verbose=False
            )
            model.fit(X_tr, y_tr, cat_features=cb_cat_idx)
            pred = model.predict(X_va).astype(int).reshape(-1)
            fold_scores.append(balanced_accuracy_score(y_va, pred))

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))

        if best_cb is None or mean_score > best_cb["cv_balanced_acc_mean"]:
            best_cb = {
                "model": "CatBoost",
                "cv_balanced_acc_mean": mean_score,
                "cv_balanced_acc_std": std_score,
                "best_params": p
            }

        print(f"  Sample {i:02d}: mean={mean_score:.4f}, std={std_score:.4f}, params={p}")

    print(f"\nBest CatBoost CV balanced accuracy: {best_cb['cv_balanced_acc_mean']:.4f} ± {best_cb['cv_balanced_acc_std']:.4f}")

    results.append({
        "model": "CatBoost",
        "cv_balanced_acc_mean": best_cb["cv_balanced_acc_mean"],
        "cv_balanced_acc_std": best_cb["cv_balanced_acc_std"],
        "best_params": {f"model__{k}": v for k, v in best_cb["best_params"].items()}
    })

    if best_cb["cv_balanced_acc_mean"] > best_overall["cv_balanced_acc_mean"]:
        best_overall = {
            "model": "CatBoost",
            "cv_balanced_acc_mean": best_cb["cv_balanced_acc_mean"],
            "cv_balanced_acc_std": best_cb["cv_balanced_acc_std"],
            "best_params": best_cb["best_params"],
            "best_estimator": None,
            "catboost_cat_cols": cb_cat_cols
        }

    # -------------------------
    # Save artifacts
    # -------------------------
    leaderboard = pd.DataFrame(results).sort_values("cv_balanced_acc_mean", ascending=False)
    leaderboard_path = ARTIFACTS_DIR / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    print(f"\nSaved leaderboard -> {leaderboard_path}")

    best_cfg = {
        "best_model": best_overall["model"],
        "cv_balanced_acc_mean": best_overall["cv_balanced_acc_mean"],
        "cv_balanced_acc_std": best_overall["cv_balanced_acc_std"],
        "best_params": best_overall["best_params"],
        "notes": "Balanced accuracy estimated via 5-fold Stratified CV on training set."
    }
    cfg_path = ARTIFACTS_DIR / "best_model.json"
    cfg_path.write_text(json.dumps(best_cfg, indent=2))
    print(f"Saved best model config -> {cfg_path}")

    # Fit best model on full train and predict test
    if best_overall["model"] == "CatBoost":
        p = best_overall["best_params"]
        model = CatBoostClassifier(
            **p,
            loss_function="Logloss",
            eval_metric="BalancedAccuracy",
            random_seed=RANDOM_STATE,
            verbose=False
        )
        model.fit(X_cb, y, cat_features=cb_cat_idx)
        test_pred = model.predict(test_cb).astype(int).reshape(-1)
    else:
        best_pipe = best_overall["best_estimator"]
        best_pipe.fit(X, y)
        test_pred = best_pipe.predict(test).astype(int)

    pred_path = ARTIFACTS_DIR / "predictions_from_best_search.csv"
    out = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: test_pred})
    out.to_csv(pred_path, index=False)
    print(f"Saved predictions from best search -> {pred_path}")

    print("\n=== BEST OVERALL ===")
    print(json.dumps(best_cfg, indent=2))

    # Optional: save probability scores if model supports it (for ROC plots, etc.)
    proba_path = ARTIFACTS_DIR / "test_scores.csv"
    try:
        if best_overall["model"] == "CatBoost":
            test_score = model.predict_proba(test_cb)[:, 1]
        else:
            if hasattr(best_pipe, "predict_proba"):
                test_score = best_pipe.predict_proba(test)[:, 1]
            elif hasattr(best_pipe, "decision_function"):
                test_score = best_pipe.decision_function(test)
            else:
                test_score = None

        if test_score is not None:
            pd.DataFrame({ID_COL: test[ID_COL], "score": test_score}).to_csv(proba_path, index=False)
            print(f"Saved test scores -> {proba_path}")
    except Exception:
        pass

    summary_path = ARTIFACTS_DIR / "summary_for_slides.json"
    summary = {
        "metric": "Balanced Accuracy (5-fold Stratified CV)",
        "n_train": int(train.shape[0]),
        "n_test": int(test.shape[0]),
        "leaderboard_top5": leaderboard.head(5).to_dict(orient="records"),
        "best_model": best_cfg
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved slide summary -> {summary_path}")


if __name__ == "__main__":
    main()

