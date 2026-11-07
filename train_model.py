import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from templates.methodology.phase3.features import FEATURE_COLUMNS, registered_domain


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "static" / "datasets" / "balanced_dataset_complete.csv"
DEFAULT_MODEL = PROJECT_ROOT / "static" / "models" / "url_phishing_logistic_regression.joblib"
DEFAULT_METRICS = PROJECT_ROOT / "static" / "models" / "training_metrics.json"
RANDOM_STATE = 42


def train_model(
    dataset_path: Path = DEFAULT_DATASET,
    model_path: Path = DEFAULT_MODEL,
    metrics_path: Path = DEFAULT_METRICS,
) -> dict:
    data = pd.read_csv(dataset_path)
    required_columns = {"normalized_url", "is_phishing", *FEATURE_COLUMNS}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    data = data.dropna(subset=["normalized_url", "is_phishing"]).copy()
    data["normalized_url"] = data["normalized_url"].astype(str).str.strip()
    data = data[data["normalized_url"] != ""]
    data["is_phishing"] = pd.to_numeric(data["is_phishing"], errors="raise").astype(int)
    if not set(data["is_phishing"].unique()).issubset({0, 1}):
        raise ValueError("is_phishing must contain only binary labels 0 and 1.")

    conflicting_urls = data.groupby("normalized_url")["is_phishing"].nunique()
    conflicting_urls = conflicting_urls[conflicting_urls > 1].index
    data = data[~data["normalized_url"].isin(conflicting_urls)].drop_duplicates(
        subset=["normalized_url"]
    )
    if data["is_phishing"].nunique() != 2:
        raise ValueError("Training data must contain both classes after duplicate cleanup.")

    features = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Feature columns contain missing or non-finite values.")

    groups = data["normalized_url"].map(registered_domain)
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    train_indices, test_indices = next(
        splitter.split(features, data["is_phishing"], groups=groups)
    )
    X_train, X_test = features.iloc[train_indices], features.iloc[test_indices]
    y_train = data["is_phishing"].iloc[train_indices]
    y_test = data["is_phishing"].iloc[test_indices]
    train_domains = set(groups.iloc[train_indices])
    test_domains = set(groups.iloc[test_indices])
    if train_domains.intersection(test_domains):
        raise RuntimeError("Registered domains leaked between training and test sets.")
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Both data splits must contain both classes.")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    phishing_probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_phishing": float(
            precision_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "recall_phishing": float(recall_score(y_test, predictions, pos_label=1)),
        "f1_phishing": float(f1_score(y_test, predictions, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_test, phishing_probabilities)),
        "brier_score": float(brier_score_loss(y_test, phishing_probabilities)),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }
    training_record = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "is_phishing",
        "positive_class": 1,
        "threshold": 0.5,
        "algorithm": "StandardScaler + LogisticRegression",
        "random_state": RANDOM_STATE,
        "split": "StratifiedGroupKFold (5 folds), first fold; grouped by registered domain",
        "dataset": dataset_path.name,
        "rows_after_cleanup": int(len(data)),
        "conflicting_urls_excluded": int(len(conflicting_urls)),
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "train_domains": int(len(train_domains)),
        "test_domains": int(len(test_domains)),
        "train_class_counts": {
            str(label): int(count)
            for label, count in y_train.value_counts().sort_index().items()
        },
        "test_class_counts": {
            str(label): int(count)
            for label, count in y_test.value_counts().sort_index().items()
        },
        "metrics": metrics,
        "probability_note": (
            "The dataset is artificially balanced; probabilities may not represent "
            "real-world phishing prevalence."
        ),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(training_record, model_path)
    metrics_path.write_text(
        json.dumps(
            {key: value for key, value in training_record.items() if key != "model"},
            indent=2,
        ),
        encoding="utf-8",
    )
    return training_record


if __name__ == "__main__":
    result = train_model()
    print(f"Model saved to: {DEFAULT_MODEL}")
    print(f"Metrics saved to: {DEFAULT_METRICS}")
    print(json.dumps(result["metrics"], indent=2))
