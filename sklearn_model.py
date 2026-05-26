from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class SklearnMetrics:
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: float


def train_sklearn(text_train: list[str], y_train: list[int], text_val: list[str], y_val: list[int]) -> tuple[Pipeline, SklearnMetrics]:
    """
    Strong baseline: TF-IDF (1-2 grams) + LinearSVC calibrated for probabilities.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    base = LinearSVC(class_weight="balanced")
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)

    pipe: Pipeline = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    pipe.fit(text_train, y_train)

    proba = pipe.predict_proba(text_val)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = SklearnMetrics(
        accuracy=float(accuracy_score(y_val, pred)),
        f1=float(f1_score(y_val, pred)),
        precision=float(precision_score(y_val, pred)),
        recall=float(recall_score(y_val, pred)),
        roc_auc=float(roc_auc_score(y_val, proba)),
    )
    return pipe, metrics


def predict_proba(model: Pipeline, texts: list[str]) -> np.ndarray:
    return model.predict_proba(texts)[:, 1]


def save(model: Pipeline, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sklearn.joblib"
    joblib.dump(model, path)
    return path


def load(model_dir: str | Path) -> Any:
    model_dir = Path(model_dir)
    return joblib.load(model_dir / "sklearn.joblib")

