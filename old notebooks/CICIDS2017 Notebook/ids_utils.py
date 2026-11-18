# Helper module saved next to the notebook so that the saved model / scaler can be re-loaded anywhere.
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


def signed_log1p(X):
    """sign(x) * log1p(|x|): tames CICIDS2017's heavy tails while keeping negative values (e.g. Init_Win_bytes = -1)."""
    X = np.asarray(X)
    if X.dtype.kind != "f":
        X = X.astype("float32")
    return np.sign(X) * np.log1p(np.abs(X))


class PrefitSoftVoting:
    """Average predict_proba of already-fitted multiclass models; prediction = argmax of the mean probability."""
    def __init__(self, models, weights=None):
        self.models = list(models)
        self.weights = weights
        self.classes_ = np.asarray(self.models[0].classes_)

    def predict_proba(self, X):
        P = np.stack([m.predict_proba(X) for m in self.models], axis=0)
        return np.average(P, axis=0, weights=self.weights)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class PrefitHardVoting:
    """Majority vote of already-fitted multiclass models (ties broken by the mean probability).
    predict_proba = fraction of models voting for each class."""
    def __init__(self, models):
        self.models = list(models)
        self.classes_ = np.asarray(self.models[0].classes_)

    def _scores(self, X):
        n, k = len(X), len(self.classes_)
        votes, soft = np.zeros((n, k)), np.zeros((n, k))
        for m in self.models:
            P = m.predict_proba(X)
            votes[np.arange(n), P.argmax(axis=1)] += 1
            soft += P / len(self.models)
        return votes / len(self.models), soft

    def predict_proba(self, X):
        return self._scores(X)[0]

    def predict(self, X):
        votes, soft = self._scores(X)
        return self.classes_[(votes + 1e-3 * soft).argmax(axis=1)]


class IDSDetector:
    """Load the saved artefacts and score RAW CICIDS2017 flow rows (same columns as the CSVs)."""
    def __init__(self, model_dir="models"):
        d = Path(model_dir)
        self.model = joblib.load(d / "final_model.joblib")
        self.preprocessor = joblib.load(d / "preprocessor.joblib")
        self.label_encoder = joblib.load(d / "label_encoder.joblib")
        self.feature_columns = json.load(open(d / "feature_columns.json"))
        self.selected_features = json.load(open(d / "selected_features.json"))

    def _prep(self, raw_df):
        X = raw_df.copy()
        X.columns = X.columns.str.strip()
        X = X[self.feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.preprocessor.transform(X)[self.selected_features].astype("float32")

    def predict_proba(self, raw_df):
        """(n_flows, n_classes) probabilities; column order = label_encoder.classes_."""
        return self.model.predict_proba(self._prep(raw_df))

    def predict(self, raw_df):
        """Attack-category names."""
        return self.label_encoder.inverse_transform(self.model.predict(self._prep(raw_df)))
