"""Feature engineering and proxy target creation for the credit risk model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGET_COLUMN = "is_high_risk"
IDENTIFIER_COLUMNS = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
]


def add_time_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic calendar features from TransactionStartTime."""
    output = transactions.copy()
    timestamps = pd.to_datetime(output["TransactionStartTime"], errors="coerce")
    output["transaction_hour"] = timestamps.dt.hour
    output["transaction_day"] = timestamps.dt.day
    output["transaction_month"] = timestamps.dt.month
    output["transaction_year"] = timestamps.dt.year
    return output


def build_customer_aggregates(transactions: pd.DataFrame) -> pd.DataFrame:
    """Create per-customer aggregate transaction features."""
    aggregates = (
        transactions.groupby("CustomerId")
        .agg(
            total_transaction_amount=("Amount", "sum"),
            average_transaction_amount=("Amount", "mean"),
            transaction_count=("TransactionId", "count"),
            std_transaction_amount=("Amount", "std"),
            total_transaction_value=("Value", "sum"),
            average_transaction_value=("Value", "mean"),
            fraud_count=("FraudResult", "sum"),
        )
        .reset_index()
    )
    aggregates["std_transaction_amount"] = aggregates[
        "std_transaction_amount"
    ].fillna(0)
    return aggregates


def calculate_rfm(
    transactions: pd.DataFrame, snapshot_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Calculate Recency, Frequency, and Monetary values per customer."""
    working = transactions.copy()
    working["TransactionStartTime"] = pd.to_datetime(
        working["TransactionStartTime"], errors="coerce"
    )
    if snapshot_date is None:
        snapshot_date = working["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = (
        working.groupby("CustomerId")
        .agg(
            recency=(
                "TransactionStartTime",
                lambda dates: (snapshot_date - dates.max()).days,
            ),
            frequency=("TransactionId", "count"),
            monetary=("Value", "sum"),
        )
        .reset_index()
    )
    return rfm


def assign_high_risk_labels(
    transactions: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Cluster RFM profiles and label the least engaged group as high risk."""
    rfm = calculate_rfm(transactions)
    feature_columns = ["recency", "frequency", "monetary"]

    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm[feature_columns])
    cluster_count = min(n_clusters, len(rfm))
    kmeans = KMeans(
        n_clusters=cluster_count,
        random_state=random_state,
        n_init=10,
    )
    rfm["risk_cluster"] = kmeans.fit_predict(scaled_rfm)

    cluster_profile = rfm.groupby("risk_cluster")[feature_columns].mean()
    risk_score = (
        cluster_profile["recency"].rank(ascending=True)
        + cluster_profile["frequency"].rank(ascending=False)
        + cluster_profile["monetary"].rank(ascending=False)
    )
    high_risk_cluster = risk_score.idxmax()
    rfm[TARGET_COLUMN] = (rfm["risk_cluster"] == high_risk_cluster).astype(int)
    return rfm[["CustomerId", TARGET_COLUMN, "recency", "frequency", "monetary"]]


class TransactionFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transform raw transactions into engineered transaction-level model rows."""

    def __init__(
        self,
        include_target: bool = True,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.include_target = include_target
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transactions = add_time_features(X)
        aggregates = build_customer_aggregates(transactions)
        rfm = assign_high_risk_labels(transactions, random_state=self.random_state)

        engineered = transactions.merge(aggregates, on="CustomerId", how="left")
        engineered = engineered.merge(rfm, on="CustomerId", how="left")

        if not self.include_target:
            engineered = engineered.drop(columns=[TARGET_COLUMN], errors="ignore")

        return engineered


@dataclass
class _WoEBin:
    edges: np.ndarray
    mapping: dict[int, float]
    default: float
    iv: float


class ModelReadyTransformer(BaseEstimator, TransformerMixin):
    """Impute, encode, scale, and add WoE/IV features as a DataFrame."""

    def __init__(
        self,
        target_column: str = TARGET_COLUMN,
        identifier_columns: Iterable[str] | None = None,
        max_categories: int = 25,
        woe_bins: int = 5,
    ) -> None:
        self.target_column = target_column
        self.identifier_columns = identifier_columns
        self.max_categories = max_categories
        self.woe_bins = woe_bins

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        data = X.copy()
        self.target_present_ = self.target_column in data.columns
        target = data[self.target_column] if self.target_present_ else y
        features = self._feature_frame(data)

        self.numeric_columns_ = features.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.categorical_columns_ = [
            col for col in features.columns if col not in self.numeric_columns_
        ]

        self.numeric_medians_ = features[self.numeric_columns_].median()
        self.numeric_means_ = features[self.numeric_columns_].mean()
        self.numeric_stds_ = (
            features[self.numeric_columns_].std().replace(0, 1).fillna(1)
        )

        self.category_values_ = {}
        for col in self.categorical_columns_:
            values = (
                features[col]
                .fillna("missing")
                .astype(str)
                .value_counts()
                .head(self.max_categories)
                .index.tolist()
            )
            self.category_values_[col] = values

        self.woe_bins_ = {}
        if target is not None:
            clean_target = (
                pd.Series(target).fillna(0).astype(int).reset_index(drop=True)
            )
            for col in self.numeric_columns_:
                self.woe_bins_[col] = self._fit_woe_feature(
                    features[col].reset_index(drop=True), clean_target
                )

        self.output_columns_ = self._transform_features(features).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        features = self._feature_frame(X)
        transformed = self._transform_features(features)
        return transformed.reindex(columns=self.output_columns_, fill_value=0)

    def _feature_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        identifier_columns = self.identifier_columns or IDENTIFIER_COLUMNS
        drop_columns = [*identifier_columns, self.target_column]
        return data.drop(columns=drop_columns, errors="ignore")

    def _transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        parts = []
        numeric = features.reindex(columns=self.numeric_columns_).copy()
        numeric = numeric.fillna(self.numeric_medians_)
        scaled = (numeric - self.numeric_means_) / self.numeric_stds_
        scaled.columns = [f"{col}_scaled" for col in scaled.columns]
        parts.append(scaled)

        if self.woe_bins_:
            woe_features = pd.DataFrame(index=features.index)
            for col, fitted_bin in self.woe_bins_.items():
                woe_features[f"{col}_woe"] = self._apply_woe_feature(
                    features[col], fitted_bin
                )
            parts.append(woe_features)

        categorical = pd.DataFrame(index=features.index)
        for col, allowed_values in self.category_values_.items():
            values = features[col].fillna("missing").astype(str)
            values = values.where(values.isin(allowed_values), "other")
            for category in [*allowed_values, "other"]:
                column_name = f"{col}_{category}"
                categorical[column_name] = (values == category).astype(int)
        parts.append(categorical)

        return pd.concat(parts, axis=1)

    def _fit_woe_feature(self, values: pd.Series, target: pd.Series) -> _WoEBin:
        filled = values.fillna(values.median())
        unique_count = filled.nunique(dropna=True)
        if unique_count <= 1:
            return _WoEBin(np.array([-np.inf, np.inf]), {0: 0.0}, 0.0, 0.0)

        quantiles = min(self.woe_bins, unique_count)
        bins = pd.qcut(filled, q=quantiles, duplicates="drop")
        categories = bins.cat.categories
        edges = np.array([categories[0].left, *[cat.right for cat in categories]])
        edges[0] = -np.inf
        edges[-1] = np.inf
        indexed_bins = pd.cut(filled, bins=edges, labels=False, include_lowest=True)

        total_good = max((target == 0).sum(), 1)
        total_bad = max((target == 1).sum(), 1)
        mapping = {}
        information_value = 0.0
        for bin_id in sorted(indexed_bins.dropna().unique()):
            mask = indexed_bins == bin_id
            good_rate = ((target[mask] == 0).sum() + 0.5) / total_good
            bad_rate = ((target[mask] == 1).sum() + 0.5) / total_bad
            woe = float(np.log(good_rate / bad_rate))
            mapping[int(bin_id)] = woe
            information_value += (good_rate - bad_rate) * woe

        default = float(np.mean(list(mapping.values()))) if mapping else 0.0
        return _WoEBin(edges, mapping, default, float(information_value))

    def _apply_woe_feature(self, values: pd.Series, fitted_bin: _WoEBin) -> pd.Series:
        bin_ids = pd.cut(
            values.fillna(values.median()),
            bins=fitted_bin.edges,
            labels=False,
            include_lowest=True,
        )
        return bin_ids.map(fitted_bin.mapping).fillna(fitted_bin.default).astype(float)


def build_feature_pipeline(include_target: bool = True) -> Pipeline:
    """Return a sklearn Pipeline that produces a model-ready DataFrame."""
    return Pipeline(
        steps=[
            (
                "feature_engineering",
                TransactionFeatureEngineer(include_target=include_target),
            ),
            ("model_ready", ModelReadyTransformer()),
        ]
    )


def process_transactions(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Fit the full processing pipeline and return a model-ready DataFrame."""
    pipeline = build_feature_pipeline(include_target=True)
    return pipeline.fit_transform(raw_data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build processed credit-risk features."
    )
    parser.add_argument("--input", default="data/raw/training.csv")
    parser.add_argument("--output", default="data/processed/processed_training.csv")
    args = parser.parse_args()

    raw_path = Path(args.input)
    output_path = Path(args.output)
    raw = pd.read_csv(raw_path)
    engineered = TransactionFeatureEngineer(include_target=True).fit_transform(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_csv(output_path, index=False)
    print(f"Wrote processed dataset with {len(engineered)} rows to {output_path}")


if __name__ == "__main__":
    main()
