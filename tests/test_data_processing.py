import pandas as pd

from src.data_processing import (
    TARGET_COLUMN,
    add_time_features,
    assign_high_risk_labels,
    build_customer_aggregates,
)


def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionId": [f"TransactionId_{idx}" for idx in range(1, 7)],
            "BatchId": ["BatchId_1"] * 6,
            "AccountId": [
                "AccountId_1",
                "AccountId_1",
                "AccountId_2",
                "AccountId_3",
                "AccountId_4",
                "AccountId_4",
            ],
            "SubscriptionId": ["SubscriptionId_1"] * 6,
            "CustomerId": [
                "CustomerId_1",
                "CustomerId_1",
                "CustomerId_2",
                "CustomerId_3",
                "CustomerId_4",
                "CustomerId_4",
            ],
            "CurrencyCode": ["UGX"] * 6,
            "CountryCode": [256] * 6,
            "ProviderId": ["ProviderId_1"] * 6,
            "ProductId": ["ProductId_1"] * 6,
            "ProductCategory": ["airtime"] * 6,
            "ChannelId": ["ChannelId_1"] * 6,
            "Amount": [100, 200, 25, 5000, 10, 20],
            "Value": [100, 200, 25, 5000, 10, 20],
            "TransactionStartTime": [
                "2018-11-01T01:00:00Z",
                "2018-11-03T05:00:00Z",
                "2018-11-02T12:00:00Z",
                "2018-11-20T18:00:00Z",
                "2018-10-01T06:00:00Z",
                "2018-10-02T07:00:00Z",
            ],
            "PricingStrategy": [2] * 6,
            "FraudResult": [0] * 6,
        }
    )


def test_add_time_features_returns_expected_columns():
    transformed = add_time_features(sample_transactions())

    assert {
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year",
    }.issubset(transformed.columns)
    assert transformed.loc[0, "transaction_hour"] == 1


def test_build_customer_aggregates_calculates_transaction_count():
    aggregates = build_customer_aggregates(sample_transactions())
    customer_1 = aggregates[aggregates["CustomerId"] == "CustomerId_1"].iloc[0]

    assert customer_1["transaction_count"] == 2
    assert customer_1["total_transaction_amount"] == 300


def test_assign_high_risk_labels_returns_binary_target():
    labels = assign_high_risk_labels(sample_transactions())

    assert TARGET_COLUMN in labels.columns
    assert set(labels[TARGET_COLUMN].unique()).issubset({0, 1})
