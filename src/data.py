"""Load and clean the Telco Customer Churn data."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

TARGET = "Churn"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL


def load_clean(path: Path = RAW_CSV) -> pd.DataFrame:
    """Return the cleaned data: one row per customer, target as 0/1."""
    df = pd.read_csv(path)

    # TotalCharges is text because 11 brand-new customers (tenure 0) have a blank value.
    # They haven't been billed yet, so their total is 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"].eq(0) & df["TotalCharges"].isna(), "TotalCharges"] = 0.0

    # SeniorCitizen is stored as 0/1 while every other yes/no field is text: make it consistent.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})

    assert df["customerID"].is_unique
    assert df[FEATURES + [TARGET]].notna().all().all()
    return df
