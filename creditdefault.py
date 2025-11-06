# creditdefault.py

import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64tz_dtype


def _normalize_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all datetime columns are parsed and timezone-naive
    to avoid 'tz-naive vs tz-aware' subtraction errors.
    """
    datetime_cols = [
        "cdate",
        "dob",
        "fund_transfer_ts",
        "due_date",
        "paid_date",
    ]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if is_datetime64tz_dtype(df[col]):
                # Convert timezone-aware datetimes to naive (no tz)
                df[col] = df[col].dt.tz_convert(None)
    return df


def _ensure_id_string(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast ID-like columns to pandas string dtype so things like
    2003023548799 never appear as 2003023548799.0.
    """
    id_cols = ["application_id", "customer_id", "loan_id", "payment_id"]
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def preprocess_raw_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess raw uploaded data before sending it into the model.
    The model pipeline should handle most transformations; this function
    keeps the logic minimal and stable.

    Steps:
    - Normalize datetime columns and drop timezone info.
    - Ensure ID columns are strings.
    - Fill numeric missing values with median.
    - Fill categorical missing values with 'missing'.
    - Select a subset of features expected by the model.
    """
    df = raw_df.copy()

    # Fix datetime columns and make them timezone-naive
    df = _normalize_datetimes(df)

    # IDs as string
    df = _ensure_id_string(df)

    # Numeric missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Categorical missing values
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("missing")

    # Select features that the model was trained on.
    # Adjust this list if your training pipeline uses a different set.
    selected_features = [
        "loan_amount",
        "loan_duration",
        "installment_amount",
        "paid_amount",
        "dpd",
        "dependent",
        "marital_status",
        "job_type",
        "job_industry",
        "address_provinsi",
        "loan_purpose",
        "loan_purpose_desc",
    ]
    available = [c for c in selected_features if c in df.columns]
    X = df[available].copy()

    return X


def predict_raw_data(model, raw_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Main entry point called by the Streamlit app.

    - Preprocess the raw DataFrame.
    - Use model.predict_proba (or decision_function / predict) to get a score.
    - Interpret this score as probability of default (PD).
    - Create a predicted label using the threshold.
    - Return a DataFrame with PD, predicted label, IDs, and some key features.

    Parameters
    ----------
    model : fitted model or pipeline
        The object loaded from credit_scoring_best_model.pkl.
    raw_df : pd.DataFrame
        Raw combined_df-style data uploaded by the user.
    threshold : float
        Cut-off for mapping PD to binary default label (0/1).

    Returns
    -------
    result_df : pd.DataFrame
        Scored records with columns:
        - customer_id, application_id, loan_id (as string)
        - pd (probability of default)
        - predicted_label (0 = non-default, 1 = default)
        - selected input features for reference
    """
    # Preprocess features
    X = preprocess_raw_data(raw_df)

    # Predict default probability
    if hasattr(model, "predict_proba"):
        pd_scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        pd_scores = model.decision_function(X)
    else:
        # As a last resort, use the raw prediction as "score"
        pd_scores = model.predict(X)

    # Binary label based on threshold
    predicted_label = (pd_scores >= threshold).astype(int)

    # Ensure IDs are strings in output as well
    raw_ids = raw_df.copy()
    raw_ids = _ensure_id_string(raw_ids)

    result_df = pd.DataFrame({
        "customer_id": raw_ids.get("customer_id", pd.Series(range(len(raw_df)))).astype("string"),
        "application_id": raw_ids.get("application_id", pd.Series([None] * len(raw_df))).astype("string"),
        "loan_id": raw_ids.get("loan_id", pd.Series([None] * len(raw_df))).astype("string"),
        "pd": pd_scores,
        "predicted_label": predicted_label,
    })

    # Attach useful context variables from the raw data
    extra_cols = [
        "loan_amount",
        "loan_duration",
        "installment_amount",
        "paid_amount",
        "dpd",
        "dependent",
        "marital_status",
        "job_type",
        "job_industry",
        "address_provinsi",
        "loan_purpose",
        "loan_purpose_desc",
    ]
    for col in extra_cols:
        if col in raw_df.columns and col not in result_df.columns:
            result_df[col] = raw_df[col]

    return result_df
