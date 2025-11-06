# creditdefault.py
import pandas as pd
import numpy as np

def preprocess_raw_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess raw uploaded data to match model feature expectations.
    Adjust column names and transformations to match the training phase.
    """

    df = raw_df.copy()

    # --- Example: standardize column names ---
    df.columns = [c.strip().lower() for c in df.columns]

    # --- Basic feature engineering example ---
    # Calculate age if dob and cdate exist
    if "dob" in df.columns and "cdate" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
        df["cdate"] = pd.to_datetime(df["cdate"], errors="coerce")
        df["age"] = ((df["cdate"] - df["dob"]).dt.days / 365.25).round(1)
    elif "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
        df["age"] = 2021 - df["dob"].dt.year

    # Handle missing values
    df = df.fillna({
        "marital_status": "missing",
        "job_type": "missing",
        "job_industry": "missing",
        "address_provinsi": "missing",
        "loan_purpose": "missing",
        "loan_purpose_desc": "missing"
    })

    # Numeric columns: fill missing with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Example categorical encoding (keep raw string for model pipeline)
    cat_cols = [
        "marital_status", "job_type", "job_industry",
        "address_provinsi", "loan_purpose", "loan_purpose_desc"
    ]
    for col in cat_cols:
        if col not in df.columns:
            df[col] = "missing"

    # --- Select only features used in model training ---
    # Adjust this list according to your model's final selected_features
    selected_features = [
        "loan_amount", "loan_duration", "installment_amount",
        "dpd", "age", "marital_status", "job_type",
        "job_industry", "address_provinsi", "loan_purpose_desc"
    ]
    available = [f for f in selected_features if f in df.columns]
    X = df[available].copy()

    return X


def predict_raw_data(model, raw_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Main function called from Streamlit.
    Takes raw_df, preprocesses it, applies model.predict_proba,
    and returns a DataFrame with probability of default and predicted class.
    """

    # Preprocess
    X = preprocess_raw_data(raw_df)

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        probs = model.decision_function(X)
    else:
        probs = model.predict(X)

    # Predict class based on threshold
    preds = (probs >= threshold).astype(int)

    # Build output DataFrame
    result_df = pd.DataFrame({
        "customer_id": raw_df.get("customer_id", pd.Series(range(len(X)))),
        "pd": probs,
        "predicted_label": preds
    })

    # Merge back some key info if needed
    keep_cols = [
        "loan_amount", "loan_duration", "installment_amount",
        "dpd", "marital_status", "job_type", "job_industry",
        "address_provinsi", "loan_purpose_desc"
    ]
    for col in keep_cols:
        if col in raw_df.columns and col not in result_df.columns:
            result_df[col] = raw_df[col]

    return result_df
