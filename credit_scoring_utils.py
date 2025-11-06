import pandas as pd
import numpy as np
import joblib
from IPython.display import display


# 1. Build Customer-Level Features from combined_df
def build_customer_features_from_combined(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw combined_df (loan + payment + customer info)
    into customer-level features for credit scoring.
    Compatible with your combined_df schema.
    """

    df = combined_df.copy()

    # Ensure identifier columns are strings (IDs are not numeric features)
    id_cols = ['application_id', 'customer_id', 'loan_id', 'payment_id']
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Ensure numeric and datetime types for main numeric/date columns
    numeric_cols = ['loan_amount', 'loan_duration', 'installment_amount',
                    'paid_amount', 'dependent', 'dpd']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

    # Create SLIK-style credit score based on DPD
    if 'dpd' not in df.columns:
        df['dpd'] = np.nan

    cond1 = (df['dpd'].isna() | (df['dpd'] <= 0))
    cond2 = df['dpd'].between(1, 90, inclusive="both")
    cond3 = df['dpd'].between(91, 120, inclusive="both")
    cond4 = df['dpd'].between(121, 180, inclusive="both")
    cond5 = df['dpd'] > 180

    df['slik_score'] = np.select(
        [cond1, cond2, cond3, cond4, cond5],
        [1, 2, 3, 4, 5],
        default=1
    )

    # Loan-level aggregation
    loan_level = (
        df.groupby(['customer_id', 'loan_id'], dropna=False)
        .agg(
            avg_loan_amount=('loan_amount', 'mean'),
            avg_loan_duration=('loan_duration', 'mean'),
            avg_installment=('installment_amount', 'mean'),
            total_paid=('paid_amount', 'sum'),
            total_due=('installment_amount', 'sum'),
            n_payments=('installment_amount', 'count'),
            n_late=('dpd', lambda x: (pd.to_numeric(x, errors='coerce') > 0).sum()),
            avg_dpd=('dpd', lambda x: pd.to_numeric(x, errors='coerce').mean()),
            max_dpd=('dpd', lambda x: pd.to_numeric(x, errors='coerce').max()),
            worst_slik_score=('slik_score', 'max')
        )
        .reset_index()
    )

    # Define loan-level default
    loan_level['default_flag_loan'] = np.where(
        (loan_level['total_paid'].isna()) |
        (loan_level['total_paid'] == 0) |
        (loan_level['max_dpd'] > 90) |
        (loan_level['total_paid'] < 0.8 * loan_level['total_due']),
        1,
        0
    )

    # Customer-level behavioral aggregation
    cust_behavior = (
        loan_level.groupby('customer_id', dropna=False)
        .agg(
            n_loans=('loan_id', 'nunique'),
            n_defaulted_loans=('default_flag_loan', 'sum'),
            default_flag_customer=('default_flag_loan', 'max'),
            avg_loan_amount=('avg_loan_amount', 'mean'),
            max_loan_amount=('avg_loan_amount', 'max'),
            min_loan_amount=('avg_loan_amount', 'min'),
            avg_loan_duration=('avg_loan_duration', 'mean'),
            avg_dpd=('avg_dpd', 'mean'),
            worst_dpd=('max_dpd', 'max'),
            worst_slik_score=('worst_slik_score', 'max'),
            sum_total_paid=('total_paid', 'sum'),
            sum_total_due=('total_due', 'sum'),
            sum_n_late=('n_late', 'sum'),
            sum_n_payments=('n_payments', 'sum')
        )
        .reset_index()
    )

    # Derived ratio features
    cust_behavior['pay_ratio_total'] = (
        cust_behavior['sum_total_paid'] /
        cust_behavior['sum_total_due'].replace(0, np.nan)
    )
    cust_behavior['late_ratio'] = (
        cust_behavior['sum_n_late'] /
        cust_behavior['sum_n_payments'].replace(0, np.nan)
    )
    cust_behavior['ontime_ratio'] = 1 - cust_behavior['late_ratio']

    for col in ['pay_ratio_total', 'late_ratio', 'ontime_ratio']:
        cust_behavior[col] = cust_behavior[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    cust_behavior = cust_behavior.drop(
        columns=['sum_total_paid', 'sum_total_due', 'sum_n_late', 'sum_n_payments']
    )

    # Safe mode for categorical aggregations
    def mode_or_unknown(x):
        x = x.dropna()
        if x.empty:
            return 'unknown'
        m = x.mode()
        return m.iloc[0] if not m.empty else 'unknown'

    cust_demo = (
        df.groupby('customer_id', dropna=False)
        .agg(
            marital_status=('marital_status', mode_or_unknown),
            job_type=('job_type', mode_or_unknown),
            job_industry=('job_industry', mode_or_unknown),
            address_provinsi=('address_provinsi', mode_or_unknown),
            main_loan_purpose=('loan_purpose', mode_or_unknown),
            avg_dependent=('dependent', 'mean'),
            dob=('dob', lambda x: pd.to_datetime(x, errors='coerce').max())
        )
        .reset_index()
    )

    # Compute age and age bucket
    cust_demo['age'] = (
        (pd.Timestamp('2022-12-31') - cust_demo['dob']).dt.days / 365.25
    )
    cust_demo['age'] = cust_demo['age'].replace([np.inf, -np.inf], np.nan).fillna(0)

    cust_demo['age_bucket'] = pd.cut(
        cust_demo['age'],
        bins=[0, 25, 35, 45, 55, 120],
        labels=['<25', '25-34', '35-44', '45-54', '55+']
    ).astype(str).fillna('unknown')

    # Merge demographic and behavioral features
    df_features = pd.merge(
        cust_behavior,
        cust_demo[['customer_id', 'marital_status', 'job_type', 'job_industry',
                   'address_provinsi', 'main_loan_purpose', 'avg_dependent',
                   'age', 'age_bucket']],
        on='customer_id',
        how='left'
    )

    return df_features


# 2. Interactive Scoring using saved model
def run_scoring_interactive(model_path: str = 'best_credit_scoring_logreg.pkl'):
    """
    Interactive scoring function for customer-level default prediction.
    Steps:
      1. Ask for input CSV (combined_df-style).
      2. Build customer-level features.
      3. Load saved model pipeline.
      4. Impute missing values in X to avoid NaN issues.
      5. Predict pd_score and summarize results.
      6. Save scored DataFrame to CSV.
    """

    input_csv = input("Enter input CSV file name (e.g. combined_df.csv): ").strip()
    if not input_csv:
        print("No input file provided.")
        return

    output_csv = input("Enter output CSV name (default: scored_customers.csv): ").strip()
    if not output_csv:
        output_csv = "scored_customers.csv"
    # Ensure .csv extension
    if not output_csv.lower().endswith(".csv"):
        output_csv = output_csv + ".csv"

    # Read CSV, try to keep IDs as strings
    try:
        combined_df = pd.read_csv(
            input_csv,
            dtype={
                'application_id': str,
                'customer_id': str,
                'loan_id': str,
                'payment_id': str
            }
        )
    except Exception:
        combined_df = pd.read_csv(input_csv)

    print(f"Loaded {len(combined_df)} rows from {input_csv}")
    display(combined_df.head())

    # Build customer-level features
    df_features = build_customer_features_from_combined(combined_df)

    # Load trained model pipeline
    model = joblib.load(model_path)

    # Prepare X for model (drop ID and target if present)
    X = df_features.drop(columns=['customer_id', 'default_flag_customer'], errors='ignore')
    X = X.copy()

    # Identify numeric and non-numeric columns for imputation
    num_cols = X.select_dtypes(include=['int64', 'float64', 'Int64', 'float32']).columns
    cat_cols = X.columns.difference(num_cols)

    # Numeric: median fill
    for col in num_cols:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)

    # Categorical: "missing"
    for col in cat_cols:
        X[col] = X[col].fillna('missing')

    # Final safety: replace any leftover NaN
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Predict PD
    if hasattr(model.named_steps['model'], "predict_proba"):
        pd_score = model.predict_proba(X)[:, 1]
    else:
        s = model.decision_function(X)
        pd_score = (s - s.min()) / (s.max() - s.min() + 1e-9)

    df_features['pd_score'] = pd_score
    df_features['pred_default_flag'] = (pd_score >= 0.5).astype(int)

    # Summary statistics
    total = len(df_features)
    n_default = int(df_features['pred_default_flag'].sum())
    pct_default = round(n_default / total * 100, 2)

    summary_df = pd.DataFrame({
        'Total_Customers': [total],
        'Predicted_Defaults': [n_default],
        'Predicted_Default_Rate_%': [pct_default]
    })
    display(summary_df)

    # Output columns for business
    cols_out = ['customer_id', 'default_flag_customer', 'pd_score']
    cols_out = [c for c in cols_out if c in df_features.columns]
    scored_df = df_features[cols_out].copy()

    scored_df.to_csv(output_csv, index=False)
    print(f"Scoring completed. Output saved as: {output_csv}")
    display(scored_df.head())

    return scored_df, summary_df
