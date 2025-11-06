import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

import creditdefault as cd  # pastikan file creditdefault.py ada di repo

MODEL_PATH = "best_credit_scoring_logreg.pkl"


@st.cache_resource
def load_model(model_path: str):
    model = joblib.load(model_path)
    return model


def basic_numeric_summary(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return pd.DataFrame()
    summary = df[num_cols].describe().T
    summary["missing"] = df[num_cols].isna().sum()
    return summary


def basic_categorical_summary(df, col, top_n=15):
    vc = df[col].value_counts(dropna=False)
    total = vc.sum()
    freq = (
        vc.head(top_n)
        .to_frame("count")
        .assign(percentage=lambda x: (x["count"] / total * 100).round(2))
    )
    return freq


def compute_deciles(df, score_col, n_deciles=10):
    df = df.copy()
    df["decile"] = pd.qcut(df[score_col], q=n_deciles, labels=False, duplicates="drop") + 1
    agg = (
        df.groupby("decile")
        .agg(
            n_customers=("decile", "size"),
            avg_score=(score_col, "mean"),
            min_score=(score_col, "min"),
            max_score=(score_col, "max")
        )
        .sort_index()
    )
    total = agg["n_customers"].sum()
    agg["share_pct"] = (agg["n_customers"] / total * 100).round(2)
    return agg


def main():
    st.title("Credit Default Prediction Dashboard")

    st.markdown(
        """
        This app allows you to:
        1. Upload raw loan data (e.g. `combined_df.csv`).
        2. Explore EDA (statistics, demographics, labels).
        3. Run credit scoring model.
        4. Analyze prediction & decile results.
        5. Download scored output as CSV.
        """
    )

    st.sidebar.header("Configuration")

    uploaded_file = st.sidebar.file_uploader("Upload raw CSV data", type=["csv"])
    threshold = st.sidebar.slider("Default threshold", 0.01, 0.99, 0.5, 0.01)
    run_scoring = st.sidebar.button("Run Scoring")

    if uploaded_file is None:
        st.info("Please upload a CSV file first.")
        return

    # Read and show input data
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("1. Input Data Preview")
    st.write(f"Rows: {raw_df.shape[0]}, Columns: {raw_df.shape[1]}")
    st.dataframe(raw_df.head())

    with st.expander("Columns and dtypes"):
        info_df = pd.DataFrame({
            "column": raw_df.columns,
            "dtype": raw_df.dtypes.astype(str),
            "missing": raw_df.isna().sum()
        })
        st.dataframe(info_df)

    # --- EDA ---
    st.subheader("2. Exploratory Data Analysis (EDA)")

    # Numeric stats
    st.markdown("**2.1 Numeric Summary Statistics**")
    num_summary = basic_numeric_summary(raw_df)
    if num_summary.empty:
        st.write("No numeric columns found.")
    else:
        st.dataframe(num_summary)

        for col in ["loan_amount", "loan_duration", "installment_amount", "paid_amount", "dpd"]:
            if col in raw_df.columns:
                fig, ax = plt.subplots()
                raw_df[col].dropna().hist(bins=30, ax=ax)
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                st.pyplot(fig)

    # Demographic
    st.markdown("**2.2 Demographic Analysis**")
    demo_cols = ["marital_status", "job_type", "job_industry", "address_provinsi", "loan_purpose"]
    for col in demo_cols:
        if col in raw_df.columns:
            st.markdown(f"Distribution by `{col}`")
            freq = basic_categorical_summary(raw_df, col)
            st.dataframe(freq)
            fig, ax = plt.subplots()
            freq["count"].plot(kind="bar", ax=ax)
            ax.set_title(col)
            ax.set_ylabel("Count")
            st.pyplot(fig)

    # Label analysis
    st.markdown("**2.3 Label Analysis (if available)**")
    label_col = None
    for c in ["default_flag_customer", "default_flag", "target", "label"]:
        if c in raw_df.columns:
            label_col = c
            break

    if label_col:
        st.write(f"Using label column: `{label_col}`")
        vc = raw_df[label_col].value_counts(dropna=False)
        label_df = (
            vc.to_frame("count")
            .assign(percentage=lambda x: (x["count"] / vc.sum() * 100).round(2))
        )
        st.dataframe(label_df)
        fig, ax = plt.subplots()
        label_df["count"].plot(kind="bar", ax=ax)
        ax.set_title("Label Distribution")
        st.pyplot(fig)
    else:
        st.write("No label column found.")

    # --- Scoring ---
    st.subheader("3. Model Scoring")
    if not run_scoring:
        st.info("Click 'Run Scoring' to continue.")
        return

    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    try:
        result_df = cd.predict_raw_data(model, raw_df, threshold=threshold)
    except Exception as e:
        st.error(f"Error in predict_raw_data: {e}")
        st.stop()

    st.markdown("**3.1 Scored Output Preview**")
    st.write(f"Scored rows: {result_df.shape[0]}")
    st.dataframe(result_df.head())

    # Download button
    csv_buf = io.StringIO()
    result_df.to_csv(csv_buf, index=False)
    st.download_button(
        "Download Scored Output",
        csv_buf.getvalue(),
        "scored_customers.csv",
        "text/csv"
    )

    # --- Prediction Summary ---
    st.markdown("**3.2 Prediction Summary and Deciles**")
    num_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
    score_col = next((c for c in num_cols if "pd" in c.lower() or "prob" in c.lower()), num_cols[0])

    result_df["predicted_bad"] = (result_df[score_col] >= threshold).astype(int)
    pred_summary = result_df["predicted_bad"].value_counts().sort_index().to_frame("count")
    pred_summary["label"] = ["Good (0)", "Bad (1)"]
    pred_summary["percentage"] = (pred_summary["count"] / pred_summary["count"].sum() * 100).round(2)
    st.dataframe(pred_summary)

    fig, ax = plt.subplots()
    pred_summary["count"].plot(kind="bar", ax=ax)
    ax.set_xticklabels(pred_summary["label"], rotation=0)
    ax.set_ylabel("Count")
    ax.set_title("Predicted Good vs Bad")
    st.pyplot(fig)

    # --- Decile ---
    st.markdown("**3.3 Decile Analysis**")
    decile_table = compute_deciles(result_df, score_col)
    st.dataframe(decile_table)

    fig, ax = plt.subplots()
    decile_table["avg_score"].plot(kind="bar", ax=ax)
    ax.set_xlabel("Decile (1=best, 10=worst)")
    ax.set_ylabel("Average Score")
    ax.set_title("Average Score by Decile")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    decile_table["share_pct"].plot(kind="bar", ax=ax)
    ax.set_xlabel("Decile")
    ax.set_ylabel("Customer Share (%)")
    ax.set_title("Customer Distribution by Decile")
    st.pyplot(fig)


if __name__ == "__main__":
    main()
