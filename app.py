import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

import creditdefault as cd  # local module with predict_raw_data()

MODEL_PATH = "credit_scoring_best_model.pkl"


@st.cache_resource
def load_model(model_path: str):
    """
    Load trained model from disk.
    """
    return joblib.load(model_path)


def cast_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure ID-like columns are stored as pandas string dtype.
    """
    id_cols = ["application_id", "customer_id", "loan_id", "payment_id"]
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def build_column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build summary of columns: dtype, missing count, missing percentage.
    """
    overview = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "missing_pct": (df.isna().sum() / len(df) * 100).round(2)
    })
    return overview


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create numeric summary (describe + missing count) for numeric columns.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return pd.DataFrame()
    summary = df[num_cols].describe().T
    summary["missing"] = df[num_cols].isna().sum()
    return summary


def cat_summary(df: pd.DataFrame, col: str, top_n: int = 10) -> pd.DataFrame:
    """
    Frequency table for a categorical column.
    """
    vc = df[col].value_counts(dropna=False)
    total = vc.sum()
    freq = (
        vc.head(top_n)
        .to_frame("count")
        .assign(percentage=lambda x: (x["count"] / total * 100).round(2))
    )
    return freq


def plot_bar_orange(values, labels, title, xlabel, ylabel):
    """
    Bar chart with an orange gradient color palette.
    """
    fig, ax = plt.subplots()
    n = len(values)
    colors = plt.cm.Oranges(np.linspace(0.5, 0.9, n))
    ax.bar(range(n), values, color=colors, edgecolor="black")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def main():
    st.title("Credit Default Prediction Dashboard")

    st.markdown(
        "This app takes raw payment-level data, performs EDA, "
        "and then runs the credit scoring model to generate default predictions."
    )

    # Sidebar settings
    st.sidebar.header("Settings")
    threshold = st.sidebar.slider(
        "Prediction threshold (default = 0.5)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )
    default_filename = st.sidebar.text_input(
        "Default output CSV file name",
        value="credit_default_predictions.csv",
    )
    show_feature_importance = st.sidebar.checkbox("Show feature importance", value=True)
    run_scoring = st.sidebar.button("Run scoring")

    # File upload
    st.subheader("1. Upload raw payment-level CSV")
    uploaded_file = st.file_uploader("Upload combined_df-style CSV", type=["csv"])

    if uploaded_file is None:
        st.info("Please upload a CSV file to continue.")
        return

    # Read and type-cast raw data
    df_raw = pd.read_csv(uploaded_file)
    df_raw = cast_id_columns(df_raw)

    st.markdown("**Raw data preview**")
    st.write(f"Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    st.dataframe(df_raw.head())

    # 2. EDA
    st.subheader("2. Exploratory Data Analysis")

    # 2.1 Numeric summary
    st.markdown("#### 2.1 Numeric summary")
    num_summary = numeric_summary(df_raw)
    if num_summary.empty:
        st.write("No numeric columns found.")
    else:
        st.dataframe(num_summary)

    # 2.1b Column overview
    st.markdown("#### 2.1b Columns overview")
    st.dataframe(build_column_overview(df_raw))

    # 2.2 Numeric distributions (orange histograms)
    st.markdown("#### 2.2 Numeric distributions")
    numeric_candidates = ["loan_amount", "installment_amount", "paid_amount", "dpd"]
    for col in numeric_candidates:
        if col in df_raw.columns:
            fig, ax = plt.subplots()
            df_raw[col].dropna().hist(bins=40, ax=ax, color="#FF8A00", edgecolor="black")
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            st.pyplot(fig)

    # 2.3 Categorical distributions (orange-dominant)
    st.markdown("#### 2.3 Categorical distributions")
    cat_cols = ["address_provinsi", "loan_purpose", "marital_status", "job_type", "job_industry", "dependent"]
    for col in cat_cols:
        if col in df_raw.columns:
            freq = cat_summary(df_raw, col, top_n=10)
            with st.expander(f"Distribution of {col}"):
                st.dataframe(freq)
                fig = plot_bar_orange(
                    values=freq["count"].values,
                    labels=freq.index.astype(str).tolist(),
                    title=f"{col} (top {len(freq)})",
                    xlabel=col,
                    ylabel="Count",
                )
                st.pyplot(fig)

    # 2.4 Simple relationships
    st.markdown("#### 2.4 Relationships between variables")
    if {"loan_amount", "installment_amount"}.issubset(df_raw.columns):
        fig, ax = plt.subplots()
        ax.scatter(df_raw["loan_amount"], df_raw["installment_amount"], alpha=0.3, color="#FFA726")
        ax.set_xlabel("loan_amount")
        ax.set_ylabel("installment_amount")
        ax.set_title("Loan amount vs Installment amount")
        st.pyplot(fig)

    if {"installment_amount", "paid_amount"}.issubset(df_raw.columns):
        fig, ax = plt.subplots()
        ax.scatter(df_raw["installment_amount"], df_raw["paid_amount"], alpha=0.3, color="#FFB74D")
        ax.set_xlabel("installment_amount")
        ax.set_ylabel("paid_amount")
        ax.set_title("Installment amount vs Paid amount")
        st.pyplot(fig)

    if {"dpd", "loan_amount"}.issubset(df_raw.columns):
        fig, ax = plt.subplots()
        ax.scatter(df_raw["loan_amount"], df_raw["dpd"], alpha=0.3, color="#FFCC80")
        ax.set_xlabel("loan_amount")
        ax.set_ylabel("dpd")
        ax.set_title("DPD vs Loan amount")
        st.pyplot(fig)

    # 2.5 Correlation heatmap (orange gradient, pure matplotlib)
    st.markdown("#### 2.5 Correlation heatmap")
    num_cols_for_corr = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols_for_corr) >= 2:
        corr = df_raw[num_cols_for_corr].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(corr.values, cmap="Oranges", vmin=-1, vmax=1)
        ax.set_xticks(range(len(num_cols_for_corr)))
        ax.set_xticklabels(num_cols_for_corr, rotation=45, ha="right")
        ax.set_yticks(range(len(num_cols_for_corr)))
        ax.set_yticklabels(num_cols_for_corr)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Correlation heatmap (numeric features)")
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.write("Not enough numeric features for a correlation heatmap.")

    # 3. Scoring and prediction-based analysis
    st.subheader("3. Scoring and label analysis based on model predictions")

    if not run_scoring:
        st.info("Set the threshold in the sidebar and click 'Run scoring' to generate predictions.")
        return

    # Load model
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model from '{MODEL_PATH}': {e}")
        return

    # Run prediction via creditdefault.py
    try:
        scored_df = cd.predict_raw_data(model, df_raw, threshold=threshold)
    except Exception as e:
        st.error(f"Error during prediction in creditdefault.predict_raw_data: {e}")
        return

    scored_df = cast_id_columns(scored_df)

    st.markdown("**Scored data preview**")
    st.write(f"Total customers scored: {scored_df.shape[0]}")
    st.dataframe(scored_df.head())

    # Determine probability column
    if "pd" in scored_df.columns:
        score_col = "pd"
    else:
        num_cols_scored = scored_df.select_dtypes(include=[np.number]).columns
        if len(num_cols_scored) == 0:
            st.error("No numeric column found in scored data to use as probability.")
            return
        score_col = num_cols_scored[0]

    scored_df["predicted_label"] = (scored_df[score_col] >= threshold).astype(int)

    # 3.1 Prediction distribution
    st.markdown("#### 3.1 Prediction distribution (model-based labels)")
    pred_counts = scored_df["predicted_label"].value_counts().sort_index()
    total_pred = pred_counts.sum()
    pred_summary = (
        pred_counts.to_frame("count")
        .assign(
            label=lambda x: ["Non-default (0)", "Default (1)"][: len(x)],
            percentage=lambda x: (x["count"] / total_pred * 100).round(2),
        )
    )
    st.dataframe(pred_summary)

    fig = plot_bar_orange(
        values=pred_summary["count"].values,
        labels=pred_summary["label"].tolist(),
        title="Prediction distribution",
        xlabel="Predicted label",
        ylabel="Number of customers",
    )
    st.pyplot(fig)

    # Pie chart (green as secondary color only)
    fig, ax = plt.subplots()
    colors = ["#66BB6A", "#FF8A00"]  # non-default = green, default = orange
    ax.pie(
        pred_summary["count"],
        labels=pred_summary["label"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors[: len(pred_summary)],
    )
    ax.set_title("Prediction share")
    st.pyplot(fig)

    # 3.2 Decile analysis using predicted PD
    st.markdown("#### 3.2 Decile analysis based on predicted PD")
    try:
        tmp_dec = scored_df.copy()
        tmp_dec["decile"] = pd.qcut(tmp_dec[score_col], 10, labels=False, duplicates="drop") + 1
        decile_pd = (
            tmp_dec.groupby("decile")
            .agg(
                n_customers=("decile", "size"),
                avg_pd=(score_col, "mean"),
                default_rate=("predicted_label", "mean"),
            )
            .sort_index()
        )
        decile_pd["avg_pd"] = decile_pd["avg_pd"].round(4)
        decile_pd["default_rate"] = (decile_pd["default_rate"] * 100).round(2)
        decile_pd["share_pct"] = (decile_pd["n_customers"] / decile_pd["n_customers"].sum() * 100).round(2)
        st.dataframe(decile_pd)

        fig = plot_bar_orange(
            values=decile_pd["avg_pd"].values,
            labels=decile_pd.index.astype(str).tolist(),
            title="Average PD by decile (1 = lowest risk)",
            xlabel="Decile",
            ylabel="Average PD",
        )
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to compute PD deciles: {e}")

    # 3.3 Feature importance (optional)
    if show_feature_importance:
        st.markdown("#### 3.3 Feature importance (if available)")
        estimator = model
        if hasattr(model, "named_steps"):
            for step_name in ["model", "clf", "classifier", "final_estimator"]:
                if step_name in model.named_steps:
                    estimator = model.named_steps[step_name]
                    break

        importances = None
        feature_names = None

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_
            if coef.ndim == 1:
                importances = np.abs(coef)
            else:
                importances = np.abs(coef[0])

        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        else:
            feature_names = [
                c for c in scored_df.columns
                if c not in ["customer_id", "application_id", "loan_id", "pd", "predicted_label"]
                and scored_df[c].dtype != "string[python]"
            ]

        if importances is None or feature_names is None:
            st.info("Feature importance is not available for this model.")
        else:
            fi = pd.DataFrame({
                "feature": feature_names,
                "importance": importances,
            }).sort_values("importance", ascending=False)

            st.dataframe(fi)

            top_fi = fi.head(10)
            fig = plot_bar_orange(
                values=top_fi["importance"].values,
                labels=top_fi["feature"].tolist(),
                title="Top feature importance",
                xlabel="Feature",
                ylabel="Importance",
            )
            st.pyplot(fig)

    # 4. Download scored output
    st.subheader("4. Download scored predictions")
    csv_buffer = io.StringIO()
    scored_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download predictions as CSV",
        data=csv_buffer.getvalue(),
        file_name=default_filename,
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
