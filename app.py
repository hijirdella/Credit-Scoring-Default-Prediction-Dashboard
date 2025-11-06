import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import seaborn as sns  # only for heatmap

import creditdefault as cd  # local module that does preprocessing + prediction

MODEL_PATH = "credit_scoring_best_model.pkl"


@st.cache_resource
def load_model(model_path: str):
    """
    Load the trained model pipeline from disk.
    """
    return joblib.load(model_path)


def cast_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that ID-like columns are stored as string dtype.
    """
    id_cols = ["application_id", "customer_id", "loan_id", "payment_id"]
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def build_column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table for columns: dtype, missing count, and missing percentage.
    """
    col_info = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "missing_pct": (df.isna().sum() / len(df) * 100).round(2)
    })
    return col_info


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a numeric summary (describe + missing count) for all numeric columns.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return pd.DataFrame()
    summary = df[num_cols].describe().T
    summary["missing"] = df[num_cols].isna().sum()
    return summary


def cat_summary(df: pd.DataFrame, col: str, top_n: int = 10) -> pd.DataFrame:
    """
    Build a frequency table for a categorical column.
    """
    vc = df[col].value_counts(dropna=False)
    total = vc.sum()
    freq = (
        vc.head(top_n)
        .to_frame("count")
        .assign(percentage=lambda x: (x["count"] / total * 100).round(2))
    )
    return freq


def plot_bar_with_orange_gradient(values, index_labels, title, xlabel, ylabel):
    """
    Plot a bar chart with an Orange gradient color.
    """
    fig, ax = plt.subplots()
    n = len(values)
    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, n))
    ax.bar(range(n), values, color=colors)
    ax.set_xticks(range(n))
    ax.set_xticklabels(index_labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def compute_deciles(df: pd.DataFrame, score_col: str, label_col: str, n: int = 10) -> pd.DataFrame:
    """
    Compute deciles on a probability score and show default rate per decile.
    """
    tmp = df.copy()
    tmp["decile"] = pd.qcut(tmp[score_col], q=n, labels=False, duplicates="drop") + 1
    agg = (
        tmp.groupby("decile")
        .agg(
            n_customers=("decile", "size"),
            avg_pd=(score_col, "mean"),
            min_pd=(score_col, "min"),
            max_pd=(score_col, "max"),
            default_rate=(label_col, "mean")
        )
        .sort_index()
    )
    agg["avg_pd"] = agg["avg_pd"].round(4)
    agg["default_rate"] = agg["default_rate"].round(4)
    agg["share_pct"] = (agg["n_customers"] / agg["n_customers"].sum() * 100).round(2)
    return agg


def get_feature_importance(model, feature_names):
    """
    Try to extract feature importance or coefficients from the model.
    Works for tree-based models with feature_importances_ or linear models with coef_.
    """
    importances = None

    # Case 1: model is a pipeline
    if hasattr(model, "named_steps"):
        # Try some common step names
        for step_name in ["model", "clf", "classifier", "final_estimator"]:
            if step_name in model.named_steps:
                estimator = model.named_steps[step_name]
                break
        else:
            estimator = model
    else:
        estimator = model

    # Tree-based models
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    # Linear models
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            importances = np.abs(coef[0])

    if importances is None:
        return None

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return fi


def main():
    st.title("Credit Default Prediction Dashboard")

    st.markdown(
        "This app takes raw payment-level data (`combined_df`-style), "
        "performs exploratory data analysis (EDA), and then runs the credit scoring model."
    )

    # Sidebar controls
    st.sidebar.header("Settings")
    threshold = st.sidebar.slider(
        "Prediction threshold (default = 0.5)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )
    default_filename = st.sidebar.text_input(
        "Default output CSV file name",
        value="credit_default_predictions.csv"
    )
    show_feature_importance = st.sidebar.checkbox("Show feature importance", value=True)
    run_scoring = st.sidebar.button("Run scoring")

    # File upload
    st.subheader("1. Upload raw payment-level CSV")
    uploaded_file = st.file_uploader("Upload combined_df-style CSV", type=["csv"])

    if uploaded_file is None:
        st.info("Please upload a CSV file to continue.")
        return

    # Read data
    df_raw = pd.read_csv(uploaded_file)
    df_raw = cast_id_columns(df_raw)

    st.markdown("**Raw data preview**")
    st.write(f"Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    st.dataframe(df_raw.head())

    # Column overview
    st.markdown("### 2. Exploratory Data Analysis")
    st.markdown("#### 2.1 Numeric summary")
    num_summary = numeric_summary(df_raw)
    if num_summary.empty:
        st.write("No numeric columns found.")
    else:
        st.dataframe(num_summary)

    st.markdown("#### 2.1b Columns overview")
    col_info = build_column_overview(df_raw)
    st.dataframe(col_info)

    # Numeric distributions
    st.markdown("#### 2.2 Numeric distributions")
    numeric_candidates = ["loan_amount", "installment_amount", "paid_amount", "dpd"]
    for col in numeric_candidates:
        if col in df_raw.columns:
            fig, ax = plt.subplots()
            df_raw[col].dropna().hist(bins=40, ax=ax)
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            st.pyplot(fig)

    # Categorical distributions
    st.markdown("#### 2.3 Categorical distributions")
    cat_cols = ["address_provinsi", "loan_purpose", "marital_status", "job_type", "job_industry", "dependent"]
    for col in cat_cols:
        if col in df_raw.columns:
            freq = cat_summary(df_raw, col, top_n=10)
            with st.expander(f"Distribution of {col}"):
                st.dataframe(freq)
                fig = plot_bar_with_orange_gradient(
                    values=freq["count"].values,
                    index_labels=freq.index.astype(str).tolist(),
                    title=f"{col} (top {len(freq)})",
                    xlabel=col,
                    ylabel="Count"
                )
                st.pyplot(fig)

    # Relationships between variables
    st.markdown("#### 2.4 Relationships between variables")
    if {"loan_amount", "installment_amount"}.issubset(df_raw.columns):
        fig, ax = plt.subplots()
        ax.scatter(df_raw["loan_amount"], df_raw["installment_amount"], alpha=0.3)
        ax.set_xlabel("loan_amount")
        ax.set_ylabel("installment_amount")
        ax.set_title("Loan amount vs Installment amount")
        st.pyplot(fig)

    if {"installment_amount", "paid_amount"}.issubset(df_raw.columns):
        fig, ax = plt.subplots()
        ax.scatter(df_raw["installment_amount"], df_raw["paid_amount"], alpha=0.3)
        ax.set_xlabel("installment_amount")
        ax.set_ylabel("paid_amount")
        ax.set_title("Installment amount vs Paid amount")
        st.pyplot(fig)

    if {"dpd", "loan_amount"}.issubset(df_raw.columns):
        fig, ax = plt.subplots()
        ax.scatter(df_raw["loan_amount"], df_raw["dpd"], alpha=0.3)
        ax.set_xlabel("loan_amount")
        ax.set_ylabel("dpd")
        ax.set_title("DPD vs Loan amount")
        st.pyplot(fig)

    if {"loan_amount", "marital_status"}.issubset(df_raw.columns):
        fig, ax = plt.subplots()
        df_raw.boxplot(column="loan_amount", by="marital_status", ax=ax)
        ax.set_title("Loan amount by Marital status")
        ax.set_xlabel("")
        ax.set_ylabel("loan_amount")
        plt.suptitle("")
        st.pyplot(fig)

    # Correlation heatmap
    st.markdown("#### 2.5 Correlation heatmap")
    num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        corr = df_raw[num_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            corr,
            ax=ax,
            cmap="Oranges",
            annot=False,
            square=True,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Correlation heatmap (numeric features)")
        st.pyplot(fig)
    else:
        st.write("Not enough numeric features for a correlation heatmap.")

    # Label analysis for raw data (if label exists)
    st.markdown("#### 2.6 Label analysis (raw data, if available)")
    label_col_raw = None
    for cand in ["default_flag_customer", "default_flag", "target", "label"]:
        if cand in df_raw.columns:
            label_col_raw = cand
            break

    if label_col_raw is not None:
        st.write(f"Detected label column in raw data: `{label_col_raw}`")
        vc = df_raw[label_col_raw].value_counts(dropna=False)
        total = vc.sum()
        label_df = (
            vc.to_frame("count")
            .assign(percentage=lambda x: (x["count"] / total * 100).round(2))
        )
        st.dataframe(label_df)

        fig = plot_bar_with_orange_gradient(
            values=label_df["count"].values,
            index_labels=label_df.index.astype(str).tolist(),
            title="Raw label distribution",
            xlabel="Label",
            ylabel="Count"
        )
        st.pyplot(fig)
    else:
        st.info(
            "The uploaded raw CSV does not contain an explicit default label. "
            "Label analysis will be based on model predictions in the next section."
        )

    # Scoring section
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

    # Predict using helper module
    try:
        scored_df = cd.predict_raw_data(model, df_raw, threshold=threshold)
    except Exception as e:
        st.error(f"Error during prediction in creditdefault.predict_raw_data: {e}")
        return

    # Ensure ID columns remain strings after prediction
    scored_df = cast_id_columns(scored_df)

    st.markdown("**Scored data preview**")
    st.write(f"Total customers scored: {scored_df.shape[0]}")
    st.dataframe(scored_df.head())

    # Decide which column is probability
    if "pd" in scored_df.columns:
        score_col = "pd"
    else:
        # Fallback: first numeric column
        num_cols_scored = scored_df.select_dtypes(include=[np.number]).columns
        if len(num_cols_scored) == 0:
            st.error("No numeric column found in scored data to use as probability.")
            return
        score_col = num_cols_scored[0]

    # Create predicted label based on threshold (even if already present)
    scored_df["predicted_label"] = (scored_df[score_col] >= threshold).astype(int)

    # Prediction distribution (label analysis from predictions)
    st.markdown("#### 3.1 Prediction distribution (model-based label analysis)")
    pred_counts = scored_df["predicted_label"].value_counts().sort_index()
    total_pred = pred_counts.sum()
    pred_summary = (
        pred_counts.to_frame("count")
        .assign(
            label=lambda x: ["Non-default (0)", "Default (1)"][: len(x)],
            percentage=lambda x: (x["count"] / total_pred * 100).round(2)
        )
    )
    st.dataframe(pred_summary)

    # Bar chart with orange gradient
    fig = plot_bar_with_orange_gradient(
        values=pred_summary["count"].values,
        index_labels=pred_summary["label"].tolist(),
        title="Prediction distribution",
        xlabel="Predicted label",
        ylabel="Number of customers"
    )
    st.pyplot(fig)

    # Pie chart
    fig, ax = plt.subplots()
    ax.pie(
        pred_summary["count"],
        labels=pred_summary["label"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Prediction share")
    st.pyplot(fig)

    # Decile analysis
    st.markdown("#### 3.2 Decile analysis based on predicted PD")
    try:
        decile_table = compute_deciles(
            scored_df,
            score_col=score_col,
            label_col="predicted_label",
            n=10
        )
        st.dataframe(decile_table)

        fig = plot_bar_with_orange_gradient(
            values=decile_table["avg_pd"].values,
            index_labels=decile_table.index.astype(str).tolist(),
            title="Average PD by decile (1 = lowest risk)",
            xlabel="Decile",
            ylabel="Average PD"
        )
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to compute deciles: {e}")

    # Feature importance section (optional)
    if show_feature_importance:
        st.markdown("#### 3.3 Feature importance from the model")
        # Try to infer feature names from model or from scored_df
        feature_names = None
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        else:
            # As a fallback, take numeric and categorical columns that are not obvious IDs
            feature_names = [
                c for c in scored_df.columns
                if c not in ["customer_id", "application_id", "loan_id", "payment_id",
                             "pd", "predicted_label"]
            ]

        fi = get_feature_importance(model, feature_names)
        if fi is None:
            st.info("Feature importance is not available for this model.")
        else:
            st.dataframe(fi)

            # Plot top features with orange gradient
            top_fi = fi.head(10)
            fig = plot_bar_with_orange_gradient(
                values=top_fi["importance"].values,
                index_labels=top_fi["feature"].tolist(),
                title="Top feature importance",
                xlabel="Feature",
                ylabel="Importance"
            )
            st.pyplot(fig)

    # Download scored results
    st.subheader("4. Download scored predictions")
    csv_buffer = io.StringIO()
    scored_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download predictions as CSV",
        data=csv_buffer.getvalue(),
        file_name=default_filename,
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
