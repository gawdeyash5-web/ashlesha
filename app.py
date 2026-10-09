
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# COSMIC IDENTITY — CONFIGURATION
# =========================================================
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "Dataset 41.csv"
OUTPUTS = BASE / "outputs"

MODEL_PATH = OUTPUTS / "cosmic_model.joblib"
METRICS_PATH = OUTPUTS / "metrics.json"

DEFAULT_FEATURES = [
    "Right_Ascension",
    "Declination",
    "U_Magnitude",
    "G_Magnitude",
    "R_Magnitude",
    "I_Magnitude",
    "Z_Magnitude",
    "Redshift",
]

CLASS_COLORS = {
    "STAR": "#FBBF24",
    "QSO": "#C084FC",
    "GALAXY": "#38BDF8",
}

st.set_page_config(
    page_title="Cosmic Identity | AstroManthan 2.0",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM STYLING
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                ellipse at top right,
                rgba(42, 57, 112, 0.24),
                transparent 48%
            ),
            #080D1B;
        color: #E5E7EB;
    }

    [data-testid="stSidebar"] {
        background: #0D1426;
        border-right: 1px solid #24304A;
    }

    [data-testid="stMetric"] {
        background: rgba(20, 30, 53, 0.88);
        border: 1px solid #293754;
        padding: 17px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #A7B5CE;
    }

    [data-testid="stMetricValue"] {
        color: #F3F7FF;
    }

    h1, h2, h3 {
        color: #F3F7FF;
    }

    .hero {
        padding: 25px 26px;
        border: 1px solid #2C4165;
        border-radius: 18px;
        background: linear-gradient(
            115deg,
            rgba(24, 39, 74, 0.98),
            rgba(15, 23, 42, 0.96)
        );
        margin-bottom: 22px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        letter-spacing: 1px;
        color: #F8FAFC;
        margin-bottom: 7px;
    }

    .hero-subtitle {
        font-size: 14px;
        color: #B7C7E5;
    }

    .eyebrow {
        font-size: 11px;
        font-weight: 700;
        color: #67E8F9;
        letter-spacing: 2px;
        margin-bottom: 9px;
    }

    .result-card {
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #345078;
        background: linear-gradient(
            120deg,
            rgba(21, 42, 72, 0.95),
            rgba(15, 23, 42, 0.98)
        );
        margin: 12px 0;
    }

    .result-class {
        font-size: 30px;
        font-weight: 800;
        color: #67E8F9;
    }

    .small-muted {
        font-size: 12px;
        color: #A7B5CE;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0891B2, #4F46E5);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        min-height: 44px;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD SAVED MODEL, METRICS AND DATA
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH, low_memory=False)


def check_project_files():
    missing = [
        path.name
        for path in [MODEL_PATH, METRICS_PATH, DATA_PATH]
        if not path.exists()
    ]

    if missing:
        st.error(
            "Required project files are missing: "
            + ", ".join(missing)
        )
        st.info(
            "Run `python main.py` first and check the outputs folder."
        )
        st.stop()


check_project_files()

try:
    saved = load_model()
    metrics = load_metrics()
    df = load_dataset()

    # Your main.py saves a dictionary with model and features.
    if isinstance(saved, dict):
        model = saved["model"]
        features = saved.get("features", DEFAULT_FEATURES)
    else:
        # Also support a directly saved sklearn estimator.
        model = saved
        features = DEFAULT_FEATURES

except Exception as exc:
    st.error(f"Unable to load the project: {exc}")
    st.stop()


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def get_metric(key, default=0):
    return metrics.get(key, default)


def get_class_counts():
    report = metrics.get("data_report", {})
    counts = report.get("class_counts", {})

    if counts:
        return pd.Series(counts, dtype="int64")

    if "Class" in df.columns:
        return df["Class"].value_counts()

    return pd.Series(dtype="int64")


def get_classifier_classes():
    # Use the exact class order known by the trained estimator.
    if hasattr(model, "classes_"):
        return [str(x) for x in model.classes_]

    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "classes_"):
                return [str(x) for x in step.classes_]

    return list(metrics.get("labels", ["GALAXY", "QSO", "STAR"]))


def get_feature_summary(feature):
    if feature not in df.columns:
        return None

    values = pd.to_numeric(df[feature], errors="coerce").dropna()

    if values.empty:
        return None

    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "q01": float(values.quantile(0.01)),
        "q99": float(values.quantile(0.99)),
    }


def format_class(name):
    return str(name).upper()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="font-size:42px;">🌌</div>
        <div style="font-size:22px;font-weight:800;color:#F8FAFC;">
            COSMIC IDENTITY
        </div>
        <div style="font-size:11px;color:#9FB3D9;letter-spacing:1px;">
            ASTROMANTHAN 2.0
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Predict an Object",
            "Model Evaluation",
            "Dataset Explorer",
        ],
        label_visibility="visible",
    )

    st.divider()

    st.markdown("**MODEL STATUS**")
    st.success("Trained model loaded")

    st.caption("Random Forest classifier")
    st.caption(f"{len(features)} input features")
    st.caption("Classes: STAR · QSO · GALAXY")

    st.divider()
    st.caption("Built for ASTROMANTHAN 2.0")


# =========================================================
# HERO HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">ASTRONOMICAL MACHINE LEARNING</div>
        <div class="hero-title">COSMIC IDENTITY</div>
        <div class="hero-subtitle">
            Stellar Object Classification Challenge
            &nbsp;·&nbsp;
            Discover whether an astronomical object is a
            Star, Quasar or Galaxy.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PAGE 1 — OVERVIEW
# =========================================================
if page == "Overview":

    st.subheader("Mission Control")
    st.write(
        "An overview of the dataset, model performance, "
        "and astronomical classification task."
    )

    data_report = metrics.get("data_report", {})
    total_rows = int(data_report.get("clean_rows", len(df)))
    original_rows = int(data_report.get("original_rows", len(df)))
    duplicates = int(data_report.get("duplicates_removed", 0))
    test_rows = int(metrics.get("testing_rows", 0))
    accuracy = float(metrics.get("accuracy", 0))

    class_counts = get_class_counts()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "TEST ACCURACY",
        f"{accuracy * 100:.2f}%",
    )

    k2.metric(
        "CLEAN RECORDS",
        f"{total_rows:,}",
    )

    k3.metric(
        "TEST SAMPLES",
        f"{test_rows:,}",
    )

    k4.metric(
        "OBJECT CLASSES",
        len(class_counts) if not class_counts.empty else 3,
    )

    st.write("")

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("Class Distribution")

        if not class_counts.empty:
            distribution = class_counts.rename_axis("Class").to_frame(
                "Records"
            )

            st.bar_chart(distribution)

            st.dataframe(
                pd.DataFrame({
                    "Class": class_counts.index,
                    "Records": class_counts.values,
                    "Share (%)": (
                        class_counts.values
                        / class_counts.sum()
                        * 100
                    ).round(2),
                }),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Class distribution is not available.")

    with right:
        st.subheader("Data Preparation")

        st.metric("Original records", f"{original_rows:,}")
        st.metric("Duplicate rows removed", f"{duplicates:,}")

        missing_after = data_report.get("missing_values_after")

        if missing_after is not None:
            st.metric(
                "Missing values after cleaning",
                f"{int(missing_after):,}",
            )

        st.info(
            "The model uses eight numerical astronomical features. "
            "Missing-value handling should follow the saved training pipeline."
        )

    st.divider()

    st.subheader("How the system works")

    steps = st.columns(4)

    workflow = [
        ("01", "LOAD", "Read astronomical survey data"),
        ("02", "CLEAN", "Inspect missing values and duplicates"),
        ("03", "TRAIN", "Fit a Random Forest classifier"),
        ("04", "PREDICT", "Classify unseen observations"),
    ]

    for column, (number, title, description) in zip(steps, workflow):
        with column:
            st.markdown(
                f"""
                <div style="
                    background:#111B30;
                    border:1px solid #2A3A58;
                    border-radius:13px;
                    padding:17px;
                    min-height:145px;
                ">
                    <div style="color:#67E8F9;font-size:12px;">
                        STEP {number}
                    </div>
                    <div style="
                        font-size:18px;
                        font-weight:800;
                        margin:8px 0;
                        color:#F8FAFC;
                    ">{title}</div>
                    <div style="
                        font-size:12px;
                        color:#A7B5CE;
                    ">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "Evaluation figures come from the saved test results. "
        "Test accuracy is not a guarantee of performance on every survey."
    )


# =========================================================
# PAGE 2 — INTERACTIVE PREDICTION
# =========================================================
elif page == "Predict an Object":

    st.subheader("Interactive Object Classifier")

    st.write(
        "Enter astronomical measurements to generate a prediction "
        "using the trained model. Values are prefilled from the dataset "
        "medians as starting points."
    )

    st.warning(
        "This is a numerical-feature demo, not an image classifier. "
        "Use realistic measurements in the same units and conventions "
        "as the training dataset."
    )

    summaries = {
        feature: get_feature_summary(feature)
        for feature in features
    }

    with st.form("prediction_form"):

        st.markdown("### Astronomical measurements")

        columns = st.columns(2)
        inputs = {}

        for index, feature in enumerate(features):
            summary = summaries.get(feature)

            if summary is None:
                st.error(
                    f"Cannot create an input for {feature}: "
                    "no valid numeric data was found."
                )
                continue

            with columns[index % 2]:
                st.markdown(f"**{feature.replace('_', ' ')}**")

                st.caption(
                    f"Dataset median: {summary['median']:.4f}"
                )

                # Use observed min/max so that values are numeric
                # and bounded to the available dataset range.
                inputs[feature] = st.number_input(
                    label=feature,
                    min_value=summary["min"],
                    max_value=summary["max"],
                    value=min(
                        max(
                            summary["median"],
                            summary["min"]
                        ),
                        summary["max"]
                    ),
                    step=0.01,
                    format="%.5f",
                    label_visibility="collapsed",
                    key=f"feature_{feature}",
                )

        submitted = st.form_submit_button(
            "🚀 Classify Object",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if len(inputs) != len(features):
            st.error(
                "Some required feature inputs could not be created. "
                "Check that your dataset contains all model features."
            )
        else:
            try:
                input_df = pd.DataFrame(
                    [[inputs[feature] for feature in features]],
                    columns=features,
                )

                # Preserve the feature order saved with the model.
                input_df = input_df[features]

                prediction = model.predict(input_df)[0]
                predicted_class = format_class(prediction)

                st.markdown("---")
                st.subheader("Classification Result")

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="small-muted">
                            PREDICTED ASTRONOMICAL CLASS
                        </div>
                        <div class="result-class">
                            {predicted_class}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Probabilities are available only if supported by
                # the fitted estimator and its preprocessing pipeline.
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_df)[0]
                    labels = get_classifier_classes()

                    probability_df = pd.DataFrame({
                        "Class": labels,
                        "Estimated probability (%)": (
                            np.asarray(probabilities) * 100
                        ).round(2),
                    })

                    probability_df = probability_df.sort_values(
                        "Estimated probability (%)",
                        ascending=False,
                    )

                    st.markdown("### Model probability estimates")

                    st.bar_chart(
                        probability_df.set_index("Class")
                    )

                    st.dataframe(
                        probability_df,
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.caption(
                        "These are model probability estimates, not "
                        "guaranteed certainty or calibrated scientific "
                        "confidence."
                    )
                else:
                    st.info(
                        "The loaded model does not expose predict_proba()."
                    )

                # Warn if the values lie outside the central 98% of
                # a feature's observed training-data distribution.
                outside_typical_range = []

                for feature, value in inputs.items():
                    summary = summaries[feature]

                    if value < summary["q01"] or value > summary["q99"]:
                        outside_typical_range.append(feature)

                if outside_typical_range:
                    st.warning(
                        "Some inputs fall outside the central 98% "
                        "of observed values for: "
                        + ", ".join(outside_typical_range)
                        + ". Interpret this prediction cautiously."
                    )

                with st.expander("View submitted measurements"):
                    st.dataframe(
                        input_df,
                        hide_index=True,
                        use_container_width=True,
                    )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.info(
                    "Check that the saved model accepts the exact "
                    "feature names and preprocessing used during training."
                )


# =========================================================
# PAGE 3 — MODEL EVALUATION
# =========================================================
elif page == "Model Evaluation":

    st.subheader("Model Performance Laboratory")

    accuracy = float(metrics.get("accuracy", 0))
    training_rows = int(metrics.get("training_rows", 0))
    testing_rows = int(metrics.get("testing_rows", 0))

    a, b, c = st.columns(3)

    a.metric("Test accuracy", f"{accuracy * 100:.2f}%")
    b.metric("Training records", f"{training_rows:,}")
    c.metric("Testing records", f"{testing_rows:,}")

    st.divider()

    left, right = st.columns(2, gap="large")

    confusion_path = OUTPUTS / "confusion_matrix.png"
    importance_path = OUTPUTS / "feature_importance.png"

    with left:
        st.subheader("Confusion Matrix")

        if confusion_path.exists():
            st.image(
                str(confusion_path),
                use_container_width=True,
            )
        else:
            st.warning(
                "Confusion matrix image not found. "
                "Run python main.py first."
            )

    with right:
        st.subheader("Feature Importance")

        if importance_path.exists():
            st.image(
                str(importance_path),
                use_container_width=True,
            )
        else:
            st.warning(
                "Feature-importance image not found. "
                "Run python main.py first."
            )

    st.divider()

    st.subheader("Classification Report")

    report = metrics.get("classification_report", {})

    if report:
        report_df = pd.DataFrame(report).T

        preferred = [
            label for label in ["STAR", "QSO", "GALAXY"]
            if label in report_df.index
        ]

        preferred += [
            label for label in [
                "macro avg",
                "weighted avg",
                "accuracy",
            ]
            if label in report_df.index
        ]

        if preferred:
            report_df = report_df.loc[preferred]

        st.dataframe(
            report_df.round(3),
            use_container_width=True,
        )

        metric_classes = [
            label for label in ["STAR", "QSO", "GALAXY"]
            if label in report
            and isinstance(report[label], dict)
        ]

        if metric_classes:
            chart_df = pd.DataFrame({
                label: {
                    metric_name: report[label].get(metric_name, 0)
                    for metric_name in ["precision", "recall", "f1-score"]
                }
                for label in metric_classes
            }).T

            st.subheader("Per-Class Performance")
            st.bar_chart(chart_df)

    else:
        st.info("Classification report is not available.")

    st.caption(
        "Precision measures the reliability of positive predictions; "
        "recall measures how many actual objects of a class were found; "
        "F1-score balances precision and recall."
    )

    st.download_button(
        "Download evaluation metrics (JSON)",
        data=json.dumps(metrics, indent=2, default=str),
        file_name="cosmic_identity_metrics.json",
        mime="application/json",
        use_container_width=False,
    )


# =========================================================
# PAGE 4 — DATASET EXPLORER
# =========================================================
elif page == "Dataset Explorer":

    st.subheader("Astronomical Dataset Explorer")

    st.write(
        "Inspect the source survey data, review column quality, "
        "and download a filtered sample for further analysis."
    )

    total_rows, total_columns = df.shape

    missing_total = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())

    a, b, c, d = st.columns(4)

    a.metric("Source rows", f"{total_rows:,}")
    b.metric("Columns", total_columns)
    c.metric("Missing cells", f"{missing_total:,}")
    d.metric("Duplicate rows", f"{duplicate_count:,}")

    st.divider()

    tab_preview, tab_quality, tab_classes = st.tabs([
        "Data Preview",
        "Data Quality",
        "Class Distribution",
    ])

    with tab_preview:
        st.markdown("### Preview records")

        max_rows = min(100, len(df))

        rows_to_show = st.slider(
            "Number of rows",
            min_value=5,
            max_value=max(5, max_rows),
            value=min(20, max(5, max_rows)),
        )

        st.dataframe(
            df.head(rows_to_show),
            use_container_width=True,
            height=400,
        )

        available_columns = df.columns.tolist()

        selected_columns = st.multiselect(
            "Select columns for CSV download",
            options=available_columns,
            default=[
                col for col in features + ["Class"]
                if col in available_columns
            ],
        )

        if selected_columns:
            download_df = df[selected_columns]

            st.download_button(
                "⬇ Download selected data",
                data=download_df.to_csv(index=False).encode("utf-8"),
                file_name="cosmic_identity_dataset_sample.csv",
                mime="text/csv",
                type="primary",
            )
        else:
            st.info("Select at least one column to enable the download.")

    with tab_quality:
        st.markdown("### Missing values by column")

        quality_df = pd.DataFrame({
            "Column": df.columns,
            "Missing values": df.isna().sum().values,
            "Missing (%)": (
                df.isna().mean().values * 100
            ).round(2),
            "Data type": df.dtypes.astype(str).values,
        })

        quality_df = quality_df.sort_values(
            "Missing values",
            ascending=False,
        )

        st.dataframe(
            quality_df,
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("### Data types")
        st.bar_chart(
            quality_df.set_index("Column")[["Missing values"]]
        )

    with tab_classes:
        st.markdown("### Class balance")

        if "Class" in df.columns:
            class_counts = df["Class"].value_counts(dropna=False)

            st.bar_chart(
                class_counts.rename_axis("Class").to_frame("Records")
            )

            st.dataframe(
                pd.DataFrame({
                    "Class": class_counts.index.astype(str),
                    "Records": class_counts.values,
                }),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.warning("No 'Class' column exists in the source dataset.")


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#8797B4;
        font-size:12px;
        padding:8px;
    ">
        COSMIC IDENTITY · ASTROMANTHAN 2.0<br>
        Machine learning for astronomical object classification
    </div>
    """,
    unsafe_allow_html=True,
)