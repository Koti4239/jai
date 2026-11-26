import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Customer Auto-Renewal Prediction - Gradient Boosting
# ---------------------------------------------------------

st.set_page_config(
    page_title="Customer Renewal Prediction",
    page_icon="🤖",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# Change this to your actual saved model filename
MODEL_PATH = BASE_DIR / "gradient_boosting_model.pkl"

DATASET_PATH = BASE_DIR / "customer_churn_dataset.csv"

# ---------------------------------------------------------
# Model input features
# ---------------------------------------------------------

FEATURES = [
    "duration_days",
    "completed_deliveries",
    "delayed_deliveries",
    "delivery_ratio",
    "total_contacts",
    "complaints",
    "complaint_ratio",
    "total_web_visits",
    "orders",
    "session_frequency",
    "order_frequency"
]

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)

@st.cache_data
def load_dataset(path, modified_time):
    return pd.read_csv(path)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}")
    st.info("Place your Gradient Boosting model in the same folder as app.py.")
    st.stop()

try:
    model = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime
    )
    st.success("Gradient Boosting model loaded successfully!")

except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

dataset = None

if DATASET_PATH.exists():
    try:
        dataset = load_dataset(
            str(DATASET_PATH),
            DATASET_PATH.stat().st_mtime
        )
    except Exception as exc:
        st.warning(f"Dataset could not be loaded: {exc}")

# ---------------------------------------------------------
# Model feature check
# ---------------------------------------------------------

saved_features = getattr(model, "feature_names_in_", None)

if saved_features is not None:
    model_features = list(saved_features)
else:
    model_features = FEATURES

st.title("🐾 Customer Auto-Renewal Prediction")
st.write(
    "Predict whether a customer is likely to renew "
    "their subscription using a Gradient Boosting model."
)

st.divider()

if hasattr(model, "n_features_in_"):
    st.info(
        f"Model expects {model.n_features_in_} input features."
    )

if len(model_features) != len(FEATURES):
    st.warning(
        "The saved model feature count differs from the manual input "
        "feature list. Make sure the model was trained using these "
        "same features before making predictions."
    )

with st.expander("View model features"):
    st.write(model_features)

# ---------------------------------------------------------
# Load default values from dataset
# ---------------------------------------------------------

def number_default(column, fallback=0.0):
    if dataset is not None and column in dataset.columns:
        values = pd.to_numeric(
            dataset[column], errors="coerce"
        ).dropna()

        if not values.empty:
            return float(values.median())

    return float(fallback)

def safe_number(row, column, fallback=0.0):
    if column in row.index and pd.notna(row[column]):
        return float(row[column])

    return float(fallback)

# ---------------------------------------------------------
# Input method
# ---------------------------------------------------------

st.header("1. Customer Information")

mode_options = ["Manual Input"]

if dataset is not None and "customer_id" in dataset.columns:
    mode_options.insert(0, "Select Customer from Dataset")

mode = st.radio(
    "Choose input method",
    mode_options,
    horizontal=True
)

values = {}
submitted = False

# ---------------------------------------------------------
# Select customer from dataset
# ---------------------------------------------------------

if mode == "Select Customer from Dataset":

    customer_ids = dataset["customer_id"].astype(str).tolist()

    chosen_id = st.selectbox(
        "Select Customer ID",
        customer_ids
    )

    customer = dataset.loc[
        dataset["customer_id"].astype(str) == chosen_id
    ].iloc[0]

    total_orders = safe_number(customer, "total_orders")
    completed = safe_number(customer, "completed_deliveries")

    delivery_ratio = (
        completed / total_orders
        if total_orders > 0
        else 0.0
    )

    csv_mapping = {
        "duration_days": "duration_days",
        "completed_deliveries": "completed_deliveries",
        "delayed_deliveries": "delayed_deliveries",
        "total_contacts": "total_contacts",
        "complaints": "complaints",
        "complaint_ratio": "complaint_ratio",
        "total_web_visits": "total_web_visits",
        "session_frequency": "session_frequency",
        "order_frequency": "order_frequency"
    }

    for feature, csv_column in csv_mapping.items():
        values[feature] = safe_number(customer, csv_column)

    values["delivery_ratio"] = delivery_ratio
    values["orders"] = total_orders

    st.subheader("Customer Details")

    st.dataframe(
        pd.DataFrame([values]),
        use_container_width=True,
        hide_index=True
    )

    submitted = st.button(
        "Predict Customer Renewal",
        type="primary",
        use_container_width=True
    )

# ---------------------------------------------------------
# Manual input
# ---------------------------------------------------------

else:

    st.caption(
        "Enter customer information below. "
        "Delivery ratio will be calculated automatically."
    )

    with st.form("customer_input_form"):

        col1, col2 = st.columns(2)

        with col1:

            duration_days = st.number_input(
                "Duration (Days)",
                min_value=0.0,
                value=number_default("duration_days", 30.0),
                step=1.0
            )

            total_orders = st.number_input(
                "Total Orders",
                min_value=0.0,
                value=number_default("total_orders", 10.0),
                step=1.0
            )

            completed_deliveries = st.number_input(
                "Completed Deliveries",
                min_value=0.0,
                value=number_default("completed_deliveries", 8.0),
                step=1.0
            )

            delayed_deliveries = st.number_input(
                "Delayed Deliveries",
                min_value=0.0,
                value=number_default("delayed_deliveries", 1.0),
                step=1.0
            )

            total_contacts = st.number_input(
                "Total Contacts",
                min_value=0.0,
                value=number_default("total_contacts", 1.0),
                step=1.0
            )

        with col2:

            complaints = st.number_input(
                "Complaints",
                min_value=0.0,
                value=number_default("complaints", 0.0),
                step=1.0
            )

            complaint_ratio = st.number_input(
                "Complaint Ratio",
                min_value=0.0,
                max_value=1.0,
                value=min(
                    1.0,
                    max(0.0, number_default("complaint_ratio", 0.0))
                ),
                step=0.01
            )

            total_web_visits = st.number_input(
                "Total Web Visits",
                min_value=0.0,
                value=number_default("total_web_visits", 10.0),
                step=1.0
            )

            session_frequency = st.number_input(
                "Session Frequency",
                min_value=0.0,
                value=number_default("session_frequency", 2.0),
                step=0.1
            )

            order_frequency = st.number_input(
                "Order Frequency",
                min_value=0.0,
                value=number_default("order_frequency", 1.0),
                step=0.1
            )

        submitted = st.form_submit_button(
            "Predict Customer Renewal",
            type="primary",
            use_container_width=True
        )

    if submitted:

        delivery_ratio = (
            completed_deliveries / total_orders
            if total_orders > 0
            else 0.0
        )

        values = {
            "duration_days": duration_days,
            "completed_deliveries": completed_deliveries,
            "delayed_deliveries": delayed_deliveries,
            "delivery_ratio": delivery_ratio,
            "total_contacts": total_contacts,
            "complaints": complaints,
            "complaint_ratio": complaint_ratio,
            "total_web_visits": total_web_visits,
            "orders": total_orders,
            "session_frequency": session_frequency,
            "order_frequency": order_frequency
        }

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if submitted:

    missing_features = [
        feature for feature in model_features
        if feature not in values
    ]

    if missing_features:
        st.error(
            "Missing model features: "
            + ", ".join(missing_features)
        )
        st.stop()

    model_input = pd.DataFrame(
        [[values[feature] for feature in model_features]],
        columns=model_features
    )

    st.header("2. Prediction Input")

    st.dataframe(
        model_input,
        use_container_width=True,
        hide_index=True
    )

    try:

        prediction = model.predict(model_input)[0]

        st.header("3. Prediction Result")

        if str(prediction) == "1":

            st.success("🐾 Customer is predicted to RENEW")

        elif str(prediction) == "0":

            st.warning("⚠️ Customer is predicted NOT to renew")

        else:

            st.success(f"Prediction: {prediction}")

        # -------------------------------------------------
        # Prediction probability
        # -------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(model_input)[0]

            classes = list(
                getattr(model, "classes_", range(len(probabilities)))
            )

            positive_index = next(
                (
                    i for i, class_value in enumerate(classes)
                    if str(class_value) == "1"
                ),
                None
            )

            if positive_index is not None:

                probability = float(probabilities[positive_index])

                st.metric(
                    "Probability of Renewal",
                    f"{probability:.2%}"
                )

                st.progress(probability)

                if probability >= 0.70:

                    st.info(
                        "This customer has a high predicted probability "
                        "of renewing."
                    )

                elif probability >= 0.40:

                    st.info(
                        "This customer has a moderate probability "
                        "of renewing."
                    )

                else:

                    st.warning(
                        "This customer has a low predicted probability "
                        "of renewing. Retention actions may be considered."
                    )

    except Exception as exc:

        st.error(f"Prediction failed: {exc}")

        st.warning(
            "Check that the input features, their names, their order, "
            "and any preprocessing steps match the original model training."
        )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Customer Auto-Renewal Prediction | Gradient Boosting Classifier"
)
```
