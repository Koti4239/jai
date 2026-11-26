import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Customer Auto-Renewal: test ONE saved model.pkl
# ---------------------------------------------------------

st.set_page_config(
    page_title="Test Saved ML Model",
    page_icon="🤖",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "gradient_boosting_model.pkl"
DATASET_PATH = BASE_DIR / "customer_churn_dataset.csv"

# These are the 11 numeric features documented in the supplied project ZIP.
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
    "order_frequency",
]

st.title("🤖 Customer Auto-Renewal — Saved Model Tester")
st.write("Load your existing `model.pkl`, enter customer details, and test its prediction.")

# ---------------------------------------------------------
# Load model and dataset
# ---------------------------------------------------------

@st.cache_resource
def load_model(path, modified_time):
    # Only load pickle files that you created or otherwise trust.
    with open(path, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_dataset(path, modified_time):
    return pd.read_csv(path)


if not MODEL_PATH.exists():
    st.error("model.pkl was not found.")
    st.info("Place model.pkl in the same folder as app.py.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load model.pkl: {exc}")
    st.stop()

dataset = None
if DATASET_PATH.exists():
    try:
        dataset = load_dataset(str(DATASET_PATH), DATASET_PATH.stat().st_mtime)
    except Exception as exc:
        st.warning(f"Dataset could not be read: {exc}")

# Check whether the saved estimator records its training feature names.
saved_features = getattr(model, "feature_names_in_", None)
if saved_features is not None:
    model_features = list(saved_features)
    st.caption("Input feature names were read from the saved model.")
else:
    model_features = FEATURES
    st.warning(
        "This model does not expose feature_names_in_. "
        "The app will use the 11 feature names documented in your project ZIP. "
        "Please confirm these match the model's training columns and order."
    )

if len(model_features) != len(FEATURES):
    st.warning(
        f"The model reports {len(model_features)} input features. "
        f"The project feature list contains {len(FEATURES)}. "
        "The manual form below uses the documented project features; "
        "the prediction may fail if your model was trained differently."
    )

if hasattr(model, "n_features_in_"):
    st.caption(f"Model expects {model.n_features_in_} input feature(s).")

with st.expander("View model input feature names"):
    st.write(model_features)

# ---------------------------------------------------------
# Input values: dataset customer or manual entry
# ---------------------------------------------------------

st.header("1. Enter customer data")

mode_options = ["Manual input"]
if dataset is not None and "customer_id" in dataset.columns:
    mode_options.insert(0, "Choose customer from CSV")

mode = st.radio("Input method", mode_options, horizontal=True)

def number_default(column, fallback=0.0):
    if dataset is not None and column in dataset.columns:
        values = pd.to_numeric(dataset[column], errors="coerce").dropna()
        if not values.empty:
            return float(values.median())
    return float(fallback)


def safe_number(row, column, fallback=0.0):
    if column in row.index and pd.notna(row[column]):
        return float(row[column])
    return float(fallback)


values = {}

if mode == "Choose customer from CSV":
    customer_ids = dataset["customer_id"].astype(str).tolist()
    chosen_id = st.selectbox("Select Customer ID", customer_ids)
    customer = dataset.loc[
        dataset["customer_id"].astype(str) == chosen_id
    ].iloc[0]

    total_orders = safe_number(customer, "total_orders")
    completed = safe_number(customer, "completed_deliveries")
    delivery_ratio = completed / total_orders if total_orders > 0 else 0.0

    # Map CSV's total_orders field to the model's documented "orders" feature.
    csv_mapping = {
        "duration_days": "duration_days",
        "completed_deliveries": "completed_deliveries",
        "delayed_deliveries": "delayed_deliveries",
        "total_contacts": "total_contacts",
        "complaints": "complaints",
        "complaint_ratio": "complaint_ratio",
        "total_web_visits": "total_web_visits",
        "session_frequency": "session_frequency",
        "order_frequency": "order_frequency",
    }

    for feature, csv_column in csv_mapping.items():
        values[feature] = safe_number(customer, csv_column)

    values["delivery_ratio"] = delivery_ratio
    values["orders"] = total_orders

    with st.expander("Review customer input values", expanded=True):
        st.dataframe(
            pd.DataFrame([values]),
            use_container_width=True,
            hide_index=True,
        )

else:
    st.info(
        "Enter the 10 values from the dataset. The app calculates "
        "delivery_ratio = completed_deliveries / total_orders."
    )

    with st.form("manual_input_form"):
        left, right = st.columns(2)

        with left:
            duration_days = st.number_input(
                "Duration days", min_value=0.0,
                value=number_default("duration_days", 30), step=1.0
            )
            total_orders = st.number_input(
                "Total orders", min_value=0.0,
                value=number_default("total_orders", 10), step=1.0
            )
            completed_deliveries = st.number_input(
                "Completed deliveries", min_value=0.0,
                value=number_default("completed_deliveries", 8), step=1.0
            )
            delayed_deliveries = st.number_input(
                "Delayed deliveries", min_value=0.0,
                value=number_default("delayed_deliveries", 1), step=1.0
            )
            total_contacts = st.number_input(
                "Total contacts", min_value=0.0,
                value=number_default("total_contacts", 1), step=1.0
            )

        with right:
            complaints = st.number_input(
                "Complaints", min_value=0.0,
                value=number_default("complaints", 0), step=1.0
            )
            complaint_ratio = st.number_input(
                "Complaint ratio", min_value=0.0, max_value=1.0,
                value=min(1.0, max(0.0, number_default("complaint_ratio", 0))),
                step=0.01, format="%.4f"
            )
            total_web_visits = st.number_input(
                "Total web visits", min_value=0.0,
                value=number_default("total_web_visits", 10), step=1.0
            )
            session_frequency = st.number_input(
                "Session frequency", min_value=0.0,
                value=number_default("session_frequency", 2), step=0.1
            )
            order_frequency = st.number_input(
                "Order frequency", min_value=0.0,
                value=number_default("order_frequency", 1), step=0.1
            )

        submitted = st.form_submit_button(
            "Test model prediction",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        st.stop()

    delivery_ratio = (
        completed_deliveries / total_orders if total_orders > 0 else 0.0
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
        "order_frequency": order_frequency,
    }

if mode == "Choose customer from CSV":
    submitted = st.button(
        "Test model prediction",
        type="primary",
        use_container_width=True,
    )

if submitted:
    # Preserve the exact feature order recorded by the model when available.
    missing = [name for name in model_features if name not in values]
    if missing:
        st.error(
            "The app does not have values for these model features: "
            + ", ".join(missing)
        )
        st.stop()

    model_input = pd.DataFrame(
        [[values[name] for name in model_features]],
        columns=model_features,
    )

    st.header("2. Model input")
    st.dataframe(model_input, use_container_width=True, hide_index=True)

    try:
        prediction = model.predict(model_input)[0]

        st.header("3. Prediction result")
        if str(prediction) == "1":
            st.success("Prediction: Customer WILL renew (1)")
        elif str(prediction) == "0":
            st.warning("Prediction: Customer WILL NOT renew (0)")
        else:
            st.success(f"Model prediction: {prediction}")

        # Show probability only if the model supports predict_proba.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(model_input)[0]
            classes = list(getattr(model, "classes_", range(len(probabilities))))
            positive_index = next(
                (i for i, class_value in enumerate(classes)
                 if str(class_value) == "1"),
                None,
            )

            if positive_index is not None:
                probability = float(probabilities[positive_index])
                st.metric("Probability of auto-renewal", f"{probability:.2%}")
            else:
                st.caption(
                    "The model provides class probabilities, but class '1' "
                    "was not found, so renewal probability was not labelled."
                )
        else:
            st.caption(
                "This model does not provide predict_proba(); only its "
                "predicted class is displayed."
            )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.write(
            "Most commonly, this means the model was trained with different "
            "feature names, feature order, or preprocessing. Upload the exact "
            "model.pkl so its expected input format can be checked."
        )

st.divider()
st.caption(
    "Test application for a saved model.pkl. Only load pickle files from a trusted source."
)
