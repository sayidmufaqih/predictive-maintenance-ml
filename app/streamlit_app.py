import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from validation_range import validate_machine_input
import torch
import torch.nn as nn

def generate_shap_summary(shap_values, feature_names):
    shap_values = np.asarray(shap_values).flatten()

    importance = np.abs(shap_values)
    top_indices = np.argsort(importance)[::-1][:3]

    summary = []

    for idx in top_indices:
        feature = feature_names[idx]
        value = shap_values[idx]

        if value > 0:
            direction = "pushed the prediction toward machine failure"
        else:
            direction = "pushed the prediction toward normal operation"

        summary.append(
            f"- **{feature}** {direction}."
        )

    return summary

st.set_page_config(
    page_title="Industrial Machine Failure Prediction",
    page_icon="⚙️",
    layout="wide"
)

# Load MLP
mlp_model = nn.Sequential(
    nn.Linear(8, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 1)
)

    

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Load Random Forest
rf_model = joblib.load(MODEL_DIR / "random_forest_ml" / "rf_final.pkl")
preprocessor = joblib.load(MODEL_DIR / "random_forest_ml" / "preprocessor.pkl")
threshold = joblib.load(MODEL_DIR / "random_forest_ml" / "threshold.pkl")


# Load MLP model
mlp_model.load_state_dict(
    torch.load(
        MODEL_DIR / "mlp" / "best_mlp.pth",
        map_location="cpu"
    )
)

mlp_model.eval()

mlp_scaler = joblib.load(MODEL_DIR / "mlp" / "scaler.pkl")
mlp_encoder = joblib.load(MODEL_DIR / "mlp" / "encoder.pkl")
mlp_threshold = joblib.load( MODEL_DIR / "mlp" / "threshold.pkl")

# SHAP explainer for Random Forest
explainer = shap.TreeExplainer(rf_model)


# Load SHAP background for MLP
mlp_background = torch.load(
    MODEL_DIR / "mlp" / "shap_background.pt",
    map_location="cpu"
)

mlp_explainer = shap.DeepExplainer(
    mlp_model,
    mlp_background
)


# Sidebar
with st.sidebar:
    st.header("About This Model")

    st.write(
        "This application uses machine learning to predict "
        "the probability of industrial machine failure "
        "based on operating conditions."
    )

    st.divider()

    st.subheader("Models")

    st.markdown(
        """
        **Random Forest**
        - Ensemble machine learning model
        - Handles nonlinear relationships
        - Explained using Tree SHAP

        **MLP**
        - Multilayer neural network
        - Trained with class weighting
        - Uses a tuned decision threshold
        - Explained using Deep SHAP
        """
    )

    st.divider()

    st.subheader("Input Features")

    st.markdown(
        """
        - Machine Type
        - Air Temperature
        - Process Temperature
        - Rotational Speed
        - Torque
        - Tool Wear
        """
    )

    st.divider()

    st.caption(
        "SHAP explanations describe model behavior "
        "and should not be interpreted as causal effects."
    )

    st.divider()

    st.caption("Author")
    st.write("**Sayid Mufaqih**")
    st.caption(
        "Machine Learning & AI Portfolio Project"
    )

st.title("⚙️ Industrial Machine Failure Prediction")

st.markdown(
    """
    **AI-powered predictive maintenance system** for detecting
    potential machine failures from operating conditions.

    This application compares a **Random Forest** model and a
    **Multilayer Perceptron (MLP)** neural network, with
    **SHAP-based explainability** to understand each prediction.
    """
)

st.divider()

model_choice = st.selectbox(
    "Select Model",
    ["Random Forest", "MLP"]
)

if model_choice == "Random Forest":
    st.info(
        "Random Forest model with SHAP-based explainability."
    )
else:
    st.info(
        "MLP neural network trained with class weighting "
        "and threshold tuning for imbalanced failure detection."
    )

#INPUT MACHINE
st.subheader("Machine Operating Conditions")

st.caption(
    "Enter the current operating conditions of the machine."
)

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temperature = st.number_input(
        "Air Temperature [K]",
        value=300.0,
        step=0.1,
        help="Valid input range: 295.3–304.5 K."
    )

    process_temperature = st.number_input(
        "Process Temperature [K]",
        value=310.0,
        step=0.1,
        help="Valid input range: 305.7–313.8 K."
    )

with col2:
    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        value=1500,
        step=10,
        help="Valid input range: 1168–2886 rpm."
    )

    torque = st.number_input(
        "Torque [Nm]",
        value=40.0,
        step=0.1,
        help="Valid input range: 3.8–76.6 Nm."
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        value=100,
        step=1,
        help="Valid input range: 0–253 min."
    )

st.info(
    "Input ranges are based on the training data used by the final model. "
    "Values outside these ranges are not supported."
)


# PREDICTION BUTTON
if st.button(
    "🔍 Predict Machine Failure",
    type="primary",
    key="predict_button"
):
    # -----------------------------
    # CREATE INPUT DATA
    # -----------------------------
    input_data = pd.DataFrame({
        "Type": [machine_type],
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    # -----------------------------
    # INPUT VALIDATION
    # -----------------------------
    errors = validate_machine_input(input_data)

    if errors:

        st.error("❌ Input validation failed.")

        for error in errors:
            st.warning(error)

    else:

        # -----------------------------
        # PREDICTION
        # -----------------------------

        if model_choice == "Random Forest":

            # Preprocessing for Random Forest
            input_processed = preprocessor.transform(
                input_data
            )

            # Random Forest prediction
            failure_probability = rf_model.predict_proba(
                input_processed
            )[0, 1]

            prediction = int(
                failure_probability >= threshold
            )

            current_threshold = threshold


        else:

            # Preprocessing for MLP
            numerical_features = [
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]"
            ]

            categorical_features = ["Type"]

            input_num = mlp_scaler.transform(
                input_data[numerical_features]
            )

            input_cat = mlp_encoder.transform(
                input_data[categorical_features]
            )

            input_processed = np.hstack([
                input_num,
                input_cat
            ])

            # Convert to PyTorch tensor
            input_tensor = torch.tensor(
                input_processed,
                dtype=torch.float32
            )

            # MLP prediction
            with torch.no_grad():
                output = mlp_model(input_tensor)

                failure_probability = torch.sigmoid(
                    output
                ).item()

            prediction = int(
                failure_probability >= mlp_threshold
            )

            current_threshold = mlp_threshold
        # -----------------------------
        # PREDICTION RESULT
        # -----------------------------
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Failure Probability",
                f"{failure_probability:.1%}"
            )

        with col2:
            st.metric(
                "Decision Threshold",
                f"{current_threshold:.1%}"
            )

        if prediction == 1:
            st.error(
                "⚠️ **Predicted Failure Risk**\n\n"
                "The model estimates that the machine may require attention."
            )
        else:
            st.success(
                "✅ **Predicted Normal Operation**\n\n"
                "The model does not identify a failure risk above the decision threshold."
            )

        st.caption(
            "The prediction is based on the selected model and its tuned decision threshold."
        )
        # -----------------------------
        # SHAP EXPLANATION
        # -----------------------------
        if model_choice == "Random Forest":

            st.subheader(
                "Why did the model make this prediction?"
            )

            # Calculate SHAP values
            shap_values = explainer.shap_values(
                input_processed
            )

            # Handle different SHAP output formats
            if isinstance(shap_values, list):

                shap_values_failure = shap_values[1][0]

            elif len(shap_values.shape) == 3:

                shap_values_failure = shap_values[0, :, 1]

            else:

                shap_values_failure = shap_values[0]

            # Feature names
            feature_names = (
                preprocessor.get_feature_names_out()
            )

            clean_feature_names = [
                name.replace("num__", "")
                    .replace("cat__", "")
                for name in feature_names
            ]

            # Create SHAP explanation
            shap_explanation = shap.Explanation(
                values=shap_values_failure,
                base_values=explainer.expected_value[1],
                data=input_processed[0],
                feature_names=clean_feature_names
            )

            # Display waterfall
            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

            shap.plots.waterfall(
                shap_explanation,
                max_display=8,
                show=False
            )

            st.pyplot(fig)

            plt.close(fig)

            st.markdown("### Key factors in this prediction")

            summary = generate_shap_summary(
                    shap_values_failure,
                    clean_feature_names)

            for item in summary:
                st.markdown(item)
            st.caption(
                "SHAP values show how each feature contributed to the model's prediction. "
                "They describe model behavior and should not be interpreted as causal effects."
            )
            

        else:

            st.subheader(
                "Why did the model make this prediction?"
            )

            # Calculate SHAP values
            shap_values = mlp_explainer.shap_values(
                input_tensor
            )

            # Handle SHAP output format
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_values = np.array(shap_values)

            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 0]

            # MLP feature names
            mlp_feature_names = [
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
                "Type_H",
                "Type_L",
                "Type_M"
            ]

            # Create SHAP explanation
            shap_explanation = shap.Explanation(
                values=shap_values[0],
                base_values=np.array(
                    mlp_explainer.expected_value
                ).flatten()[0],
                data=input_processed[0],
                feature_names=mlp_feature_names
            )

            # Display waterfall plot
            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

            shap.plots.waterfall(
                shap_explanation,
                max_display=8,
                show=False
            )

            st.pyplot(fig)

            plt.close(fig)
            st.markdown("### Key factors in this prediction")

            summary = generate_shap_summary(
                shap_values[0],
                mlp_feature_names
            )

            for item in summary:
                st.markdown(item)

            st.caption(
                "SHAP values show how each feature contributed to the model's prediction. "
                "They describe model behavior and should not be interpreted as causal effects."
            )

st.divider()

st.caption(
    "Industrial Predictive Maintenance | Machine Learning & Explainable AI"
)