# Industrial Machine Failure Prediction

## Project Overview
This project develops an end-to-end predictive maintenance system for detecting industrial machine failures from operating conditions. It compares a Random Forest model with a Multilayer Perceptron (MLP) and integrates SHAP-based explainability into a Streamlit application.

## Dataset
- **Source:** AI4I 2020 Predictive Maintenance dataset
- **Description:** Synthetic data representing industrial machine operating conditions.
- **Features:** Machine Type, Air Temperature, Process Temperature, Rotational Speed, Torque, and Tool Wear
- **Target:** Machine Failure (0 = normal, 1 = failure)

## Objective
The goal is to develop machine learning and deep learning models that can identify potential machine failures from operating conditions and provide interpretable predictions for predictive maintenance.

## Methodology

### 1. Data Preprocessing

- Removed identifier and failure-mode columns that were not used as predictive features.
- Encoded categorical machine type using One-Hot Encoding.
- Standardized numerical features for the MLP model.
- Split the data into training, validation, and test sets.

### 2. Machine Learning — Random Forest

A Random Forest classifier was trained to capture nonlinear relationships between machine operating conditions and failure risk.

- Class imbalance was considered during model development.
- Decision threshold was tuned using the validation set to improve failure detection.
- Feature importance and SHAP were used to interpret model behavior.

### 3. Deep Learning — Multilayer Perceptron

A Multilayer Perceptron (MLP) neural network was developed as a deep learning approach.

- Class weighting was used to address the imbalanced target.
- The decision threshold was tuned using validation data.
- SHAP was used to explain individual predictions.

### 4. Evaluation

Models were evaluated using:

- Precision
- Recall
- F1-score
- ROC AUC
- Precision-Recall AUC

## Model Performance

The final models were evaluated on the held-out test set. Because machine failure is a minority class, particular attention was given to failure recall and F1-score.

| Model | Accuracy | Failure Precision | Failure Recall | Failure F1-score |
|---|---:|---:|---:|---:|
| Random Forest | — | — | — | — |
| MLP | 95% | 39% | 92% | 55% |

The MLP achieved a high failure recall on the test set, indicating that it was able to identify most of the machines labeled as failures. The Random Forest and MLP models use separately tuned decision thresholds to account for the imbalanced failure class.

## Key Findings

- Mechanical and operational factors such as **Torque, Rotational Speed, and Tool Wear** are the most influential predictors of machine failure.  
- Threshold tuning improved failure recall from **0.56 → 0.67**, balancing detection and precision.  
- The model demonstrates strong performance in distinguishing between normal and failure conditions, even with class imbalance.

## Explainability

SHAP (SHapley Additive exPlanations) was used to interpret model predictions.

- **Global explainability:** Identifies features that have the greatest influence on model predictions.
- **Local explainability:** Explains how individual machine conditions contribute to a specific failure prediction.
- SHAP explanations describe model behavior and should not be interpreted as causal effects.

The explainability component is integrated into the Streamlit application, allowing users to inspect the key factors behind each prediction.


## Project Structure

predictive-maintenance-ml/
├── app/
│   └── streamlit_app.py
├── data/
│   └── ai4i2020.csv
├── models/
│   ├── random_forest_ml/
│   │   ├── rf_final.pkl
│   │   ├── preprocessor.pkl
│   │   └── threshold.pkl
│   └── mlp/
│       ├── best_mlp.pth
│       ├── scaler.pkl
│       ├── encoder.pkl
│       ├── threshold.pkl
│       └── shap_background.pt
├── notebooks/
│   ├── machine_learning.ipynb
│   └── deep_learning.ipynb
├── src/
│   └── validation_range.py
├── README.md
└── requirements.txt

## Tools & Technologies

- Python
- Pandas & NumPy
- Scikit-learn
- PyTorch
- SHAP
- Matplotlib
- Streamlit
- Joblib


## How to Run

### 1. Clone the repository

git clone https://github.com/sayidmufaqih/predictive-maintenance-ml.git
cd predictive-maintenance-ml

## Reproducibility

The notebooks contain the complete workflow for data preprocessing, model training, evaluation, threshold tuning, and explainability.

- `machine_learning.ipynb` — Random Forest development
- `deep_learning.ipynb` — MLP development
- `app/streamlit_app.py` — Model deployment and interactive prediction

The trained model artifacts are included in the `models/` directory for running the application without retraining the models.

## Future Work

- Incorporate time-series sensor data for sequential machine condition monitoring.
- Explore advanced deep learning architectures for predictive maintenance.
- Integrate real-time sensor data for continuous failure monitoring.
- Improve model calibration and probability estimation for operational decision support.

## Kaggle Notebook

The original exploratory and modeling work is also available on Kaggle:

[View the Kaggle Notebook](https://www.kaggle.com/code/sayidmufaqih/predictive-maintenance-for-industrial-machines)