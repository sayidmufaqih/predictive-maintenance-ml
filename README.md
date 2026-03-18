# Predictive Maintenance for Industrial Machines

## Project Overview
Predictive maintenance is a critical component in modern manufacturing systems, enabling early detection of equipment failure and reducing unplanned downtime. This project presents a data-driven approach to predicting machine failure using machine learning techniques applied to industrial sensor data.

## Dataset
- **Source:** AI4I 2020 Predictive Maintenance dataset  
- **Description:** Synthetic sensor data representing real industrial machine conditions.  
- **Features:** Torque, Rotational Speed, Air Temperature, Process Temperature, Tool Wear, and Machine Type  
- **Target:** Machine Failure (binary classification: 0 = normal, 1 = failure)  

## Objective
The goal of this project is to develop a robust classification model to predict machine failure and support predictive maintenance strategies, improving equipment reliability and reducing downtime.

## Methodology
- **Model:** Random Forest Classifier  
- **Reason:** Handles nonlinear relationships, robust to structured industrial data  
- **Techniques Applied:**
  - Threshold tuning to improve recall for minority class (failure)  
  - Feature importance analysis to identify key predictors of failure  
- **Evaluation Metrics:**
  - Accuracy, Precision, Recall, F1-score  
  - ROC AUC = 0.971  
  - Precision-Recall AUC = 0.710  

## Key Findings
- Mechanical and operational factors such as **Torque, Rotational Speed, and Tool Wear** are the most influential predictors of machine failure.  
- Threshold tuning improved failure recall from **0.56 → 0.67**, balancing detection and precision.  
- The model demonstrates strong performance in distinguishing between normal and failure conditions, even with class imbalance.

## Tools & Technologies
- Python  
- Scikit-learn  
- Pandas & NumPy  
- Matplotlib / Seaborn  

## Project Structure
predictive-maintenance-ml/
- notebook.ipynb  : Main analysis and modeling
- README.md  : Project overview and documentation
- requirements.txt  : Python dependencies
- data/  : dataset link or placeholder

## Future Work
- Explore advanced models such as **Gradient Boosting** or **Deep Learning approaches**  
- Incorporate **time-series analysis** for sequential sensor readings  
- Deploy the model for **real-time predictive maintenance systems**

## Kaggle Notebook
[Link to Kaggle Notebook](https://www.kaggle.com/code/sayidmufaqih/predictive-maintenance-for-industrial-machines)