# Cars-Evaluation-Analysis-CS431-
## Overview
Most AI focuses on accuracy, leaving decision "logic" hidden. This project uses the Car Evaluation Dataset to find "deal-breakers" like safety or cost. We aim to ensure model priorities align with human experts, making automated decisions transparent and reliable.

## Project Structure
```
.
├── data/               # Source datasets (car.data)
├── results/Images/     # Output visualizations (Confusion Matrix, Feature Importance)
├── cars_eval.py        # Primary script for model execution and evaluation
├── requirements.txt    # Python dependencies
└── README.md           # Documentation

```

## Prerequisites
```
1. scikit-learn
2. pandas
3. numpy
4. matplotlib
5. seaborn
```
## Installation
### 1. Clone the repository
```
git clone https://github.com/VanshSachdeva-04/Car-Evaluation-Analysis.git
cd Car-Evaluation-Analysis/
```

### 2. Install the required dependencies
```
pip install -r requirements.txt
```

## How to Run
```
Execute the primary script:
python3 cars_eval.py

This script automates the following workflow:
1. Data Loading: Ingests and preprocesses the raw dataset.
2. Model Training: Executes the Random Forest training process using optimized hyperparameters.
3. Validation: Calculates Accuracy and Macro F1-Score via Stratified 10-Fold CV.
4. Visualization: Exports the confusion matrix (figure_1.png) and feature importance (figure_2.png) to the project folder.
```

## Key Findings
```
1. High Preformance: Achieved 97% accuracy, verified through rigorous stratified validation.
2. Interpretablity: Identified that Safety (30.3%) and Passenger Capacity (23.9%) are the primary driver decisions.
3. Logic Robustness: Sensitivity analysis confirmed that the model incorporates 'Safety' as a non-negotiable threshold, mirroring human decision-making hierarchies.
```

## Authors
```
Vansh Sachdeva, Wiem Boubaker