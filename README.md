# Cosmic Identity — Stellar Object Classification

A hackathon-ready Python project for classifying astronomical survey records into STAR, QSO, and GALAXY.

## Included
- `Dataset 41.csv` — supplied dataset
- `main.py` — train/evaluate model and save reports
- `app.py` — interactive Streamlit dashboard
- `cosmic_ml.py` — shared cleaning and ML pipeline
- `requirements.txt` — dependencies
- `run_model.bat` and `run_dashboard.bat` — Windows launch helpers

## Requirements
- Python 3.10 or newer recommended
- VS Code with the Python extension

## Run in VS Code (Windows)
Open this folder in VS Code, then open Terminal and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

To start the dashboard:

```powershell
python -m streamlit run app.py
```

The browser opens the dashboard. Click **Train Random Forest model**.

If PowerShell blocks environment activation, use Command Prompt and run:
```bat
.venv\Scripts\activate.bat
```

## Run with the included batch files
1. Double-click `run_model.bat` (or run it from the VS Code terminal).
2. After the model completes, run `run_dashboard.bat`.

## Pipeline
1. Load and inspect the CSV.
2. Standardize target labels and remove exact duplicate records.
3. Use selected astronomical numeric features only.
4. Convert non-numeric values to missing, validate coordinates, and handle invalid values.
5. Impute missing numeric values using a median imputer fitted only on training data.
6. Split labeled observations into stratified 80% training and 20% testing subsets.
7. Train a Random Forest classifier and evaluate accuracy, per-class precision/recall/F1 and confusion matrix.
8. Save charts, metrics, sample predictions and the fitted model in `outputs/`.

## Important notes
- The target is `Class`. The duplicate label column `class_label` is intentionally excluded to prevent target leakage.
- This is a first-pass model. Confirm the dataset documentation for invalid sentinel values and astronomy-specific measurement quality rules.
- Report only the accuracy and metrics printed by your own run; do not promise a particular score in advance.
- Do not use test results to tune the model repeatedly. For stronger validation, use cross-validation on training data and keep the held-out test set for final evaluation.
