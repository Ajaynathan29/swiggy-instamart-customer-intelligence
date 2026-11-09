import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"
MODEL_PATH = BASE_DIR / "data" / "processed" / "random_forest.joblib"
THRESHOLD_PATH = BASE_DIR / "data" / "processed" / "best_threshold.txt"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "final_model_evaluation.csv"


# --------------------------------------------------
# 2. LOAD DATA AND MODEL
# --------------------------------------------------

print("=" * 70)
print("PART B: FINAL LAPSE MODEL EVALUATION")
print("=" * 70)

for path in [DATA_PATH, MODEL_PATH, THRESHOLD_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

print("\n--- DATASET OVERVIEW ---")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# --------------------------------------------------
# 3. IDENTIFY EXACT TRAINING FEATURES
# --------------------------------------------------

if not hasattr(model, "feature_names_in_"):
    raise ValueError(
        "The saved model does not contain feature_names_in_. "
        "Cannot verify the original training feature order."
    )

feature_columns = list(model.feature_names_in_)

missing_columns = set(feature_columns) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Model-required features missing from dataset: "
        f"{sorted(missing_columns)}"
    )

if "lapsed" not in df.columns:
    raise ValueError("Target column 'lapsed' is missing.")

if "customer_id" in feature_columns or "lapsed" in feature_columns:
    raise ValueError(
        "Data leakage detected: customer_id or lapsed is included "
        "in the model features."
    )

X = df[feature_columns]
y = df["lapsed"]

print("\n--- MODEL FEATURE VALIDATION ---")
print(f"Number of model features: {len(feature_columns)}")
print("Features:", feature_columns)

assert X.columns.tolist() == feature_columns
assert len(X) == len(y)
assert set(y.unique()).issubset({0, 1})

print("PASS: Model features match the saved training pipeline.")
print("PASS: Target is binary.")
print("PASS: Customer ID and target are excluded from features.")


# --------------------------------------------------
# 4. RECREATE ORIGINAL TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n--- ORIGINAL TRAIN-TEST SPLIT ---")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

assert len(X_train) == 924
assert len(X_test) == 232

print("PASS: Original train-test split reproduced.")


# --------------------------------------------------
# 5. LOAD OPTIMIZED THRESHOLD
# --------------------------------------------------

with open(THRESHOLD_PATH, "r") as file:
    best_threshold = float(file.read().strip())

if not 0 < best_threshold < 1:
    raise ValueError(
        f"Invalid optimized threshold: {best_threshold}"
    )

print(f"\nOptimized threshold: {best_threshold:.4f}")


# --------------------------------------------------
# 6. PREDICT PROBABILITIES
# --------------------------------------------------

probabilities = model.predict_proba(X_test)[:, 1]

if len(probabilities) != len(y_test):
    raise ValueError("Prediction count does not match test rows.")

if not np.isfinite(probabilities).all():
    raise ValueError("Non-finite prediction probabilities detected.")

if ((probabilities < 0) | (probabilities > 1)).any():
    raise ValueError("Prediction probabilities must be between 0 and 1.")

roc_auc = roc_auc_score(y_test, probabilities)
pr_auc = average_precision_score(y_test, probabilities)


# --------------------------------------------------
# 7. EVALUATE DEFAULT AND OPTIMIZED THRESHOLDS
# --------------------------------------------------

thresholds = [0.50, best_threshold]

results = []

for threshold in thresholds:

    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    })

    print("\n" + "=" * 60)
    print(f"FINAL TEST EVALUATION - THRESHOLD {threshold:.4f}")
    print("=" * 60)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print(f"PR AUC:    {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


# --------------------------------------------------
# 8. SAVE FINAL EVALUATION
# --------------------------------------------------

comparison = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

comparison.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n--- FINAL THRESHOLD COMPARISON ---")
print(comparison.round(4).to_string(index=False))


# --------------------------------------------------
# 9. FINAL VALIDATION
# --------------------------------------------------

saved_results = pd.read_csv(OUTPUT_PATH)

assert len(saved_results) == 2
assert list(saved_results.columns) == [
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc"
]

print("\n--- FINAL VALIDATION ---")
print("PASS: Original test split reproduced.")
print("PASS: Model predictions generated successfully.")
print("PASS: Default threshold evaluated.")
print("PASS: Optimized threshold evaluated.")
print("PASS: ROC-AUC and PR-AUC calculated.")
print("PASS: Evaluation results saved and verified.")

print(f"\nOutput: {OUTPUT_PATH}")

print("\nALL FINAL MODEL EVALUATIONS PASSED!")