import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("=" * 70)
print("INSTAMART LAPSE MODEL TRAINING AND HYPERPARAMETER TUNING")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

print("\nDataset shape:", df.shape)


# --------------------------------------------------
# 3. DEFINE FEATURES AND TARGET
# --------------------------------------------------

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "city",
    "gender",
    "favourite_category"
]

feature_columns = numeric_features + categorical_features

X = df[feature_columns].copy()
y = df["lapsed"].copy()

# Customer ID is retained separately for later reporting.
customer_ids = df["customer_id"].copy()


# --------------------------------------------------
# 4. DATA VALIDATION
# --------------------------------------------------

assert len(df) == 1156

assert y.value_counts().to_dict() == {
    0: 713,
    1: 443
}

assert not X.isnull().any().any(), (
    "Missing values found in model features"
)

assert "segment_id" not in X.columns

assert "segment_name" not in X.columns

assert "is_premium" not in X.columns

print("\nPASS: Dataset and feature validation")

print("Total customers:", len(X))

print("Total features:", len(feature_columns))

print("Full-cohort frequency mean:", round(X["frequency"].mean(), 4))


# --------------------------------------------------
# 5. TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(

    X,
    y,
    customer_ids,

    test_size=0.2,

    stratify=y,

    random_state=42

)

print("\n--- TRAIN TEST SPLIT ---")

print("Training rows:", len(X_train))

print("Testing rows:", len(X_test))

print("\nTraining distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting distribution:")
print(y_test.value_counts().sort_index())

assert len(X_train) == 924
assert len(X_test) == 232


# --------------------------------------------------
# 6. PREPROCESSING PIPELINE
# --------------------------------------------------

numeric_pipeline = Pipeline([

    ("scaler", StandardScaler())

])

categorical_pipeline = Pipeline([

    (
        "encoder",
        OneHotEncoder(
            drop="first",
            handle_unknown="ignore"
        )
    )

])

preprocessor = ColumnTransformer([

    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),

    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )

])


# --------------------------------------------------
# 7. BASELINE MODEL
# --------------------------------------------------

print("\n" + "=" * 70)
print("BASELINE: ALWAYS PREDICT NOT LAPSED")
print("=" * 70)

baseline_predictions = np.zeros(len(y_test), dtype=int)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

baseline_precision = precision_score(
    y_test,
    baseline_predictions,
    zero_division=0
)

baseline_recall = recall_score(
    y_test,
    baseline_predictions,
    zero_division=0
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
    zero_division=0
)

print("Accuracy:", round(baseline_accuracy, 4))
print("Lapsed precision:", round(baseline_precision, 4))
print("Lapsed recall:", round(baseline_recall, 4))
print("Lapsed F1:", round(baseline_f1, 4))

print("\nBaseline confusion matrix:")
print(confusion_matrix(y_test, baseline_predictions))


# --------------------------------------------------
# 8. LOGISTIC REGRESSION GRID SEARCH
# --------------------------------------------------

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION GRID SEARCH")
print("=" * 70)

logistic_pipeline = Pipeline([

    ("preprocessor", preprocessor),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )

])

logistic_parameters = {

    "classifier__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ]

}

logistic_search = GridSearchCV(

    estimator=logistic_pipeline,

    param_grid=logistic_parameters,

    cv=5,

    scoring="f1",

    n_jobs=-1,

    verbose=1

)

logistic_search.fit(X_train, y_train)

best_logistic = logistic_search.best_estimator_

print("\nBest Logistic Regression parameters:")
print(logistic_search.best_params_)

print(
    "Best Logistic Regression CV F1:",
    round(logistic_search.best_score_, 4)
)


# --------------------------------------------------
# 9. RANDOM FOREST GRID SEARCH
# --------------------------------------------------

print("\n" + "=" * 70)
print("RANDOM FOREST GRID SEARCH")
print("=" * 70)

forest_pipeline = Pipeline([

    ("preprocessor", preprocessor),

    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )

])

forest_parameters = {

    "classifier__max_depth": [
        3,
        5,
        8
    ],

    "classifier__min_samples_leaf": [
        5,
        10,
        20
    ]

}

forest_search = GridSearchCV(

    estimator=forest_pipeline,

    param_grid=forest_parameters,

    cv=5,

    scoring="f1",

    n_jobs=-1,

    verbose=1

)

forest_search.fit(X_train, y_train)

best_forest = forest_search.best_estimator_

print("\nBest Random Forest parameters:")
print(forest_search.best_params_)

print(
    "Best Random Forest CV F1:",
    round(forest_search.best_score_, 4)
)


# --------------------------------------------------
# 10. TRAINING-ONLY SCALER VALIDATION
# --------------------------------------------------

print("\n" + "=" * 70)
print("PREPROCESSING VALIDATION")
print("=" * 70)

fitted_scaler = (
    best_logistic
    .named_steps["preprocessor"]
    .named_transformers_["numeric"]
    .named_steps["scaler"]
)

frequency_index = numeric_features.index("frequency")

training_frequency_mean = fitted_scaler.mean_[frequency_index]

full_frequency_mean = X["frequency"].mean()

print(
    "Training-only frequency mean:",
    round(training_frequency_mean, 4)
)

print(
    "Full-cohort frequency mean:",
    round(full_frequency_mean, 4)
)

print(
    "These means differ because the scaler was fitted only on training rows, "
    "while the full-cohort mean includes the held-out test rows."
)


# --------------------------------------------------
# 11. TEST SET EVALUATION
# --------------------------------------------------

def evaluate_model(name, model):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("\nAccuracy:", round(accuracy, 4))

    print("Lapsed precision:", round(precision, 4))

    print("Lapsed recall:", round(recall, 4))

    print("Lapsed F1:", round(f1, 4))

    print("\nConfusion matrix:")
    print(matrix)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return {

        "model": name,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "confusion_matrix": matrix.tolist()

    }


logistic_results = evaluate_model(
    "Tuned Logistic Regression",
    best_logistic
)

forest_results = evaluate_model(
    "Tuned Random Forest",
    best_forest
)


# --------------------------------------------------
# 12. SAVE RESULTS
# --------------------------------------------------

results = pd.DataFrame([

    {

        "Model": "Logistic Regression",

        "Best Parameters": str(logistic_search.best_params_),

        "Best CV F1": logistic_search.best_score_,

        "Test Accuracy": logistic_results["accuracy"],

        "Test Precision": logistic_results["precision"],

        "Test Recall": logistic_results["recall"],

        "Test F1": logistic_results["f1"]

    },

    {

        "Model": "Random Forest",

        "Best Parameters": str(forest_search.best_params_),

        "Best CV F1": forest_search.best_score_,

        "Test Accuracy": forest_results["accuracy"],

        "Test Precision": forest_results["precision"],

        "Test Recall": forest_results["recall"],

        "Test F1": forest_results["f1"]

    }

])

print("\n--- FINAL MODEL COMPARISON ---")

print(
    results.round(4).to_string(index=False)
)


# --------------------------------------------------
# 13. SAVE TUNED MODELS
# --------------------------------------------------

joblib.dump(

    best_logistic,

    OUTPUT_DIR / "logistic_regression.joblib"

)

joblib.dump(

    best_forest,

    OUTPUT_DIR / "random_forest.joblib"

)

results.to_csv(

    OUTPUT_DIR / "model_comparison.csv",

    index=False

)


# --------------------------------------------------
# 14. SAVE TEST PREDICTIONS FOR LATER TASKS
# --------------------------------------------------

prediction_output = pd.DataFrame({

    "customer_id": id_test.values,

    "actual_lapsed": y_test.values,

    "logistic_prediction": best_logistic.predict(X_test),

    "logistic_probability": best_logistic.predict_proba(X_test)[:, 1],

    "random_forest_prediction": best_forest.predict(X_test),

    "random_forest_probability": best_forest.predict_proba(X_test)[:, 1]

})

prediction_output.to_csv(

    OUTPUT_DIR / "test_predictions.csv",

    index=False

)


# --------------------------------------------------
# 15. FINAL VALIDATION
# --------------------------------------------------

assert best_logistic is not None

assert best_forest is not None

assert logistic_results["f1"] >= 0

assert forest_results["f1"] >= 0

print("\n" + "=" * 70)
print("MODEL TRAINING AND GRID SEARCH COMPLETED")
print("=" * 70)

print("\nSaved files:")

print(OUTPUT_DIR / "logistic_regression.joblib")

print(OUTPUT_DIR / "random_forest.joblib")

print(OUTPUT_DIR / "model_comparison.csv")

print(OUTPUT_DIR / "test_predictions.csv")