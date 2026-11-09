import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_TICKETS = BASE_DIR / "data" / "raw" / "tickets.csv"

TRIAGED_TICKETS = (
    BASE_DIR / "data" / "processed" / "tickets_triaged.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVALUATION_FILE = OUTPUT_DIR / "triage_evaluation.csv"
METRICS_FILE = OUTPUT_DIR / "triage_metrics.csv"
CONFUSION_FILE = OUTPUT_DIR / "triage_confusion_matrix.csv"


# ============================================================
# 2. CONFIGURATION
# ============================================================

CATEGORIES = [
    "billing",
    "delivery",
    "technical",
    "fraud"
]


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("PART C: CLASSIFICATION EVALUATION")
print("=" * 70)

print("\nLoading original ticket labels...")

raw_df = pd.read_csv(
    RAW_TICKETS,
    keep_default_na=False
)

triaged_df = pd.read_csv(
    TRIAGED_TICKETS,
    keep_default_na=False
)

print("Original tickets:", len(raw_df))
print("Triaged tickets:", len(triaged_df))


# ============================================================
# 4. DATA VALIDATION
# ============================================================

required_raw = {
    "ticket_id",
    "true_category"
}

required_triaged = {
    "ticket_id",
    "category",
    "status",
    "classification_method"
}

assert required_raw.issubset(raw_df.columns)
assert required_triaged.issubset(triaged_df.columns)

assert raw_df["ticket_id"].is_unique
assert triaged_df["ticket_id"].is_unique

print("\nPASS: Required columns and unique IDs verified.")


# ============================================================
# 5. MERGE ORIGINAL LABELS WITH PREDICTIONS
# ============================================================

evaluation_df = raw_df[
    ["ticket_id", "true_category"]
].merge(
    triaged_df[
        [
            "ticket_id",
            "category",
            "status",
            "classification_method"
        ]
    ],
    on="ticket_id",
    how="inner",
    validate="one_to_one"
)

assert len(evaluation_df) == 164

evaluation_df["true_category"] = (
    evaluation_df["true_category"]
    .astype(str)
    .str.strip()
    .str.lower()
)

evaluation_df["category"] = (
    evaluation_df["category"]
    .astype(str)
    .str.strip()
    .str.lower()
)

print("Matched tickets:", len(evaluation_df))

print("\n--- ORIGINAL CATEGORY DISTRIBUTION ---")
print(evaluation_df["true_category"].value_counts())


# ============================================================
# 6. SEPARATE SUCCESSFUL, INVALID AND SKIPPED TICKETS
# ============================================================

successful = evaluation_df[
    evaluation_df["status"] == "success"
].copy()

invalid = evaluation_df[
    evaluation_df["status"] == "invalid"
].copy()

skipped = evaluation_df[
    evaluation_df["status"] == "skipped"
].copy()

print("\n--- PROCESSING SUMMARY ---")

print("Successfully classified:", len(successful))
print("Invalid model responses:", len(invalid))
print("Skipped tickets:", len(skipped))


# ============================================================
# 7. PREPARE EVALUATION DATA
# ============================================================

# Only evaluate tickets with a valid ground-truth label
# and a successful classification.

valid_evaluation = successful[
    successful["true_category"].isin(CATEGORIES)
    & successful["category"].isin(CATEGORIES)
].copy()

y_true = valid_evaluation["true_category"]
y_pred = valid_evaluation["category"]

assert len(valid_evaluation) == len(successful)

print("\nTickets included in classification evaluation:", len(valid_evaluation))


# ============================================================
# 8. ACCURACY
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

correct_predictions = int((y_true == y_pred).sum())

evaluated_count = len(valid_evaluation)

eligible_count = len(evaluation_df[
    evaluation_df["true_category"].isin(CATEGORIES)
    & ~evaluation_df["status"].eq("skipped")
])

coverage = (
    len(successful) / eligible_count
    if eligible_count else 0
)

effective_accuracy = (
    correct_predictions / eligible_count
    if eligible_count else 0
)

print("\n" + "=" * 70)
print("CLASSIFICATION ACCURACY")
print("=" * 70)

print("Correct predictions:", correct_predictions)
print("Evaluated successful classifications:", evaluated_count)

print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Classification coverage: {coverage:.4f} ({coverage * 100:.2f}%)")

print(
    f"Effective accuracy including invalid responses: "
    f"{effective_accuracy:.4f} ({effective_accuracy * 100:.2f}%)"
)


# ============================================================
# 9. PRECISION, RECALL AND F1-SCORE
# ============================================================

precision, recall, f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=CATEGORIES,
    zero_division=0
)

print("\n" + "=" * 70)
print("CATEGORY-WISE PERFORMANCE")
print("=" * 70)

report = classification_report(
    y_true,
    y_pred,
    labels=CATEGORIES,
    target_names=CATEGORIES,
    zero_division=0
)

print(report)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=CATEGORIES
)

confusion_df = pd.DataFrame(
    matrix,
    index=[f"Actual_{category}" for category in CATEGORIES],
    columns=[f"Predicted_{category}" for category in CATEGORIES]
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(confusion_df)


# ============================================================
# 11. INVALID AND SKIPPED ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("INVALID AND SKIPPED TICKETS")
print("=" * 70)

print("\nInvalid response ground-truth categories:")
print(invalid["true_category"].value_counts())

print("\nSkipped ticket ground-truth categories:")
print(skipped["true_category"].value_counts())


# ============================================================
# 12. SAVE PER-TICKET EVALUATION
# ============================================================

evaluation_df["correct"] = (
    (evaluation_df["status"] == "success")
    & (evaluation_df["true_category"] == evaluation_df["category"])
)

# Store no original customer message text.
evaluation_output = evaluation_df[
    [
        "ticket_id",
        "true_category",
        "category",
        "status",
        "classification_method",
        "correct"
    ]
].copy()

evaluation_output.to_csv(
    EVALUATION_FILE,
    index=False
)


# ============================================================
# 13. SAVE CATEGORY METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "category": CATEGORIES,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "support": support
})

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


# ============================================================
# 14. SAVE CONFUSION MATRIX
# ============================================================

confusion_df.to_csv(CONFUSION_FILE)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION SUMMARY")
print("=" * 70)

print(f"Total tickets: {len(evaluation_df)}")
print(f"Successful classifications: {len(successful)}")
print(f"Invalid responses: {len(invalid)}")
print(f"Skipped tickets: {len(skipped)}")

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Coverage: {coverage * 100:.2f}%")
print(f"Effective accuracy: {effective_accuracy * 100:.2f}%")

print("\nSaved evaluation files:")

print(EVALUATION_FILE)
print(METRICS_FILE)
print(CONFUSION_FILE)

print("\nPART C CLASSIFICATION EVALUATION COMPLETED!")
print("=" * 70)