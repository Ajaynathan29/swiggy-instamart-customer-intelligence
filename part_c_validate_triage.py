import re
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FILE = BASE_DIR / "data" / "processed" / "tickets_triaged.csv"

MAX_TICKET_CHARS = 500
MAX_ATTEMPTS = 4

ALLOWED_CATEGORIES = {
    "billing",
    "delivery",
    "technical",
    "fraud",
    "unclassified"
}

EXPECTED_STATUS_COUNTS = {
    "success": 147,
    "invalid": 9,
    "skipped": 8
}

EXPECTED_METHOD_COUNTS = {
    "json": 135,
    "keyword_fallback": 12,
    "invalid_response": 9,
    "none": 8
}

EXPECTED_PRIORITY = {
    "fraud": "critical"
}

EXPECTED_ROUTING = {
    "fraud": "fraud_specialist",
    "billing": "billing_support",
    "delivery": "delivery_support",
    "technical": "technical_support",
    "unclassified": "manual_review"
}


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 65)
print("PART C: INDEPENDENT TRIAGE VALIDATION")
print("=" * 65)

assert FILE.exists(), f"Missing output file: {FILE}"

df = pd.read_csv(FILE, keep_default_na=False)

print("\n--- DATASET OVERVIEW ---")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

required_columns = {
    "ticket_id",
    "customer_id",
    "redacted_text",
    "category",
    "reason",
    "status",
    "classification_method",
    "attempts",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "priority",
    "routing_queue"
}

missing_columns = required_columns - set(df.columns)

assert not missing_columns, (
    f"Missing required columns: {missing_columns}"
)

print("\nPASS: All required columns exist.")


# ============================================================
# 4. ROW COUNT AND UNIQUE TICKET IDs
# ============================================================

assert len(df) == 164

assert df["ticket_id"].is_unique

print("\n--- TICKET ID VALIDATION ---")
print("Total tickets:", len(df))
print("Unique ticket IDs:", df["ticket_id"].nunique())
print("Duplicate ticket IDs: 0")

print("PASS: All 164 tickets retained with unique IDs.")


# ============================================================
# 5. CATEGORY VALIDATION
# ============================================================

invalid_categories = (
    set(df["category"].dropna()) - ALLOWED_CATEGORIES
)

assert not invalid_categories, (
    f"Unexpected categories: {invalid_categories}"
)

print("\n--- CATEGORY DISTRIBUTION ---")
print(df["category"].value_counts())

print("\nPASS: All categories are valid.")


# ============================================================
# 6. PROCESSING STATUS VALIDATION
# ============================================================

actual_status_counts = df["status"].value_counts().to_dict()

print("\n--- PROCESSING STATUS ---")
print(actual_status_counts)

for status, expected_count in EXPECTED_STATUS_COUNTS.items():

    actual_count = actual_status_counts.get(status, 0)

    assert actual_count == expected_count, (
        f"{status}: expected {expected_count}, got {actual_count}"
    )

assert sum(actual_status_counts.values()) == 164

print("PASS: Processing status counts match.")


# ============================================================
# 7. CLASSIFICATION METHOD VALIDATION
# ============================================================

actual_method_counts = (
    df["classification_method"].value_counts().to_dict()
)

print("\n--- CLASSIFICATION METHODS ---")
print(actual_method_counts)

for method, expected_count in EXPECTED_METHOD_COUNTS.items():

    actual_count = actual_method_counts.get(method, 0)

    assert actual_count == expected_count, (
        f"{method}: expected {expected_count}, got {actual_count}"
    )

assert sum(actual_method_counts.values()) == 164

print("PASS: Classification method counts match.")


# ============================================================
# 8. ATTEMPT AND RETRY VALIDATION
# ============================================================

df["attempts"] = pd.to_numeric(
    df["attempts"],
    errors="raise"
)

assert df["attempts"].min() >= 0
assert df["attempts"].max() <= MAX_ATTEMPTS

skipped = df[df["status"] == "skipped"]

assert (skipped["attempts"] == 0).all()

print("\n--- RETRY VALIDATION ---")
print("Maximum attempts:", df["attempts"].max())
print("Tickets skipped before model calls:", len(skipped))

print("PASS: Retry limit respected.")
print("PASS: Skipped tickets made zero model attempts.")


# ============================================================
# 9. TOKEN ACCOUNTING VALIDATION
# ============================================================

token_columns = [
    "prompt_tokens",
    "completion_tokens",
    "total_tokens"
]

for column in token_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="raise"
    )

assert (df[token_columns] >= 0).all().all()

assert (
    df["total_tokens"]
    == df["prompt_tokens"] + df["completion_tokens"]
).all()

assert (
    skipped["total_tokens"] == 0
).all()

print("\n--- TOKEN ACCOUNTING ---")
print("Prompt tokens:", df["prompt_tokens"].sum())
print("Completion tokens:", df["completion_tokens"].sum())
print("Total tokens:", df["total_tokens"].sum())

print("PASS: Token accounting is correct.")


# ============================================================
# 10. INPUT GUARDRAIL VALIDATION
# ============================================================

text = df["redacted_text"].fillna("").astype(str)

empty_mask = text.str.strip().eq("")
long_mask = text.str.len() > MAX_TICKET_CHARS

empty_count = int(empty_mask.sum())
long_count = int(long_mask.sum())

assert empty_count == 4
assert long_count == 4

assert (df.loc[empty_mask, "status"] == "skipped").all()
assert (df.loc[long_mask, "status"] == "skipped").all()

assert int((df["status"] == "skipped").sum()) == 8

print("\n--- INPUT GUARDRAILS ---")
print("Empty tickets:", empty_count)
print("Excessively long tickets:", long_count)
print("Total skipped tickets:", int((df["status"] == "skipped").sum()))

print("PASS: Empty and long tickets were skipped.")


# ============================================================
# 11. PII REDACTION VALIDATION
# ============================================================

email_pattern = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

phone_pattern = re.compile(
    r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"
)

order_pattern = re.compile(
    r"\bORD-\d{6}\b",
    re.IGNORECASE
)

emails_found = text.str.contains(email_pattern).sum()
phones_found = text.str.contains(phone_pattern).sum()
orders_found = text.str.contains(order_pattern).sum()

print("\n--- PII REDACTION VALIDATION ---")
print("Unredacted emails:", emails_found)
print("Unredacted phone numbers:", phones_found)
print("Unredacted order IDs:", orders_found)

assert emails_found == 0
assert phones_found == 0
assert orders_found == 0

print("PASS: No unredacted email, phone or order IDs found.")


# ============================================================
# 12. ORIGINAL TEXT PRIVACY VALIDATION
# ============================================================

assert "original_text" not in df.columns

print("\nPASS: Original ticket text excluded from output.")


# ============================================================
# 13. PRIORITY VALIDATION
# ============================================================

fraud_tickets = df[df["category"] == "fraud"]

assert (
    fraud_tickets["priority"] == "critical"
).all()

premium_tickets = df[
    (df["category"] != "fraud")
    & (df["priority"] == "high")
]

print("\n--- PRIORITY DISTRIBUTION ---")
print(df["priority"].value_counts())

print("Fraud tickets:", len(fraud_tickets))
print("High-priority non-fraud tickets:", len(premium_tickets))

print("PASS: All fraud tickets have critical priority.")


# ============================================================
# 14. ROUTING QUEUE VALIDATION
# ============================================================

for category, expected_queue in EXPECTED_ROUTING.items():

    category_rows = df[df["category"] == category]

    assert (
        category_rows["routing_queue"] == expected_queue
    ).all(), (
        f"Incorrect routing for category: {category}"
    )

print("\n--- ROUTING QUEUES ---")
print(df["routing_queue"].value_counts())

print("PASS: All categories routed to their correct queues.")


# ============================================================
# 15. SUCCESS / INVALID RESPONSE CONSISTENCY
# ============================================================

successful = df[df["status"] == "success"]
invalid = df[df["status"] == "invalid"]

assert (
    successful["classification_method"].isin(
        ["json", "keyword_fallback"]
    )
).all()

assert (
    invalid["classification_method"] == "invalid_response"
).all()

assert (
    invalid["category"] == "unclassified"
).all()

assert (
    skipped["classification_method"] == "none"
).all()

print("\nPASS: Status and classification methods are consistent.")


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL VALIDATION SUMMARY")
print("=" * 65)

checks = [
    "Required columns",
    "164 tickets retained",
    "Unique ticket IDs",
    "Allowed categories",
    "Processing status counts",
    "Classification method counts",
    "Retry limit",
    "Skipped tickets made zero model calls",
    "Token accounting",
    "Input guardrails",
    "PII redaction",
    "Original text excluded",
    "Priority assignment",
    "Routing queue assignment",
    "Status-method consistency"
]

for check in checks:
    print("[PASS]", check)

print("\nALL 15 VALIDATION GROUPS PASSED!")
print("=" * 65)