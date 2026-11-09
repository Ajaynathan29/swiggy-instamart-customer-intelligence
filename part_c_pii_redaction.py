import re
from pathlib import Path

import pandas as pd


# ==========================================================
# 1. PATH CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "raw" / "tickets.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "tickets_redacted.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. LOAD RAW TICKETS
# ==========================================================

print("=" * 65)
print("PART C: PII DETECTION AND REDACTION")
print("=" * 65)

tickets = pd.read_csv(
    INPUT_FILE,
    keep_default_na=False
)

print("\n--- ORIGINAL TICKET DATA ---")
print("Total tickets:", len(tickets))

required_columns = {
    "ticket_id",
    "customer_id",
    "order_id",
    "text",
    "true_category"
}

missing_columns = required_columns - set(tickets.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ==========================================================
# 3. DEFINE PII DETECTION PATTERNS
# ==========================================================

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"
)

ORDER_ID_PATTERN = re.compile(
    r"\bORD-\d{6}\b",
    re.IGNORECASE
)


# ==========================================================
# 4. REDACTION FUNCTION
# ==========================================================

def redact_pii(text):

    if pd.isna(text):
        return text

    text = str(text)

    # Redact email addresses
    text = EMAIL_PATTERN.sub(
        "[EMAIL_REDACTED]",
        text
    )

    # Redact Indian mobile numbers
    text = PHONE_PATTERN.sub(
        "[PHONE_REDACTED]",
        text
    )

    # Redact order IDs
    text = ORDER_ID_PATTERN.sub(
        "[ORDER_ID_REDACTED]",
        text
    )

    return text


# ==========================================================
# 5. PRESERVE ORIGINAL TEXT LOCALLY
# ==========================================================

tickets["original_text"] = tickets["text"].astype(str)

tickets["redacted_text"] = tickets["original_text"].apply(
    redact_pii
)


# ==========================================================
# 6. COUNT DETECTED PII
# ==========================================================

original_text = tickets["original_text"]

email_count = original_text.str.count(
    EMAIL_PATTERN
).sum()

phone_count = original_text.str.count(
    PHONE_PATTERN
).sum()

order_id_count = original_text.str.count(
    ORDER_ID_PATTERN
).sum()


# ==========================================================
# 7. VALIDATE REDACTION
# ==========================================================

redacted_text = tickets["redacted_text"].fillna("").astype(str)

remaining_emails = redacted_text.str.contains(
    EMAIL_PATTERN
).sum()

remaining_phones = redacted_text.str.contains(
    PHONE_PATTERN
).sum()

remaining_order_ids = redacted_text.str.contains(
    ORDER_ID_PATTERN
).sum()


# ==========================================================
# 8. DISPLAY REDACTION SUMMARY
# ==========================================================

print("\n" + "=" * 65)
print("PII REDACTION SUMMARY")
print("=" * 65)

print("Total tickets:", len(tickets))

print("\nDetected PII:")
print("Email addresses:", email_count)
print("Phone numbers:", phone_count)
print("Order IDs:", order_id_count)

print(
    "Total PII occurrences:",
    email_count + phone_count + order_id_count
)

print("\nRemaining PII after redaction:")
print("Emails:", remaining_emails)
print("Phone numbers:", remaining_phones)
print("Order IDs:", remaining_order_ids)


# ==========================================================
# 9. ASSERT REDACTION SUCCESS
# ==========================================================

assert remaining_emails == 0, "Email redaction failed"

assert remaining_phones == 0, "Phone redaction failed"

assert remaining_order_ids == 0, "Order ID redaction failed"

assert len(tickets) == 164, "Unexpected ticket count"

assert tickets["ticket_id"].is_unique, "Duplicate ticket IDs found"

print("\nPASS: All PII patterns successfully redacted.")


# ==========================================================
# 10. SAVE PROCESSED DATA
# ==========================================================

tickets.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nRedacted tickets saved to:")
print(OUTPUT_FILE)


# ==========================================================
# 11. DISPLAY SAFE SAMPLES
# ==========================================================

print("\n--- REDACTED TICKET SAMPLES ---")

for _, row in tickets.head(5).iterrows():

    print("\nTicket ID:", row["ticket_id"])
    print("Redacted text:", row["redacted_text"])


print("\n" + "=" * 65)
print("PII REDACTION COMPLETED SUCCESSFULLY!")
print("=" * 65)