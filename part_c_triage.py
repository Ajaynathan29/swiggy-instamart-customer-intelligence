import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import mock_llm


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "tickets_redacted.csv"

SEGMENT_FILE = BASE_DIR / "outputs" / "customer_segments.csv"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

TRIAGED_FILE = PROCESSED_DIR / "tickets_triaged.csv"
RESULTS_FILE = OUTPUT_DIR / "triage_results.csv"
BUDGET_FILE = OUTPUT_DIR / "triage_budget_demo.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


ALLOWED_CATEGORIES = {
    "billing",
    "delivery",
    "technical",
    "fraud"
}

MAX_TICKET_CHARS = 500

MAX_RETRIES = 3

# Few-shot is the default. Change to "zero_shot" to
# experiment with the zero-shot prompt.
PROMPT_MODE = "few_shot"

# Demonstration budget, not a limit on the main classification run.
DEMO_TOKEN_BUDGET = 2000

# Fixed random generator makes retry delays reproducible.
random_generator = random.Random(42)


# ==========================================================
# 2. PROMPT TEMPLATES
# ==========================================================

ZERO_SHOT_PROMPT = """
CLASSIFY

You are a customer support ticket classifier.

Classify the customer's issue into exactly one category:

billing
delivery
technical
fraud

Category guidance:

billing:
Payments, invoices, charges, refunds and deductions.

delivery:
Late deliveries, missing orders, riders and delivery problems.

technical:
App crashes, login problems, OTP errors and technical failures.

fraud:
Unauthorized transactions, stolen accounts and suspicious activity.

Return ONLY valid JSON in this format:

{
    "category": "billing",
    "reason": "Brief explanation"
}

Do not include markdown or additional text.
"""


FEW_SHOT_PROMPT = """
CLASSIFY

You are a customer support ticket classifier.

Classify the customer's issue into exactly one category:

billing
delivery
technical
fraud

Category guidance:

billing:
Payments, invoices, charges, refunds and deductions.

delivery:
Late deliveries, missing orders, riders and delivery problems.

technical:
App crashes, login problems, OTP errors and technical failures.

fraud:
Unauthorized transactions, stolen accounts and suspicious activity.

EXAMPLES:

Ticket: I was charged twice for my order.
Output: {"category":"billing","reason":"Duplicate charge"}

Ticket: My delivery has not arrived.
Output: {"category":"delivery","reason":"Late delivery"}

Ticket: The application crashes when I log in.
Output: {"category":"technical","reason":"Application failure"}

Ticket: Someone made an unauthorized transaction.
Output: {"category":"fraud","reason":"Unauthorized transaction"}

Return ONLY valid JSON in this format:

{
    "category": "billing",
    "reason": "Brief explanation"
}

Do not include markdown or additional text.
"""


if PROMPT_MODE == "few_shot":
    SYSTEM_PROMPT = FEW_SHOT_PROMPT
elif PROMPT_MODE == "zero_shot":
    SYSTEM_PROMPT = ZERO_SHOT_PROMPT
else:
    raise ValueError("PROMPT_MODE must be zero_shot or few_shot")


# ==========================================================
# 3. KEYWORD FALLBACK
# ==========================================================

KEYWORDS = {
    "fraud": {
        "unauthorized",
        "fraud",
        "hacked",
        "stolen"
    },

    "technical": {
        "app",
        "crash",
        "login",
        "otp",
        "error",
        "freezes"
    },

    "billing": {
        "charged",
        "invoice",
        "bill",
        "refund",
        "deducted",
        "payment"
    },

    "delivery": {
        "arrived",
        "rider",
        "delivery",
        "delivered",
        "late"
    }
}


def keyword_classification(ticket_text):

    words = set(
        re.findall(
            r"[a-z]+",
            str(ticket_text).lower()
        )
    )

    # Same precedence as the mock LLM's few-shot mode.
    category_order = [
        "fraud",
        "technical",
        "billing",
        "delivery"
    ]

    for category in category_order:

        if KEYWORDS[category] & words:

            return category

    return "technical"


# ==========================================================
# 4. CUSTOMER SEGMENT ROUTING
# ==========================================================

def load_customer_segments():

    if not SEGMENT_FILE.exists():

        raise FileNotFoundError(
            f"Customer segment file not found: {SEGMENT_FILE}"
        )

    segments = pd.read_csv(SEGMENT_FILE)

    required = {
        "customer_id",
        "segment_name",
        "is_premium"
    }

    missing = required - set(segments.columns)

    if missing:

        raise ValueError(
            f"Missing segment columns: {missing}"
        )

    if segments["customer_id"].duplicated().any():

        raise ValueError(
            "Duplicate customer IDs in segment file"
        )

    return segments[
        [
            "customer_id",
            "segment_name",
            "is_premium"
        ]
    ]


def determine_priority(category, segment_name, is_premium):

    segment = str(segment_name).lower()

    # Fraud cases receive the highest priority.
    if category == "fraud":

        return "critical"

    # Premium customers receive high priority.
    if is_premium == 1:

        return "high"

    if "frequent" in segment:

        return "medium"

    if "regular" in segment:

        return "medium"

    return "normal"


def determine_routing_queue(category, priority):

    if category == "fraud":

        return "fraud_specialist"

    if category == "billing":

        return "billing_support"

    if category == "delivery":

        return "delivery_support"

    if category == "technical":

        return "technical_support"

    return "manual_review"


# ==========================================================
# 5. INPUT GUARDRAILS
# ==========================================================

def validate_ticket_input(ticket_text):

    if pd.isna(ticket_text):

        return False, "Empty ticket text"

    text = str(ticket_text)

    if not text.strip():

        return False, "Empty ticket text"

    if len(text) > MAX_TICKET_CHARS:

        return False, "Ticket exceeds maximum input length"

    return True, "Valid input"


# ==========================================================
# 6. CLASSIFICATION RESPONSE VALIDATION
# ==========================================================

def validate_json_response(content):

    if not content:

        return None, "Empty model response"

    try:

        result = json.loads(content)

    except json.JSONDecodeError:

        return None, "Invalid JSON response"

    if not isinstance(result, dict):

        return None, "JSON response must be an object"

    category = result.get("category")

    if category not in ALLOWED_CATEGORIES:

        return None, f"Invalid category: {category}"

    reason = result.get("reason")

    if not isinstance(reason, str) or not reason.strip():

        return None, "Missing or invalid reason"

    return result, None


# ==========================================================
# 7. CLASSIFICATION FUNCTION
# ==========================================================

def classify_ticket(ticket_text):

    valid, validation_reason = validate_ticket_input(ticket_text)

    # Guardrails execute before calling the model.
    if not valid:

        return {
            "category": "unclassified",
            "reason": validation_reason,
            "status": "skipped",
            "classification_method": "none",
            "attempts": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": str(ticket_text)
        }
    ]

    total_prompt_tokens = 0
    total_completion_tokens = 0

    last_error = "Unknown error"

    for attempt in range(1, MAX_RETRIES + 2):

        try:

            response = mock_llm.chat(messages)

            total_prompt_tokens += response.usage.prompt_tokens

            total_completion_tokens += response.usage.completion_tokens

            content = response.choices[0].message.content

            result, error = validate_json_response(content)

            # --------------------------------------------------
            # VALID JSON RESPONSE
            # --------------------------------------------------

            if result is not None:

                return {
                    "category": result["category"],
                    "reason": result["reason"],
                    "status": "success",
                    "classification_method": "json",
                    "attempts": attempt,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": (
                        total_prompt_tokens
                        + total_completion_tokens
                    )
                }

            # --------------------------------------------------
            # NON-JSON RESPONSE: KEYWORD FALLBACK
            # --------------------------------------------------

            if error == "Invalid JSON response":

                fallback_category = keyword_classification(
                    ticket_text
                )

                return {
                    "category": fallback_category,
                    "reason": "Keyword-based fallback after non-JSON response",
                    "status": "success",
                    "classification_method": "keyword_fallback",
                    "attempts": attempt,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": (
                        total_prompt_tokens
                        + total_completion_tokens
                    )
                }

            # --------------------------------------------------
            # INVALID CATEGORY OR INVALID JSON STRUCTURE
            # --------------------------------------------------

            return {
                "category": "unclassified",
                "reason": error,
                "status": "invalid",
                "classification_method": "invalid_response",
                "attempts": attempt,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": (
                    total_prompt_tokens
                    + total_completion_tokens
                )
            }

        except mock_llm.TransientLLMError as error:

            last_error = str(error)

            # Exponential backoff with bounded random jitter.
            if attempt <= MAX_RETRIES:

                base_delay = 0.2 * (2 ** (attempt - 1))

                jitter = random_generator.uniform(
                    0.0,
                    0.1
                )

                delay = base_delay + jitter

                print(
                    f"Transient error on attempt {attempt}. "
                    f"Retrying in {delay:.2f} seconds."
                )

                time.sleep(delay)

        except Exception as error:

            last_error = f"Unexpected error: {error}"

            break

    # All retry attempts exhausted.
    return {
        "category": "unclassified",
        "reason": last_error,
        "status": "failed",
        "classification_method": "failed",
        "attempts": MAX_RETRIES + 1,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": (
            total_prompt_tokens
            + total_completion_tokens
        )
    }


# ==========================================================
# 8. TOKEN BUDGET DEMONSTRATION
# ==========================================================

def create_budget_demo(results_df):

    budget = DEMO_TOKEN_BUDGET

    used_tokens = 0

    records = []

    budget_exhausted = False

    for _, row in results_df.iterrows():

        tokens = int(row["total_tokens"])

        if row["status"] == "skipped":

            decision = "skipped_guardrail"

        elif budget_exhausted:

            decision = "rejected_budget_exhausted"

        elif used_tokens + tokens <= budget:

            used_tokens += tokens

            decision = "processed_within_budget"

        else:

            decision = "rejected_budget_exceeded"

            budget_exhausted = True

        records.append({

            "ticket_id": row["ticket_id"],

            "total_tokens": tokens,

            "token_budget": budget,

            "cumulative_tokens": used_tokens,

            "budget_status": decision

        })

    budget_df = pd.DataFrame(records)

    budget_df.to_csv(
        BUDGET_FILE,
        index=False
    )

    print("\n--- TOKEN BUDGET DEMONSTRATION ---")

    print("Token budget:", budget)

    print("Tokens used:", used_tokens)

    print(
        "Tickets processed within budget:",
        (
            budget_df["budget_status"]
            == "processed_within_budget"
        ).sum()
    )

    print(
        "Tickets rejected:",
        budget_df["budget_status"].isin(
            [
                "rejected_budget_exceeded",
                "rejected_budget_exhausted"
            ]
        ).sum()
    )

    print("Budget demonstration saved:", BUDGET_FILE)


# ==========================================================
# 9. MAIN PIPELINE
# ==========================================================

def main():

    print("=" * 70)

    print("PART C: CUSTOMER SUPPORT TICKET TRIAGE")

    print("=" * 70)

    # ------------------------------------------------------
    # LOAD INPUTS
    # ------------------------------------------------------

    print("\nLoading redacted tickets...")

    df = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False
    )

    print("Total tickets:", len(df))

    if "redacted_text" not in df.columns:

        raise ValueError(
            "Input file must contain redacted_text"
        )

    if "customer_id" not in df.columns:

        raise ValueError(
            "Input file must contain customer_id"
        )

    segments = load_customer_segments()

    # ------------------------------------------------------
    # MERGE CUSTOMER SEGMENTS
    # ------------------------------------------------------

    df = df.merge(
        segments,
        on="customer_id",
        how="left",
        validate="many_to_one"
    )

    df["segment_name"] = df["segment_name"].fillna(
        "Unknown"
    )

    df["is_premium"] = (
        pd.to_numeric(
            df["is_premium"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    # ------------------------------------------------------
    # INPUT GUARDRAIL SUMMARY
    # ------------------------------------------------------

    lengths = df["redacted_text"].astype(str).str.len()

    empty_count = (
        df["redacted_text"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    long_count = (
        lengths > MAX_TICKET_CHARS
    ).sum()

    print("\n--- INPUT GUARDRAILS ---")

    print("Maximum ticket length:", MAX_TICKET_CHARS)

    print("Empty tickets:", empty_count)

    print("Excessively long tickets:", long_count)

    print(
        "Eligible tickets:",
        len(df) - empty_count - long_count
    )

    # ------------------------------------------------------
    # RESET DETERMINISTIC MOCK
    # ------------------------------------------------------

    mock_llm.reset()

    random_generator.seed(42)

    results = []

    print("\n--- CLASSIFICATION STARTED ---")

    # ------------------------------------------------------
    # CLASSIFY EACH TICKET
    # ------------------------------------------------------

    for index, row in df.iterrows():

        result = classify_ticket(
            row["redacted_text"]
        )

        results.append(result)

        if (index + 1) % 25 == 0:

            print(
                f"Processed {index + 1}/{len(df)} tickets"
            )

    results_df = pd.DataFrame(results)

    # ------------------------------------------------------
    # COMBINE CLASSIFICATION RESULTS
    # ------------------------------------------------------

    output_df = pd.concat(
        [
            df.drop(
                columns=["original_text"],
                errors="ignore"
            ).reset_index(drop=True),

            results_df.reset_index(drop=True)
        ],
        axis=1
    )

    # ------------------------------------------------------
    # CUSTOMER PRIORITY AND ROUTING
    # ------------------------------------------------------

    output_df["priority"] = output_df.apply(

        lambda row: determine_priority(
            row["category"],
            row["segment_name"],
            row["is_premium"]
        ),

        axis=1
    )

    output_df["routing_queue"] = output_df.apply(

        lambda row: determine_routing_queue(
            row["category"],
            row["priority"]
        ),

        axis=1
    )

    # ------------------------------------------------------
    # SAVE COMPATIBILITY OUTPUT
    # ------------------------------------------------------

    output_df.to_csv(
        TRIAGED_FILE,
        index=False
    )

    # ------------------------------------------------------
    # SAVE REQUIRED EVALUATION OUTPUT
    # ------------------------------------------------------

    output_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    # ------------------------------------------------------
    # CLASSIFICATION SUMMARY
    # ------------------------------------------------------

    print("\n" + "=" * 70)

    print("TRIAGE SUMMARY")

    print("=" * 70)

    print("\nTotal tickets:", len(output_df))

    print("\nCategory distribution:")

    print(
        output_df["category"].value_counts(
            dropna=False
        )
    )

    print("\nProcessing status:")

    print(
        output_df["status"].value_counts(
            dropna=False
        )
    )

    print("\nClassification methods:")

    print(
        output_df["classification_method"].value_counts(
            dropna=False
        )
    )

    print("\nPriority distribution:")

    print(
        output_df["priority"].value_counts(
            dropna=False
        )
    )

    print("\nRouting queue distribution:")

    print(
        output_df["routing_queue"].value_counts(
            dropna=False
        )
    )

    # ------------------------------------------------------
    # TOKEN ACCOUNTING
    # ------------------------------------------------------

    print("\n--- TOKEN ACCOUNTING ---")

    print(
        "Total attempts:",
        output_df["attempts"].sum()
    )

    print(
        "Prompt tokens:",
        output_df["prompt_tokens"].sum()
    )

    print(
        "Completion tokens:",
        output_df["completion_tokens"].sum()
    )

    print(
        "Total tokens:",
        output_df["total_tokens"].sum()
    )

    # ------------------------------------------------------
    # CLASSIFICATION METHOD VALIDATION
    # ------------------------------------------------------

    print("\n--- CLASSIFICATION METHOD VALIDATION ---")

    method_counts = (
        output_df["classification_method"]
        .value_counts()
    )

    print(
        "JSON:",
        method_counts.get("json", 0)
    )

    print(
        "Keyword fallback:",
        method_counts.get("keyword_fallback", 0)
    )

    print(
        "Invalid responses:",
        method_counts.get("invalid_response", 0)
    )

    # ------------------------------------------------------
    # SAVE TOKEN BUDGET DEMONSTRATION
    # ------------------------------------------------------

    create_budget_demo(output_df)

    # ------------------------------------------------------
    # FINAL VALIDATIONS
    # ------------------------------------------------------

    print("\n" + "=" * 70)

    print("FINAL TRIAGE VALIDATIONS")

    print("=" * 70)

    assert len(output_df) == 164

    assert output_df["ticket_id"].is_unique

    assert (
        output_df["total_tokens"]
        ==
        output_df["prompt_tokens"]
        + output_df["completion_tokens"]
    ).all()

    assert "original_text" not in output_df.columns

    assert empty_count == 4

    assert long_count == 4

    assert (
        output_df["status"] == "skipped"
    ).sum() == 8

    assert (
        output_df["attempts"][
            output_df["status"] == "skipped"
        ] == 0
    ).all()

    assert (
        output_df["classification_method"]
        .value_counts()
        .get("json", 0)
        +
        output_df["classification_method"]
        .value_counts()
        .get("keyword_fallback", 0)
        +
        output_df["classification_method"]
        .value_counts()
        .get("invalid_response", 0)
    ) == 156

    print("PASS: All 164 tickets retained.")

    print("PASS: Four empty tickets skipped.")

    print("PASS: Four long tickets skipped.")

    print("PASS: Eight tickets skipped before model calls.")

    print("PASS: Token accounting is correct.")

    print("PASS: Original text excluded from outputs.")

    print("PASS: Classification methods account for all eligible tickets.")

    print("\nSaved files:")

    print(TRIAGED_FILE)

    print(RESULTS_FILE)

    print(BUDGET_FILE)

    print("\n" + "=" * 70)

    print("PART C TRIAGE PIPELINE COMPLETED!")

    print("=" * 70)


if __name__ == "__main__":

    main()