import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "raw" / "tickets.csv"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

# Preserve empty ticket text as empty strings.
tickets = pd.read_csv(
    INPUT_PATH,
    keep_default_na=False
)


print("=" * 70)
print("RAW TICKETS DATA QUALITY ANALYSIS")
print("=" * 70)


# --------------------------------------------------
# 3. DATASET OVERVIEW
# --------------------------------------------------

print("\n--- DATASET OVERVIEW ---")

print("Rows:", tickets.shape[0])
print("Columns:", tickets.shape[1])

print("\nColumn names:")
print(tickets.columns.tolist())

print("\nData types:")
print(tickets.dtypes)


# --------------------------------------------------
# 4. MISSING VALUE ANALYSIS
# --------------------------------------------------

print("\n--- MISSING VALUE ANALYSIS ---")

missing_report = pd.DataFrame({
    "Missing Count": tickets.isna().sum(),
    "Missing Percentage": (
        tickets.isna().mean() * 100
    ).round(2)
})

print(missing_report)


# --------------------------------------------------
# 5. EMPTY STRING ANALYSIS
# --------------------------------------------------

print("\n--- EMPTY STRING ANALYSIS ---")

for column in tickets.columns:

    if tickets[column].dtype == "object":

        empty_count = (
            tickets[column]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        print(f"{column}: {empty_count}")


# --------------------------------------------------
# 6. DATASET PREVIEW
# --------------------------------------------------

print("\n--- DATASET PREVIEW ---")

print(
    tickets.head().to_string(index=False)
)


# --------------------------------------------------
# 7. DUPLICATE ANALYSIS
# --------------------------------------------------

print("\n--- DUPLICATE ANALYSIS ---")

print(
    "Exact duplicate rows:",
    tickets.duplicated().sum()
)

if "ticket_id" in tickets.columns:

    print(
        "Unique ticket IDs:",
        tickets["ticket_id"].nunique()
    )

    print(
        "Duplicate ticket IDs:",
        tickets["ticket_id"].duplicated().sum()
    )


# --------------------------------------------------
# 8. TICKET CATEGORY ANALYSIS
# --------------------------------------------------

print("\n--- TICKET CATEGORY DISTRIBUTION ---")

if "true_category" in tickets.columns:

    print(
        tickets["true_category"]
        .value_counts(dropna=False)
    )


# --------------------------------------------------
# 9. TICKET TEXT ANALYSIS
# --------------------------------------------------

print("\n--- TICKET TEXT ANALYSIS ---")

if "text" in tickets.columns:

    text_lengths = (
        tickets["text"]
        .astype(str)
        .str.len()
    )

    print(
        "Empty ticket texts:",
        (text_lengths == 0).sum()
    )

    print(
        "Whitespace-only ticket texts:",
        tickets["text"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        "Shortest ticket text:",
        text_lengths.min()
    )

    print(
        "Longest ticket text:",
        text_lengths.max()
    )

    print(
        "Average ticket text length:",
        round(text_lengths.mean(), 2)
    )


# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("RAW TICKETS ANALYSIS COMPLETED")
print("=" * 70)
