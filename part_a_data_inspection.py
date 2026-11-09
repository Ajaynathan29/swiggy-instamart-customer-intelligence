from pathlib import Path

import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"


# --------------------------------------------------
# EXPECTED REFERENCE VALUES FROM THE TUTOR BRIEF
# --------------------------------------------------

EXPECTED_AUDIT = {
    "Raw order rows": 13084,
    "Exact duplicate order rows": 119,
    "Pending order rows": 656,
    "Distinct order IDs": 12309,
    "Non-numeric order amounts": 120,
    "Negative delivery times": 106,
    "Blank delivery times": 2541,
    "Blank customer ages": 61,
    "Ages outside 10-100": 6,
    "Distinct raw city spellings": 24,
    "Distinct non-blank raw gender spellings": 10,
    "Blank customer genders": 30,
}


# --------------------------------------------------
# CSV LOADING
# --------------------------------------------------

def load_raw_csv(filename):
    """
    Load a raw CSV file.

    tickets.csv must use keep_default_na=False so empty
    ticket text remains an empty string rather than NaN.
    """

    file_path = RAW_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required raw file not found: {file_path}"
        )

    if filename == "tickets.csv":
        return pd.read_csv(
            file_path,
            keep_default_na=False
        )

    return pd.read_csv(file_path)


# --------------------------------------------------
# GENERIC DATA INSPECTION
# --------------------------------------------------

def inspect_dataframe(name, df):

    print("\n" + "=" * 75)
    print(f"DATA INSPECTION: {name}")
    print("=" * 75)

    # Shape
    print("\n--- SHAPE ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Data types
    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    # Missing values
    print("\n--- MISSING VALUE REPORT ---")

    missing_count = df.isna().sum()

    missing_percentage = (
        missing_count / len(df) * 100
        if len(df) > 0
        else missing_count
    )

    missing_report = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage": missing_percentage.round(2)
    })

    print(missing_report)

    # Blank strings
    print("\n--- BLANK STRING REPORT ---")

    blank_counts = {}

    for column in df.columns:

        if (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ):

            blank_counts[column] = int(
                df[column]
                .astype("string")
                .str.strip()
                .eq("")
                .fillna(False)
                .sum()
            )

        else:
            blank_counts[column] = 0

    print(pd.Series(
        blank_counts,
        name="Blank String Count"
    ))

    # Exact duplicate rows
    print("\n--- DUPLICATE ROW ANALYSIS ---")

    duplicate_rows = int(df.duplicated().sum())

    print(f"Exact duplicate rows: {duplicate_rows}")

    # Identifier checks
    print("\n--- IDENTIFIER ANALYSIS ---")

    possible_id_columns = [
        column for column in df.columns
        if column.lower().endswith("_id")
    ]

    if not possible_id_columns:
        print("No ID columns found.")

    for column in possible_id_columns:

        non_missing = df[column].dropna()

        print(f"\nColumn: {column}")
        print(f"Non-missing IDs: {len(non_missing)}")
        print(f"Unique IDs: {non_missing.nunique()}")
        print(
            "Repeated ID values:",
            int(non_missing.duplicated().sum())
        )


# --------------------------------------------------
# REFERENCE AUDIT HELPER
# --------------------------------------------------

def count_blank_values(series):
    """
    Count both actual missing values and blank/whitespace strings.
    """

    text_values = series.astype("string")

    blank_mask = (
        text_values.isna()
        | text_values.str.strip().eq("")
    )

    return int(blank_mask.sum())


# --------------------------------------------------
# CALCULATE REFERENCE AUDIT
# --------------------------------------------------

def calculate_reference_audit(customers, orders):
    """
    Calculate the exact raw-data figures required by the tutor.

    Raw city spellings are counted without normalizing them.
    Missing and non-numeric order amounts are both counted
    as values that cannot be converted to numeric.
    """

    results = {}

    # --------------------------------------------------
    # ORDERS AUDIT
    # --------------------------------------------------

    results["Raw order rows"] = len(orders)

    results["Exact duplicate order rows"] = int(
        orders.duplicated().sum()
    )

    # Pending order records
    if "status" in orders.columns:

        status_values = (
            orders["status"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        results["Pending order rows"] = int(
            status_values.eq("pending").sum()
        )

    else:
        results["Pending order rows"] = 0

    # Distinct order IDs
    if "order_id" in orders.columns:

        order_ids = (
            orders["order_id"]
            .astype("string")
            .str.strip()
        )

        valid_order_ids = order_ids[
            order_ids.notna()
            & order_ids.ne("")
        ]

        results["Distinct order IDs"] = int(
            valid_order_ids.nunique()
        )

    else:
        results["Distinct order IDs"] = 0

    # --------------------------------------------------
    # NON-NUMERIC ORDER AMOUNTS
    # --------------------------------------------------

    if "order_amount" in orders.columns:

        numeric_amounts = pd.to_numeric(
            orders["order_amount"],
            errors="coerce"
        )

        # Count all values that cannot be converted to numeric,
        # including existing missing values.
        results["Non-numeric order amounts"] = int(
            numeric_amounts.isna().sum()
        )

    else:
        results["Non-numeric order amounts"] = 0

    # --------------------------------------------------
    # NEGATIVE DELIVERY TIMES
    # --------------------------------------------------

    if "delivery_minutes" in orders.columns:

        delivery_values = pd.to_numeric(
            orders["delivery_minutes"],
            errors="coerce"
        )

        results["Negative delivery times"] = int(
            delivery_values.lt(0).sum()
        )

    else:
        results["Negative delivery times"] = 0

    # --------------------------------------------------
    # BLANK DELIVERY TIMES
    # --------------------------------------------------

    if "delivery_minutes" in orders.columns:

        results["Blank delivery times"] = count_blank_values(
            orders["delivery_minutes"]
        )

    else:
        results["Blank delivery times"] = 0

    # --------------------------------------------------
    # CUSTOMERS AUDIT
    # --------------------------------------------------

    # Blank ages
    if "age" in customers.columns:

        results["Blank customer ages"] = count_blank_values(
            customers["age"]
        )

        age_values = pd.to_numeric(
            customers["age"],
            errors="coerce"
        )

        results["Ages outside 10-100"] = int(
            (
                age_values.notna()
                & (
                    age_values.lt(10)
                    | age_values.gt(100)
                )
            ).sum()
        )

    else:
        results["Blank customer ages"] = 0
        results["Ages outside 10-100"] = 0

    # --------------------------------------------------
    # RAW CITY SPELLING COUNT
    # --------------------------------------------------

    if "city" in customers.columns:

        # Keep original raw spellings.
        # Do not strip or title-case before nunique().
        city_values = customers["city"].astype("string")

        # Exclude missing and whitespace-only values.
        # Original nonblank values remain unchanged for nunique().
        non_blank_cities = city_values[
            city_values.notna()
            & city_values.str.strip().ne("")
        ]

        results["Distinct raw city spellings"] = int(
            non_blank_cities.nunique()
        )

    else:
        results["Distinct raw city spellings"] = 0

    # --------------------------------------------------
    # RAW GENDER SPELLING COUNT
    # --------------------------------------------------

    if "gender" in customers.columns:

        gender_values = customers["gender"].astype("string")

        non_blank_genders = gender_values[
            gender_values.notna()
            & gender_values.str.strip().ne("")
        ]

        results[
            "Distinct non-blank raw gender spellings"
        ] = int(non_blank_genders.nunique())

        results["Blank customer genders"] = count_blank_values(
            customers["gender"]
        )

    else:
        results["Distinct non-blank raw gender spellings"] = 0
        results["Blank customer genders"] = 0

    return results


# --------------------------------------------------
# REFERENCE CHECKPOINT VALIDATION
# --------------------------------------------------

def validate_reference_audit(results):

    print("\n" + "=" * 75)
    print("REFERENCE CHECKPOINT VALIDATION")
    print("=" * 75)

    passed = 0
    failed = 0

    for check_name, expected_value in EXPECTED_AUDIT.items():

        actual_value = results.get(check_name)

        if actual_value == expected_value:

            print(
                f"PASS: {check_name} | "
                f"Expected={expected_value} | "
                f"Actual={actual_value}"
            )

            passed += 1

        else:

            print(
                f"FAIL: {check_name} | "
                f"Expected={expected_value} | "
                f"Actual={actual_value}"
            )

            failed += 1

    print("\n" + "-" * 75)
    print(f"Passed checks: {passed}")
    print(f"Failed checks: {failed}")
    print("-" * 75)

    if failed == 0:

        print("\nALL REFERENCE AUDIT CHECKPOINTS PASSED!")

    else:

        print(
            "\nSome reference checkpoints did not match. "
            "Do not modify the raw CSV files to force a match."
        )

    return failed == 0


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("=" * 75)
    print("INSTAMART CUSTOMER INTELLIGENCE")
    print("PART A - RAW DATA INSPECTION")
    print("=" * 75)

    filenames = [
        "customers.csv",
        "orders.csv",
        "products.csv",
        "tickets.csv",
        "policy_eval.csv",
        "agent_scenarios.csv",
    ]

    raw_data = {}

    for filename in filenames:

        df = load_raw_csv(filename)

        raw_data[filename] = df

        inspect_dataframe(filename, df)

    # --------------------------------------------------
    # TUTOR'S REQUIRED RAW DATA AUDIT FIGURES
    # --------------------------------------------------

    customers = raw_data["customers.csv"]
    orders = raw_data["orders.csv"]

    audit_results = calculate_reference_audit(
        customers,
        orders
    )

    print("\n" + "=" * 75)
    print("TUTOR'S REQUIRED RAW DATA AUDIT FIGURES")
    print("=" * 75)

    for check_name, actual_value in audit_results.items():

        print(f"{check_name}: {actual_value}")

    # --------------------------------------------------
    # VALIDATE REFERENCE VALUES
    # --------------------------------------------------

    all_passed = validate_reference_audit(
        audit_results
    )

    print("\n" + "=" * 75)
    print("RAW DATA INSPECTION COMPLETED")
    print("=" * 75)

    if all_passed:
        print("Result: PASS")
    else:
        print("Result: REVIEW REQUIRED")


if __name__ == "__main__":
    main()