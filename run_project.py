import csv
import json
import shutil
import subprocess
import sys

from datetime import datetime
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

ARCHIVE_DIR = BASE_DIR / "run_archives"

PYTHON = sys.executable

passed = []
failed = []


# ============================================================
# DISPLAY
# ============================================================

def heading(title):

    print("\n" + "=" * 75)
    print(title.center(75))
    print("=" * 75)


def record_pass(name):

    passed.append(name)
    print(f"\n[PASS] {name}")


def record_fail(name, reason):

    failed.append((name, str(reason)))
    print(f"\n[FAIL] {name}: {reason}")


# ============================================================
# ARCHIVE PREVIOUS PART D RESULTS
# ============================================================

def archive_previous_agent_results():

    files = [
        OUTPUT_DIR / "agent_runs.jsonl",
        OUTPUT_DIR / "refund_ledger.jsonl"
    ]

    existing = [
        path for path in files
        if path.exists()
    ]

    if not existing:
        print("No previous Part D logs to archive.")
        return

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    archive_path = ARCHIVE_DIR / timestamp

    archive_path.mkdir(
        parents=True,
        exist_ok=True
    )

    for path in existing:

        shutil.copy2(
            path,
            archive_path / path.name
        )

        path.unlink()

        print("Archived:", path.name)

    print("Archive location:", archive_path)


# ============================================================
# RUN PYTHON SCRIPT
# ============================================================

def run_script(script_name):

    script_path = BASE_DIR / script_name

    heading(f"RUNNING {script_name}")

    if not script_path.exists():

        record_fail(
            script_name,
            "Script does not exist."
        )

        return False

    try:

        result = subprocess.run(

            [PYTHON, str(script_path)],

            cwd=BASE_DIR,

            text=True,

            check=False

        )

        if result.returncode != 0:

            record_fail(
                script_name,
                f"Exit code {result.returncode}"
            )

            return False

        record_pass(script_name)

        return True

    except Exception as error:

        record_fail(
            script_name,
            error
        )

        return False


# ============================================================
# EXECUTION PIPELINE
# ============================================================

def execute_pipeline():

    scripts = [

        # ---------------- PART A ----------------

        "part_a_data_inspection.py",
        "part_a_orders_analysis.py",
        "part_a_tickets_analysis.py",

        "part_a_clean_customers.py",
        "part_a_clean_orders.py",

        "part_a_create_database.py",


        # ---------------- PART B ----------------

        "part_b_inspect_data.py",
        "part_b_cohort_analysis.py",
        "part_b_target_analysis.py",

        "part_b_build_features.py",
        "part_b_validate_features.py",

        "part_b_segmentation.py",
        "part_b_segment_analysis.py",

        "part_b_train_models.py",
        "part_b_threshold_optimization.py",

        "part_b_feature_importance.py",
        "part_b_final_evaluation.py",


        # ---------------- PART C ----------------

        "part_c_pii_redaction.py",
        "part_c_triage.py",

        "part_c_validate_triage.py",
        "part_c_evaluate_triage.py",


        # ---------------- PART D ----------------

        "part_d_policy_rag.py",
        "part_d_evaluate_rag.py",

        "part_d_refund_agent.py",

        "part_d_agent_safety_tests.py",
        "part_d_agent_resume_test.py"

    ]

    for script in scripts:

        if not run_script(script):

            print(
                "\nPipeline stopped because "
                "a required script failed."
            )

            return False

    return True


# ============================================================
# ACCEPTANCE CHECKS
# ============================================================

def check_file(relative_path):

    path = BASE_DIR / relative_path

    if not path.exists():

        record_fail(
            f"File exists: {relative_path}",
            "Required file missing."
        )

        return False

    if path.stat().st_size == 0:

        record_fail(
            f"File exists: {relative_path}",
            "File is empty."
        )

        return False

    record_pass(
        f"File exists: {relative_path}"
    )

    return True


def check_csv_rows(relative_path, expected_rows):

    path = BASE_DIR / relative_path

    try:

        with open(
            path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            rows = list(csv.DictReader(file))

        actual = len(rows)

        if actual != expected_rows:

            record_fail(
                f"{relative_path} row count",
                f"Expected {expected_rows}, got {actual}"
            )

            return False

        record_pass(
            f"{relative_path}: {actual} rows"
        )

        return True

    except Exception as error:

        record_fail(
            f"{relative_path} row count",
            error
        )

        return False


def check_agent_runs():

    path = OUTPUT_DIR / "agent_runs.jsonl"

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        if len(records) != 7:

            record_fail(
                "Part D scenario count",
                f"Expected 7, got {len(records)}"
            )

            return

        expected_statuses = {

            "S1": "REFUND",
            "S2": "REFUND",
            "S3": "ESCALATE",
            "S4": "DENIED",
            "S5": "DENIED",
            "S6": "NOT_FOUND",
            "S7": "FALLBACK_MAX_ITERATIONS"

        }

        for record in records:

            scenario = record["scenario_id"]

            expected = expected_statuses.get(
                scenario
            )

            actual = record["status"]

            if actual != expected:

                record_fail(
                    f"Scenario {scenario}",
                    f"Expected {expected}, got {actual}"
                )

                return

        record_pass(
            "Part D: All seven scenario outcomes"
        )

    except Exception as error:

        record_fail(
            "Part D scenario validation",
            error
        )


def check_refund_ledger():

    path = OUTPUT_DIR / "refund_ledger.jsonl"

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        if len(records) != 4:

            record_fail(
                "Refund ledger",
                f"Expected 4 entries (3 scenarios + 1 resume test), got {len(records)}"
            )

            return

        record_pass(
            "Refund ledger: 4 expected transactions"
        )

    except Exception as error:

        record_fail(
            "Refund ledger validation",
            error
        )


def run_acceptance_checks():

    heading("FINAL ACCEPTANCE TESTS")

    required_files = [

        "data/clean/customers.csv",
        "data/clean/orders.csv",
        "data/clean/products.csv",

        "data/instamart.db",
        "data/processed/instamart.db",

        "data/processed/customer_features.csv",
        "data/processed/customer_segments.csv",

        "data/processed/tickets_redacted.csv",
        "data/processed/tickets_triaged.csv",

        "outputs/customer_segments.csv",
        "outputs/lapse_model.joblib",

        "outputs/retention_list.csv",

        "outputs/triage_results.csv",
        "outputs/triage_metrics.csv",

        "outputs/agent_runs.jsonl",
        "outputs/refund_ledger.jsonl"

    ]

    for file_path in required_files:

        check_file(file_path)


    # Dataset row-count checks

    check_csv_rows(
        "data/clean/customers.csv",
        1200
    )

    check_csv_rows(
        "data/clean/orders.csv",
        12309
    )

    check_csv_rows(
        "data/processed/tickets_redacted.csv",
        164
    )

    check_csv_rows(
        "outputs/triage_results.csv",
        164
    )

    # Part D checks

    check_agent_runs()

    check_refund_ledger()


# ============================================================
# FINAL REPORT
# ============================================================

def final_report():

    heading("PROJECT EXECUTION SUMMARY")

    total = len(passed) + len(failed)

    print("Total checks :", total)
    print("Passed       :", len(passed))
    print("Failed       :", len(failed))

    print("\nPASSED CHECKS")

    for item in passed:

        print("  [PASS]", item)

    if failed:

        print("\nFAILED CHECKS")

        for name, reason in failed:

            print(
                f"  [FAIL] {name}: {reason}"
            )

    print("\n" + "=" * 75)

    if failed:

        print("PROJECT RESULT: FAILED")

        print("Fix the failures before submission.")

    else:

        print("PROJECT RESULT: ALL CHECKS PASSED")

        print("Parts A, B, C and D completed.")

    print("=" * 75)

    return len(failed) == 0


# ============================================================
# MAIN
# ============================================================

def main():

    heading(
        "INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT"
    )

    print("Python:", PYTHON)
    print("Project:", BASE_DIR)

    # Check essential raw inputs before execution.

    required_inputs = [

        "data/raw/customers.csv",
        "data/raw/orders.csv",
        "data/raw/products.csv",
        "data/raw/tickets.csv",
        "data/raw/agent_scenarios.csv",
        "data/raw/policy_eval.csv"

    ]

    missing = [

        item for item in required_inputs
        if not (BASE_DIR / item).exists()

    ]

    if missing:

        for item in missing:

            record_fail(
                "Required input",
                f"Missing {item}"
            )

        final_report()

        return 1

    # Archive old agent records before a fresh run.

    try:

        archive_previous_agent_results()

    except Exception as error:

        record_fail(
            "Part D log archival",
            error
        )

        final_report()

        return 1

    # Execute all four parts.

    if not execute_pipeline():

        final_report()

        return 1

    # Verify outputs.

    run_acceptance_checks()

    # Final result.

    return 0 if final_report() else 1


if __name__ == "__main__":

    sys.exit(main())