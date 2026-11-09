"""Demonstrate checkpointing and resuming the S1 refund agent."""

import csv
import uuid

import mock_llm

from part_d_refund_agent import (
    run_agent,
    resume_agent,
    SimulatedAgentCrash
)


# --------------------------------------------------
# LOAD S1 SCENARIO
# --------------------------------------------------

SCENARIO_FILE = "data/raw/agent_scenarios.csv"


def get_s1_message():

    with open(
        SCENARIO_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        scenarios = list(csv.DictReader(file))

    for scenario in scenarios:

        if scenario["scenario_id"] == "S1":

            return scenario["message"]

    raise ValueError("S1 scenario not found.")


# --------------------------------------------------
# CRASH AND RESUME TEST
# --------------------------------------------------

def test_checkpoint_resume():

    print("\n" + "=" * 65)
    print("CHECKPOINT AND RESUME TEST")
    print("=" * 65)

    user_message = get_s1_message()

    run_id = str(uuid.uuid4())

    print("\nRun ID:", run_id)

    # ----------------------------------------------
    # PHASE 1: INITIAL EXECUTION
    # ----------------------------------------------

    print("\nPHASE 1: INITIAL EXECUTION")

    mock_llm.reset()

    try:

        run_agent(

            user_message=user_message,

            scenario_id="S1",

            run_id=run_id,

            crash_after_tools=2

        )

    except SimulatedAgentCrash as error:

        print("\nSIMULATED CRASH:", error)

    else:

        raise AssertionError(
            "Expected simulated crash did not occur."
        )

    # ----------------------------------------------
    # PHASE 2: RESUME
    # ----------------------------------------------

    print("\nPHASE 2: RESUMING FROM CHECKPOINT")

    result = resume_agent(run_id)

    # ----------------------------------------------
    # RESULTS
    # ----------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL RESUME RESULTS")
    print("=" * 65)

    print("Run ID:", result["run_id"])

    print("Status:", result["status"])

    print("Final answer:", result["final_answer"])

    print("Model calls:", result["llm_calls"])

    print("Tokens used:", result["tokens_used"])

    print("\nTools executed in order:")

    for index, tool_name in enumerate(
        result["tools_executed"],
        start=1
    ):

        print(f"{index}. {tool_name}")

    print("\nTotal tool executions:", len(result["steps"]))

    # ----------------------------------------------
    # ACCEPTANCE CHECKS
    # ----------------------------------------------

    expected_tools = [

        "get_order",

        "search_policy",

        "calculate_refund",

        "issue_refund"

    ]

    assert result["status"] == "REFUND"

    assert "128.25" in result["final_answer"]

    assert result["tools_executed"] == expected_tools

    assert len(result["steps"]) == 4

    assert result["llm_calls"] == 5

    print("\n" + "=" * 65)
    print("CHECKPOINT AND RESUME TEST PASSED")
    print("=" * 65)

    print("Correct refund: INR 128.25")

    print("Exactly four tool executions.")

    print("No previously executed tool was repeated.")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    test_checkpoint_resume()