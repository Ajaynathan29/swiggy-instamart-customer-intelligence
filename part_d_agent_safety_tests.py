"""part_d_agent_safety_tests.py

Tests the refund agent's safety limits:
1. Token limit
2. Timeout
3. Max iterations (S7 already verified in the main agent run)
"""

import pandas as pd
import mock_llm

from part_d_refund_agent import run_agent


SCENARIO_FILE = "data/raw/agent_scenarios.csv"


def get_s1_message():
    """Load the user message for scenario S1."""

    df = pd.read_csv(SCENARIO_FILE)

    row = df[df["scenario_id"] == "S1"]

    if row.empty:
        raise ValueError("Scenario S1 not found.")

    possible_columns = [
        "user_message",
        "message",
        "user_prompt",
        "prompt",
        "request",
        "scenario",
    ]

    for column in possible_columns:
        if column in df.columns:
            return str(row.iloc[0][column])

    raise ValueError(
        "Could not find the user message column. "
        f"Available columns: {list(df.columns)}"
    )


def test_token_limit(user_message):

    print("\n" + "=" * 50)
    print("TEST 1: TOKEN LIMIT")
    print("=" * 50)

    mock_llm.reset()

    result = run_agent(
        user_message=user_message,
        scenario_id="S1_TOKEN_LIMIT",
        token_limit=200,
    )

    print("Status:", result["status"])
    print("Model calls:", result["llm_calls"])
    print("Tokens used:", result["tokens_used"])
    print("Final answer:", result["final_answer"])
    print("Tools executed:", len(result["steps"]))

    assert result["status"] == "FALLBACK_TOKEN_LIMIT"

    # A tool may have executed in an earlier iteration.
    # The tool requested by the call exceeding the budget
    # must NOT be executed.
    assert len(result["steps"]) == 1

    print("Previously executed tool:", result["steps"][0])

    print(
        "\nPASS: Agent stopped after exceeding the token limit "
        "without executing the next requested tool."
    )


def test_timeout(user_message):

    print("\n" + "=" * 50)
    print("TEST 2: TIMEOUT")
    print("=" * 50)

    mock_llm.reset()

    result = run_agent(
        user_message=user_message,
        scenario_id="S1_TIMEOUT",
        time_limit_seconds=0,
    )

    print("Status:", result["status"])
    print("Model calls:", result["llm_calls"])
    print("Tokens used:", result["tokens_used"])
    print("Final answer:", result["final_answer"])
    print("Tools executed:", len(result["steps"]))

    assert result["status"] == "FALLBACK_TIMEOUT"
    assert result["llm_calls"] == 0
    assert len(result["steps"]) == 0

    print(
        "\nPASS: Agent stopped before making an LLM call "
        "or executing a tool."
    )


if __name__ == "__main__":

    s1_message = get_s1_message()

    test_token_limit(s1_message)

    test_timeout(s1_message)

    print("\n" + "=" * 50)
    print("ALL SAFETY TESTS PASSED")
    print("=" * 50)

    print("S7 max-iteration test: already verified.")
    print("Expected status: FALLBACK_MAX_ITERATIONS")