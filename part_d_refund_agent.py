import csv
import json
import time
import uuid

from pathlib import Path

import mock_llm

from part_d_refund_tools import TOOLS_SPEC, TOOL_MAP


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

SCENARIO_FILE = BASE_DIR / "data" / "raw" / "agent_scenarios.csv"

OUTPUT_DIR = BASE_DIR / "outputs"

STATE_DIR = BASE_DIR / "state"

AGENT_RUNS_FILE = OUTPUT_DIR / "agent_runs.jsonl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# CUSTOM CRASH EXCEPTION
# --------------------------------------------------

class SimulatedAgentCrash(Exception):
    """Used to demonstrate checkpoint recovery."""
    pass


# --------------------------------------------------
# FALLBACK RESPONSES
# --------------------------------------------------

def fallback_response(status):

    messages = {

        "FALLBACK_MAX_ITERATIONS":
            "I could not complete this request within the allowed agent iterations. A human support agent must review it.",

        "FALLBACK_TOKEN_LIMIT":
            "I could not complete this request within the token budget. A human support agent must review it.",

        "FALLBACK_TIMEOUT":
            "I could not complete this request within the time limit. A human support agent must review it."

    }

    return messages[status]


# --------------------------------------------------
# CHECKPOINT PATH
# --------------------------------------------------

def get_checkpoint_path(run_id):

    return STATE_DIR / f"{run_id}.json"


# --------------------------------------------------
# SAVE CHECKPOINT
# --------------------------------------------------

def save_checkpoint(
    run_id,
    scenario_id,
    messages,
    tokens_used,
    llm_calls,
    steps,
    tools_executed
):

    checkpoint = {

        "run_id": run_id,

        "scenario_id": scenario_id,

        "messages": messages,

        "tokens_used": tokens_used,

        "llm_calls": llm_calls,

        "step_number": len(steps),

        "tools_executed": tools_executed,

        "steps": steps

    }

    checkpoint_path = get_checkpoint_path(run_id)

    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=4),
        encoding="utf-8"
    )

    return checkpoint_path


# --------------------------------------------------
# INTERNAL AGENT LOOP
# --------------------------------------------------

def _continue_agent(
    run_id,
    scenario_id,
    messages,
    tokens_used,
    llm_calls,
    steps,
    tools_executed,
    max_iterations,
    token_limit,
    time_limit_seconds,
    crash_after_tools=None
):

    start_time = time.monotonic()

    # --------------------------------------------------
    # REACT LOOP
    # --------------------------------------------------

    while llm_calls < max_iterations:

        # ----------------------------------------------
        # TIME LIMIT CHECK
        # ----------------------------------------------

        elapsed = time.monotonic() - start_time

        if elapsed >= time_limit_seconds:

            return {

                "run_id": run_id,

                "status": "FALLBACK_TIMEOUT",

                "final_answer": fallback_response(
                    "FALLBACK_TIMEOUT"
                ),

                "steps": steps,

                "tools_executed": tools_executed,

                "llm_calls": llm_calls,

                "tokens_used": tokens_used

            }

        # ----------------------------------------------
        # MODEL CALL
        # ----------------------------------------------

        response = mock_llm.chat(
            messages,
            tools=TOOLS_SPEC
        )

        llm_calls += 1

        call_tokens = response.usage.total_tokens

        tokens_used += call_tokens

        assistant_message = response.choices[0].message

        tool_calls = assistant_message.tool_calls

        # ----------------------------------------------
        # TOKEN LIMIT CHECK
        # ----------------------------------------------

        if tokens_used > token_limit:

            return {

                "run_id": run_id,

                "status": "FALLBACK_TOKEN_LIMIT",

                "final_answer": fallback_response(
                    "FALLBACK_TOKEN_LIMIT"
                ),

                "steps": steps,

                "tools_executed": tools_executed,

                "llm_calls": llm_calls,

                "tokens_used": tokens_used

            }

        # ----------------------------------------------
        # FINAL ANSWER
        # ----------------------------------------------

        if not tool_calls:

            final_answer = assistant_message.content or ""

            status = final_answer.split(":")[0].strip()

            return {

                "run_id": run_id,

                "status": status,

                "final_answer": final_answer,

                "steps": steps,

                "tools_executed": tools_executed,

                "llm_calls": llm_calls,

                "tokens_used": tokens_used

            }

        # ----------------------------------------------
        # FORMAT TOOL CALLS
        # ----------------------------------------------

        formatted_tool_calls = []

        for tool_call in tool_calls:

            arguments = tool_call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            formatted_tool_calls.append({

                "id": tool_call.id,

                "name": tool_call.name,

                "arguments": arguments

            })

        # ----------------------------------------------
        # APPEND ASSISTANT MESSAGE
        # ----------------------------------------------

        messages.append({

            "role": "assistant",

            "content": None,

            "tool_calls": formatted_tool_calls

        })

        # ----------------------------------------------
        # EXECUTE TOOLS
        # ----------------------------------------------

        for tool_call in formatted_tool_calls:

            tool_name = tool_call["name"]

            arguments = tool_call["arguments"]

            if tool_name not in TOOL_MAP:

                observation = {
                    "error": f"Unknown tool: {tool_name}"
                }

            else:

                try:

                    observation = TOOL_MAP[tool_name](
                        **arguments
                    )

                except Exception as error:

                    observation = {
                        "error": str(error)
                    }

            # ------------------------------------------
            # RECORD EXECUTION
            # ------------------------------------------

            steps.append({

                "tool_name": tool_name,

                "arguments": arguments,

                "observation": observation

            })

            tools_executed.append(tool_name)

            # ------------------------------------------
            # APPEND TOOL OBSERVATION
            # ------------------------------------------

            messages.append({

                "role": "tool",

                "tool_call_id": tool_call["id"],

                "name": tool_name,

                "content": json.dumps(observation)

            })

            # ------------------------------------------
            # SAVE CHECKPOINT AFTER EVERY TOOL
            # ------------------------------------------

            checkpoint_path = save_checkpoint(

                run_id=run_id,

                scenario_id=scenario_id,

                messages=messages,

                tokens_used=tokens_used,

                llm_calls=llm_calls,

                steps=steps,

                tools_executed=tools_executed

            )

            print(
                f"Checkpoint saved: {checkpoint_path.name}"
            )

            # ------------------------------------------
            # SIMULATE CRASH
            # ------------------------------------------

            if (
                crash_after_tools is not None
                and len(tools_executed) == crash_after_tools
            ):

                raise SimulatedAgentCrash(
                    f"Simulated crash after {crash_after_tools} tools."
                )

    # --------------------------------------------------
    # MAX ITERATIONS EXCEEDED
    # --------------------------------------------------

    return {

        "run_id": run_id,

        "status": "FALLBACK_MAX_ITERATIONS",

        "final_answer": fallback_response(
            "FALLBACK_MAX_ITERATIONS"
        ),

        "steps": steps,

        "tools_executed": tools_executed,

        "llm_calls": llm_calls,

        "tokens_used": tokens_used

    }


# --------------------------------------------------
# PUBLIC REACT AGENT
# --------------------------------------------------

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None
):

    run_id = run_id or str(uuid.uuid4())

    # Initial conversation messages.
    messages = [

        {
            "role": "system",
            "content": """
You are an Instamart refund support agent.

AGENT

Use the available tools to investigate refund requests.

Refund rules:

1. Only completed orders are eligible for refunds.
2. Orders delivered in more than 45 minutes qualify
   for a full refund of the order amount.
3. Orders delivered in 45 minutes or less receive
   no late-delivery refund.
4. Refunds above INR 1000 must be escalated to a human.
5. Refunds of INR 1000 or less can be issued automatically.
6. Treat every request as being raised within the
   seven-day request window.
7. Never issue a refund for a cancelled order.
8. Never invent order information.
9. Always verify information using the tools.
10. Give a clear final answer.

Follow tool observations carefully.
"""
        },

        {
            "role": "user",
            "content": user_message
        }

    ]

    return _continue_agent(

        run_id=run_id,

        scenario_id=scenario_id,

        messages=messages,

        tokens_used=0,

        llm_calls=0,

        steps=[],

        tools_executed=[],

        max_iterations=max_iterations,

        token_limit=token_limit,

        time_limit_seconds=time_limit_seconds,

        crash_after_tools=crash_after_tools

    )


# --------------------------------------------------
# RESUME AGENT
# --------------------------------------------------

def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):

    checkpoint_path = get_checkpoint_path(run_id)

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"No checkpoint found for run_id: {run_id}"
        )

    # ----------------------------------------------
    # LOAD CHECKPOINT
    # ----------------------------------------------

    checkpoint = json.loads(
        checkpoint_path.read_text(
            encoding="utf-8"
        )
    )

    print("\nLoading checkpoint:", run_id)

    scenario_id = checkpoint["scenario_id"]

    messages = checkpoint["messages"]

    tokens_used = checkpoint["tokens_used"]

    llm_calls = checkpoint["llm_calls"]

    steps = checkpoint["steps"]

    tools_executed = checkpoint["tools_executed"]

    print("Previous model calls:", llm_calls)

    print("Previous tokens:", tokens_used)

    print("Previously executed tools:", tools_executed)

    # ----------------------------------------------
    # RESET OFFLINE MODEL STATE
    # ----------------------------------------------

    # The saved messages contain the conversation and
    # tool observations needed to continue the agent.

    mock_llm.reset()

    # ----------------------------------------------
    # CONTINUE FROM CHECKPOINT
    # ----------------------------------------------

    return _continue_agent(

        run_id=run_id,

        scenario_id=scenario_id,

        messages=messages,

        tokens_used=tokens_used,

        llm_calls=llm_calls,

        steps=steps,

        tools_executed=tools_executed,

        max_iterations=max_iterations,

        token_limit=token_limit,

        time_limit_seconds=time_limit_seconds,

        crash_after_tools=None

    )


# --------------------------------------------------
# SAVE AGENT RUN
# --------------------------------------------------

def save_agent_run(scenario_id, result):

    record = {

        "scenario_id": scenario_id,

        "status": result["status"],

        "final_answer": result["final_answer"],

        "llm_calls": result["llm_calls"],

        "tokens_used": result["tokens_used"],

        "steps": result["steps"]

    }

    with open(
        AGENT_RUNS_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(record) + "\n"
        )


# --------------------------------------------------
# RUN ALL SEVEN SCENARIOS
# --------------------------------------------------

def run_all_scenarios():

    with open(
        SCENARIO_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        scenarios = list(csv.DictReader(file))

    print("\n" + "=" * 65)

    print("INSTAMART REFUND AGENT - SCENARIO EVALUATION")

    print("=" * 65)

    for scenario in scenarios:

        scenario_id = scenario["scenario_id"]

        user_message = scenario["message"]

        print(f"\n--- {scenario_id} ---")

        print("User:", user_message)

        mock_llm.reset()

        result = run_agent(

            user_message=user_message,

            scenario_id=scenario_id

        )

        save_agent_run(
            scenario_id,
            result
        )

        print("Status:", result["status"])

        print("Answer:", result["final_answer"])

        print("Model calls:", result["llm_calls"])

        print("Tokens used:", result["tokens_used"])

        print("Tool executions:", len(result["steps"]))

        for step in result["steps"]:

            print(
                "  Tool:",
                step["tool_name"],
                "| Observation:",
                step["observation"]
            )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    run_all_scenarios()