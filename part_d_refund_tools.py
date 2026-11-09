import json
import sqlite3

from datetime import datetime, timezone
from pathlib import Path

import part_d_policy_rag as rag


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "data" / "processed" / "instamart.db"

OUTPUT_DIR = BASE_DIR / "outputs"

LEDGER_FILE = OUTPUT_DIR / "refund_ledger.jsonl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# TOOL 1: GET ORDER
# --------------------------------------------------

def get_order(order_id):

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                order_id,
                status,
                order_amount,
                delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        )

        row = cursor.fetchone()

        if row is None:
            return {"error": "order not found"}

        return {
            "order_id": row["order_id"],
            "status": row["status"],
            "order_amount": row["order_amount"],
            "delivery_minutes": row["delivery_minutes"]
        }

    finally:
        connection.close()


# --------------------------------------------------
# TOOL 2: SEARCH POLICY
# --------------------------------------------------

def search_policy(query):

    results = rag.retrieve(query, k=3)

    return {
        "results": results
    }


# --------------------------------------------------
# TOOL 3: CALCULATE REFUND
# --------------------------------------------------

def calculate_refund(order_amount, delivery_minutes):

    amount = float(order_amount)
    delivery = float(delivery_minutes)

    if delivery <= 45:

        return {
            "refund_inr": 0.0
        }

    return {
        "refund_inr": round(amount, 2)
    }


# --------------------------------------------------
# COMMON LEDGER WRITER
# --------------------------------------------------

def write_ledger(tool_name, order_id, refund_inr):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with open(LEDGER_FILE, "a", encoding="utf-8") as file:

        file.write(json.dumps(entry) + "\n")


# --------------------------------------------------
# TOOL 4: ISSUE REFUND
# --------------------------------------------------

def issue_refund(order_id, refund_inr):

    amount = round(float(refund_inr), 2)

    write_ledger(
        tool_name="issue_refund",
        order_id=order_id,
        refund_inr=amount
    )

    return {
        "refund_inr": amount
    }


# --------------------------------------------------
# TOOL 5: ESCALATE TO HUMAN
# --------------------------------------------------

def escalate_to_human(order_id, refund_inr):

    amount = round(float(refund_inr), 2)

    write_ledger(
        tool_name="escalate_to_human",
        order_id=order_id,
        refund_inr=amount
    )

    return {
        "refund_inr": amount
    }


# --------------------------------------------------
# TOOL SPECIFICATIONS
# --------------------------------------------------

TOOLS_SPEC = [

    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Fetch order status, amount, and delivery time from the SQLite database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The unique order ID."
                    }
                },
                "required": ["order_id"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search the policy documents and return the three most relevant chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The policy question or search query."
                    }
                },
                "required": ["query"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate the eligible refund using the order amount and delivery time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {
                        "type": "number",
                        "description": "The order amount in INR."
                    },
                    "delivery_minutes": {
                        "type": "number",
                        "description": "The delivery time in minutes."
                    }
                },
                "required": [
                    "order_amount",
                    "delivery_minutes"
                ]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Record an approved automatic refund in the refund ledger.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string"
                    },
                    "refund_inr": {
                        "type": "number"
                    }
                },
                "required": [
                    "order_id",
                    "refund_inr"
                ]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Record a refund requiring human approval in the refund ledger.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string"
                    },
                    "refund_inr": {
                        "type": "number"
                    }
                },
                "required": [
                    "order_id",
                    "refund_inr"
                ]
            }
        }
    }

]


# --------------------------------------------------
# TOOL NAME -> PYTHON FUNCTION MAPPING
# --------------------------------------------------

TOOL_MAP = {

    "get_order": get_order,

    "search_policy": search_policy,

    "calculate_refund": calculate_refund,

    "issue_refund": issue_refund,

    "escalate_to_human": escalate_to_human

}


# --------------------------------------------------
# BASIC TESTS
# --------------------------------------------------

if __name__ == "__main__":

    print("--- TOOL 1: GET ORDER ---")

    print(get_order("ORD-100048"))

    print("\n--- TOOL 2: SEARCH POLICY ---")

    policy_result = search_policy("refund policy late delivery")

    for result in policy_result["results"]:

        print({
            "source": result["source"],
            "score": round(result["score"], 4)
        })

    print("\n--- TOOL 3: CALCULATE REFUND ---")

    print("Late delivery:", calculate_refund(128.25, 65))

    print("On-time delivery:", calculate_refund(109.65, 20))

    print("\n--- TOOL REGISTRY ---")

    print("Number of tools:", len(TOOL_MAP))

    print("Tool names:", list(TOOL_MAP.keys()))

    print("\nBasic tool tests completed.")