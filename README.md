# Instamart Customer Intelligence & Support Copilot

An end-to-end Python project combining customer analytics, lapse prediction, support-ticket triage, policy-based RAG, and an AI-assisted refund workflow.

**Tech stack:** Python, Pandas, NumPy, Scikit-learn, SQLite, TF-IDF, and a mock LLM.

## Overview

| Part | Focus |
|---|---|
| A | Data cleaning, quality analysis, SQL, and SQLite |
| B | Customer segmentation and lapse prediction |
| C | PII redaction and support-ticket classification |
| D | Policy RAG and refund-support agent |

This is an independent portfolio project inspired by an Instamart-style business scenario. It is not affiliated with Swiggy.

## Reproduce

**Requirements:** Python 3.11+, SQLite 3.25+, and Windows PowerShell.

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

2. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Generate synthetic datasets and policy files:

   ```powershell
   python generate_data.py
   ```

5. Run the complete pipeline:

   ```powershell
   python run_project.py
   ```

The runner executes Parts A–D, runs safety and recovery tests, and validates generated outputs.

**Latest validation: 50/50 acceptance checks passed.**

## Part A — Data Preparation and SQL Analysis

- Audited raw customer, order, product, and ticket datasets.
- Removed duplicates and handled missing, invalid, and inconsistent values.
- Created cleaned CSV datasets and a SQLite database.
- Added relational constraints, indexes, and a completed-orders view.
- Developed seven SQL queries using joins, aggregations, subqueries, CTEs, and window functions.

**Results:**
- Raw orders: 13,084; exact duplicate rows removed: 119.
- Final unique orders: 12,309.
- Customers: 1,200; products: 40.
- All seven SQL queries executed successfully.
- SQL result counts: 30, 30, 11, 25, 550, 1,175, and 183 rows.

## Part B — Customer Intelligence

- Built RFM-based customer features.
- Applied K-Means clustering and identified four customer segments.
- Trained Logistic Regression and Random Forest lapse-prediction models.
- Evaluated thresholds, feature importance, and model performance.

**Results:**
- 1,175 customers included in RFM segmentation.
- Four segments: Regular Customers, Premium, Frequent Shoppers, and Occasional Shoppers.
- Final model dataset: 1,156 customers.
- Training rows: 924; test rows: 232.
- Random Forest test ROC-AUC: **0.7808**.
- At threshold 0.50: precision **0.6591**, recall **0.6517**, and F1-score **0.6554**.
- Top Random Forest features: frequency (**0.2796**), monetary value (**0.2712**), and recency (**0.1151**).

**Evaluation note:** The premium retention list is illustrative. Its premium flag uses full-window RFM information and is not a leakage-free prospective evaluation.

## Part C — Support Ticket Intelligence

- Redacted email addresses, Indian mobile numbers, and order IDs.
- Classified tickets into delivery, billing, fraud, and technical categories.
- Implemented retry handling, invalid-response handling, and keyword fallback.
- Added input guardrails, priority assignment, and routing queues.
- Evaluated classification quality and token usage.

**Results:**
- 164 tickets processed: 147 successful, 9 invalid, and 8 skipped.
- 278 PII occurrences detected and redacted.
- Successful classifications: 147/147 correct.
- Classification coverage: **94.23%**.
- Total attempts: 175.
- Total token usage: 28,432.
- Token-budget demonstration: 2,000-token limit, 10 tickets processed, 146 rejected.

The 100% accuracy applies only to successfully classified tickets. Results use synthetic data and a deterministic mock LLM.

## Part D — Policy RAG and Refund Agent

- Built a TF-IDF and cosine-similarity retrieval pipeline over five policy documents.
- Compared chunking strategies with and without 10-word overlap.
- Implemented order lookup, policy search, refund calculation, refund issuance, and human escalation.
- Added token, timeout, and iteration limits.
- Implemented checkpointing and resume support.

**RAG evaluation:**
- Five policy documents produced 15 chunks.
- With 10-word overlap: 12/12 answerable questions retrieved the required policy information.
- Without overlap: 12/12 document retrieval, but fact coverage fell to 10/12.
- Unsupported-question fallback: 2/2.

**Seven agent scenarios:**

| Scenario | Outcome | Model calls | Tokens |
|---|---|---:|---:|
| S1 | REFUND — INR 128.25 | 5 | 1,612 |
| S2 | REFUND — INR 135.00 | 5 | 1,612 |
| S3 | ESCALATE — INR 1,999.00 | 5 | 1,599 |
| S4 | DENIED | 4 | 1,177 |
| S5 | DENIED | 3 | 766 |
| S6 | NOT_FOUND | 2 | 358 |
| S7 | FALLBACK_MAX_ITERATIONS | 6 | 3,858 |

**Safety tests:**

| Test | Result |
|---|---|
| Token limit | `FALLBACK_TOKEN_LIMIT` — passed |
| Timeout | `FALLBACK_TIMEOUT` — passed |
| Maximum iterations | `FALLBACK_MAX_ITERATIONS` — passed |

**Checkpoint and resume:** A simulated crash after two tools was recovered successfully. The agent resumed without repeating completed tools and issued a refund of INR 128.25.

Tools executed in order:
1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`

## Limitations and Next Steps

- Uses synthetic datasets and a deterministic mock LLM, not real customer conversations.
- Model and classification results require evaluation on representative real-world data.
- Refund actions use a local demonstration ledger, not a live payment provider.
- The retention list is illustrative and not a leakage-free prospective evaluation.
- Checkpointing supports recovery but does not guarantee transactional exactly-once execution.

**Future improvements:** Evaluate with real-world data, integrate production support and payment APIs, and add model-performance monitoring.