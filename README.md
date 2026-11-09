# Instamart Customer Intelligence & Support Copilot

An end-to-end Python project combining customer analytics, churn prediction, support-ticket triage, policy-based RAG, and an AI-assisted refund workflow.

**Tech stack:** Python, Pandas, NumPy, Scikit-learn, SQLite, TF-IDF, RAG, and a mock LLM.

## Overview

| Part | Focus |

|---|---|

| A | Data cleaning, quality analysis, and SQLite database |

| B | Customer segmentation and lapse prediction |

| C | PII redaction and support-ticket classification |

| D | Policy RAG and refund-support agent |

## Reproduce

Tested with Python 3.14 on Windows.

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

4. Run the complete pipeline:

   ```powershell

   python run_project.py

   ```

The runner executes Parts A–D and performs final acceptance checks.

**Latest run: 48/48 acceptance checks passed.**

## Part A — Data Preparation

- Audited raw customer, order, product, and ticket datasets.

- Removed duplicates and handled missing or invalid values.

- Created cleaned datasets and a SQLite database.

- Added relational tables, constraints, indexes, and a completed-orders view.

**Result:** 12,309 unique orders, 1,200 customers, and 40 products.

## Part B — Customer Intelligence

- Built RFM-based customer features.

- Applied K-Means clustering and identified four customer segments.

- Trained and evaluated Logistic Regression and Random Forest lapse models.

- Compared thresholds and analyzed feature importance.

**Results:**

- Four customer segments identified.

- Random Forest test ROC-AUC: 0.7808.

- At threshold 0.50: F1-score 0.6554 and recall 0.6517.

## Part C — Support Ticket Intelligence

- Redacted email addresses, mobile numbers, and order IDs.

- Classified tickets into delivery, billing, fraud, and technical categories.

- Added retry handling, invalid-response handling, and keyword fallback.

- Evaluated classification quality and token-budget handling.

**Results:**

- 164 tickets processed.

- 278 PII occurrences detected and redacted.

- 94.23% classification coverage.

- 100% accuracy among successfully classified tickets.

## Part D — RAG and Refund Agent

- Built a policy retrieval pipeline using TF-IDF and cosine similarity.

- Compared chunking strategies with and without overlap.

- Implemented refund calculation, policy lookup, and human escalation tools.

- Added safety limits, checkpointing, and resume testing.

**Results:**

- 15 policy chunks across five policy documents.

- 12/12 answerable questions retrieved the required policy information with overlapping chunks.

- Seven refund-agent scenarios tested.

- Safety and checkpoint/resume tests passed.

## Limitations and Next Steps

- Uses synthetic datasets and a deterministic mock LLM; results do not represent production performance.

- RAG and classification require evaluation against real customer conversations.

- Refund actions use a local demonstration ledger, not a live payment provider.

- The premium retention list is illustrative and is not a leakage-free model evaluation.

- Checkpointing demonstrates recovery but does not guarantee transactional exactly-once execution.

**Future improvements:** Integrate a production LLM, real support APIs, payment-provider integrations, and model-performance monitoring.
