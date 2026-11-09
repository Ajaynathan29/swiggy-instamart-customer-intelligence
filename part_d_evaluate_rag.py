import pandas as pd
import part_d_policy_rag as rag


EVAL_FILE = rag.BASE_DIR / "data" / "raw" / "policy_eval.csv"


def evaluate(overlap):

    print("\n" + "=" * 60)
    print(f"RAG EVALUATION - OVERLAP: {overlap}")
    print("=" * 60)

    # Rebuild the index for this overlap setting.
    rag.CHUNKS, rag.VECTORIZER, rag.MATRIX = rag.build_index(
        overlap=overlap
    )

    df = pd.read_csv(EVAL_FILE)

    retrieval_correct = 0
    fact_correct = 0
    fallback_correct = 0

    supported = df[df["expected_doc"] != "NONE"]
    unsupported = df[df["expected_doc"] == "NONE"]

    failures = []

    for _, row in df.iterrows():

        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        # Retrieve top 3 chunks.
        retrieved = rag.retrieve(question, k=3)

        retrieved_sources = [
            item["source"] for item in retrieved
        ]

        # Generate answer.
        answer, sources = rag.answer(question)

        if expected_doc != "NONE":

            # Check expected document among top 3.
            if expected_doc in retrieved_sources:
                retrieval_correct += 1

            # Case-sensitive expected fact comparison.
            if isinstance(expected_fact, str) and expected_fact in answer:
                fact_correct += 1
            else:
                failures.append({
                    "question": question,
                    "expected_fact": expected_fact,
                    "answer": answer
                })

        else:

            # Check exact required fallback prefix.
            if answer.startswith("I could not find"):
                fallback_correct += 1

    # Print results.
    print("\n--- RESULTS ---")

    print(
        f"Document retrieval: "
        f"{retrieval_correct}/{len(supported)}"
    )

    print(
        f"Expected fact coverage: "
        f"{fact_correct}/{len(supported)}"
    )

    print(
        f"Unsupported fallback: "
        f"{fallback_correct}/{len(unsupported)}"
    )

    print("\n--- ANSWER FAILURES ---")

    if not failures:
        print("No answerable-question failures.")
    else:
        for failure in failures:
            print("\nQuestion:", failure["question"])
            print("Expected:", failure["expected_fact"])
            print("Answer:", failure["answer"])

    return {
        "overlap": overlap,
        "retrieval": f"{retrieval_correct}/{len(supported)}",
        "fact_coverage": f"{fact_correct}/{len(supported)}",
        "unsupported_fallback": f"{fallback_correct}/{len(unsupported)}"
    }


if __name__ == "__main__":

    # Evaluation with 10-word overlap.
    overlap_results = evaluate(10)

    # Evaluation without overlap.
    zero_overlap_results = evaluate(0)

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    comparison = pd.DataFrame([
        overlap_results,
        zero_overlap_results
    ])

    print(comparison.to_string(index=False))