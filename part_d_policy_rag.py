from pathlib import Path

import mock_llm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
POLICY_DIR = BASE_DIR / "data" / "policies"


# --------------------------------------------------
# TASK 1: LOAD AND CHUNK POLICY DOCUMENTS
# --------------------------------------------------

def chunk_documents(overlap=10):
    """
    Split policy documents into 40-word chunks.

    Default overlap = 10 words.
    Chunk 1: words 0-39
    Chunk 2: words 30-69
    Chunk 3: words 60-99
    """

    chunk_size = 40
    step = chunk_size - overlap

    chunks = []

    policy_files = sorted(POLICY_DIR.glob("*.txt"))

    for policy_file in policy_files:

        text = policy_file.read_text(encoding="utf-8")

        words = text.split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_words = words[start:end]

            chunk_text = " ".join(chunk_words)

            chunks.append({
                "source": policy_file.name,
                "text": chunk_text
            })

            if end >= len(words):
                break

            start += step

    return chunks


# --------------------------------------------------
# TASK 2: TF-IDF VECTOR SEARCH
# --------------------------------------------------

def build_index(overlap=10):

    chunks = chunk_documents(overlap)

    documents = [chunk["text"] for chunk in chunks]

    # Use the default TfidfVectorizer configuration.
    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform(documents)

    return chunks, vectorizer, matrix


# Build the default index using 10-word overlap.
CHUNKS, VECTORIZER, MATRIX = build_index(overlap=10)


def retrieve(query, k=3):

    query_vector = VECTORIZER.transform([query])

    similarities = cosine_similarity(
        query_vector,
        MATRIX
    ).flatten()

    top_indices = similarities.argsort()[::-1][:k]

    results = []

    for index in top_indices:

        results.append({
            "source": CHUNKS[index]["source"],
            "score": float(similarities[index]),
            "text": CHUNKS[index]["text"]
        })

    return results


# --------------------------------------------------
# TASK 3: POLICY QUESTION ANSWERING
# --------------------------------------------------

def answer(question):

    retrieved = retrieve(question, k=3)

    context_parts = []

    for item in retrieved:

        context_parts.append(
            f"[{item['source']}]\n{item['text']}"
        )

    context = "\n".join(context_parts)

    system_message = """
You are a policy assistant.

POLICY ASSISTANT

Answer questions using ONLY the policy context supplied by the user.

Rules:
1. Do not use outside knowledge.
2. Do not invent policy rules, amounts, or conditions.
3. If the context does not contain enough information,
   respond beginning exactly with:
   I could not find
4. Give a clear and concise answer.
5. Follow the policy wording and conditions accurately.
"""

    user_message = f"""CONTEXT:
{context}
QUESTION:
{question}"""

    # Exactly two messages are sent to mock_llm.chat.
    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    response = mock_llm.chat(messages)

    model_answer = response.choices[0].message.content

    sources_used = list(dict.fromkeys(
        item["source"] for item in retrieved
    ))

    return model_answer, sources_used


# --------------------------------------------------
# QUICK VALIDATION
# --------------------------------------------------

if __name__ == "__main__":

    print("--- POLICY CHUNK SUMMARY ---")

    source_counts = {}

    for chunk in CHUNKS:
        source = chunk["source"]
        source_counts[source] = source_counts.get(source, 0) + 1

    for source, count in source_counts.items():
        print(f"{source}: {count} chunks")

    print(f"\nTotal chunks: {len(CHUNKS)}")

    print("\n--- RETRIEVAL TEST ---")

    query = "refund policy late delivery"

    results = retrieve(query, k=3)

    for number, result in enumerate(results, start=1):

        print(f"\nResult {number}")
        print("Source:", result["source"])
        print("Score:", round(result["score"], 4))
        print("Text:", result["text"])

    print("\n--- POLICY ANSWER TEST ---")

    question = "What happens if an order is delivered late?"

    response, sources = answer(question)

    print("Question:", question)
    print("Answer:", response)
    print("Sources:", sources)