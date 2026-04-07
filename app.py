import os
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from openai import OpenAI

print("STARTING APP...")

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key="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")

CSV_PATH = "sample_data/customer_support_data.csv"
EMBED_MODEL = "text-embedding-3-small"

def load_data():
    df = pd.read_csv(CSV_PATH)
    df["text"] = (
        df["category"].astype(str) + " | " +
        df["question"].astype(str) + " | " +
        df["answer"].astype(str)
    )
    return df

def get_embedding(text):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

def build_index(texts):
    embeddings = [get_embedding(t) for t in texts]
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search(query, df, index, top_k=2):
    query_embedding = np.array([get_embedding(query)], dtype="float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        results.append(df.iloc[i]["text"])
    return results

def generate_answer(query, context_chunks):
    context = "\n".join(context_chunks)

    prompt = f"""
Answer the user's question using only the context below.
If the answer is not in the context, say you could not find it.

Context:
{context}

Question:
{query}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )
    return response.output_text

def main():
    print("Loading data...")
    df = load_data()

    print("Creating index...")
    index = build_index(df["text"].tolist())

    print("Chatbot ready! Type 'exit' to quit.\n")

    while True:
        query = input("Ask: ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        context = search(query, df, index)
        answer = generate_answer(query, context)

        print("\nAnswer:")
        print(answer)
        print("\n" + "-" * 40 + "\n")

if __name__ == "__main__":
    main()