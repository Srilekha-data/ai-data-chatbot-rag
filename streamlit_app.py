import streamlit as st
import pandas as pd
import ollama

# Load data
df = pd.read_csv("sample_data/customer_support_data.csv")

df["text"] = (
    df["category"].astype(str) + " " +
    df["question"].astype(str) + " " +
    df["answer"].astype(str)
)

# Get context from data
def get_context(query):
    for text in df["text"]:
        if query.lower() in text.lower():
            return text
    return ""

# Ask Ollama
def ask_ai(query):
    context = get_context(query)

    prompt = f"""
    You are a helpful customer support assistant.

    Use this context to answer:
    {context}

    Question:
    {query}
    """

    response = ollama.chat(
        model="gemma",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"].replace("Orders", "").strip()

# UI
st.title("💬 AI Customer Support Chatbot")

user_input = st.text_input("Ask your question:")

if user_input:
    answer = ask_ai(user_input)
    st.write("### Answer:")
    st.write(answer)