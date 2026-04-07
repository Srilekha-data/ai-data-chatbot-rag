import streamlit as st
import pandas as pd

# Title
st.title("💬 AI Customer Support Chatbot")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("sample_data/customer_support_data.csv")
    return df

data = load_data()

# Simple search function
def get_answer(user_question):
    user_question = user_question.lower()

    for i, row in data.iterrows():
        if user_question in row["question"].lower():
            return row["answer"]

    return "Sorry, I couldn't find an answer. Please contact support."

# Input box
user_input = st.text_input("Ask your question:")

# Show answer
if user_input:
    answer = get_answer(user_input)
    st.write("### Answer:")
    st.write(answer)