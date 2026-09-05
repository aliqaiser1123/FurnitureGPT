import streamlit as st
from groq import Groq
from dotenv import load_dotenv

try:
    load_dotenv
    llm_client = Groq()
except Exception as ex:
    st.error(str(ex))


def generate_answer(messages):
    response = llm_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=1.5,
        max_completion_tokens=3000,
        top_p=0.95,
    )

    reply = response.choices[0].message.content
    return reply
