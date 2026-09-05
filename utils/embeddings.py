import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-base-en-v1.5")


def generate_embeddings(dataset):
    with st.spinner("Loading Model..."):
        en_model = load_model()

    embeddings = en_model.encode([data["text"] for data in dataset])
    return embeddings.tolist()
