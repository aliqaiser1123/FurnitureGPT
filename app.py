import time
import fitz
import streamlit as st
import numpy as np


from utils.llm import generate_answer
from utils.database import create_database
from utils.template import context_template
from utils.embeddings import generate_embeddings
from utils.compressor import compress_embeddings
from utils.local_pdf_parser import parse_for_dataset
from utils.retreival import original_retrieval, compressed_retrieval

from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer


@st.dialog("Product Details", width="medium")
def show_product_details(image, metadata):
    st.image(image)
    st.title(metadata["product"].upper())
    for key, value in metadata.items():
        if key == "product" or key == "page":
            continue
        st.write(f"**{key.title()}**: {value.title()}")


@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-base-en-v1.5")


with st.spinner("Loading Model..."):
    em_model = load_model()


st.set_page_config(page_title="Furniture GPT", layout="wide")
st.title("FURNITURE GPT", text_alignment="center")
st.divider()

pdf = fitz.open("Furniture Catalog.pdf")

_, col1, _, col2, _, col3, _ = st.columns(7)
with col1:
    with st.spinner("Creating dataset"):
        try:
            dataset = parse_for_dataset(pdf)
        except Exception as ex:
            st.error(str(ex))

        if dataset:
            st.metric("Dataset", "**:green[LOADED]**", "Successfully")


if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "compressed_embeddings" not in st.session_state:
    st.session_state.compressed_embeddings = None

if "codebooks" not in st.session_state:
    st.session_state.codebooks = []

with col2:
    if st.session_state.embeddings is None:
        with st.spinner("Generating Embeddings..."):
            try:
                st.session_state.embeddings = generate_embeddings(dataset)
                embeddings = st.session_state.embeddings[:]
            except Exception as ex:
                st.error(str(ex))
    else:
        embeddings = st.session_state.embeddings[:]

    if st.session_state.compressed_embeddings is None:
        with st.spinner("Compressing Embeddings..."):
            time.sleep(2)
            st.session_state.codebooks, st.session_state.compressed_embeddings = (
                compress_embeddings(embeddings)
            )
    else:
        pass

    if st.session_state.compressed_embeddings:
        st.metric("Embeddings", "**:green[GENERATED]**", "Successfully")

    compressed_embeddings = st.session_state.compressed_embeddings[:]


# st.write(sub_embeddings)
# st.write(st.session_state.compressed_embeddings)
# st.write(codebooks)

if "database" not in st.session_state:
    st.session_state.database = None

with col3:
    with st.spinner("Creating Database..."):
        try:
            if st.session_state.database is None:
                st.session_state.database = create_database(
                    dataset, embeddings, compressed_embeddings
                )

        except Exception as ex:
            st.error(str(ex))

        if st.session_state.database:
            st.metric("Database", "**:green[CREATED]**", "Successfully")

with st.expander("Expand to see Embeddings metrics"):
    c1, c2, c3 = st.columns(3)
    original_embeddings_size = (
        len(st.session_state.embeddings) * (4 * len(st.session_state.embeddings[0]))
    ) / 1000
    compressed_embeddings_size = (
        len(st.session_state.compressed_embeddings)
        * (2 * len(st.session_state.compressed_embeddings[0]))
    ) / 1000
    compressed_ratio = int(original_embeddings_size // compressed_embeddings_size)
    with c1:
        st.metric(
            "Original Embeddings Size",
            f"{original_embeddings_size} KB",
        )
    with c2:
        st.metric(
            "Compressed Embeddings Size",
            f"{compressed_embeddings_size} KB",
        )
    with c3:
        st.metric(
            "Compressed Ratio",
            f"{compressed_ratio} % Compressed",
        )

st.divider()

query = st.chat_input("Hi! Welcome to Furniture GPT, Ask Anything:")
st.subheader("**TRY :green[ASKING]**", text_alignment="center")
_, _, col1, col2, _, _ = st.columns(6)

with col1:
    if st.button("Office chair", icon="🪑"):
        query = "Office chair"
    if st.button("Furniture under $1000", icon="💰"):
        query = "Furniture under $1000"
    if st.button("Best sofa for small apartment", icon="🛋"):
        query = "Best sofa for small apartment"

with col2:
    if st.button("Scandinavian style", icon="🎨"):
        query = "Scandinavian style"
    if st.button("Furnish my bedroom", icon="🏡"):
        query = "Furnish my bedroom"
    if st.button("Beautiful TV Console", icon="📺"):
        query = "Beautiful TV Console"

with st.sidebar:
    top_k = st.slider("Select top k:", min_value=1, max_value=len(dataset), value=5)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

if "query" not in st.session_state:
    st.session_state.query = None

if "query_vector" not in st.session_state:
    st.session_state.query_vector = None

if "original_retrieved_data" not in st.session_state:
    st.session_state.original_retrieved_data = []

if "compressed_retrieved_data" not in st.session_state:
    st.session_state.compressed_retrieved_data = []

if "total_original_retrieval_time" not in st.session_state:
    st.session_state.total_original_retrieval_time = 0

if "total_compressed.retrieval_time" not in st.session_state:
    st.session_state.total_compressed_retrieval_time = 0

original_results = st.session_state.original_retrieved_data
compressed_results = st.session_state.compressed_retrieved_data
compress = st.checkbox("Use compressed embeddings")

col1, col2 = st.columns((7, 3))

if query:
    st.session_state.query = query
    st.session_state.query_vector = em_model.encode(query)

    query_subvector = np.split(st.session_state.query_vector, 16)

    lookup_table = []

    for subvector_index, subvector in enumerate(query_subvector):
        subvector_similarities = []
        for vector in st.session_state.codebooks[subvector_index]:
            subvector_similarities.append(
                cos_sim(
                    [float(value) for value in subvector],
                    [float(value) for value in vector],
                )
            )
        lookup_table.append(subvector_similarities)
    # st.write(lookup_table)

    original_retrieval_start = time.perf_counter()
    st.session_state.original_retrieved_data = original_retrieval(
        st.session_state.query_vector, st.session_state.database
    )
    original_retrieval_end = time.perf_counter()
    st.session_state.total_original_retrieval_time = (
        original_retrieval_end - original_retrieval_start
    )

    compressed_retrieval_start = time.perf_counter()
    st.session_state.compressed_retrieved_data = compressed_retrieval(
        st.session_state.query_vector, st.session_state.database, lookup_table
    )
    compressed_retrieval_end = time.perf_counter()
    st.session_state.total_compressed_retrieval_time = (
        compressed_retrieval_end - compressed_retrieval_start
    )

    original_results = st.session_state.original_retrieved_data[:top_k]
    compressed_results = st.session_state.compressed_retrieved_data[:top_k]

    with st.spinner("Generating Response..."):
        context = context_template + "\n\nContext: \n"
        if compress:
            for data in compressed_results:
                context += st.session_state.database[data["Id"]]["text"]
                context += "Source: \n"
                context += (
                    "Category: "
                    + st.session_state.database[data["Id"]]["metadata"]["category"]
                )
                context += "Page: " + str(
                    st.session_state.database[data["Id"]]["metadata"]["page"]
                )
        else:
            for data in original_results:
                context += st.session_state.database[data["Id"]]["text"]
                context += "Source: \n"
                context += (
                    "Category: "
                    + st.session_state.database[data["Id"]]["metadata"]["category"]
                )
                context += "Page: " + str(
                    st.session_state.database[data["Id"]]["metadata"]["page"]
                )

        context += "Query: \n"
        st.session_state.messages.append({"role": "user", "content": context + query})

        reply = generate_answer(st.session_state.messages)
        st.session_state.messages[-1]["content"] = query

        st.session_state.messages.append({"role": "assistant", "content": reply})

if st.session_state.messages:
    for response in st.session_state.messages:
        if response["role"] == "user":
            st.chat_message("User").write(response["content"])
        if response["role"] == "assistant":
            st.chat_message("Assistant").write(response["content"])
        st.divider()
    database = st.session_state.database[:]
    st.subheader("Recomended Products")

    col1, col2, col3 = st.columns(3)

    if compress:
        with col1:
            st.image(database[compressed_results[0]["Id"]]["image"])
            col_1, col_2 = st.columns((7, 4))
            with col_1:
                st.subheader(
                    f"**{database[compressed_results[0]['Id']]['metadata']['product']}**"
                )
                st.write(
                    f"""Category: {
                        database[compressed_results[0]["Id"]]["metadata"]["category"]
                    }"""
                )
            with col_2:
                if st.button("See Product Details", key="product1"):
                    show_product_details(
                        database[compressed_results[0]["Id"]]["image"],
                        database[compressed_results[0]["Id"]]["metadata"],
                    )
        with col2:
            st.image(database[compressed_results[1]["Id"]]["image"])
            col_1, col_2 = st.columns((7, 4))
            with col_1:
                st.subheader(
                    f"**{database[compressed_results[1]['Id']]['metadata']['product']}**"
                )
                st.write(
                    f"Category: {
                        database[compressed_results[1]['Id']]['metadata']['category']
                    }"
                )
            with col_2:
                if st.button("See Product Details", key="product2"):
                    show_product_details(
                        database[compressed_results[1]["Id"]]["image"],
                        database[compressed_results[1]["Id"]]["metadata"],
                    )
        with col3:
            st.image(database[compressed_results[2]["Id"]]["image"])
            col_1, col_2 = st.columns((7, 4))
            with col_1:
                st.subheader(
                    f"**{database[compressed_results[2]['Id']]['metadata']['product']}**"
                )
                st.write(
                    f"Category: {
                        database[compressed_results[2]['Id']]['metadata']['category']
                    }"
                )
            with col_2:
                if st.button("See Product Details", key="product3"):
                    show_product_details(
                        database[compressed_results[2]["Id"]]["image"],
                        database[compressed_results[2]["Id"]]["metadata"],
                    )
        with st.expander("See more recomendations"):
            col1, col2 = st.columns(2)
            with col1:
                st.image(database[compressed_results[3]["Id"]]["image"])
                col_1, col_2 = st.columns((7, 4))
                with col_1:
                    st.subheader(
                        f"**{database[compressed_results[3]['Id']]['metadata']['product']}**"
                    )
                    st.write(
                        f"Category: {
                            database[compressed_results[3]['Id']]['metadata'][
                                'category'
                            ]
                        }"
                    )
                with col_2:
                    if st.button("See Product Details", key="product4"):
                        show_product_details(
                            database[compressed_results[3]["Id"]]["image"],
                            database[compressed_results[3]["Id"]]["metadata"],
                        )
            with col2:
                st.image(database[compressed_results[4]["Id"]]["image"])
                col_1, col_2 = st.columns((7, 4))
                with col_1:
                    st.subheader(
                        f"**{database[compressed_results[4]['Id']]['metadata']['product']}**"
                    )
                    st.write(
                        f"Category: {
                            database[compressed_results[4]['Id']]['metadata'][
                                'category'
                            ]
                        }"
                    )
                with col_2:
                    if st.button("See Product Details", key="product5"):
                        show_product_details(
                            database[compressed_results[4]["Id"]]["image"],
                            database[compressed_results[4]["Id"]]["metadata"],
                        )

        st.write(
            f":green[{len(st.session_state.compressed_retrieved_data)} Documents Searched in {st.session_state.total_compressed_retrieval_time:.4f} seconds.]"
        )

    else:
        with col1:
            st.image(database[original_results[0]["Id"]]["image"])
            col_1, col_2 = st.columns((7, 4))
            with col_1:
                st.subheader(
                    f"**{database[original_results[0]['Id']]['metadata']['product']}**"
                )
                st.write(
                    f"""Category: {
                        database[original_results[0]["Id"]]["metadata"]["category"]
                    }"""
                )
            with col_2:
                if st.button("See Product Details", key="product1"):
                    show_product_details(
                        database[original_results[0]["Id"]]["image"],
                        database[original_results[0]["Id"]]["metadata"],
                    )
        with col2:
            st.image(database[original_results[1]["Id"]]["image"])
            col_1, col_2 = st.columns((7, 4))
            with col_1:
                st.subheader(
                    f"**{database[original_results[1]['Id']]['metadata']['product']}**"
                )
                st.write(
                    f"Category: {
                        database[original_results[1]['Id']]['metadata']['category']
                    }"
                )
            with col_2:
                if st.button("See Product Details", key="product2"):
                    show_product_details(
                        database[original_results[1]["Id"]]["image"],
                        database[original_results[1]["Id"]]["metadata"],
                    )
        with col3:
            st.image(database[original_results[2]["Id"]]["image"])
            col_1, col_2 = st.columns((7, 4))
            with col_1:
                st.subheader(
                    f"**{database[original_results[2]['Id']]['metadata']['product']}**"
                )
                st.write(
                    f"Category: {
                        database[original_results[2]['Id']]['metadata']['category']
                    }"
                )
            with col_2:
                if st.button("See Product Details", key="product3"):
                    show_product_details(
                        database[original_results[2]["Id"]]["image"],
                        database[original_results[2]["Id"]]["metadata"],
                    )
        st.write(
            f":green[{len(st.session_state.original_retrieved_data)} Documents Searched in {st.session_state.total_original_retrieval_time:.4f} seconds.]"
        )

if "recalls" not in st.session_state:
    st.session_state.recalls = []

if st.session_state.messages:
    if st.button("Clear Chat History"):
        st.session_state.messages = []

        st.toast("  **:green[Chat Cleared] ✔** ")
        st.session_state.recalls = []
        time.sleep(2)
        st.rerun()

    if compress:
        if query:
            with st.expander("Expand to see metrices"):
                ground_truth = [
                    original_results[i]["Id"] for i in range(len(original_results))
                ]
                pq_results = [
                    compressed_results[i]["Id"] for i in range(len(compressed_results))
                ]
                c1, c2 = st.columns(2)
                recall = len(set(ground_truth).intersection(set(pq_results))) / len(
                    ground_truth
                )
                with c1:
                    st.metric(
                        "CURRENT QUERY RECALL",
                        f"{recall * 100:.2f}% Accurate",
                    )
                    st.session_state.recalls.append(recall)
                    avg_recall = np.mean(st.session_state.recalls)
                with c2:
                    st.metric("AVG. RECALL", f"{avg_recall * 100:.2f}% Accurate")
                st.info(
                    f"{avg_recall * 100:.2f}% Accuracy obtained on {compressed_ratio}% Embeddings Compression over {len(st.session_state.recalls)} queries."
                )
                if len(st.session_state.recalls) > 10:
                    st.session_state.recalls = []
