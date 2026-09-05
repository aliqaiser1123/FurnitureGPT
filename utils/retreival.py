import numpy as np
import streamlit as st

from sentence_transformers.util import cos_sim
from utils.kmeans import tune_centroids


def original_retrieval(query_vector, database):
    original_similarties = []
    for i in range(len(database)):
        original_similarties.append(
            {"Id": i, "Score": cos_sim(query_vector, database[i]["embeddings"])}
        )
    original_similarties.sort(key=lambda sim: sim["Score"], reverse=True)
    return original_similarties


def compressed_retrieval(query_vector, database, lookup_table):
    compressed_embeddings = [
        database[i]["compressed embeddings"] for i in range(len(database))
    ]

    embeddings = [database[i]["embeddings"] for i in range(len(database))]

    compressed_similarties = []
    clusters_similarities = []
    n_list = 5
    clusters_centroids, vector_indices_clusters = tune_centroids(
        np.array(embeddings), n_list, epochs=5
    )

    for index, centroid in enumerate(clusters_centroids):
        clusters_similarities.append(
            {
                "Id": index,
                "Score": cos_sim(
                    [float(value) for value in query_vector],
                    [float(value) for value in centroid],
                ),
            }
        )
    clusters_similarities.sort(key=lambda sim: sim["Score"], reverse=True)

    nprobe = 3
    final_clusters = [clusters_similarities[i]["Id"] for i in range(nprobe)]

    for cluster in final_clusters:
        for vector_index in vector_indices_clusters[cluster]:
            em_sim = 0
            for dimension_index, dimension in enumerate(
                compressed_embeddings[vector_index]
            ):
                em_sim += lookup_table[dimension_index][dimension]
            compressed_similarties.append({"Id": vector_index, "Score": em_sim})
    compressed_similarties.sort(key=lambda sim: sim["Score"], reverse=True)

    return compressed_similarties
