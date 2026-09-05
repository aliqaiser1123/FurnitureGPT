import streamlit as st
import numpy as np
from utils.kmeans import tune_centroids
from sentence_transformers.util import cos_sim


def compress_embeddings(embeddings):
    total_sub_embeddings = 16
    sub_embeddings_centroids = 25
    epochs = 10

    sub_embeddings = np.split(np.array(embeddings), total_sub_embeddings, axis=1)

    codebooks = []
    compressed_embeddings = [[] for _ in range(len(embeddings))]
    for i in range(len(sub_embeddings)):
        centroids, _ = tune_centroids(
            sub_embeddings[i], sub_embeddings_centroids, epochs
        )
        codebooks.append(centroids)
        for vector_index, vector in enumerate(sub_embeddings[i]):
            sim = []
            for centroid_index, centroid in enumerate(codebooks[i]):
                sim.append(cos_sim(vector, centroid))
            compressed_embeddings[vector_index].append(int(np.argmax(sim)))
    return codebooks, compressed_embeddings
