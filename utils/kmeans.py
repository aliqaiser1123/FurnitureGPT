import streamlit as st
import numpy as np


@st.cache_resource
def tune_centroids(vectors, total_centroids, epochs):
    initial_centroids = vectors[
        np.random.choice(len(vectors), total_centroids, replace=False)
    ]
    vector_indices_clusters = [[] for _ in range(len(initial_centroids))]
    vectors_clusters = [[] for _ in range(len(initial_centroids))]
    new_centroids = []
    for i in range(epochs):
        for vector_index, vector in enumerate(vectors):
            distances = []
            for centroid in initial_centroids:
                distances.append(np.linalg.norm(vector - centroid))
            nearest_centroid = np.argmin(distances)
            vector_indices_clusters[nearest_centroid].append(vector_index)
            vectors_clusters[nearest_centroid].append(vector)

        new_centroids = []
        for cluster in vectors_clusters:
            if len(cluster) > 0:
                new_centroid = np.mean(np.array(cluster), axis=0)
            else:
                new_centroid = vectors[np.random.randint(len(vectors))]
            new_centroids.append(np.array(new_centroid))

        initial_centroids = new_centroids
        new_centroids = np.array(new_centroids)

    vector_indices_clusters = [[] for _ in range(len(initial_centroids))]
    vectors_clusters = [[] for _ in range(len(initial_centroids))]
    for vector_index, vector in enumerate(vectors):
        distances = []
        for centroid in new_centroids:
            distances.append(np.linalg.norm(vector - centroid))
        nearest_centroid = np.argmin(distances)
        vector_indices_clusters[nearest_centroid].append(vector_index)
        vectors_clusters[nearest_centroid].append(vector)

    return new_centroids, vector_indices_clusters
