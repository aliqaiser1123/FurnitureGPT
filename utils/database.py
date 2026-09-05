import json


def create_database(dataset, embeddings, compressed_embeddings):
    vector_database = []
    for i in range(len(dataset)):
        dataset[i]["embeddings"] = embeddings[i]
        dataset[i]["compressed embeddings"] = compressed_embeddings[i]
        vector_database.append(dataset[i])

    with open("vector_database.json", "w") as f:
        json.dump(vector_database, f, indent=4)
    return vector_database
