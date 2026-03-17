import faiss
import pickle
import numpy as np
import os

dimension = 384

index_path = "vector_db/faiss_index.bin"
data_path = "vector_db/data_store.pkl"

# load FAISS index
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    index = faiss.IndexFlatL2(dimension)

# load stored Q&A
if os.path.exists(data_path):
    with open(data_path,"rb") as f:
        data_store = pickle.load(f)
else:
    data_store = []


def add_vector(embedding, qa):

    #  duplicate question check
    for item in data_store:
        if item["question"].strip().lower() == qa["question"].strip().lower():
            print("Duplicate question skipped:", qa["question"])
            return {"message": "Question already trained"}

    # add new vector
    index.add(np.array([embedding]))

    data_store.append(qa)

    # save FAISS index
    faiss.write_index(index,index_path)

    # save data
    with open(data_path,"wb") as f:
        pickle.dump(data_store,f)

    return {"message": "Training data added"}


def search_vector(query_embedding):

    D,I = index.search(np.array([query_embedding]),k=1)

    distance = float(D[0][0])

    result = data_store[I[0][0]]

    return result,distance