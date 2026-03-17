import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# reset old DB
if os.path.exists("vector_db/faiss_index.bin"):
    os.remove("vector_db/faiss_index.bin")

if os.path.exists("vector_db/data_store.pkl"):
    os.remove("vector_db/data_store.pkl")

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.IndexFlatL2(384)
data_store = []

with open("data/company_qa.json", encoding="utf-8") as f:
    data = json.load(f)
    
for item in data:

    embedding = model.encode(item["question"])

    index.add(np.array([embedding]))

    data_store.append(item)

faiss.write_index(index,"vector_db/faiss_index.bin")

with open("vector_db/data_store.pkl","wb") as f:
    pickle.dump(data_store,f)

print("Training Completed")
print("Total trained:",len(data_store))