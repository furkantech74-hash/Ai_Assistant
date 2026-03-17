import json
from app.services.embedding_service import create_embedding
from app.services.vector_db_service import add_vector

def load_data():

    with open("data/company_qa.json") as f:
        data = json.load(f)

    for item in data:

        question = item["question"]

        embedding = create_embedding(question)

        add_vector(embedding, item)