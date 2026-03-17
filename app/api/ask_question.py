from fastapi import APIRouter
from app.services.embedding_service import create_embedding
from app.services.vector_db_service import search_vector

router = APIRouter()

@router.post("/ask")
def ask_question(question: str):

    embedding = create_embedding(question)

    result, distance = search_vector(embedding)

    # similarity threshold check
    if distance > 1.2:
        return {
            "question": question,
            "answer": "Sorry, no relevant answer found."
        }

    return {
        "question": question,
        "answer": result["answer"],
        "similarity_score": float(distance)
    }