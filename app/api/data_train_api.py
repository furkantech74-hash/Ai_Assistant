from fastapi import APIRouter
from pydantic import BaseModel

from app.services.embedding_service import create_embedding
from app.services.vector_db_service import add_vector

router = APIRouter()

class QARequest(BaseModel):
    question: str
    answer: str

@router.post("/data_training")
def train_data(data: QARequest):

    embedding = create_embedding(data.question)

    add_vector(embedding,{
        "question":data.question,
        "answer":data.answer
    })

    return {"message":"Training data added"}