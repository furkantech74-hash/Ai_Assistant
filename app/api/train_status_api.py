from fastapi import APIRouter
import pickle
import os

router = APIRouter()

@router.get("/training-status")
def training_status():

    file_path = "vector_db/data_store.pkl"

    if not os.path.exists(file_path):
        return {
            "status": "No training data found",
            "trained_questions": 0
        }

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    return {
        "status": "Training completed",
        "trained_questions": len(data)
    }