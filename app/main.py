from fastapi import FastAPI
from app.api.ask_question import router
from app.utils.loader import load_data
from app.api.data_train_api import router as upload_router #new knowledge learning
from app.api.ask_question import router as ask_router
from app.api.train_status_api import router as status_router

app = FastAPI()

load_data()

@app.get("/")
def home():
    return {"status": "running"}

# app.include_router(router)
app.include_router(upload_router) #Data upload and training

app.include_router(ask_router) #ask question and get answer

app.include_router(status_router) #training status
