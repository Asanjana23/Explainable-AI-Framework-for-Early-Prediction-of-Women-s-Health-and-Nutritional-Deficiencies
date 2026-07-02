from fastapi import FastAPI
from app.routes.predict import router as predict_router
from app.routes.chat import router as chat_router

app = FastAPI()

app.include_router(predict_router)
app.include_router(chat_router)