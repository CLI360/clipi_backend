# Código del backend con Gemini Flash
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

@app.get("/")
def root():
    return {"message": "CLIpi backend activo"}

@app.post("/preguntar")
def responder(query: Query):
    prompt = f"""
    Eres CLIpi, un experto en comercio exterior, aduanas y logística.
    Responde con base en los documentos cargados y tu conocimiento del área.

    Pregunta: {query.question}
    """
    response = requests.post(GEMINI_API_URL, json={
        "contents": [{"parts": [{"text": prompt}]}]
    })

    if response.status_code == 200:
        result = response.json()
        return {"respuesta": result["candidates"][0]["content"]["parts"][0]["text"]}
    else:
        return {"error": response.text}