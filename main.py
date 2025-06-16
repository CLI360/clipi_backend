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
    question = query.question

    # Cargar index de vectores
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    vectordb = Chroma(persist_directory="embeddings", embedding_function=embeddings)

    # Recuperar contexto
    docs = vectordb.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    Eres CLIpi, un asistente experto en comercio exterior, aduanas y logística.
    Usa el siguiente contexto para responder la pregunta:

    {context}

    Pregunta: {question}
    """

    response = requests.post(GEMINI_API_URL, json={
        "contents": [{"parts": [{"text": prompt}]}]
    })

    if response.status_code == 200:
        result = response.json()
        return {"respuesta": result["candidates"][0]["content"]["parts"][0]["text"]}
    else:
        return {"error": response.text}

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import shutil
import tempfile
import urllib.request

# Nueva ruta para incrustar documentos (solo lo haces 1 vez por documento)
@app.post("/cargar_documentos")
def cargar_docs(urls: List[str]):
    documents = []

    for url in urls:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
            urllib.request.urlretrieve(url, tmp_path)

            if url.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            elif url.endswith(".docx"):
                loader = Docx2txtLoader(tmp_path)
            else:
                continue

            docs = loader.load()
            documents.extend(docs)

    # Divide los textos largos en fragmentos
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Crear embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

    # Guardar vectorstore
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="embeddings")
    vectordb.persist()

    return {"status": "Documentos cargados y embebidos"}
