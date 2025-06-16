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

from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
import tempfile

@app.post("/cargar_documentos")
def cargar_documentos(urls: list[str]):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

    all_docs = []

    for url in urls:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(response.content)
                    tmp_file_path = tmp_file.name

                if url.endswith(".pdf"):
                    loader = UnstructuredPDFLoader(tmp_file_path)
                elif url.endswith(".docx"):
                    loader = UnstructuredWordDocumentLoader(tmp_file_path)
                else:
                    continue  # Ignora formatos no compatibles

                docs = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(docs)
                all_docs.extend(chunks)
        except Exception as e:
            print(f"Error al procesar {url}: {e}")

    if all_docs:
        vectordb = Chroma.from_documents(documents=all_docs, embedding=embeddings, persist_directory="embeddings")
        vectordb.persist()
        return {"mensaje": "Documentos cargados correctamente", "total_chunks": len(all_docs)}
    else:
        return {"mensaje": "No se pudo cargar ningún documento"}
