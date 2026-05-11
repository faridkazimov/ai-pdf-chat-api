from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import os
from openai import OpenAI
from fastapi import UploadFile, File
from pypdf import PdfReader
from io import BytesIO



app = FastAPI()

qdrant = QdrantClient(host="qdrant", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class AskRequest(BaseModel):
    question: str

class AddTextRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/qdrant-health")
def qdrant_health():
    collections = qdrant.get_collections()
    return {
        "status": "qdrant connected",
        "collections": collections
    }


@app.post("/create-collection")
def create_collection():
    qdrant.recreate_collection(
        collection_name="documents",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    return {
        "status": "collection created",
        "collection": "documents"
    }

@app.post("/ask")
def ask(request: AskRequest):
    question_vector = model.encode(request.question).tolist()

    results = qdrant.query_points(
        collection_name="documents",
        query=question_vector,
        limit=1
    )

    if not results.points:
        return {
            "question": request.question,
            "answer": "Bu soruyla ilgili bilgi bulunamadı."
        }

    matched_text = results.points[0].payload["text"]

    prompt = f"""
    Aşağıdaki bilgiye göre soruyu cevapla.

    Bilgi:
    {matched_text}

    Soru:
    {request.question}
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return {
        "question": request.question,
        "matched_text": matched_text,
        "answer": response.output_text
    }

@app.post("/add-text")
def add_text(request: AddTextRequest):
    vector = model.encode(request.text).tolist()

    qdrant.upsert(
        collection_name="documents",
        points=[
            PointStruct(
                id=1,
                vector=vector,
                payload={
                    "text": request.text
                }
            )
        ]
    )

    return {
        "status": "text added",
        "text": request.text
    }
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    reader = PdfReader(BytesIO(pdf_bytes))

    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    chunks = [
        full_text[i:i+500]
        for i in range(0, len(full_text), 500)
    ]

    points = []

    for index, chunk in enumerate(chunks):
        vector = model.encode(chunk).tolist()

        points.append(
            PointStruct(
                id=index + 100,
                vector=vector,
                payload={
                    "text": chunk,
                    "source": file.filename
                }
            )
        )

    qdrant.upsert(
        collection_name="documents",
        points=points
    )

    return {
        "status": "pdf uploaded",
        "filename": file.filename,
        "chunks": len(chunks)
    }