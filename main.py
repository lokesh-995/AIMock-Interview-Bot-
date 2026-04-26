from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import random

app = FastAPI()

# -------- CORS --------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- GROQ CLIENT --------
client = OpenAI(
    api_key="your api key ",
    base_url="https://api.groq.com/openai/v1"
)

# -------- EMBEDDING MODEL --------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
embeddings = []

# -------- REQUEST MODEL --------
class AnswerRequest(BaseModel):
    question: str
    answer: str


# -------- PDF TEXT EXTRACTION --------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


# -------- EMBEDDING --------
def get_embedding(text):
    return np.array(embed_model.encode(text), dtype="float32")


# -------- BUILD FAISS INDEX --------
def build_index():
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index


# -------- RETRIEVE CONTEXT (RAG) --------
def retrieve_context(query, k=3):
    if not embeddings:
        return "No resume data available"

    index = build_index()
    query_vec = get_embedding(query)

    D, I = index.search(np.array([query_vec]), k)
    return "\n".join([documents[i] for i in I[0]])


# -------- 1. UPLOAD RESUME --------
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    global documents, embeddings
    documents = []
    embeddings = []

    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file.file)
    else:
        content = await file.read()
        text = content.decode("utf-8")

    if not text.strip():
        return {"error": "No text extracted from resume"}

    # Better chunking (paragraph-based)
    chunks = text.split("\n")

    for chunk in chunks:
        chunk = chunk.strip()
        if chunk and len(chunk) > 10:
            documents.append(chunk)
            embeddings.append(get_embedding(chunk))

    return {"message": "Resume uploaded", "chunks": len(documents)}


# -------- 2. GENERATE QUESTIONS --------
@app.get("/generate-question")
def generate_question():

    if not documents:
        return {"error": "Upload resume first"}

    # 🔥 Random retrieval for variety
    queries = [
        "skills",
        "projects",
        "experience",
        "technologies used",
        "backend development",
        "machine learning",
        "spring boot project",
        "react frontend"
    ]

    context = retrieve_context(random.choice(queries))

    prompt = f"""
    You are a technical interviewer.

    Candidate Resume Context:
    {context}

    TASK:
    Generate 5 DIFFERENT interview questions.

    RULES:
    - Questions must be based ONLY on the given context
    - Focus on skills, projects, technologies
    - Avoid generic questions
    - All questions must be different

    Output format:
    1.
    2.
    3.
    4.
    5.
    """

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"questions": res.choices[0].message.content}


# -------- 3. EVALUATE ANSWER --------
@app.post("/evaluate")
def evaluate(data: AnswerRequest):

    prompt = f"""
    Evaluate this answer.

    Question: {data.question}
    Answer: {data.answer}

    Return STRICT JSON:
    {{
      "score": number (1-10),
      "strengths": "...",
      "improvements": "..."
    }}
    """

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"result": res.choices[0].message.content}