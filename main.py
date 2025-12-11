from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import fastembed
from fastembed import TextEmbedding
import faiss
import numpy as np
import json
import uuid
import datetime
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ================
# Load tiny embedder (NO TORCH)
# ================
print("🔥 Loading FastEmbed MiniLM-L6 embedder...")
embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("✅ FastEmbed ready!")

# Load FAISS + metadata
index = faiss.read_index("faiss.index")
metadata = json.load(open("metadata.json", "r"))

LOG_FILE = "chat_logs.jsonl"

class ChatRequest(BaseModel):
    question: str

def log(q, a, ref):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "question": q,
            "answer": a,
            "reference": ref
        }) + "\n")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    q = req.question
    
    # Embed text
    vec = list(embedder.embed([q]))[0].astype("float32").reshape(1, -1)
    
    # Search
    distances, ids = index.search(vec, k=3)
    refs = [metadata[i] for i in ids[0]]

    answer = refs[0]["text"]

    log(q, answer, refs)

    return {
        "answer": answer,
        "reference": refs
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <body>
        <h2>LearnAI Chatbot</h2>
        <button onclick="location.href='/static/dashboard.html'">Open Chat</button>
      </body>
    </html>
    """

