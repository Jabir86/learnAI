from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import statistics
import datetime
import uuid
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ===========================
# PRELOAD LIGHTWEIGHT MODEL
# ===========================
print("⚡ Loading MiniLM-L3-v2 model at startup...")
model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device="cpu")
print("✅ Model loaded!")

# Load FAISS index + metadata
index = faiss.read_index("faiss.index")
metadata = json.load(open("metadata.json", "r"))

LOG_FILE = "chat_logs.jsonl"

class ChatRequest(BaseModel):
    question: str

def log_interaction(question, answer, ref):
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "chunks_used": [r["id"] for r in ref],
        "titles": [r["title"] for r in ref],
        "reference": ref
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    question = req.question

    # Embed question
    q_vec = model.encode([question]).astype("float32")

    # FAISS search
    distances, ids = index.search(np.array(q_vec), k=3)

    matched_refs = [metadata[i] for i in ids[0]]

    # Simple answer
    answer = matched_refs[0]["text"] if matched_refs[0]["text"] else matched_refs[1]["text"]

    # Log result
    log_interaction(question, answer, matched_refs)

    return {
        "question": question,
        "answer": answer,
        "reference": matched_refs
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <body>
        <h2>LearnAI Chatbot</h2>
        <button onclick="window.location.href='/static/chat.html'">Open Chat</button>
    </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
