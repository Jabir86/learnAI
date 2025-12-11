from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastembed import TextEmbedding
import numpy as np
import faiss
import json
import uuid
import datetime
import statistics
import uvicorn
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================
# Load FastEmbed model (NO TORCH)
# ============================
print("🔥 Loading FastEmbed MiniLM-L6 model...")
embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("✅ FastEmbed loaded!")

# Load FAISS index + metadata
index = faiss.read_index("faiss.index")
metadata = json.load(open("metadata.json", "r"))

LOG_FILE = "chat_logs.jsonl"


class ChatRequest(BaseModel):
    question: str


def log_interaction(question, answer, refs):
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "titles": [r["title"] for r in refs],
        "reference": refs
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ============================
# Chat API
# ============================
@app.post("/api/chat")
async def chat(req: ChatRequest):
    q = req.question

    # Embed using FastEmbed
    q_vec = list(embedder.embed([q]))[0].astype("float32").reshape(1, -1)

    # Search FAISS
    distances, ids = index.search(q_vec, k=3)
    refs = [metadata[i] for i in ids[0]]

    answer = refs[0]["text"]

    # Log
    log_interaction(q, answer, refs)

    return {
        "answer": answer,
        "reference": refs
    }


# ============================
# Dashboard route
# ============================
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <html>
    <body style='font-family: Arial; padding: 40px;'>
        <h2>LearnEngg AI Dashboard</h2>
        <button onclick="window.location.href='/static/chat.html'"
                style="padding:12px 20px; font-size:18px;">
            Open Chat
        </button>
    </body>
    </html>
    """


# ============================
# Analytics routes
# ============================
@app.get("/api/analytics/daily_count")
def daily_count():
    from collections import Counter
    counter = Counter()
    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                date = entry["timestamp"][:10]
                counter[date] += 1
            except:
                pass
    return dict(counter)


@app.get("/api/analytics/top_chunks")
def top_chunks():
    from collections import Counter
    counter = Counter()
    with open(LOG_FILE, "r") as f:
        for line in f:
            entry = json.loads(line)
            for cid in entry["titles"]:
                counter[cid] += 1
    return counter.most_common(20)


@app.get("/api/analytics/answer_length")
def answer_length():
    lengths = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            entry = json.loads(line)
            lengths.append(len(entry["answer"]))
    return {
        "avg": statistics.mean(lengths),
        "min": min(lengths),
        "max": max(lengths)
    }


@app.get("/api/analytics/top_questions")
def top_questions():
    from collections import Counter
    counter = Counter()
    with open(LOG_FILE, "r") as f:
        for line in f:
            entry = json.loads(line)
            q = entry["question"].strip().lower()
            counter[q] += 1
    return counter.most_common(20)


# ============================
# Run server (correct port)
# ============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
