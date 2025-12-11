from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
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

# ================
# LAZY LOAD MODEL
# ================
model = None

def get_model():
    global model
    if model is None:
        print("⚡ Loading embedding model (MiniLM-L3-v2)...")
        model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device="cpu")
    return model

# Load FAISS + metadata
index = faiss.read_index("faiss.index")
metadata = json.load(open("metadata.json", "r"))

# Log file
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

    # Lazy load the model here
    embedder = get_model()

    # Embed question
    q_vec = embedder.encode([question]).astype("float32")

    # FAISS search
    distances, ids = index.search(np.array(q_vec), k=3)

    matched_refs = [metadata[i] for i in ids[0]]

    # Simple answer logic
    answer = matched_refs[0]["text"] if matched_refs[0]["text"] else matched_refs[1]["text"]

    # Log it
    log_interaction(question, answer, matched_refs)

    return {
        "question": question,
        "answer": answer,
        "reference": matched_refs
    }


@app.get("/", response_class=HTMLResponse)
def validation():
    url = "/static/dashboard.html"
    html_for_link = f"""
    <html>
    <head><title>Base Page</title></head>
    <body>
    <div>
        <button onclick="window.location.href='{url}'">Dashboard</button>
    </div>
    </body>
    </html>
    """
    return html_for_link


# Analytics API (unchanged)
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
                continue
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
    return {"avg": statistics.mean(lengths), "min": min(lengths), "max": max(lengths)}


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


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)


