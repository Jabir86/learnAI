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

# 💡 FIX 1: Add the missing import
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    # This happens if 'sentence-transformers' and 'torch' are missing in requirements.txt
    print("CRITICAL: 'sentence_transformers' not found. Please check requirements.txt.")
    SentenceTransformer = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ==================================================
# 💡 FIX 2: Load Model at Startup to prevent timeouts 
# ==================================================
model = None
if SentenceTransformer:
    try:
        print("⚡ Loading embedding model (MiniLM-L3-v2) at startup...")
        # Note: If this fails, it's almost always an Out-of-Memory (OOM) error.
        model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device="cpu")
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load SentenceTransformer model! This is often an OOM error on small servers. Error: {e}")
        model = None

# Load FAISS + metadata
try:
    index = faiss.read_index("faiss.index")
    metadata = json.load(open("metadata.json", "r"))
except Exception as e:
    print(f"❌ Failed to load FAISS index or metadata! Error: {e}")
    index = None
    metadata = {}

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

    # 💡 FIX 3: Check if the model failed to load during startup
    if model is None or index is None:
        return {
            "question": question,
            "answer": "Error: The AI model or FAISS index failed to load on the server. Check the deployment logs for an Out-of-Memory (OOM) error or missing files.",
            "reference": []
        }

    # Embed question
    q_vec = model.encode([question]).astype("float32")

    # FAISS search
    distances, ids = index.search(np.array(q_vec), k=3)

    matched_refs = [metadata[str(i)] for i in ids[0]] # ensure keys are strings if metadata keys were strings

    # Simple answer logic
    # Added a check to prevent IndexError if matched_refs is empty
    answer = "No relevant documents found."
    if matched_refs:
        # Prioritize the first match
        answer = matched_refs[0].get("text") or "No relevant text in the top match."
        
        # Fallback to the second match if the first one is empty
        if not matched_refs[0].get("text") and len(matched_refs) > 1:
             answer = matched_refs[1].get("text") or "No relevant text in the second match."


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
        <p>Your service is running.</p>
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

    # NOTE: This will fail if LOG_FILE is missing
    with open(LOG_FILE, "r") as f:
        for line in f:
            entry = json.loads(line)
            # Assuming 'titles' in the log entry is a list of strings
            for cid in entry.get("titles", []): 
                counter[cid] += 1
    return counter.most_common(20)


@app.get("/api/analytics/answer_length")
def answer_length():
    lengths = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            entry = json.loads(line)
            lengths.append(len(entry["answer"]))
    
    if not lengths:
        return {"avg": 0, "min": 0, "max": 0}

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
