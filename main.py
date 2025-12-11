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

# FIX 1: Add the missing import
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("CRITICAL: 'sentence_transformers' not found. Check requirements.txt.")
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

# =================================================================
# FIX 2 & 3: Load Model at Startup & Use a Smaller Model (all-MiniLM-L6-v2)
# =================================================================
model = None
MODEL_NAME = "all-MiniLM-L6-v2" # This model is much smaller (approx 80MB)

if SentenceTransformer:
    try:
        print(f"⚡ Loading smaller embedding model ({MODEL_NAME}) at startup...")
        # Switched from paraphrase-MiniLM-L3-v2 to all-MiniLM-L6-v2 to save memory
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load SentenceTransformer model! Likely OOM or file error. Error: {e}")
        model = None

# Load FAISS + metadata
try:
    index = faiss.read_index("faiss.index")
    # Note: Using str(i) is a common fix if the keys in metadata are strings
    metadata_raw = json.load(open("metadata.json", "r"))
    # Convert keys to strings if necessary, to match FAISS index returns
    metadata = {str(k): v for k, v in metadata_raw.items()} 

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
        "chunks_used": [r.get("id", "N/A") for r in ref],
        "titles": [r.get("title", "N/A") for r in ref],
        "reference": ref
    }
    # Ensure log file exists before writing
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Warning: Could not write to log file. Error: {e}")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    question = req.question

    # Check if model/index failed to load due to OOM
    if model is None or index is None:
        return {
            "question": question,
            "answer": "Error: The AI model or FAISS index failed to load. Check server logs for Out-of-Memory (OOM) errors.",
            "reference": []
        }

    # Embed question
    q_vec = model.encode([question]).astype("float32")

    # FAISS search
    distances, ids = index.search(np.array(q_vec), k=3)

    # Get references using string keys
    matched_refs = [metadata.get(str(i)) for i in ids[0] if metadata.get(str(i)) is not None]

    # Simple answer logic
    answer = "No relevant documents found."
    if matched_refs:
        # Prioritize the first match
        answer = matched_refs[0].get("text")
        
        # Fallback to the second match if the first one is empty
        if not answer and len(matched_refs) > 1:
             answer = matched_refs[1].get("text")

    # Final check for empty answer
    if not answer:
        answer = "No relevant text found in the top documents."


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
        <p>Your service is running. Model: {MODEL_NAME if model else 'Failed to Load'}</p>
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

    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    date = entry["timestamp"][:10]
                    counter[date] += 1
                except:
                    continue
        return dict(counter)
    except FileNotFoundError:
        return {}


@app.get("/api/analytics/top_chunks")
def top_chunks():
    from collections import Counter
    counter = Counter()

    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                entry = json.loads(line)
                for title in entry.get("titles", []): 
                    counter[title] += 1
        return counter.most_common(20)
    except FileNotFoundError:
        return []


@app.get("/api/analytics/answer_length")
def answer_length():
    lengths = []
    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                entry = json.loads(line)
                lengths.append(len(entry["answer"]))
    except FileNotFoundError:
        pass
    
    if not lengths:
        return {"avg": 0, "min": 0, "max": 0}

    return {"avg": statistics.mean(lengths), "min": min(lengths), "max": max(lengths)}


@app.get("/api/analytics/top_questions")
def top_questions():
    from collections import Counter
    counter = Counter()

    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                entry = json.loads(line)
                q = entry["question"].strip().lower()
                counter[q] += 1
        return counter.most_common(20)
    except FileNotFoundError:
        return []


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
