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
    allow_origins = ["*"],
    allow_credentials =True,
    allow_methods = ["*"],
    allow_headers =["*"]
)

app.mount("/static",StaticFiles(directory="static"), name ="static")

# Load Embedding model
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Load faiss index
index = faiss.read_index("faiss.index")

# Reference metadata.json contains reference texts/ files
# references = json.load(open("corpus.json", "r"))
metadata = json.load(open("metadata.json", "r"))
# Logger file
LOG_FILE = "chat_logs.jsonl"

# Conversation History
conversation_history = []
MAX_HISTORY = 4

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

# ---------------------------------------------
# Generate Prompt
# ---------------------------------------------


# ---------------------------------------------
# Chat Conversation (Request a Question)
# ---------------------------------------------
@app.post("/api/chat")
async def chat(req:ChatRequest):
    global conversation_history

    question = req.question
    
    # Embed question
    q_vec = model.encode([question])
    q_vec = np.array(q_vec).astype("float32")

    # Search FAISS
    distances, ids = index.search(q_vec, k=3)

    # Get reference texts
    
    matched_refs = [metadata[i] for i in ids[0]]

    # Generate simple answer (could replace with LLM)
    answer = matched_refs[0]["text"] if len(matched_refs[0]["text"]) > 2  else matched_refs[1]["text"] if len(matched_refs[1]["text"]) > 2 else matched_refs[2]["text"]
    # results = []
    # for i, score in zip(ids, distances):
    #     entry = metadata[i]
    #     results.append({
    #         "id": entry["id"],
    #         "title": entry["title"],
    #         "path": entry["path"],
    #         "text": entry["text"],
    #         "full_section": entry["aggregated_content"]
    #     })
    
    # Log Q/A
    log_interaction(question, answer, matched_refs)
    
    # if str(answer) != "":
    #     answer =  "Could not understand! Try Again!!"
    #     matched_refs[0]['path'] = ""
    print( {"question": question,
        "answer": answer,
        "reference": matched_refs})
    return {
        "question": question,
        "answer": answer,
        "reference": matched_refs
    }


# ----------------------------------------------
# Analytics API Calls
# ----------------------------------------------
@app.get("/api/analytics/daily_count")
def daily_count():
    from collections import Counter
    counter = Counter()
    
    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                date = entry["timestamp"][:10]
                counter[date] += 1
            except Exception as e:
                print("JSON parse error:", e, "-->", line)
                continue
    
    return dict(counter)

@app.get("/", response_class=HTMLResponse)
def validation():
    url = "http://192.168.5.56:8000/static/dashboard.html"
    html_for_link = f"""
    <html>
    <head>
    <title>Base Page</title>
    </head>
    <body>
    <div>
    <button src={url} onclick="window.location.href='http://192.168.5.56:8000/static/dashboard.html'">Dashboard</button>
    </div>
    </body>
    </html>        
"""
    return html_for_link


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
        "min":min(lengths),
        "max":max(lengths)
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)