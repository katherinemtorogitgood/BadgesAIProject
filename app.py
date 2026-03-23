from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
import ollama
import re
import sys

DB_DIR = "./chroma_db"
COLLECTION_NAME = "documents"
CHAT_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

def embed(text: str):
    res = ollama.embed(model=EMBED_MODEL, input=text)
    return res["embeddings"][0]

def retrieve(collection, question: str, n_results: int):
    query_embedding = embed(question)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

def answer_question(question: str, n_results: int = 10):
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return {"answer": "Collection not found. Run ingestion first."}

    total_chunks = collection.count()
    if total_chunks == 0:
        return {"answer": "Collection is empty. Run ingestion first."}

    initial_k = min(max(n_results, 10), total_chunks)
    results = retrieve(collection, question, initial_k)

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    distance_threshold = 0.52
    max_context_chars = 6000

    candidates = []
    question_terms = set(re.findall(r"\w+", question.lower()))

    for doc, meta, dist in zip(docs, metadatas, distances):
        if dist > distance_threshold:
            continue

        doc_terms = set(re.findall(r"\w+", doc.lower()))
        keyword_overlap = len(question_terms & doc_terms)

        candidates.append({
            "doc": doc,
            "meta": meta,
            "dist": dist,
            "keyword_overlap": keyword_overlap,
        })

    if not candidates:
        for doc, meta, dist in zip(docs[:5], metadatas[:5], distances[:5]):
            doc_terms = set(re.findall(r"\w+", doc.lower()))
            keyword_overlap = len(question_terms & doc_terms)
            candidates.append({
                "doc": doc,
                "meta": meta,
                "dist": dist,
                "keyword_overlap": keyword_overlap,
            })

    candidates.sort(key=lambda x: (-x["keyword_overlap"], x["dist"]))

    selected_chunks = []
    current_len = 0

    for item in candidates:
        meta = item["meta"]
        doc = item["doc"]
        dist = item["dist"]
        overlap = item["keyword_overlap"]

        chunk_text = (
            f"[Source: {meta.get('source_file', 'unknown')} | "
            f"Chunk {meta.get('chunk_index', '?') + 1}/{meta.get('total_chunks', '?')} | "
            f"Distance: {dist:.4f} | Overlap: {overlap}]\n"
            f"{doc}"
        )

        if current_len + len(chunk_text) > max_context_chars:
            continue

        selected_chunks.append(chunk_text)
        current_len += len(chunk_text)

    context = "\n\n".join(selected_chunks)

    prompt = f"""You are answering using only the provided context.
If the answer is not explicitly in the context, say: "Not found in provided context."

Context:
{context}

Question:
{question}

Answer clearly and mention which source/chunk you used."""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer": response["message"]["content"],
        "chunks_used": len(selected_chunks),
    }

@app.get("/")
def home():
    return {"message": "AI backend is running"}

@app.post("/ask")
def ask(request: QuestionRequest):
    return answer_question(request.question)