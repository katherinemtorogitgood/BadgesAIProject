from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
import ollama
import re

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
        return {"answer": "Collection not found. Run ingestion first.", "sources": []}

    total_chunks = collection.count()
    if total_chunks == 0:
        return {"answer": "Collection is empty. Run ingestion first.", "sources": []}

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

        candidates.append(
            {
                "doc": doc,
                "meta": meta,
                "dist": dist,
                "keyword_overlap": keyword_overlap,
            }
        )

    if not candidates:
        for doc, meta, dist in zip(docs[:5], metadatas[:5], distances[:5]):
            doc_terms = set(re.findall(r"\w+", doc.lower()))
            keyword_overlap = len(question_terms & doc_terms)

            candidates.append(
                {
                    "doc": doc,
                    "meta": meta,
                    "dist": dist,
                    "keyword_overlap": keyword_overlap,
                }
            )

    candidates.sort(key=lambda x: (-x["keyword_overlap"], x["dist"]))

    selected_chunks = []
    used_sources = []
    current_len = 0

    for item in candidates:
        meta = item["meta"]
        doc = item["doc"]
        source_file = meta.get("source_file", "unknown")

        if current_len + len(doc) > max_context_chars:
            continue

        selected_chunks.append(doc)
        current_len += len(doc)

        if source_file not in used_sources:
            used_sources.append(source_file)

    if not selected_chunks:
        return {"answer": "Not found in provided context.", "sources": []}

    context = "\n\n".join(selected_chunks)

    lower_question = question.lower()
    wants_elaboration = any(
        phrase in lower_question
        for phrase in [
            "elaborate",
            "explain",
            "what does that mean",
            "more detail",
            "in simpler terms",
        ]
    )

    if wants_elaboration:
        prompt = f"""You are answering using only the provided context.
If the answer is not explicitly in the context, say exactly: Not found in provided context.

Context:
{context}

Question:
{question}

Give a clear answer in 2 to 4 sentences using only the facts stated in the context.
Do not add unsupported interpretation.
Do not mention chunk numbers, distances, overlap scores, retrieval details, or internal metadata.
Do not include the source in the answer text."""
    else:
        prompt = f"""You are answering using only the provided context.
If the answer is not explicitly in the context, say exactly: Not found in provided context.

Context:
{context}

Question:
{question}

Answer with an exact quote from the context whenever the answer is present.
Keep it brief.
Do not paraphrase unless needed for grammar.
Do not add commentary or interpretation.
Do not mention chunk numbers, distances, overlap scores, retrieval details, or internal metadata.
Do not include the source in the answer text."""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    answer_text = response["message"]["content"].strip()

    if answer_text != "Not found in provided context." and used_sources:
        answer_text = f"{answer_text} (Source: {', '.join(used_sources[:3])})"

    return {
        "answer": answer_text,
        "sources": used_sources[:3],
        "chunks_used": len(selected_chunks),
    }


@app.get("/")
def home():
    return {"message": "AI backend is running"}


@app.post("/ask")
def ask(request: QuestionRequest):
    return answer_question(request.question)