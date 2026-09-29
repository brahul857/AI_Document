"""Local web server for the document chat page."""

import os
from functools import lru_cache
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI

from chat import (
    COLLECTION_NAME,
    DB_DIR,
    OPENAI_MODEL,
    SYSTEM_PROMPT,
    TOP_K,
    build_prompt,
)

load_dotenv()

app = Flask(__name__, static_folder=None)
ROOT_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def get_resources():
    """Load the local embedding model and indexed Chroma collection."""
    if not os.path.isdir(ROOT_DIR / DB_DIR):
        raise RuntimeError("No index found. Run ingest.py first.")
    chroma_client = chromadb.PersistentClient(path=str(ROOT_DIR / DB_DIR))
    collection = chroma_client.get_collection(COLLECTION_NAME)
    return collection


@app.get("/")
def index():
    return send_from_directory(ROOT_DIR, "chat.html")


@app.post("/api/chat")
def chat_endpoint():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "").strip()
    if not question:
        return jsonify(error="Question is required."), 400
    if not os.environ.get("OPENAI_API_KEY"):
        return jsonify(error="OPENAI_API_KEY is not configured on the server."), 500

    try:
        collection = get_resources()
        results = collection.query(
            query_texts=[question],
            n_results=TOP_K,
        )
        retrieved_chunks = results["documents"][0]
        sources = sorted({meta["source"] for meta in results["metadatas"][0]})
        if not retrieved_chunks:
            return jsonify(answer="I couldn't find anything relevant in the documents.", sources=[])

        response = OpenAI().chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=10000,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, retrieved_chunks)},
            ],
        )
        return jsonify(answer=response.choices[0].message.content, sources=sources) # sources
    except Exception as error:
        app.logger.exception("Chat request failed")
        return jsonify(error=str(error)), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
