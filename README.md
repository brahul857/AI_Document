# Personal RAG Chatbot (Practice Project)

A beginner-friendly Retrieval-Augmented Generation (RAG) chatbot. Drop your
own text files into `data/`, build an index, and ask questions about them.

## How it works

1. **Ingest** (`ingest.py`) — reads every `.txt` file in `data/`, splits it
  into overlapping chunks, embeds each chunk with Chroma's local ONNX
  embedding model, and stores the vectors in a local
   Chroma database (`chroma_db/`).
2. **Chat** (`chat.py`) — takes your question, embeds it the same way,
   retrieves the most relevant chunks from Chroma, stuffs them into a
   prompt as context, and calls the OpenAI API to generate an answer
   grounded in your documents.
3. **HTML chat** (`app.py` + `chat.html`) — serves a browser chat page while
  keeping the OpenAI API key on the Python server.

This is the core pattern behind almost every "chat with your docs" product.

## Setup

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"    # on Windows: set OPENAI_API_KEY=your-key-here
```

Get an API key at https://platform.openai.com/api-keys if you don't have one.

By default this uses `gpt-4o-mini` (cheap and fast). To use a different
model, change `OPENAI_MODEL` at the top of `chat.py`.

## Usage

```bash
# 1. Add your own .txt files to the data/ folder
#    (a sample file is already included so you can test immediately)

# 2. Build the index (run this again whenever you change data/)
python ingest.py

# 3. Start chatting — pick one:

# Option A: terminal chat
python chat.py

# Option B: browser-based HTML UI
python app.py
```

The HTML option opens a chat page at `http://127.0.0.1:5000` with the same
retrieval and answer logic as the terminal version, plus source filenames.

Type `exit` or `quit` to leave the terminal chat loop.

## What to try next (once this works)

- Swap in your own documents (class notes, a book, project docs) in `data/`.
- Try PDF or `.md` files by adding a loader for them in `ingest.py`.
- Tune `CHUNK_SIZE` / `CHUNK_OVERLAP` in `ingest.py` and see how retrieval
  quality changes.
- Print out *which* chunks were retrieved in `chat.py` so you can see why
  the model answered the way it did.
- Add a simple web UI with Streamlit (`pip install streamlit`).
- Swap the local embedding model for the OpenAI embeddings API
  (`text-embedding-3-small`).

## Project structure

```
rag-chatbot/
├── data/              # put your .txt files here
│   └── sample.txt     # example document to test with
├── ingest.py          # builds the vector index
├── chat.py            # interactive Q&A loop (terminal)
├── app.py             # Flask server for the browser chat
├── chat.html          # browser chat interface
├── requirements.txt
└── README.md
```
