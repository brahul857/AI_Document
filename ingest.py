"""
ingest.py — Reads text from supported files in data/, splits it into
overlapping chunks, embeds each chunk with Chroma's local ONNX model,
and stores the vectors in a local Chroma database.

Run this once at the start, and again any time you change files in data/.
"""

import os
from io import BytesIO
from html.parser import HTMLParser
from pathlib import Path

import chromadb
import pdf2image
import pytesseract
import xlrd
from PIL import Image
from openpyxl import load_workbook
from pypdf import PdfReader

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION_NAME = "documents"

def find_poppler_path():
    """Return the Poppler binary directory for Windows installs when PATH is not configured."""
    candidates = [
        r"C:\Users\Rahul Biswas\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin",
        r"C:\Program Files\Poppler\Library\bin",
        r"C:\Program Files (x86)\Poppler\Library\bin",
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "pdftoppm.exe")):
            return path
    return None

POPPLER_PATH = find_poppler_path()


def ocr_page(page, preferred_lang="ben"):
    """OCR a page, falling back to English if the preferred language pack is unavailable."""
    langs = [preferred_lang, "eng"] if preferred_lang != "eng" else ["eng"]
    last_error = None
    for lang in langs:
        try:
            return pytesseract.image_to_string(page, lang=lang)
        except Exception as exc:
            last_error = exc
    raise last_error


# --- Tunable parameters -----------------------------------------------
CHUNK_SIZE = 300       # characters per chunk
CHUNK_OVERLAP = 50     # characters of overlap between consecutive chunks
SUPPORTED_EXTENSIONS = {".xls", ".xlsx", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
# ------------------------------------------------------------------------


class TableParser(HTMLParser):
    """Extract rows and cells from HTML tables saved with an .xls extension."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self.current_row = []
        self.current_cell = []
        self.in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.current_row = []
        elif tag == "td":
            self.current_cell = []
            self.in_cell = True

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.in_cell:
            value = " ".join("".join(self.current_cell).split())
            self.current_row.append(value)
            self.in_cell = False
        elif tag == "tr" and self.current_row:
            self.rows.append(self.current_row)

def load_documents(data_dir):
    """Extract text from supported files and return a list of (filename, text)."""
    docs = []
    paths = sorted(Path(data_dir).iterdir()) if os.path.isdir(data_dir) else []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        print(path.suffix.lower())
        suffix = path.suffix.lower()
        print(suffix)

        if suffix == ".pdf":
            reader = PdfReader(str(path))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

            if not text.strip():
                try:
                    kwargs = {"dpi": 200}
                    if POPPLER_PATH:
                        kwargs["poppler_path"] = POPPLER_PATH
                    pages = pdf2image.convert_from_path(str(path), **kwargs)
                    text = "\n\n".join(
                        page_text
                        for page_text in (ocr_page(page, "ben") for page in pages)
                        if page_text.strip()
                    )
                except Exception:
                    text = ""

        elif suffix == ".xls":
            raw = path.read_bytes()
            if raw.lstrip().startswith(b"<"):
                parser = TableParser()
                parser.feed(raw.decode("utf-8", errors="replace"))
                text = "\n".join("\t".join(cell for cell in row if cell) for row in parser.rows)
            elif raw.startswith(b"PK"):
                workbook = load_workbook(BytesIO(raw), read_only=True, data_only=True)
                rows = []
                for worksheet in workbook.worksheets:
                    for row in worksheet.iter_rows(values_only=True):
                        values = [str(value) for value in row if value is not None]
                        if values:
                            rows.append("\t".join(values))
                workbook.close()
                text = "\n".join(rows)
            else:
                workbook = xlrd.open_workbook(path, on_demand=True)
                rows = []
                for worksheet in workbook.sheets():
                    for row in worksheet.get_rows():
                        values = [str(value) for value in row if value.value not in (None, "")]
                        if values:
                            rows.append("\t".join(values))
                workbook.release_resources()
                text = "\n".join(rows)
        elif suffix == ".xlsx":
            workbook = load_workbook(path, read_only=True, data_only=True)
            rows = []
            for worksheet in workbook.worksheets:
                for row in worksheet.iter_rows(values_only=True):
                    values = [str(value) for value in row if value is not None]
                    if values:
                        rows.append("\t".join(values))
            workbook.close()
            text = "\n".join(rows)
        else:
            text = pytesseract.image_to_string(Image.open(path))

        if text.strip():
            docs.append((path.name, text))
    return docs


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping fixed-size chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def main():
    docs = load_documents(DATA_DIR)
    if not docs:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        print(f"No supported files found in '{DATA_DIR}/'. Supported types: {supported}")
        return

    print(f"Loaded {len(docs)} document(s) from '{DATA_DIR}/'.")
    # initialize persist database in local disk
    client = chromadb.PersistentClient(path=DB_DIR)
    # Start fresh each time so re-running ingest.py doesn't duplicate chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # create or featch unique collection similer to database table
    collection = client.create_collection(COLLECTION_NAME)

    all_chunks, all_ids, all_metadatas = [], [], []
    chunk_counter = 0

    for filename, text in docs:
        chunks = chunk_text(text)
        print(f"  {filename}: {len(chunks)} chunk(s)")
        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_counter}")
            all_metadatas.append({"source": filename})
            chunk_counter += 1

    print(f"Embedding {len(all_chunks)} chunk(s)...")

    # Chroma embeds documents with the collection's embedding function and
    # persists the resulting vectors in the PersistentClient database.
    collection.add(
        ids=all_ids,
        documents=all_chunks,
        metadatas=all_metadatas,
    )

    stored = collection.get(ids=[all_ids[0]], include=["embeddings"])
    embedding_dimension = len(stored["embeddings"][0])
    print(
        f"Done. Indexed {len(all_chunks)} chunks with "
        f"{embedding_dimension}-dimensional embeddings into '{DB_DIR}/'."
    )
    print("Now run: python chat.py")


if __name__ == "__main__":
    main()
