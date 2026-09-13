import os
import re
import uuid
import base64
import fitz
import docx
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Load environment variables
load_dotenv()

# Lazy-load Groq client
def get_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

def ocr_page_with_groq(page):
    """Convert a scanned PDF page to text using Groq vision model."""
    try:
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        client = get_groq_client()
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        },
                        {
                            "type": "text",
                            "text": "Extract all text from this document image. Return only the extracted text, nothing else."
                        }
                    ]
                }
            ]
        )
        raw_text = response.choices[0].message.content

        # Clean up <think> tags
        raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
        raw_text = re.sub(r'<think>.*', '', raw_text, flags=re.DOTALL)

        return raw_text.strip()
    except Exception as e:
        print(f"OCR failed for page: {e}")
        return ""

# Lazy-load Gemini client for embeddings
def get_embedding_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embeddings(texts):
    """texts can be a single string or a list of strings. Always returns a list of vectors."""
    client = get_embedding_client()
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )
    return [e.values for e in result.embeddings]

# ---------------- Qdrant Cloud setup ----------------
QDRANT_COLLECTION = "financial_docs"
EMBEDDING_SIZE = 3072  # gemini-embedding-001 default dimension

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

def ensure_collection():
    """Create the collection once if it doesn't already exist. Never deletes it."""
    if not qdrant.collection_exists(QDRANT_COLLECTION):
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
        )

ensure_collection()

# ---------------- PDF Reading ----------------
def read_pdf(file_path):
    import concurrent.futures
    results = []
    doc = fitz.open(file_path)
    pages_to_ocr = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            results.append((page_num + 1, text))
        else:
            pages_to_ocr.append((page_num + 1, page))

    if pages_to_ocr:
        print(f"Running OCR on {len(pages_to_ocr)} scanned pages...")
        def ocr_single_page(page_data):
            page_num, page = page_data
            text = ocr_page_with_groq(page)
            time.sleep(4)
            return (page_num, text)

        # Keep concurrency low to save memory
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            ocr_results = list(executor.map(ocr_single_page, pages_to_ocr))

        for page_num, text in ocr_results:
            if text.strip():
                results.append((page_num, text))

    results.sort(key=lambda x: x[0])
    doc.close()
    return results

def read_docx(file_path):
    doc = docx.Document(file_path)
    return [(i + 1, para.text) for i, para in enumerate(doc.paragraphs) if para.text.strip()]

def read_excel(file_path):
    xl = pd.ExcelFile(file_path)
    results = []
    for i, sheet_name in enumerate(xl.sheet_names):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        text = f"Sheet: {sheet_name}\n{df.to_string()}"
        results.append((i + 1, text))
    return results

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return [(1, df.to_string())]

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [(1, f.read())]

def load_document(file_path):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return read_pdf(file_path)
    elif ext == ".docx":
        return read_docx(file_path)
    elif ext in [".xlsx", ".xls"]:
        return read_excel(file_path)
    elif ext == ".csv":
        return read_csv(file_path)
    elif ext == ".txt":
        return read_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ---------------- Text Chunking ----------------
def chunk_text(text, chunk_size=200, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def prepare_chunks(pages, filename):
    doc_id = str(uuid.uuid4())
    all_chunks = []
    for page_num, text in pages:
        for chunk in chunk_text(text):
            all_chunks.append({
                "chunk_text": chunk,
                "page_number": page_num,
                "filename": os.path.basename(filename),
                "doc_id": doc_id
            })
    return all_chunks

# ---------------- Qdrant Storage ----------------
def store_chunks(chunks):
    """Adds new chunks to the collection WITHOUT deleting existing documents.
    Multiple uploaded documents can now coexist."""
    ensure_collection()

    texts = [c["chunk_text"] for c in chunks]
    embeddings = get_embeddings(texts)

    points = []
    for i, chunk in enumerate(chunks):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={
                "chunk_text": chunk["chunk_text"],
                "page_number": chunk["page_number"],
                "filename": chunk["filename"],
                "doc_id": chunk["doc_id"],
            }
        ))

    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)

def query_collection(query_text, top_k=3):
    """Returns results in the same documents/metadatas shape the rest of
    the app already expects (matches the old ChromaDB return format)."""
    query_embedding = get_embeddings([query_text])[0]

    results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        limit=top_k,
    )

    documents = [point.payload["chunk_text"] for point in results.points]
    metadatas = [
        {
            "page_number": point.payload["page_number"],
            "filename": point.payload["filename"],
            "doc_id": point.payload["doc_id"],
        }
        for point in results.points
    ]

    return {"documents": [documents], "metadatas": [metadatas]}