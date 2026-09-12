import os
import re
import uuid
import base64
import fitz
import docx
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from groq import Groq

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

# Lazy-load embedding model
from sentence_transformers import SentenceTransformer
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./vector_store")
collection = chroma_client.get_or_create_collection("financial_docs")

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

# ---------------- ChromaDB Storage ----------------
def store_chunks(chunks):
    global collection
    chroma_client.delete_collection("financial_docs")
    collection = chroma_client.get_or_create_collection("financial_docs")

    texts = [c["chunk_text"] for c in chunks]
    model = get_model()
    embeddings = model.encode(texts).tolist()

    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"{chunk['doc_id']}_{chunk['page_number']}_{i}"],
            documents=[chunk["chunk_text"]],
            embeddings=[embeddings[i]],
            metadatas=[{
                "page_number": chunk["page_number"],
                "filename": chunk["filename"],
                "doc_id": chunk["doc_id"]
            }]
        )

def query_collection(query_text, top_k=3):
    model = get_model()
    query_embedding = model.encode([query_text]).tolist()
    return collection.query(query_embeddings=query_embedding, n_results=top_k)
