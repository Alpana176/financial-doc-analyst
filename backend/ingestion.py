import re
from time import time

import fitz
import chromadb
import os
import uuid
import docx
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import base64
from groq import Groq

# Load environment variables
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ocr_page_with_groq(page):
    """Convert a scanned PDF page to text using Groq vision model."""
    try:
        # Render page as image
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        
        # Convert to base64
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # Send to Groq vision model
        response = groq_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
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

        # Remove thinking tags if present
        import re
        # Remove thinking tags and everything between them
        raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
        # Also remove any remaining think tags
        raw_text = re.sub(r'<think>.*', '', raw_text, flags=re.DOTALL)

        return raw_text.strip()
    except Exception as e:
        print(f"OCR failed for page: {e}")
        return ""

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./vector_store")
collection = chroma_client.get_or_create_collection("financial_docs")

# ---------------- PDF Reading ----------------
def read_pdf(file_path):
    import concurrent.futures
    
    results = []
    doc = fitz.open(file_path)
    pages_to_ocr = []
    
    # First pass — extract text from normal pages
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        if text.strip():
            results.append((page_num + 1, text))
        else:
            # Mark for OCR
            pages_to_ocr.append((page_num + 1, page))
    
    # Second pass — OCR all scanned pages in parallel
    if pages_to_ocr:
        print(f"Running OCR on {len(pages_to_ocr)} scanned pages in parallel...")
        
        def ocr_single_page(page_data):
            import time
            page_num, page = page_data
            time.sleep(1)  # small delay to avoid rate limit
            text = ocr_page_with_groq(page)
            return (page_num, text)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            ocr_results = list(executor.map(ocr_single_page, pages_to_ocr))
        
        for page_num, text in ocr_results:
            if text.strip():
                results.append((page_num, text))
    
    # Sort by page number
    results.sort(key=lambda x: x[0])
    doc.close()
    return results

def read_docx(file_path):
    doc = docx.Document(file_path)
    results = []
    for page_num, para in enumerate(doc.paragraphs):
        if para.text.strip():
            results.append((page_num + 1, para.text))
    return results

def read_excel(file_path):
    results = []
    xl = pd.ExcelFile(file_path)
    for sheet_num, sheet_name in enumerate(xl.sheet_names):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        text = f"Sheet: {sheet_name}\n{df.to_string()}"
        results.append((sheet_num + 1, text))
    return results

def read_csv(file_path):
    df = pd.read_csv(file_path)
    text = df.to_string()
    return [(1, text)]

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [(1, text)]

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
        chunks = chunk_text(text)
        for chunk in chunks:
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

    # Reset the collection so only the current document's chunks remain.
    # Without this, every new upload just piles on top of old documents,
    # so stale data from previous uploads leaks into new answers.
    chroma_client.delete_collection("financial_docs")
    collection = chroma_client.get_or_create_collection("financial_docs")

    texts = [chunk["chunk_text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts).tolist()

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
    query_embedding = embedding_model.encode([query_text]).tolist()
    return collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

# ---------------- Main ----------------
if __name__ == "__main__":
    file_path = "uploads/tata-motor-IAR-2024-25.pdf"

    print("Reading PDF...")
    pages = read_pdf(file_path)
    print(f"Total pages: {len(pages)}")

    print("Preparing chunks...")
    chunks = prepare_chunks(pages, file_path)
    print(f"Total chunks: {len(chunks)}")

    print("Storing in Chroma...")
    store_chunks(chunks)
    print("Done! PDF ingested successfully.")