import os
import sys
import shutil
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.ingestion import load_document, prepare_chunks, store_chunks, list_uploaded_filenames
from backend.agent import run_agent

load_dotenv()

app = FastAPI(title="Financial Document Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Matches the formats frontend/app.py's file_uploader accepts, and that
# load_document() in ingestion.py knows how to read.
ALLOWED_EXTENSIONS = (".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt")

class QueryRequest(BaseModel):
    question: str
    conversation_history: list = []
    filename: Optional[str] = None
    session_id: Optional[str] = None

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Financial Document Analyst API is running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...) , session_id: str = Form(...)):
    try:
        if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Reading document: {file.filename}")
        pages = load_document(file_path)
        
        print(f"Preparing chunks from {len(pages)} pages...")
        chunks = prepare_chunks(pages, file_path, session_id)
        
        if len(chunks) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No readable text found in '{file.filename}'. "
                       f"The file may be blank, corrupted, or contain only images/scans that couldn't be processed."
            )
        
        print(f"Storing {len(chunks)} chunks in Chroma...")
        store_chunks(chunks)
        
        return {
            "status": "success",
            "message": f"Document '{file.filename}' ingested successfully",
            "pages": len(pages),
            "chunks": len(chunks)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_document(request: QueryRequest):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        result = run_agent(
            user_question=request.question,
            conversation_history=request.conversation_history,
            selected_filename=request.filename,
            session_id=request.session_id
        )
        
        return {
            "status": "success",
            "answer": result["answer"],
            "reasoning_steps": result["reasoning_steps"],
            "conversation_history": result["conversation_history"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("=== FULL ERROR ===")
        print(error_details)
        print("=== END ERROR ===")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def get_documents(session_id: str):
    try:
        filenames = list_uploaded_filenames(session_id=session_id)
        return {"filenames": filenames}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))  # Render sets PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
