# 📊 AI-Powered Financial Document Analyst

An intelligent document analysis system that lets users upload financial 
documents and get instant answers, structured data extraction, live market 
data, and financial calculations — powered by RAG pipelines and LLM agents.
🖼️ Preview

![Financial Document Analyst](images/your-image.png)


## 🚀 Live Demo
👉 Try the app 👉 Backend API docs

👉 [Try it live](https://financial-doc-analyst-5x7qrt4xi2nid5neffs44j.streamlit.app/)
🎯 What it does

Upload financial documents — PDFs, Word files, Excel sheets, CSVs, or plain text — and ask natural-language questions about them. An AI agent decides which tool to use for each question and returns a grounded answer, complete with page-level citations from your actual uploaded files.

📄 Multi-format ingestion — PDF, DOCX, XLSX, CSV, TXT, including OCR for scanned pages
💬 Conversational Q&A — ask anything about an uploaded document, with full conversation history
📚 Multi-document support — upload several files and pick exactly which one to query via a document selector
🧮 Financial calculator — EMI, compound interest, and CAGR calculations built in
📈 Live market data — fetch real-time stock prices by ticker
🔍 Transparent agent reasoning — see exactly which tool was called and why, for every answer

🧠 Tech Stack
Technology	Purpose
FastAPI	Backend REST API
Streamlit	Web UI
Groq (openai/gpt-oss-120b)	LLM for tool routing and answer generation
Groq Vision (qwen/qwen3.8-27b)	OCR for scanned PDF pages
Gemini (gemini-embedding-001)	Text embeddings for semantic search
Qdrant Cloud	Vector database for document storage and retrieval
PyMuPDF (fitz)	PDF text extraction
python-docx	Word document parsing
pandas / openpyxl	Excel and CSV parsing
yfinance	Live stock market data
Python	Core language

🏗️ Project Structure
financial-doc-analyst/
├── backend/
│   ├── main.py          # FastAPI app, upload/query/documents endpoints
│   ├── ingestion.py      # Document parsing, chunking, embeddings, Qdrant storage
│   ├── agent.py           # Tool-routing agent logic
│   └── tools.py           # search, extraction, market data, calculator tools
├── frontend/
│   └── app.py              # Streamlit UI
├── requirements.txt
└── .env.example
⚙️ How to run locally

1. Clone the repo
git clone https://github.com/Alpana176/financial-doc-analyst.git
cd financial-doc-analyst

2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Add your API keys

Create a .env file in the root folder:
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_URL=your_qdrant_cluster_url_here
QDRANT_API_KEY=your_qdrant_api_key_here

5. Run the backend
python backend/main.py

6. Run the frontend (in a separate terminal)
streamlit run frontend/app.py

🔑 Get your API keys
Groq: 👉 console.groq.com
Gemini: 👉 Google AI Studio
Qdrant Cloud: 👉 cloud.qdrant.io

📌 Features
✅ Multi-format document ingestion with OCR fallback
✅ Agentic tool routing (search, extraction, calculation, market data)
✅ Multi-document knowledge base with per-document filtering
✅ Financial calculator (EMI, compound interest, CAGR)
✅ Live stock market data lookup
✅ Transparent agent reasoning view
✅ Deployed as a live full-stack app (FastAPI backend + Streamlit frontend)

🙋‍♀️ Author

Alpana Choubey LinkedIn • GitHub

