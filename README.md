# financial-doc-analyst

Short description: a backend API (backend/) and frontend (frontend/) for analyzing financial documents.

## Quick start (local)
1. Create and activate a Python virtualenv.
2. Install backend deps: pip install -r backend/requirements.txt
3. Run backend: uvicorn backend.main:app --reload
4. Run frontend: follow frontend README

## Notes
- Do not commit real documents or secrets.
- Use .env for local secrets; see .env.example.
