import traceback
from backend.ingestion import query_collection

try:
    result = query_collection("what is this doc about", top_k=3)
    print("SUCCESS")
    print(result)
except Exception as e:
    print("\n=== REAL ERROR ===")
    traceback.print_exc()