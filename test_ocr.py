import sys
import time
sys.path.insert(0, '.')
from backend.ingestion import read_pdf

print("Starting OCR test with threading...")
start = time.time()

pages = read_pdf('uploads/26.pdf')

end = time.time()
print(f"\nTotal pages extracted: {len(pages)}")
print(f"Time taken: {round(end - start, 2)} seconds")

for page_num, text in pages[:2]:
    print(f"\n--- Page {page_num} ---")
    print(text[:300])