import os
import traceback
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

print("Testing Groq connection...")
print(f"API key loaded: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")

try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "say hi"}]
    )
    print("SUCCESS:", response.choices[0].message.content)
except Exception as e:
    print("\n=== FULL TRACEBACK ===")
    traceback.print_exc()
    print("\n=== UNDERLYING CAUSE ===")
    print(repr(e.__cause__))
    print("\n=== EXCEPTION TYPE ===")
    print(type(e).__name__)