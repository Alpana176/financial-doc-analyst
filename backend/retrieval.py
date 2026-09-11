import os
from dotenv import load_dotenv
from groq import Groq
from ingestion import query_collection

# Load environment variables
load_dotenv()

# ---------------- Groq Client ----------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_question(query_text, top_k=3):
    """Query ChromaDB and answer using Groq model."""
    results = query_collection(query_text, top_k=top_k)

    # Flatten list of lists
    context_chunks = [doc for docs in results["documents"] for doc in docs]
    context = "\n\n".join(context_chunks)

    prompt = f"""
    You are an assistant answering questions based on financial documents.
    Use the following context to answer the question truthfully.

    Context:
    {context}

    Question: {query_text}
    Answer:
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",  # ✅ updated model name
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# ---------------- Main ----------------
if __name__ == "__main__":
    question = "What is the total revenue of Tata Motors?"
    answer = answer_question(question, top_k=3)
    print(answer)
