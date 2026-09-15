import os
from dotenv import load_dotenv
from groq import Groq
import yfinance as yf
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.ingestion import query_collection

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def search_documents(query, top_k=3, filename=None,session_id=None):
    results = query_collection(query, top_k=top_k, filename=filename, session_id=session_id)

    
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    
    formatted = []
    for doc, meta in zip(documents, metadatas):
        formatted.append({
            "text": doc,
            "page_number": meta["page_number"],
            "filename": meta["filename"]
        })
    
    return formatted
def extract_financial_data(topic, filename=None, session_id=None):
    search_results = search_documents(topic, top_k=5, filename=filename, session_id=session_id)
    
    context = "\n\n".join([r["text"] for r in search_results])
    
    prompt = f"""
    You are a financial data extractor.
    Extract all financial figures related to: {topic}
    From this context: {context}
    
    Return ONLY a valid JSON object. No explanation, no markdown, no extra text.
    Example format:
    {{"metric_name": "value", "metric_name2": "value2"}}
    """
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw = response.choices[0].message.content.strip()
    
    try:
        data = json.loads(raw)
        return data
    except:
        return {"raw_response": raw}

def get_market_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info
        
        return {
            "ticker": ticker,
            "current_price": round(info.last_price, 2),
            "currency": info.currency,
            "market_cap": info.market_cap
        }
    except Exception as e:
        return {"error": f"Could not fetch data for {ticker}: {str(e)}"}

def calculate_financial_metric(metric_type, **kwargs):
    if metric_type == "compound_interest":
        principal = kwargs["principal"]
        rate = kwargs["rate"]
        years = kwargs["years"]
        amount = principal * (1 + rate/100) ** years
        interest = amount - principal
        return {
            "principal": principal,
            "rate": f"{rate}%",
            "years": years,
            "final_amount": round(amount, 2),
            "interest_earned": round(interest, 2)
        }
    
    elif metric_type == "emi":
        principal = kwargs["principal"]
        annual_rate = kwargs["rate"]
        years = kwargs["years"]
        
        monthly_rate = annual_rate / (12 * 100)
        months = years * 12
        
        emi = principal * monthly_rate * (1 + monthly_rate)**months / ((1 + monthly_rate)**months - 1)
        total_payment = emi * months
        total_interest = total_payment - principal
        
        return {
            "principal": principal,
            "annual_rate": f"{annual_rate}%",
            "tenure": f"{years} years",
            "monthly_emi": round(emi, 2),
            "total_interest": round(total_interest, 2),
            "total_payment": round(total_payment, 2)
        }
    
    elif metric_type == "cagr":
        initial = kwargs["initial"]
        final = kwargs["final"]
        years = kwargs["years"]
        cagr = ((final/initial) ** (1/years) - 1) * 100
        return {
            "initial_value": initial,
            "final_value": final,
            "years": years,
            "cagr": f"{round(cagr, 2)}%"
        }
    
    else:
        return {"error": f"Unknown metric type: {metric_type}"}



    