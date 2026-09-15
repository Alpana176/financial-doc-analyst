
import os
import uuid
import streamlit as st
import requests
import json


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Financial Document Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Financial Document Analyst")
st.markdown("Upload financial documents and get intelligent analysis powered by AI")

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

tab1, tab2, tab3 = st.tabs(["📄 Upload Documents", "💬 Ask Questions", "🧮 Calculator"])

with tab1:
    st.header("Upload Financial Documents")
    st.write("Supported formats: PDF, Word (.docx), Excel (.xlsx), CSV, Text (.txt)")
    
    uploaded_file = st.file_uploader(
        "Choose a financial document",
        type=["pdf", "docx", "xlsx", "xls", "csv", "txt"]
    )
    
    if uploaded_file is not None:
        if st.button("Upload and Process", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"session_id": st.session_state.session_id}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        st.info(f"📄 Pages processed: {result['pages']}")
                        st.info(f"🔍 Chunks created: {result['chunks']}")
                    else:
                        st.error(f"❌ Error: {response.json()['detail']}")
                        
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")

with tab2:
    st.header("Ask Questions About Your Documents")
    try:

        docs_response = requests.get(f"{API_URL}/documents", params={"session_id": st.session_state.session_id})
        available_docs = docs_response.json().get("filenames", []) if docs_response.status_code == 200 else []
    except Exception:
        available_docs = []

    doc_options = ["All documents"] + available_docs
    selected_doc = st.selectbox("Ask about:", doc_options)
    selected_filename = None if selected_doc == "All documents" else selected_doc
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            if message["role"] == "assistant" and "reasoning" in message:
                with st.expander("🔍 View Agent Reasoning"):
                    for step in message["reasoning"]:
                        st.write(f"🔧 Tool called: **{step['tool_called']}**")
                        st.write(f"📥 Arguments: {step['arguments']}")
                        st.divider()
    
    if prompt := st.chat_input("Ask anything about your financial documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    payload = {
                        "question": prompt,
                        "conversation_history": st.session_state.conversation_history,
                        "filename": selected_filename,
                        "session_id": st.session_state.session_id
                    }
                    response = requests.post(f"{API_URL}/query", json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result["answer"]
                        reasoning = result["reasoning_steps"]
                        st.session_state.conversation_history = result["conversation_history"]
                        
                        st.write(answer)
                        
                        if reasoning:
                            with st.expander("🔍 View Agent Reasoning"):
                                for step in reasoning:
                                    st.write(f"🔧 Tool called: **{step['tool_called']}**")
                                    st.write(f"📥 Arguments: {step['arguments']}")
                                    st.divider()
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "reasoning": reasoning
                        })
                    else:
                        st.error(f"❌ Error: {response.json()['detail']}")
                        
                except Exception as e:
                    st.error(f"❌ Connection error: Make sure FastAPI server is running. {str(e)}")
    
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()

with tab3:
    st.header("Financial Calculator")
    st.write("Calculate EMI, Compound Interest, and CAGR directly")
    
    calc_type = st.selectbox(
        "Select Calculation Type",
        ["EMI (Loan Calculator)", "Compound Interest", "CAGR"]
    )
    
    if calc_type == "EMI (Loan Calculator)":
        col1, col2, col3 = st.columns(3)
        with col1:
            principal = st.number_input("Loan Amount (₹)", min_value=0.0, value=500000.0)
        with col2:
            rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, value=10.5)
        with col3:
            years = st.number_input("Tenure (Years)", min_value=1, value=15)
        
        if st.button("Calculate EMI", type="primary"):
            question = f"Calculate EMI for principal {principal}, rate {rate}%, {years} years"
            response = requests.post(f"{API_URL}/query", json={"question": question, "conversation_history": []})
            if response.status_code == 200:
                st.success(response.json()["answer"])
    
    elif calc_type == "Compound Interest":
        col1, col2, col3 = st.columns(3)
        with col1:
            principal = st.number_input("Principal Amount (₹)", min_value=0.0, value=100000.0)
        with col2:
            rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, value=8.5)
        with col3:
            years = st.number_input("Time Period (Years)", min_value=1, value=5)
        
        if st.button("Calculate", type="primary"):
            question = f"Calculate compound interest for principal {principal}, rate {rate}%, {years} years"
            response = requests.post(f"{API_URL}/query", json={"question": question, "conversation_history": []})
            if response.status_code == 200:
                st.success(response.json()["answer"])
    
    elif calc_type == "CAGR":
        col1, col2, col3 = st.columns(3)
        with col1:
            initial = st.number_input("Initial Value (₹)", min_value=0.0, value=100000.0)
        with col2:
            final = st.number_input("Final Value (₹)", min_value=0.0, value=150000.0)
        with col3:
            years = st.number_input("Number of Years", min_value=1, value=5)
        
        if st.button("Calculate CAGR", type="primary"):
            question = f"Calculate CAGR with initial value {initial}, final value {final}, {years} years"
            response = requests.post(f"{API_URL}/query", json={"question": question, "conversation_history": []})
            if response.status_code == 200:
                st.success(response.json()["answer"])

