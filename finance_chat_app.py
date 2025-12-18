import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yfinance as yf
import re
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# ==========================================
# 1. Configuration & Device Logic (M4/Windows)
# ==========================================
st.set_page_config(page_title="FinChat: AML Group Project", page_icon="📈", layout="wide")

@st.cache_resource
def get_device_map():
    """
    Detects the best available hardware.
    - MacOS M-series: Uses 'mps' (Metal Performance Shaders)
    - Windows w/ NVIDIA: Uses 'cuda'
    - Default: 'cpu'
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device_map()

# ==========================================
# 2. Model Loading
# ==========================================
@st.cache_resource
def load_model():
    model_name = "microsoft/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 1. Detect Device
    if torch.backends.mps.is_available():
        run_device = "mps"
    elif torch.cuda.is_available():
        run_device = "cuda"
    else:
        run_device = "cpu"

    # 2. Load Model with the FIX (trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map=None,
        trust_remote_code=False,  # <--- Fixes the 'DynamicCache' error
    ).to(run_device)

    # 3. Return exactly 2 values to match your code
    return tokenizer, model

@st.cache_resource
def load_retrieval_resources():
    """
    Safely loads RAG resources. Returns None if files are missing 
    to prevent app crash during development.
    """
    try:
        if not os.path.exists("financeqa_df.pkl") or not os.path.exists("financeqa_index.faiss"):
            return None, None, None
            
        df = pd.read_pickle("financeqa_df.pkl")
        index = faiss.read_index("financeqa_index.faiss")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return df, index, embed_model
    except Exception as e:
        st.error(f"Error loading RAG resources: {e}")
        return None, None, None

tokenizer, model = load_model()
df_rag, index_rag, embed_model = load_retrieval_resources()

# ==========================================
# 3. Feature Modules (The "New" Requirements)
# ==========================================

def get_financial_statements(ticker):
    """
    Feature 1: Financial Statements
    Retrieves key financial statement data.
    """
    try:
        t = yf.Ticker(ticker)
        # Get last year's financials (simplified)
        balance_sheet = t.balance_sheet
        income_stmt = t.income_stmt
        
        summary = []
        if not income_stmt.empty:
            rev = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else "N/A"
            net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else "N/A"
            summary.append(f"Most recent Annual Revenue: {rev}, Net Income: {net_income}")
            
        if not balance_sheet.empty:
            assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else "N/A"
            liab = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else "N/A"
            summary.append(f"Total Assets: {assets}, Total Liabilities: {liab}")
            
        return " | ".join(summary)
    except Exception:
        return "Financial statement data unavailable."

def get_ticker_live_data(ticker):
    """
    Retrieves live price, PE, and Sector.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get("currentPrice", "N/A")
        pe = info.get("trailingPE", "N/A")
        sector = info.get("sector", "N/A")
        # New: Add forward PE and Beta for "Recommendation" context
        f_pe = info.get("forwardPE", "N/A")
        beta = info.get("beta", "N/A")
        
        return (f"Live Market Data for {ticker}: Price={price}, Trailing PE={pe}, "
                f"Forward PE={f_pe}, Beta={beta}, Sector={sector}.")
    except Exception:
        return f"Could not fetch live data for {ticker}."

def extract_ticker(text):
    # Improved regex to catch $MSFT or MSFT
    candidates = re.findall(r"\b\$?[A-Z]{1,5}\b", text)
    blacklist = {"WHAT", "IS", "ARE", "THE", "AND", "ETF", "STOCK", "CAN", "YOU", "HELP", "BUY", "SELL"}
    tickers = [c.replace("$", "") for c in candidates if c.replace("$", "") not in blacklist]
    return tickers[0] if tickers else None

def retrieve_context(query, k=3):
    if df_rag is None: 
        return ""
    
    q_emb = embed_model.encode([query])
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    scores, idxs = index_rag.search(q_emb, k)
    idxs = idxs[0]
    
    snippets = []
    for i in idxs:
        if 0 <= i < len(df_rag):
            snippets.append(df_rag.iloc[i]["CONTEXT"])
    return "\n\n".join(snippets)

# ==========================================
# 4. Core Logic: Prompt Engineering & Generation
# ==========================================

def build_prompt(history, context_data):
    """
    Constructs the prompt. context_data contains combined Live Data + RAG Context.
    """
    system_msg = (
        "You are a highly accurate financial assistant for a Master's level project. "
        "Strictly adhere to the following rules:\n"
        "1. VERIFICATION: If CONTEXT is provided, use those numbers exactly. Do not hallucinate metrics.\n"
        "2. STRUCTURE: Answer in three sections: (A) Summary, (B) Detailed Analysis, (C) Risks/Caveats.\n"
        "3. TONE: Professional, objective, and precise.\n"
        "4. RECOMMENDATIONS: If asked for advice, analyze the data (PE, Beta, Financials) but finish with "
        "a strict disclaimer: 'This is not financial advice'.\n"
    )
    
    # Inject Context if available
    if context_data:
        system_msg += f"\n### CONTEXT (FACTUAL SOURCE OF TRUTH):\n{context_data}\n### END CONTEXT\n"

    conversation = ""
    for role, msg in history[-4:]: # Keep context window manageable
        conversation += f"{role}: {msg}\n"

    return f"<|user|>\n{system_msg}\n{conversation}\n<|assistant|>\n"

def generate_response(history, context_data=""):
    # --- Step 1: Draft Generation ---
    prompt = build_prompt(history, context_data)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6, # Lower temp for more precision
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Cleanup to get just the assistant's latest response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    elif "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
        
    return response

# ==========================================
# 5. Streamlit UI
# ==========================================

with st.sidebar:
    st.header("AML Project Control")
    st.markdown(f"**Device:** `{DEVICE.upper()}`")
    st.markdown("**Modules Active:**")
    st.checkbox("RAG Retrieval", value=(df_rag is not None), disabled=True)
    st.checkbox("Live Data (Yahoo)", value=True, disabled=True)
    st.checkbox("Financial Stmts", value=True, disabled=True)
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Team:\n* Shreya Shetty (svs2148)\n* Shruti Shetty (ss7592)\n* Anamika Mishra (akm2259)\n* Akriti Agarwal (aa5807)")

if "history" not in st.session_state:
    st.session_state.history = []

# Display Chat
for role, msg in st.session_state.history:
    if role == "User":
        st.chat_message("user").write(msg)
    elif role == "Assistant":
        st.chat_message("assistant").write(msg)
    elif role == "System":
        with st.expander("System Context (Verified Data)", expanded=False):
            st.markdown(f"_{msg}_")

user_msg = st.chat_input("Ask about a stock, market concept, or financial data...")

if user_msg:
    st.chat_message("user").write(user_msg)
    st.session_state.history.append(("User", user_msg))

    # --- Pre-processing & Retrieval ---
    context_buffer = []
    
    # 1. Identify Ticker
    ticker = extract_ticker(user_msg.upper())
    if ticker:
        # Fetch Live Data
        live_data = get_ticker_live_data(ticker)
        context_buffer.append(live_data)
        
        # Fetch Financial Statements (Feature 1)
        fin_stmts = get_financial_statements(ticker)
        context_buffer.append(f"Financial Statements for {ticker}: {fin_stmts}")
        
        st.toast(f"Pulled live data for {ticker}", icon="📡")

    # 2. RAG Retrieval
    if df_rag is not None:
        rag_text = retrieve_context(user_msg)
        if rag_text:
            context_buffer.append(f"RAG Knowledge Base:\n{rag_text}")

    full_context = "\n".join(context_buffer)
    
    # Log context to history (hidden in expander)
    if full_context:
        st.session_state.history.append(("System", full_context))

    # --- Generation & Verification ---
    with st.chat_message("assistant"):
        with st.spinner("Analyzing financials & Verifying data..."):
            reply = generate_response(st.session_state.history, full_context)
            st.write(reply)
            
    st.session_state.history.append(("Assistant", reply))