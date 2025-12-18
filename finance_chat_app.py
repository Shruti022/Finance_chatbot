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
# 1. App Configuration & Device Logic
# ==========================================
st.set_page_config(page_title="FinChat: AML Group Project", page_icon="💸", layout="wide")

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
# 2. Model Loading (Fixed for Phi-3)
# ==========================================
@st.cache_resource
def load_model():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    run_device = get_device_map()
    
    # trust_remote_code=False is CRITICAL to avoid the 'DynamicCache' error
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32, 
        device_map=None,
        trust_remote_code=False, 
    ).to(run_device)

    return tokenizer, model

@st.cache_resource
def load_retrieval_resources():
    """
    Safely loads RAG resources. Returns None if files are missing.
    """
    try:
        if not os.path.exists("financeqa_df.pkl") or not os.path.exists("financeqa_index.faiss"):
            return None, None, None
        df = pd.read_pickle("financeqa_df.pkl")
        index = faiss.read_index("financeqa_index.faiss")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return df, index, embed_model
    except Exception as e:
        return None, None, None

tokenizer, model = load_model()
df_rag, index_rag, embed_model = load_retrieval_resources()

# ==========================================
# 3. Feature Modules (Data Fetching)
# ==========================================
def get_financial_statements(ticker):
    """Fetches key financial metrics for context."""
    try:
        t = yf.Ticker(ticker)
        income_stmt = t.income_stmt
        balance_sheet = t.balance_sheet
        summary = []
        
        # Income Statement Data
        if not income_stmt.empty:
            rev = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else "N/A"
            net = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else "N/A"
            summary.append(f"Most Recent Revenue: {rev}, Net Income: {net}")
            
        # Balance Sheet Data
        if not balance_sheet.empty:
            assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else "N/A"
            liab = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else "N/A"
            summary.append(f"Total Assets: {assets}, Total Liabilities: {liab}")
            
        return " | ".join(summary)
    except:
        return "Financial Statement Data Unavailable"

def get_ticker_live_data(ticker):
    """Fetches live price, PE, and Beta for recommendations."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get("currentPrice", "N/A")
        pe = info.get("trailingPE", "N/A")
        f_pe = info.get("forwardPE", "N/A")
        beta = info.get("beta", "N/A")
        sector = info.get("sector", "N/A")
        rec_key = info.get("recommendationKey", "none")
        
        return (f"Live Market Data for {ticker}: Price={price}, Trailing PE={pe}, "
                f"Forward PE={f_pe}, Beta={beta}, Sector={sector}, Analyst Consensus={rec_key}.")
    except:
        return f"Live data unavailable for {ticker}."

def extract_ticker(text):
    """
    Extracts tickers while ignoring common English words (Stopwords).
    Fixes the 'OF' issue.
    """
    # 1. Regex: Look for uppercase words 2-5 chars long
    candidates = re.findall(r"\b[A-Z]{2,5}\b", text)
    
    # 2. Strict Blacklist (Common uppercase words in questions)
    blacklist = {
        "WHAT", "IS", "ARE", "THE", "AND", "FOR", "CAN", "YOU", "HELP", 
        "BUY", "SELL", "HOW", "WHO", "WHY", "ETF", "STOCK", "SHARE", 
        "CAPITAL", "EQUITY", "PRICE", "DATE", "YEAR", "DATA", "LONG", 
        "TERM", "DEBT", "RATIO", "COST", "CASH", "FLOW", "OF", "IN", "TO", "MY"
    }
    
    # Filter candidates
    valid_tickers = [c for c in candidates if c not in blacklist]
    
    # Return the first valid ticker found, or None
    return valid_tickers[0] if valid_tickers else None

def retrieve_context(query, k=3):
    if df_rag is None: return ""
    q_emb = embed_model.encode([query])
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    scores, idxs = index_rag.search(q_emb, k)
    snippets = [df_rag.iloc[i]["CONTEXT"] for i in idxs[0] if 0 <= i < len(df_rag)]
    return "\n\n".join(snippets)

# ==========================================
# 4. Prompt Engineering (The "Brain")
# ==========================================
def build_prompt(history, context_data):
    """
    Constructs a prompt that enforces the 'Summary, Details, Risks' structure.
    Crucially, it ignores 'System' messages from history to prevent lag/confusion.
    """
    system_msg = (
        "You are a professional financial assistant. "
        "Your goal is to answer questions using the provided CONTEXT data.\n"
        "STRICT RESPONSE FORMAT:\n"
        "1. **Summary**: A direct answer in 1-2 sentences.\n"
        "2. **Details**: Explain the data, citing numbers from the Context exactly.\n"
        "3. **Risks/Caveats**: Mention risks or disclaimers (e.g., 'This is not financial advice').\n\n"
        "RULES:\n"
        "- If CONTEXT is provided, it is the absolute truth. Do not invent numbers.\n"
        "- If asked for a recommendation, analyze the PE and Beta but end with 'This is not financial advice'.\n"
    )
    
    # Inject ONLY the current context (not old context)
    if context_data:
        system_msg += f"\n### CURRENT CONTEXT (Source of Truth):\n{context_data}\n### END CONTEXT\n"

    # Start Prompt Construction (Phi-3 Format)
    prompt_text = f"<|user|>\n{system_msg}\n\n"
    
    # Add History (Filter out 'System' role to reduce lag)
    # Only keep last 4 turns to keep it fast
    relevant_history = [h for h in history if h[0] != "System"][-4:]
    
    for role, msg in relevant_history:
        if role == "User":
            # If not start, add separator
            if not prompt_text.endswith("\n\n"): prompt_text += "<|end|>\n<|user|>\n"
            prompt_text += f"{msg}"
        elif role == "Assistant":
            prompt_text += f"<|end|>\n<|assistant|>\n{msg}"

    # Prepare for generation
    prompt_text += "<|end|>\n<|assistant|>\n"
    return prompt_text

def generate_response(history, context_data=""):
    prompt = build_prompt(history, context_data)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=350, # Sufficient for Summary/Details/Risks
            do_sample=True,
            temperature=0.3,    # Low temp = more factual/stable
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse out only the new response
    if "Assistant" in full_out:
        response = full_out.split("Assistant")[-1].strip()
        response = response.lstrip(": ") # Clean leading colons
    else:
        response = full_out
        
    return response

# ==========================================
# 5. Streamlit UI (The Interface)
# ==========================================
with st.sidebar:
    st.header("AML Project Control")
    st.markdown(f"**Device:** `{DEVICE.upper()}`")
    if st.button("Clear Conversation"):
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.markdown("### Team:\n* Shreya Shetty (svs2148)\n* Shruti Shetty (ss7592)\n* Anamika Mishra (akm2259)\n* Akriti Agarwal (aa5807)")

if "history" not in st.session_state:
    st.session_state.history = []

# Display Chat History
for role, msg in st.session_state.history:
    if role == "User":
        st.chat_message("user").write(msg)
    elif role == "Assistant":
        st.chat_message("assistant").write(msg)
    elif role == "System":
        # Collapsible System Context (Verification Layer)
        with st.expander("🔍 Verified Context Data (Click to View)", expanded=False):
            st.code(msg, language="text")

# Chat Input
user_msg = st.chat_input("Ask a question (e.g., 'What is the PE of MSFT?' or 'Equity share capital?')...")

if user_msg:
    # 1. Show User Message
    st.chat_message("user").write(user_msg)
    st.session_state.history.append(("User", user_msg))

    # 2. Fast Greeting Check (No LLM needed)
    lower_msg = user_msg.lower().strip()
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if lower_msg in greetings or (len(lower_msg) < 15 and any(lower_msg.startswith(g) for g in greetings)):
        reply = (
            "Hello! I am your financial assistant. I can help with:\n"
            "- **Live Stock Data** (e.g., 'Check MSFT')\n"
            "- **Financial Concepts** (e.g., 'What is P/E?')\n"
            "- **RAG Analysis** (Using your class dataset)\n"
            "How can I help you today?"
        )
        st.chat_message("assistant").write(reply)
        st.session_state.history.append(("Assistant", reply))
        st.stop() # Exit successfully

    # 3. Context Builder
    context_buffer = []
    
    # A. Check for Ticker
    ticker = extract_ticker(user_msg.upper())
    if ticker:
        live = get_ticker_live_data(ticker)
        stmts = get_financial_statements(ticker)
        context_buffer.append(live)
        context_buffer.append(f"Financial Statements for {ticker}: {stmts}")
        st.toast(f"Retrieved live data for {ticker}", icon="📡")

    # B. Check RAG (if dataset loaded)
    if df_rag is not None:
        rag_text = retrieve_context(user_msg)
        if rag_text: 
            context_buffer.append(f"Internal Knowledge Base (RAG):\n{rag_text}")

    full_context = "\n\n".join(context_buffer)

    # 4. Save Context to History (For UI verification only)
    if full_context:
        st.session_state.history.append(("System", full_context))
        # We re-display it immediately for the current turn
        with st.expander("🔍 Verified Context Data (Click to View)", expanded=False):
            st.code(full_context, language="text")

    # 5. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing financials..."):
            reply = generate_response(st.session_state.history, full_context)
            st.write(reply)

    # 6. Save Answer
    st.session_state.history.append(("Assistant", reply))