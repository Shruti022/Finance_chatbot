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
    # MPS (Apple Silicon) often needs explicit fallback if ops aren't supported
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
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
    
    # trust_remote_code=False is CRITICAL to avoid the 'DynamicCache' error
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32, 
        device_map=None,
        trust_remote_code=False, 
    ).to(DEVICE)

    return tokenizer, model

@st.cache_resource
def load_retrieval_resources():
    try:
        if not os.path.exists("financeqa_df.pkl") or not os.path.exists("financeqa_index.faiss"):
            return None, None, None
        df = pd.read_pickle("financeqa_df.pkl")
        index = faiss.read_index("financeqa_index.faiss")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return df, index, embed_model
    except Exception:
        return None, None, None

tokenizer, model = load_model()
df_rag, index_rag, embed_model = load_retrieval_resources()

# ==========================================
# 3. Feature Modules (Data Fetching)
# ==========================================
def get_financial_statements(ticker):
    try:
        t = yf.Ticker(ticker)
        income_stmt = t.income_stmt
        balance_sheet = t.balance_sheet
        summary = []
        if not income_stmt.empty:
            rev = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else "N/A"
            net = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else "N/A"
            summary.append(f"Revenue: {rev}, Net Income: {net}")
        if not balance_sheet.empty:
            assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else "N/A"
            summary.append(f"Total Assets: {assets}")
        return " | ".join(summary)
    except:
        return "Data Unavailable"

def get_ticker_live_data(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get("currentPrice", "N/A")
        pe = info.get("trailingPE", "N/A")
        beta = info.get("beta", "N/A")
        return f"Live Market Data: Price={price}, PE={pe}, Beta={beta}."
    except:
        return f"Live data unavailable."

def extract_ticker(text):
    # Improved ticker extraction with blacklist
    candidates = re.findall(r"\b[A-Z]{2,5}\b", text)
    blacklist = {
        "WHAT", "IS", "ARE", "THE", "AND", "FOR", "CAN", "YOU", "HELP", 
        "BUY", "SELL", "HOW", "WHO", "WHY", "ETF", "STOCK", "SHARE", 
        "CAPITAL", "EQUITY", "PRICE", "DATE", "YEAR", "DATA", "LONG", 
        "TERM", "DEBT", "RATIO", "COST", "CASH", "FLOW", "OF", "IN", "TO", "MY"
    }
    valid_tickers = [c for c in candidates if c not in blacklist]
    return valid_tickers[0] if valid_tickers else None

def retrieve_context(query, k=3):
    if df_rag is None: return ""
    q_emb = embed_model.encode([query])
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    _, idxs = index_rag.search(q_emb, k)
    snippets = [df_rag.iloc[i]["CONTEXT"] for i in idxs[0] if 0 <= i < len(df_rag)]
    return "\n".join(snippets)

# ==========================================
# 4. Prompt Engineering (Simplified to fix hang)
# ==========================================
def build_prompt(history, context_data):
    # Simplified system message to prevent confusing the model
    system_msg = (
        "You are a helpful financial assistant for a Master's project. "
        "Use the provided CONTEXT data to answer accurately. "
        "If the answer is not in the context, state that you do not know. "
        "Do not provide financial advice."
    )
    
    # Inject current context
    if context_data:
        system_msg += f"\n### CONTEXT:\n{context_data}\n### END CONTEXT\n"

    # Build Phi-3 chat format
    prompt_text = f"<|user|>\n{system_msg}\n\n"
    
    # Filter history: keep last 3 turns, remove system messages
    relevant_history = [h for h in history if h[0] != "System"][-3:]
    
    for role, msg in relevant_history:
        if role == "User":
             # Ensure correct spacing between turns
            if not prompt_text.endswith("\n\n"): prompt_text += "<|end|>\n<|user|>\n"
            prompt_text += f"{msg}"
        elif role == "Assistant":
            prompt_text += f"<|end|>\n<|assistant|>\n{msg}"

    prompt_text += "<|end|>\n<|assistant|>\n"
    return prompt_text

def generate_response(history, context_data=""):
    prompt = build_prompt(history, context_data)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        # Reduced max tokens to prevent long hangs
        outputs = model.generate(
            **inputs,
            max_new_tokens=256, 
            do_sample=True,
            temperature=0.2, # Lower temp for stability
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant" in full_out:
        response = full_out.split("Assistant")[-1].strip()
    else:
        response = full_out
        
    return response

# ==========================================
# 5. Streamlit UI
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
        with st.expander("🔍 Verified Context Data (Click to View)", expanded=False):
            st.code(msg, language="text")

# Chat Input
user_msg = st.chat_input("Ask a question...")

if user_msg:
    st.chat_message("user").write(user_msg)
    st.session_state.history.append(("User", user_msg))

    # --- Fast Greeting Check ---
    lower_msg = user_msg.lower().strip()
    greetings = ["hi", "hello", "hey", "good morning"]
    if lower_msg in greetings or (len(lower_msg) < 15 and any(lower_msg.startswith(g) for g in greetings)):
        reply = "Hello! I am your financial assistant. Ask me about stocks or financial concepts."
        st.chat_message("assistant").write(reply)
        st.session_state.history.append(("Assistant", reply))
        st.stop()

    # --- Build Context ---
    context_buffer = []
    
    ticker = extract_ticker(user_msg.upper())
    if ticker:
        live = get_ticker_live_data(ticker)
        stmts = get_financial_statements(ticker)
        context_buffer.append(f"Live Data for {ticker}: {live}")
        context_buffer.append(f"Financials for {ticker}: {stmts}")
        st.toast(f"Retrieved data for {ticker}", icon="✅")

    if df_rag is not None:
        rag_text = retrieve_context(user_msg)
        if rag_text: context_buffer.append(f"RAG Context:\n{rag_text}")

    full_context = "\n\n".join(context_buffer)

    # --- Generate & Display ---
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            # Save context to history for UI display
            if full_context:
                 st.session_state.history.append(("System", full_context))
                 
            # Generate
            reply = generate_response(st.session_state.history, full_context)
            st.write(reply)

    st.session_state.history.append(("Assistant", reply))
    # Force rerun to show the new system message in the expander above
    st.rerun()