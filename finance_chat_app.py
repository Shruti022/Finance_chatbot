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

# ---------- Load Model (cached across reruns) ----------
@st.cache_resource
def load_model():
    model_name = "microsoft/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # CRITICAL FIX: trust_remote_code=False prevents the 'DynamicCache' error
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=False, 
    ).to(device)

    return tokenizer, model, device

# --- NEW: Feature 1 (Financial Statements) ---
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
        return "Financial statements unavailable."

def get_ticker_summary(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info  
        price = info.get("currentPrice", "N/A")
        pe = info.get("trailingPE", "N/A")
        sector = info.get("sector", "N/A")
        
        # --- NEW: Feature 2 (Data for Recommendation) ---
        beta = info.get("beta", "N/A")
        f_pe = info.get("forwardPE", "N/A")
        
        return (f"Live data for {ticker}: price={price}, PE={pe}, Forward PE={f_pe}, "
                f"Beta={beta}, sector={sector}.")
    except Exception:
        return f"Could not fetch live data for {ticker}."


def extract_ticker(text):
    # CRITICAL FIX: The "SHARE" bug happened because SHARE is 5 letters.
    # We explicitly ignore it here.
    candidates = re.findall(r"\b[A-Z]{2,5}\b", text)
    blacklist = {
        "WHAT", "IS", "ARE", "THE", "AND", "ETF", "STOCK", "FOR", 
        "BUY", "SELL", "SHARE", "DATA", "DATE", "YEAR", "LONG", "TERM",
        "OF", "IN", "TO", "MY", "HELP", "WHO", "HOW"
    }
    tickers = [c for c in candidates if c not in blacklist]
    return tickers[0] if tickers else None


# ---------- Prompt building (financial chat style) ----------

def build_prompt(history):
    system = (
        "Instruction: You are a friendly, professional financial assistant. "
        "Your main job is to answer questions about finance, investing, markets, and the economy. "
        "When CONTEXT is provided, treat it as the primary source of truth and copy numbers exactly. "
        "Use live market data (price, PE, Beta) to provide analysis. "
        # --- NEW: Recommendation Logic ---
        "If asked for a recommendation (Buy/Sell), analyze the PE and Beta, but YOU MUST END WITH: 'This is not financial advice'. "
        "Structure finance answers as: (1) brief summary, (2) important details, (3) risks or caveats.\n"
    )

    conversation = ""
    # CRITICAL FIX: Only use the last 6 messages to prevent "Thinking Mode" hang
    for role, msg in history[-6:]:
        # We skip printing "System" lines in the prompt to keep it clean for the model
        # The relevant context is usually injected freshly in the last turn anyway
        if role != "System":
            conversation += f"{role}: {msg}\n"

    return system + conversation + "Assistant:"


# ---------- Model generation ----------
def generate_response(history, tokenizer, model, device):
    # ----- Step 1: draft answer -----
    prompt_draft = build_prompt(history)
    inputs = tokenizer(prompt_draft, return_tensors="pt").to(device)

    with torch.no_grad():
        draft_outputs = model.generate(
            **inputs,
            max_new_tokens=150, 
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
        )

    decoded_draft = tokenizer.decode(draft_outputs[0], skip_special_tokens=True)
    if "Assistant:" in decoded_draft:
        draft_answer = decoded_draft.split("Assistant:")[-1].strip()
    else:
        draft_answer = decoded_draft.strip()

    # ----- Step 2: refine answer -----
    refine_prompt = (
        "Instruction: You are improving a draft answer from a financial assistant. "
        "Task: Fix incomplete sentences, make the explanation clearer, "
        "ensure the structure is (1) Summary, (2) Details, (3) Risks. "
        "If recommending, add 'This is not financial advice'.\n\n"
        f"User question:\n{history[-1][1]}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        "Improved answer:"
    )

    refine_inputs = tokenizer(refine_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        refine_outputs = model.generate(
            **refine_inputs,
            max_new_tokens=200,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
        )

    refined_decoded = tokenizer.decode(refine_outputs[0], skip_special_tokens=True)
    if "Improved answer:" in refined_decoded:
        refined_answer = refined_decoded.split("Improved answer:")[-1].strip()
    else:
        refined_answer = refined_decoded.strip()

    return refined_answer


#Rag
@st.cache_resource
def load_retrieval():
    try:
        if not os.path.exists("financeqa_df.pkl"): return None, None, None
        df = pd.read_pickle("financeqa_df.pkl")
        index = faiss.read_index("financeqa_index.faiss")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return df, index, embed_model
    except:
        return None, None, None

def retrieve_context(query, k=3):
    df, index, embed_model = load_retrieval()
    if df is None: return ""
    q_emb = embed_model.encode([query])
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, k)
    idxs = idxs[0]
    snippets = [df.iloc[i]["CONTEXT"] for i in idxs if 0 <= i < len(df)]
    return "\n\n".join(snippets)


# ---------- Streamlit UI (ChatGPT-style) ----------

st.set_page_config(page_title="FinChat: AML Group Project", page_icon="💸", layout="wide")
st.title("💸🦙 Financial Chat Assistant (LLaMA)")

tokenizer, model, device = load_model()

with st.sidebar:
    st.header("AML Project Control")
    st.caption("Model: `Phi-3-mini-4k`")
    st.caption(f"Device: **{device.upper()}**")
    if st.button("Clear chat history"):
        st.session_state.history = []
        st.rerun()
    
    st.markdown("---")
    # --- NEW: Feature 3 (Team Names) ---
    st.markdown("### Team:\n* Akriti Agarwal (aa5807)\n* Shreya Shetty (svs2148)\n* Shruti Shetty (ss7592)\n* Anamika Mishra (akm2259)")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Show previous messages
for role, msg in st.session_state.history:
    if role == "User":
        st.chat_message("user").write(msg)
    elif role == "Assistant":
        st.chat_message("assistant").write(msg)
    elif role == "System":
        # --- NEW: Feature 4 (Verification UI) ---
        with st.expander("🔍 Verified Context (Click to see data used)", expanded=False):
            st.markdown(f"_{msg}_")

user_msg = st.chat_input("Ask your finance question...")

if user_msg:
    # Add user message
    st.session_state.history.append(("User", user_msg))
    st.chat_message("user").write(user_msg)

    # --- NEW: detect greetings ---
    lower_msg = user_msg.strip().lower()
    greeting_triggers = ["hi", "hello", "hey", "good morning", "how are you"]
    
    is_greeting = any(lower_msg == g or lower_msg.startswith(g + " ") for g in greeting_triggers)

    if is_greeting:
        reply = (
            "Hi! I’m your financial assistant. "
            "You can ask me about stocks, company fundamentals, or request an investment recommendation (educational only)."
        )
        st.chat_message("assistant").write(reply)
        st.session_state.history.append(("Assistant", reply))
        st.stop()

    # Existing Yahoo Finance part
    ticker = extract_ticker(user_msg.upper())
    if ticker:
        live_context = get_ticker_summary(ticker)
        # --- NEW: Add Financial Statements ---
        fin_stmts = get_financial_statements(ticker)
        
        full_context = f"{live_context}\nFinancial Statements: {fin_stmts}"
        st.session_state.history.append(("System", full_context))
        
        # Show Verification immediately
        with st.expander("🔍 Verified Context (Click to see data used)", expanded=False):
             st.markdown(f"_{full_context}_")

    # Existing RAG part
    rag_context = retrieve_context(user_msg)
    if rag_context:
        st.session_state.history.append(
            ("System",
             "CONTEXT from financial reports (use these numbers exactly, do not change them):\n"
             + rag_context)
        )
        # Show Verification immediately
        with st.expander("🔍 Verified Context (Click to see data used)", expanded=False):
             st.markdown(f"_{rag_context}_")

    # Model reply
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Financials..."):
            reply = generate_response(
                st.session_state.history,
                tokenizer,
                model,
                device,
            )
        st.write(reply)

    # Save reply to history
    st.session_state.history.append(("Assistant", reply))
