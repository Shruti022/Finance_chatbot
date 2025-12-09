from turtle import pd
from xml.parsers.expat import model
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yfinance as yf
import re


import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------- Load Model (cached across reruns) ----------


@st.cache_resource
def load_model():
    # Tiny chat model that is OK on CPU
    model_name = "microsoft/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    device = "cpu"
    dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)

    return tokenizer, model, device




def get_ticker_summary(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info  # may be .fast_info in newer versions
        price = info.get("currentPrice")
        pe = info.get("trailingPE")
        sector = info.get("sector")
        return f"Live data for {ticker}: price={price}, PE={pe}, sector={sector}."
    except Exception as e:
        return f"Could not fetch live data for {ticker}."


def extract_ticker(text):
    # super naive: look for ALL‑CAPS 1–5 letter tokens
    # candidates = re.findall(r"\b[A-Z]{1,5}\b", text)
    candidates = re.findall(r"\b[A-Z]{2,5}\b", text)
    blacklist = {"WHAT", "IS", "ARE", "THE", "AND", "ETF", "STOCK"}
    tickers = [c for c in candidates if c not in blacklist]
    return tickers[0] if tickers else None


# ---------- Prompt building (financial chat style) ----------

def build_prompt(history):


    system = (
        "Instruction: You are a friendly, professional financial assistant. "
        "Your main job is to answer questions about finance, investing, markets, and the economy. "
        "You can also respond briefly and naturally to greetings or small talk (like 'hi', 'hello', 'how are you') "
        "and then gently steer the conversation back to finance. "
        "When CONTEXT from financial reports is provided, treat it as the primary source of truth for factual numbers "
        "and copy numeric values exactly rather than inventing them. "
        "Use any live market data (prices, PE, sector) as additional color. "
        "Structure finance answers as: (1) brief summary, (2) important details, (3) risks or caveats.\n"
    )




    conversation = ""
    for role, msg in history:
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
            max_new_tokens=48,
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
        "Task: Fix any incomplete sentences, make the explanation clearer, "
        "and add at least one concrete risk or caveat if relevant.\n\n"
        f"User question:\n{history[-1][1]}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        "Improved answer:"
    )

    refine_inputs = tokenizer(refine_prompt, return_tensors="pt").to(device)

    with torch.no_grad():

        refine_outputs = model.generate(
            **refine_inputs,
            max_new_tokens=96,
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
    df = pd.read_pickle("financeqa_df.pkl")
    index = faiss.read_index("financeqa_index.faiss")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, index, embed_model

def retrieve_context(query, k=3):
    df, index, embed_model = load_retrieval()
    q_emb = embed_model.encode([query])
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, k)
    idxs = idxs[0]
    snippets = [df.iloc[i]["CONTEXT"] for i in idxs]
    return "\n\n".join(snippets)





# ---------- Streamlit UI (ChatGPT-style) ----------

st.set_page_config(page_title="Finance LLaMA Chat", page_icon="💸", layout="wide")
st.title("💸🦙 Financial Chat Assistant (LLaMA)")

st.write(
    "Ask any finance-related question (markets, investing, macro, personal finance, etc.). "
    "This uses your fine-tuned LLaMA model under the hood."
)

# Load model once
# tokenizer, model, device, newline_token_id = load_model()
# Load model once
tokenizer, model, device = load_model()


with st.sidebar:
    st.header("Model Info")
    st.caption("Model: `oopere/Llama-FinSent-S`")
    st.caption(f"Device: **{device.upper()}**")
    if st.button("Clear chat history"):
        st.session_state.history = []

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, message)

# Show previous messages
for role, msg in st.session_state.history:
    if role == "User":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)

# Input box at bottom (like ChatGPT)
user_msg = st.chat_input("Ask your finance question...")



if user_msg:
    # Add user message
    st.session_state.history.append(("User", user_msg))
    st.chat_message("user").write(user_msg)

    # --- NEW: detect greetings / very general small talk ---
    lower_msg = user_msg.strip().lower()
    greeting_triggers = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
    help_triggers = ["can you help", "what can you do", "i'm new", "im new", "i am new"]

    is_greeting = any(lower_msg == g or lower_msg.startswith(g + " ") for g in greeting_triggers)
    is_helpish = any(p in lower_msg for p in help_triggers)

    if is_greeting:
        reply = (
            "Hi! I’m your financial assistant. "
            "You can ask me about stocks, ETFs, company fundamentals, or general investing questions."
        )
        st.chat_message("assistant").write(reply)
        st.session_state.history.append(("Assistant", reply))
        st.stop()

    if is_helpish and not is_greeting:
        reply = (
            "I can help explain investing concepts, analyze companies at a high level, "
            "and use some live market data to discuss stocks and ETFs. "
            "What finance or investing question would you like to start with?"
        )
        st.chat_message("assistant").write(reply)
        st.session_state.history.append(("Assistant", reply))
        st.stop()
    # -------------------------------------------------------

    # Existing Yahoo Finance part (if you have it)
    ticker = extract_ticker(user_msg)
    if ticker:
        live_context = get_ticker_summary(ticker)
        st.session_state.history.append(("System", live_context))

    # Existing RAG part
    rag_context = retrieve_context(user_msg)
    if rag_context:
        st.session_state.history.append(
            ("System",
             "CONTEXT from financial reports (use these numbers exactly, do not change them):\n"
             + rag_context)
        )

    # Model reply (draft+refine inside generate_response)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = generate_response(
                st.session_state.history,
                tokenizer,
                model,
                device,
            )
        st.write(reply)

    st.session_state.history.append(("Assistant", reply))




    # Save reply to history
    st.session_state.history.append(("Assistant", reply))
    # MAX_TURNS = 5
    # if len(st.session_state.history) > 2 * MAX_TURNS:
    #     st.session_state.history = st.session_state.history[-2 * MAX_TURNS:]



