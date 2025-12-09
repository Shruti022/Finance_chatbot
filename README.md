Uses a small instruction LLM to answer finance questions in a clear, structured way (summary, details, risks).
Pulls live stock data (price, P/E, sector) from Yahoo Finance when the user mentions a ticker.
Uses RAG on the FinanceQA dataset by retrieving relevant CONTEXT passages with a FAISS index and feeding them to the model for more grounded answers.​​
Runs in a Streamlit chat UI with simple rules for greetings and “can you help me” so it feels conversational.
