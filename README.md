Uses a small instruction LLM to answer finance questions in a clear, structured way (summary, details, risks).
Pulls live stock data (price, P/E, sector) from Yahoo Finance when the user mentions a ticker.
Uses RAG on the FinanceQA dataset by retrieving relevant CONTEXT passages with a FAISS index and feeding them to the model for more grounded answers.​​
Runs in a Streamlit chat UI with simple rules for greetings and “can you help me” so it feels conversational.

Start with running in the terminal:

```bash
pip install -r requirements.txt
```

To run the project, run the following file in colab notebook:
```bash
Finance_chatbot/Finance_chatbot_aml_v2.ipynb

```

Diagram 1: System Architecture (Main Pipeline)
![Diagram 1: System Architecture (Main Pipeline)](Images/Diagram1.png)

Diagram 2: Multi-Agent Workflow (Flowchart)
![Diagram 2: Multi-Agent Workflow (Flowchart)](Images/Diagram2.png)

Diagram 3: RAG Retrieval Process
![Diagram 3: RAG Retrieval Process](Images/Diagram3.png)
