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




***

# 🏦 Lumiq: Multi-Agent AI Platform for Financial Analysis



![Status](https://img.shields.io/badge/status-active-successYour Journey to Intelligent Investing Begins Here.**

**A multi-agent RAG-based system that democratizes professional-grade stock analysis and financial document processing through collaborative AI.**[1][2]

***

## 🌐 Live Demo

**Try it now:** [https://huggingface.co/spaces/Shruti02222/ai-investment-analyst](https://huggingface.co/spaces/Shruti02222/ai-investment-analyst)

***

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [API Keys Required](#api-keys-required)
- [License](#license)
- [Acknowledgments](#acknowledgments)

***

## 🎯 Overview

Lumiq addresses the gap between expensive professional financial tools (Bloomberg Terminal: $20,000+/year) and free but limited alternatives. It employs **6 specialized AI agents** that analyze investments from complementary perspectives:[2][1]

- 📊 **Fundamental Analyst** - Financial health metrics
- 📈 **Market Data Analyst** - Price trends and positioning
- ⚠️ **Risk Analyst** - Threat assessment
- 📉 **Technical Analyst** - Chart patterns and indicators
- 👔 **Chief Investment Officer** - Final recommendation synthesis
- 📄 **Financial Statement Analyst** - Document parsing and interpretation

***

## ✨ Features

### 1️⃣ **Stock Analysis Workflow**
- Natural language queries: *"Should I invest in AAPL?"*
- Multi-perspective analysis from 4 specialist agents[1]
- **BUY/SELL/HOLD** recommendations with conviction scores (1-10)[1]
- Actionable plans: entry prices, stop-loss levels, timelines[1]
- Transparent reasoning showing how each agent contributed[1]

### 2️⃣ **Financial Statement Analysis**
- Upload **PDF, Excel (.xlsx/.xls), or CSV** files[1]
- Automatic extraction of key metrics (P/E, ROE, Debt-to-Equity, margins)[1]
- 6-section structured reports:[1]
  - 📋 Statement Overview
  - 💰 Key Financial Metrics
  - 💪 Strengths & Positive Indicators
  - ⚠️ Cautions & Red Flags
  - 🎯 Crucial Points for Investors
  - 📈 Recommendations

### 3️⃣ **RAG-Based Knowledge Grounding**
- **128-document knowledge base:**[2][1]
  - 115 verified Q&A from FinanceBench dataset
  - 10 live stocks from Yahoo Finance (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, V, WMT)
  - 3 core financial concept definitions
- Two-stage retrieval: FAISS vector search → Cross-encoder reranking[2]
- Reduces hallucination by grounding responses in real data[1]

***

## 🏗️ Architecture

```

```

**Scoring Logic:**[2][1]
- Each specialist agent assigns: **+1** (Bullish), **0** (Neutral), **-1** (Bearish)
- Total score = Sum of 4 agents
- **Total ≥ +2** → BUY
- **Total ≤ -2** → SELL
- **Otherwise** → HOLD

***

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Llama-3.3-70B-versatile (Groq API) [2] |
| **Fallback LLM** | Llama-3.1-8B-instant (rate limit handling) [2] |
| **Embeddings** | all-MiniLM-L6-v2 (384-dim) [2] |
| **Vector Store** | FAISS (IndexFlatIP) [2] |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 [2] |
| **Document Parsing** | pdfplumber, PyPDF2, openpyxl, pandas [1] |
| **Market Data** | yfinance (Yahoo Finance API) [2] |
| **Dataset** | PatronusAI/financebench [2] |
| **UI Framework** | Gradio [2] |
| **Deployment** | Hugging Face Spaces [2] |
| **Language** | Python 3.10+ |

***

## 📦 Installation

### **Prerequisites**
- Python 3.10 or higher
- Google Colab account (recommended) OR local Jupyter environment
- API Keys:
  - **Groq API Key** (get free at [console.groq.com](https://console.groq.com))
  - **Hugging Face Token** (get at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

### **Setup Steps**

#### **Option 1: Google Colab (Recommended)**

1. **Open the notebook:**
   ```
   https://github.com/Shruti022/Finance_chatbot/blob/main/Finance_chatbot_aml_v2-6.ipynb
   ```

2. **Click "Open in Colab"**

3. **Add your API keys to Colab Secrets:**
   - Click the 🔑 key icon in the left sidebar
   - Add two secrets:
     - Name: `GROQ_API_KEY` → Value: Your Groq API key
     - Name: `HF_TOKEN` → Value: Your Hugging Face token

4. **Run all cells sequentially:**
   - The notebook will automatically:
     - Install dependencies
     - Download FinanceBench dataset[2]
     - Fetch live Yahoo Finance data[2]
     - Create embeddings
     - Build FAISS index
     - Generate `financebench_enhanced.pkl` and `financebench_enhanced.faiss` files
     - Launch Gradio interface

5. **Access the interface:**
   - A public Gradio link will appear at the bottom of the last cell
   - Example: `https://xxxxx.gradio.live`

#### **Option 2: Local Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shruti022/Finance_chatbot.git
   cd Finance_chatbot
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables:**
   ```bash
   export GROQ_API_KEY="your-groq-api-key"
   export HF_TOKEN="your-huggingface-token"
   ```

5. **Run the notebook:**
   ```bash
   jupyter notebook Finance_chatbot_aml_v2-6.ipynb
   ```

6. **Execute all cells to:**
   - Build knowledge base
   - Generate FAISS index
   - Launch Gradio UI

***

## 🚀 Usage

### **Stock Analysis Example**

**Input:**
```
Should I invest in SNAP?
```

**Output:**
```




***

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- **API Keys**:
  - **Groq API Key** (get free at [console.groq.com](https://console.groq.com/))
  - **Hugging Face Token** (get at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) - *Only needed for Colab*

***

### Option 1: Use Live Hugging Face Spaces Deployment (No Setup Required!)

**Just visit and use:** [https://huggingface.co/spaces/Shruti02222/ai-investment-analyst](https://huggingface.co/spaces/Shruti02222/ai-investment-analyst)

- ✅ **No API keys needed from users**
- ✅ **No installation required**
- ✅ **Ready to use immediately**
- The app is pre-configured with the GROQ_API_KEY as a secret variable by the developer
- Pre-built RAG files (`financebench_enhanced.pkl` and `financebench_enhanced.faiss`) are already uploaded to the Space

***

### Option 2: Run in Google Colab (Full Build from Scratch)

1. **Open the notebook:**
   ```
   https://github.com/Shruti022/Finance_chatbot/blob/main/Finance_chatbot_aml_v2-6.ipynb
   ```
   Click **"Open in Colab"**

2. **Add your API keys to Colab Secrets:**
   - Click the 🔑 **key icon** in the left sidebar
   - Add **TWO secrets**:
     - **Secret 1:**
       - Name: `GROQ_API_KEY`
       - Value: Your Groq API key
     - **Secret 2:**
       - Name: `HF_TOKEN`
       - Value: Your Hugging Face token

3. **Run all cells sequentially:**
   The notebook will:
   - Install dependencies
   - Download FinanceBench dataset from Hugging Face (requires HF_TOKEN)
   - Fetch live Yahoo Finance data for 10 stocks
   - Create embeddings using all-MiniLM-L6-v2
   - Build FAISS index
   - **Generate `financebench_enhanced.pkl` and `financebench_enhanced.faiss` files**
   - Launch Gradio interface

4. **Access your Gradio interface:**
   - A public link will appear: `https://xxxxx.gradio.live`

***

### Option 3: Deploy Your Own Hugging Face Space

1. **First, generate the RAG files in Google Colab** (follow Option 2 above to create the `.pkl` and `.faiss` files)

2. **Download the generated files:**
   - `financebench_enhanced.pkl`
   - `financebench_enhanced.faiss`

3. **Create a new Hugging Face Space:**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space)
   - Choose **Gradio** as SDK

4. **Upload files to your Space:**
   - Upload `app.py` (modified version that LOADS pre-built files)
   - Upload `financebench_enhanced.pkl`
   - Upload `financebench_enhanced.faiss`
   - Upload `requirements.txt`

5. **Key difference in `app.py` for Hugging Face:**
   ```python
   # LOAD PRE-BUILT RAG (created in Colab)
   print("📦 Loading RAG system...")
   
   df_rag = pd.read_pickle("financebench_enhanced.pkl")
   index_rag = faiss.read_index("financebench_enhanced.faiss")
   embed_model = SentenceTransformer("all-MiniLM-L6-v2")
   
   # Build agents (NO dataset downloading, NO FAISS building)
   agent1 = FundamentalAnalyst()
   agent2 = MarketDataAnalyst()
   # ... etc
   ```

6. **Add GROQ_API_KEY as a Space secret:**
   - Go to Space Settings → Variables and secrets
   - Add: `GROQ_API_KEY` = your key

7. **Your Space is live!**

***

### Option 4: Local Installation (Advanced)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shruti022/Finance_chatbot.git
   cd Finance_chatbot
   ```

2. **You need the pre-built RAG files:**
   - Either: Generate them using the Colab notebook first (Option 2)
   - Or: Download them from the Hugging Face Space repository

3. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set environment variables:**
   ```bash
   export GROQ_API_KEY="your-groq-api-key"
   ```

6. **Run the app:**
   ```bash
   python app.py
   ```
   (Assuming you have an `app.py` that loads the pre-built files)

***






