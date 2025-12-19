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




# 🏦 Lumiq: Multi-Agent AI Platform for Financial Analysis

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

**A multi-agent RAG-based system that democratizes professional-grade stock analysis and financial document processing through collaborative AI.**

---

## 🌐 Live Demo

**Try it now:** [https://huggingface.co/spaces/Shruti02222/ai-investment-analyst](https://huggingface.co/spaces/Shruti02222/ai-investment-analyst)

---

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
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

Lumiq addresses the gap between expensive professional financial tools (Bloomberg Terminal: $20,000+/year) and free but limited alternatives. It employs **6 specialized AI agents** that analyze investments from complementary perspectives:

- 📊 **Fundamental Analyst** - Financial health metrics
- 📈 **Market Data Analyst** - Price trends and positioning
- ⚠️ **Risk Analyst** - Threat assessment
- 📉 **Technical Analyst** - Chart patterns and indicators
- 👔 **Chief Investment Officer** - Final recommendation synthesis
- 📄 **Financial Statement Analyst** - Document parsing and interpretation

---

## ✨ Features

### 1️⃣ **Stock Analysis Workflow**
- Natural language queries: *"Should I invest in AAPL?"*
- Multi-perspective analysis from 4 specialist agents
- **BUY/SELL/HOLD** recommendations with conviction scores (1-10)
- Actionable plans: entry prices, stop-loss levels, timelines
- Transparent reasoning showing how each agent contributed

### 2️⃣ **Financial Statement Analysis**
- Upload **PDF, Excel (.xlsx/.xls), or CSV** files
- Automatic extraction of key metrics (P/E, ROE, Debt-to-Equity, margins)
- 6-section structured reports:
  - 📋 Statement Overview
  - 💰 Key Financial Metrics
  - 💪 Strengths & Positive Indicators
  - ⚠️ Cautions & Red Flags
  - 🎯 Crucial Points for Investors
  - 📈 Recommendations

### 3️⃣ **RAG-Based Knowledge Grounding**
- **128-document knowledge base:**
  - 115 verified Q&A from FinanceBench dataset
  - 10 live stocks from Yahoo Finance (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, V, WMT)
  - 3 core financial concept definitions
- Two-stage retrieval: FAISS vector search → Cross-encoder reranking
- Reduces hallucination by grounding responses in real data

---

## 🏗️ Architecture




