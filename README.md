<<<<<<< HEAD
# Autonomous Research Assistant

An autonomous research agent built with **LangChain**, **Streamlit**, and **Ollama**. It combines Retrieval-Augmented Generation (RAG) with web scraping to provide comprehensive answers to your research questions.

## Prerequisites

1. **Install Ollama**: Download and install from [ollama.com](https://ollama.com/).
2. **Pull the Model**:
   ```bash
   ollama pull llama3
   ```
3. **Start Ollama**: Ensure the Ollama service is running (`ollama serve`).

## Setup

1. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Start the Streamlit interface:
```bash
streamlit run app.py
```

## Features

- **Local RAG**: Ingest text and PDF files into a local ChromaDB vector store.
- **Autonomous Agent**: Uses a ReAct agent to decide whether to query the local database or scrape the web.
- **Web Scraping**: Automatically scrapes and ingests content from URLs when needed.
- **Source Attribution**: Shows exactly which sources were used to generate the answer.
=======
# autonomus_research_agent
>>>>>>> a57a48fbe2319fa25db06a2caaca045914be5e91
