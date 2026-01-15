# 🩺 Professional Medical Chatbot

A powerful, RAG-based (Retrieval-Augmented Generation) Medical Chatbot built with **LangChain**, **Hugging Face**, and **Streamlit**. This assistant provides professional, clear, and direct medical information based on provided documentation, ensuring a natural "ChatGPT-like" experience without exposing raw data retrieval processes.

## 🚀 Features

- **Professional AI Responses**: Refined prompt engineering ensures responses are clinical, direct, and helpful.
- **Context-Aware Information**: Uses RAG to retrieve and synthesize medical information from your local PDF database.
- **No Meta-References**: Unlike basic bots, this assistant never mentions "based on the context" or "according to the documents," acting as a knowledgeable professional.
- **Chat History**: Maintains session state for ongoing medical queries.
- **Fast Vector Retrieval**: Powered by **FAISS** for near-instant retrieval from large medical datasets.
- **Modern UI**: Clean and intuitive Streamlit interface.

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM Framework**: [LangChain](https://www.langchain.com/)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via Hugging Face
- **LLM**: `mistralai/Mistral-7B-Instruct-v0.2` via Hugging Face Hub
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss)
- **Environment Management**: Pipenv

## 📋 Setup Instructions

### 1. Prerequisites

Ensure you have Python 3.12+ and `pipenv` installed.

### 2. Install Dependencies

```bash
pipenv install
```

### 3. Environment Variables

Create a `.env` file in the root directory and add your Hugging Face API token:

```env
HF_TOKEN=your_huggingface_token_here
```

### 4. Prepare the Medical Memory (Optional)

If you add new PDFs to the `data/` folder, run the following script to create/update the vector store:

```bash
python create_memory_for_llm.py
```

### 5. Run the Application

```bash
streamlit run medibot.py
```

## 🔒 Security Note

The `.env` file and `vectorstore/` directory are included in `.gitignore` to prevent sensitive API tokens and raw data from being pushed to public repositories.

---

_Disclaimer: This chatbot is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment._
