# 📄 PDF Q&A Chatbot

A fully local RAG-powered chatbot that lets you upload any PDF and ask questions about it — no API keys, no data leaves your machine.

Built as a practical demonstration of Retrieval-Augmented Generation (RAG) using LangChain, FAISS, and Ollama. Exposed via both a Streamlit UI and a FastAPI backend.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| API | FastAPI |
| LLM | Ollama (llama3.2) |
| Embeddings | Ollama (nomic-embed-text) |
| Vector Store | FAISS |
| PDF Loader | LangChain PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |

---

## ⚙️ Architecture

```
PDF Upload
   │
   ▼
PyPDFLoader → RecursiveCharacterTextSplitter
   │
   ▼
OllamaEmbeddings (nomic-embed-text)
   │
   ▼
FAISS Vector Index
   │
   ▼
User Question → Retriever (top-k chunks)
   │
   ▼
PromptTemplate + Ollama LLM (llama3.2)
   │
   ▼
Answer + Source Chunks
```

---

## 🚀 Setup & Running

### 1. Install Ollama
Download and install from [https://ollama.com/download](https://ollama.com/download)

### 2. Pull the required models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3. Clone the repo
```bash
git clone https://github.com/sassyredfox/pdf-qa-chatbot.git
cd pdf-qa-chatbot
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5a. Run the Streamlit UI
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

### 5b. Run the FastAPI backend
```bash
uvicorn main:app --reload
```
Opens at `http://localhost:8000`  
Interactive API docs at `http://localhost:8000/docs`

---

## 📦 Requirements

```
streamlit
fastapi
uvicorn
python-multipart
langchain
langchain-community
langchain-core
langchain-text-splitters
pypdf
faiss-cpu
```

---

## ✨ Features

- **Upload any PDF** and start chatting with it instantly
- **Fully local** — runs on your machine, no API keys needed
- **Source chunk viewer** — see exactly which part of the PDF the answer came from
- **Streamlit UI** for interactive use
- **FastAPI backend** for programmatic/API access
- **Configurable settings** via sidebar:
  - Swap Ollama models on the fly
  - Adjust chunk size and overlap
  - Control how many chunks are retrieved (k)
- **Conversation history** preserved within the session
- **Clear chat** button to reset and upload a new PDF

---

## 🔌 API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload and index a PDF |
| `POST` | `/ask` | Ask a question about an uploaded PDF |
| `GET` | `/sessions` | List all active sessions |
| `DELETE` | `/sessions/{session_id}` | Delete a session and free memory |

### Example Usage

**Upload a PDF:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@yourfile.pdf"
```

**Ask a question:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "yourfile",
    "question": "What is this document about?",
    "model_name": "llama3.2",
    "k_docs": 3
  }'
```

---

## 🖼️ Demo

> Upload a PDF → Ask a question → See the answer with source chunks

![demo](demo.png)

---

## 📁 Project Structure

```
pdf-qa-chatbot/
│
├── app.py              # Streamlit UI
├── main.py             # FastAPI backend
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md           # You are here
```

---

## 💡 How RAG Works Here

1. **Ingestion** — PDF is loaded and split into overlapping chunks
2. **Embedding** — Each chunk is converted to a vector using nomic-embed-text
3. **Indexing** — Vectors are stored in a FAISS index for fast similarity search
4. **Retrieval** — On each question, the top-k most similar chunks are fetched
5. **Generation** — The chunks are injected into a prompt and sent to llama3.2
6. **Response** — The LLM answers strictly based on the retrieved context

---

## 🔧 Troubleshooting

| Error | Fix |
|---|---|
| `ollama: command not found` | Install Ollama from ollama.com |
| `connection refused` | Run `ollama serve` in a separate terminal |
| `module not found` | Run `pip install -r requirements.txt` |
| Slow first response | Model is loading into memory, subsequent responses are faster |

---

## 👤 Author

**Soham Das**  
[github.com/sassyredfox](https://github.com/sassyredfox)  
soham.d.info@gmail.com
