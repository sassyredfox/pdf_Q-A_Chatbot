from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

app = FastAPI(
    title="PDF Q&A API",
    description="RAG-powered PDF question answering using LangChain + FAISS + Ollama",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory store ───────────────────────────────────────────────────────────
vectorstores = {}

# ── Models ────────────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    session_id: str
    question: str
    model_name: str = "llama3.2"
    k_docs: int = 3

class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "message": "PDF Q&A API is live"}


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    embed_model: str = "nomic-embed-text"
):
    """Upload a PDF and index it. Returns a session_id to use for questions."""

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Load and split
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        os.unlink(tmp_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(documents)

        # Embed and index
        embeddings = OllamaEmbeddings(model=embed_model)
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Store with session id
        session_id = file.filename.replace(".pdf", "").replace(" ", "_")
        vectorstores[session_id] = vectorstore

        return {
            "session_id": session_id,
            "filename": file.filename,
            "chunks_indexed": len(chunks),
            "message": "PDF indexed successfully. Use session_id to ask questions."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    """Ask a question about an uploaded PDF using its session_id."""

    if request.session_id not in vectorstores:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' not found. Please upload a PDF first."
        )

    try:
        vectorstore = vectorstores[request.session_id]
        retriever = vectorstore.as_retriever(search_kwargs={"k": request.k_docs})

        # Retrieve relevant chunks
        docs = retriever.invoke(request.question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Build prompt
        prompt_template = """You are a helpful assistant answering questions based strictly on the provided context.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Call LLM
        llm = Ollama(model=request.model_name, temperature=0.1)
        final_prompt = prompt.format(context=context, question=request.question)
        answer = llm.invoke(final_prompt)

        # Extract source snippets
        sources = [doc.page_content[:200] + "..." for doc in docs]

        return QuestionResponse(
            answer=answer,
            sources=sources,
            session_id=request.session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
def list_sessions():
    """List all active sessions (uploaded PDFs)."""
    return {
        "active_sessions": list(vectorstores.keys()),
        "count": len(vectorstores)
    }


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a session and free memory."""
    if session_id not in vectorstores:
        raise HTTPException(status_code=404, detail="Session not found.")
    del vectorstores[session_id]
    return {"message": f"Session '{session_id}' deleted successfully."}