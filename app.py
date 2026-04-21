import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF Q&A Chatbot")
st.caption("Powered by LangChain · FAISS · Ollama")

# ── Sidebar config ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Ollama Model",
        ["llama3.2", "llama3", "mistral", "phi3", "gemma2"],
        index=0
    )
    embed_model = st.selectbox(
        "Embedding Model",
        ["nomic-embed-text", "mxbai-embed-large"],
        index=0
    )
    chunk_size = st.slider("Chunk Size", 256, 1024, 512, step=128)
    chunk_overlap = st.slider("Chunk Overlap", 0, 256, 64, step=32)
    k_docs = st.slider("Chunks to retrieve (k)", 1, 8, 3)
    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ── Helper: build vectorstore ─────────────────────────────────────────────────
def build_vectorstore(uploaded_file, chunk_size, chunk_overlap, embed_model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=embed_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)

# ── Helper: build QA chain ────────────────────────────────────────────────────
def build_qa_chain(vectorstore, model_name, k_docs):
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

    llm = Ollama(model=model_name, temperature=0.1)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_docs})

    def run_chain(question):
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        final_prompt = prompt.format(context=context, question=question)
        answer = llm.invoke(final_prompt)
        return answer, docs

    return run_chain

# ── PDF Upload ────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
    with st.spinner("⏳ Processing PDF — chunking and embedding..."):
        try:
            vs, n_chunks = build_vectorstore(
                uploaded_file, chunk_size, chunk_overlap, embed_model
            )
            st.session_state.vectorstore = vs
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.messages = []
            st.success(f"✅ Ready! Indexed **{n_chunks}** chunks from `{uploaded_file.name}`")
        except Exception as e:
            st.error(f"❌ Error processing PDF: {e}")

# ── Chat interface ────────────────────────────────────────────────────────────
if st.session_state.vectorstore is None:
    st.info("👆 Upload a PDF to get started.")
else:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Source chunks"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.caption(f"**Chunk {i} — Page {src.metadata.get('page', '?') + 1}**")
                        st.write(src.page_content)

    # User input
    if question := st.chat_input("Ask something about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    chain = build_qa_chain(
                        st.session_state.vectorstore, model_name, k_docs
                    )
                    answer, sources = chain(question)

                    st.markdown(answer)
                    if sources:
                        with st.expander("📎 Source chunks"):
                            for i, src in enumerate(sources, 1):
                                st.caption(f"**Chunk {i} — Page {src.metadata.get('page', '?') + 1}**")
                                st.write(src.page_content)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    err = f"❌ Error: {e}"
                    st.error(err)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err
                    })