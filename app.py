"""
CiteMind – Local RAG Study Companion
Upload PDFs → Ask questions → Get answers grounded in your materials with citations.
"""

import streamlit as st
import os
import tempfile
from typing import List, Tuple
import numpy as np

# Lazy imports to speed up startup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CiteMind – Local RAG Study Companion",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    """Load a lightweight, high-quality embedding model (runs locally)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from an uploaded PDF file."""
    reader = PdfReader(uploaded_file)
    text_parts = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(f"[Page {i + 1}]\n{page_text}")
    return "\n\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks suitable for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def build_faiss_index(chunks: List[str], model: SentenceTransformer):
    """Embed chunks and build a FAISS index."""
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings


def retrieve(
    query: str,
    index,
    chunks: List[str],
    model: SentenceTransformer,
    top_k: int = 4,
) -> List[Tuple[str, float]]:
    """Return top-k most relevant chunks with their distances."""
    query_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_emb, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            results.append((chunks[idx], float(dist)))
    return results


def generate_answer_with_groq(
    query: str,
    context_chunks: List[str],
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
) -> str:
    """Generate a grounded answer using Groq (free tier available)."""
    from groq import Groq

    client = Groq(api_key=api_key)

    context = "\n\n---\n\n".join(
        [f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )

    system_prompt = """You are CiteMind, a careful study assistant.
Your ONLY job is to answer the student's question using the provided source material.
Rules:
1. Base every claim on the sources. If the answer is not in the sources, say "I could not find this information in the uploaded materials."
2. Always cite the source number(s) you used, e.g. (Source 2).
3. Be concise, clear, and educational.
4. If the question is ambiguous, ask for clarification.
5. Never invent facts, numbers, or quotes that are not present in the sources."""

    user_prompt = f"""Sources:
{context}

Student question: {query}

Answer (with citations):"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Sidebar – configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 CiteMind")
    st.markdown("**Local RAG Study Companion**")
    st.markdown("---")

    st.subheader("🔑 API Key")
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get a free key at https://console.groq.com (very fast & free tier)",
        value=os.getenv("GROQ_API_KEY", ""),
    )

    st.subheader("⚙️ Retrieval Settings")
    top_k = st.slider("Number of chunks to retrieve", 2, 8, 4)
    chunk_size = st.slider("Chunk size (characters)", 400, 1200, 800, 100)

    st.markdown("---")
    st.markdown(
        """
        **How it works**
        1. Upload one or more PDFs  
        2. CiteMind builds a local vector index  
        3. Ask any question about the content  
        4. Answers are grounded + cited
        """
    )
    st.markdown("---")
    st.caption("Made for midterm project • No data leaves your machine except the LLM call")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("CiteMind")
st.markdown("### Upload your lecture notes or textbooks → Ask questions → Get cited answers")

uploaded_files = st.file_uploader(
    "Upload PDF file(s)",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload multiple PDFs. They will be processed together.",
)

# Session state for the index
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "model" not in st.session_state:
    st.session_state.model = None

process_btn = st.button("🔨 Build Knowledge Base", type="primary", use_container_width=True)

if process_btn and uploaded_files:
    with st.spinner("Extracting text and building local index… (first time downloads embedding model)"):
        all_text = []
        for f in uploaded_files:
            text = extract_text_from_pdf(f)
            all_text.append(text)

        full_text = "\n\n".join(all_text)
        chunks = chunk_text(full_text, chunk_size=chunk_size)

        model = load_embedding_model()
        index, _ = build_faiss_index(chunks, model)

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.model = model

        st.success(f"✅ Knowledge base ready! {len(chunks)} chunks indexed from {len(uploaded_files)} file(s).")

# Chat interface
st.markdown("---")
st.subheader("💬 Ask a question")

if st.session_state.index is None:
    st.info("👆 Upload PDF(s) and click **Build Knowledge Base** to start.")
else:
    query = st.text_input(
        "Your question",
        placeholder="e.g. What is the main idea of chapter 3? Explain the key concept of X.",
        key="query_input",
    )

    if st.button("Get Answer", type="primary") and query.strip():
        if not groq_key:
            st.error("Please enter your Groq API key in the sidebar (free at console.groq.com).")
        else:
            with st.spinner("Retrieving relevant passages and generating answer…"):
                results = retrieve(
                    query,
                    st.session_state.index,
                    st.session_state.chunks,
                    st.session_state.model,
                    top_k=top_k,
                )
                context_chunks = [chunk for chunk, _ in results]

                try:
                    answer = generate_answer_with_groq(query, context_chunks, groq_key)
                    st.markdown("### Answer")
                    st.markdown(answer)

                    with st.expander("📎 Retrieved source passages (what the AI saw)"):
                        for i, (chunk, dist) in enumerate(results, 1):
                            st.markdown(f"**Source {i}** (distance: {dist:.3f})")
                            st.text(chunk[:800] + ("…" if len(chunk) > 800 else ""))
                            st.markdown("---")
                except Exception as e:
                    st.error(f"Error calling Groq: {e}")
                    st.info("Make sure your API key is valid and you have remaining free credits.")

# Footer
st.markdown("---")
st.caption(
    "CiteMind • Local embeddings (all-MiniLM-L6-v2) + FAISS • Generation via Groq • "
    "Your documents never leave this machine except for the final generation call."
)
