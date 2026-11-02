"""
CiteMind – Local RAG Study Companion
"""

import streamlit as st
from typing import List, Tuple
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="CiteMind – Local RAG Study Companion",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text_parts = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(f"[Page {i + 1}]\n{page_text}")
    return "\n\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def build_faiss_index(chunks: List[str], model: SentenceTransformer):
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings


def retrieve(query: str, index, chunks: List[str], model: SentenceTransformer, top_k: int = 4):
    query_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_emb, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            results.append((chunks[idx], float(dist)))
    return results


def generate_answer_with_groq(query: str, context_chunks: List[str], api_key: str) -> str:
    from groq import Groq

    client = Groq(api_key=api_key.strip())

    context = "\n\n---\n\n".join(
        [f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )

    system_prompt = """You are CiteMind, a careful study assistant.
Your ONLY job is to answer the student's question using the provided source material.
Rules:
1. Base every claim on the sources. If the answer is not in the sources, say "I could not find this information in the uploaded materials."
2. Always cite the source number(s) you used, e.g. (Source 2).
3. Be concise, clear, and educational.
4. Never invent facts that are not present in the sources."""

    user_prompt = f"""Sources:
{context}

Student question: {query}

Answer (with citations):"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("📚 CiteMind")
    st.markdown("**Local RAG Study Companion**")
    st.markdown("---")

    st.subheader("🔑 API Key")
    groq_key = st.text_input(
        "Paste your Groq API Key here",
        type="password",
        help="Get a free key at https://console.groq.com"
    )

    st.subheader("⚙️ Retrieval Settings")
    top_k = st.slider("Number of chunks to retrieve", 2, 8, 4)
    chunk_size = st.slider("Chunk size (characters)", 400, 1200, 800, 100)

    st.markdown("---")
    st.caption("Embeddings stay local • Generation via Groq")


# ---------- MAIN ----------
st.title("CiteMind")
st.markdown("### Upload your lecture notes → Ask questions → Get cited answers")

uploaded_files = st.file_uploader(
    "Upload PDF file(s)",
    type=["pdf"],
    accept_multiple_files=True,
)

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "model" not in st.session_state:
    st.session_state.model = None

if st.button("🔨 Build Knowledge Base", type="primary", use_container_width=True) and uploaded_files:
    with st.spinner("Building knowledge base..."):
        all_text = [extract_text_from_pdf(f) for f in uploaded_files]
        full_text = "\n\n".join(all_text)
        chunks = chunk_text(full_text, chunk_size=chunk_size)

        model = load_embedding_model()
        index, _ = build_faiss_index(chunks, model)

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.model = model
        st.success(f"✅ Ready! {len(chunks)} chunks indexed.")

st.markdown("---")
st.subheader("💬 Ask a question")

if st.session_state.index is None:
    st.info("👆 Upload PDF(s) and click **Build Knowledge Base** first.")
else:
    query = st.text_input("Your question", placeholder="Ask something about your notes...")

    if st.button("Get Answer", type="primary") and query.strip():
        if not groq_key:
            st.error("Please paste your Groq API key in the sidebar.")
        else:
            with st.spinner("Thinking..."):
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

                    with st.expander("📎 Sources used"):
                        for i, (chunk, dist) in enumerate(results, 1):
                            st.markdown(f"**Source {i}**")
                            st.text(chunk[:700] + ("..." if len(chunk) > 700 else ""))
                except Exception as e:
                    st.error(f"Error calling Groq: {e}")
