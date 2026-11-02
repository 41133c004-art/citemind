# CiteMind – Local RAG Study Companion
**Live App:** [click here to open the app](https://citemind-2uzgd54edwpso9xnvkgjtm.streamlit.app/) 
**Upload your lecture notes or PDFs → Ask questions → Get answers that are grounded in your materials and always cite the source.**

CiteMind is a lightweight Retrieval-Augmented Generation (RAG) tool designed for students. It keeps your documents private (embeddings and search run fully locally) and only sends the retrieved passages + question to a fast free LLM for the final answer.

---

## Why CiteMind?

Most chatbots hallucinate or mix general knowledge with your specific course content.  
CiteMind forces the model to answer **only** from the PDFs you upload and requires it to cite the exact source passages it used.

This makes it ideal for:
- Studying from lecture notes / textbooks
- Checking understanding before exams
- Creating reliable summaries and explanations
- Demonstrating modern, responsible AI techniques (RAG + grounding)

---

## Features

- 📄 Multi-PDF upload
- 🔍 Local semantic search (sentence-transformers + FAISS)
- 🧠 Grounded answers with mandatory citations
- 🎛️ Adjustable chunk size and number of retrieved passages
- ⚡ Fast generation via Groq (free tier)
- 🔒 Documents stay on your machine except for the final LLM call

---

## Quick Start

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/citemind.git
cd citemind
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a free Groq API key
1. Go to [https://console.groq.com](https://console.groq.com)
2. Create an account
3. Generate an API key (free tier is very generous and extremely fast)

### 3. Run the app
```bash
streamlit run app.py
```

Open the browser URL that appears (usually http://localhost:8501).

Paste your Groq key in the sidebar → upload one or more PDFs → click **Build Knowledge Base** → start asking questions.

---

## How it works (technical overview)

1. **Extraction** – `pypdf` extracts text page by page and tags each page.
2. **Chunking** – Recursive character splitter creates overlapping chunks (~800 characters).
3. **Embedding** – `all-MiniLM-L6-v2` (local, fast, good quality) turns every chunk into a vector.
4. **Indexing** – FAISS `IndexFlatL2` stores the vectors for exact nearest-neighbor search.
5. **Retrieval** – At query time the question is embedded and the top-k most similar chunks are fetched.
6. **Generation** – The retrieved chunks + question are sent to Groq (Llama 3.3 70B) with a strict system prompt that forces citation and forbids hallucination outside the sources.

All heavy lifting except the final generation happens on your machine.

---

## Project Structure

```
citemind/
├── app.py              # Main Streamlit application
├── requirements.txt    # Dependencies
├── README.md           # This file
├── .gitignore
└── report/             # Project report (optional)
```

---

## Evaluation & Limitations

**What was checked**
- Citation accuracy on sample lecture PDFs
- Behavior when the answer is *not* in the documents (correctly refuses)
- Latency and quality with different chunk sizes / top-k values

**Known limitations**
- Pure text extraction (no OCR for scanned PDFs yet)
- Quality depends on the clarity of the source PDF
- Requires internet only for the final LLM call (embeddings are local)

---

## Future Improvements

- Support for scanned PDFs (OCR)
- Multi-turn conversation memory
- Automatic quiz / flashcard generation from the same knowledge base
- Option for fully offline generation (Ollama)
- Export of Q&A sessions

---

## License

MIT – feel free to use, modify, and share.

---

Built as a midterm project demonstrating practical RAG and responsible AI use.
