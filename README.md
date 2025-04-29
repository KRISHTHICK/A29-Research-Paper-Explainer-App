# A29-Research-Paper-Explainer-App
GenAI

Awesome! Here's a **Research Paper Explainer App** using Streamlit + Ollama + LangChain + OCR for PDFs.

---

### ✅ Features:
- Upload multiple research papers (PDFs)
- Extracts both text & images (OCR)
- Asks questions like:
  - “Summarize contributions”
  - “What methods are used?”
  - “Any dataset names or results?”
- All works offline (no API keys)

---

### 🧠 Code: `research_explainer.py`

```python
# Required packages:
# pip install streamlit langchain chromadb ollama pdfplumber pdf2image pytesseract
# sudo apt-get install poppler-utils tesseract-ocr

import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import tempfile

def extract_text_and_ocr(uploaded_files):
    all_text = ""
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"

        images = convert_from_path(tmp_path)
        for img in images:
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                all_text += ocr_text + "\n"

    return all_text

@st.cache_resource
def create_vector_store(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    embed_model = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma.from_documents(docs, embed_model)

# UI
st.set_page_config(page_title="📚 Research Paper Explainer", layout="wide")
st.title("📚 Research Paper Explainer")
st.caption("Ask anything about uploaded research papers!")

uploaded_files = st.sidebar.file_uploader("Upload Research PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents..."):
        full_text = extract_text_and_ocr(uploaded_files)
        vector_store = create_vector_store(full_text)
        retriever = vector_store.as_retriever()
        llm = Ollama(model="llama3")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.chat_input("Ask about your paper (methods, results, summary...)")

    if user_input:
        with st.spinner("Thinking..."):
            response = qa_chain.run(user_input)
            st.session_state.history.append((user_input, response))

    for q, a in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
else:
    st.info("⬅️ Upload one or more research PDFs to begin.")

st.sidebar.markdown("🔍 Suggested Prompts:")
st.sidebar.markdown("- What is the main contribution?\n- Summarize methodology\n- Mention datasets used\n- Describe experimental setup")
```

---

### 🚀 How to Run Locally

1. Save this code as `research_explainer.py`
2. Create a folder (e.g., `paper-rag-bot`)
3. Open in VS Code
4. Run:

```bash
pip install streamlit langchain chromadb ollama pdfplumber pdf2image pytesseract
sudo apt install poppler-utils tesseract-ocr
streamlit run research_explainer.py
```

> You’ll also need Ollama running with:
```bash
ollama run llama3
ollama run nomic-embed-text
```

---

### 📌 Ready for GitHub?

If you'd like, I can:
- Help structure this into a GitHub-ready project
- Add a `README.md` and `.gitignore`
- Help deploy it (e.g., Hugging Face Spaces, Render)

Want me to package and push this to GitHub with docs?
