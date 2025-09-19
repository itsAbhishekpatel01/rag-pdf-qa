Nice — let’s build the Level-1 PDF Q&A RAG project using Gemini Pro. Below you’ll get a step-by-step procedure, the exact file structure, runnable code for ingestion + query, and the commands to run everything.

I’ll assume you’re on Linux/macOS (Windows notes included where relevant). I’ll also give two embedding options (Google embeddings or local HuggingFace) so you can pick.

What I’ll give you

Project file structure

requirements.txt (exact packages to install)

.env.example and env var notes (Gemini API key guidance). 
Google AI for Developers
+1

ingest.py — load a PDF → split → embed → build & save FAISS index. 
LangChain

query.py — load index → run RetrievalQA with Gemini Pro (LangChain wrapper ChatGoogleGenerativeAI). 
LangChain

Quick run commands and tips.

1) Project file structure
rag-pdf-qa/
├─ data/                     # put your PDFs here (e.g., my_notes.pdf)
├─ index/                    # saved FAISS index will go here
├─ ingest.py                 # build the index from PDF(s)
├─ query.py                  # simple CLI to ask questions
├─ requirements.txt
├─ .env.example
└─ README.md                 # short usage notes

2) Install & env setup

Create virtualenv & install:

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt


Important env vars

GEMINI_API_KEY — used by Google’s official SDK / quickstart. Set this for the google SDK calls. 
Google AI for Developers

GOOGLE_API_KEY — LangChain’s langchain-google-genai integration will read this (or you can pass the key in code). Setting both to the same key is simplest. 
LangChain

Example:

export GEMINI_API_KEY="ya...your_key..."
export GOOGLE_API_KEY="$GEMINI_API_KEY"

3) requirements.txt
langchain
langchain-google-genai
google-genai
faiss-cpu
pypdf2
sentence-transformers
python-dotenv


(If you plan to use a Streamlit demo later add streamlit.)

langchain-google-genai is the LangChain integration for Gemini/Gemini-embeddings. 
PyPI

google-genai is Google’s official Python SDK / Gemini client. 
Google AI for Developers

4) ingest.py — ingest PDF and build FAISS index

Create ingest.py:

# ingest.py
import os
import argparse
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Option A: Google embeddings (Gemini embeddings via langchain-google-genai)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Option B (fallback/local): HuggingFace embeddings (no external API)
# from langchain.embeddings import HuggingFaceEmbeddings as LocalEmbeddings

def build_index(pdf_path: str, index_path: str = "index", use_google_embeddings: bool = True):
    # 1) load PDF -> documents
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2) split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = text_splitter.split_documents(docs)

    # 3) embeddings
    if use_google_embeddings:
        # model name can be adjusted according to docs (e.g. "models/embedding-001" or specific gemini embedding)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    else:
        # local option (uncomment import at top if you use this)
        # embeddings = LocalEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        raise RuntimeError("Local embeddings path currently disabled. Toggle code to enable.")

    # 4) create FAISS vectorstore and save locally
    vector_store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_path, exist_ok=True)
    vector_store.save_local(index_path)
    print(f"Saved FAISS index to '{index_path}' with {len(chunks)} chunks.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pdf", help="path to pdf (e.g., data/my_notes.pdf)")
    p.add_argument("--index", default="index", help="folder to save index")
    p.add_argument("--use-google-embeddings", action="store_true", help="use Gemini/Google embeddings (default)")
    args = p.parse_args()
    build_index(args.pdf, args.index, use_google_embeddings=True)


Notes:

PyPDFLoader extracts text and sets source metadata.

The GoogleGenerativeAIEmbeddings class is provided by langchain-google-genai (you can swap to HuggingFace locally if you want to avoid embedding API calls). 
LangChain

5) query.py — CLI to ask questions (loads saved FAISS index)

Create query.py:

# query.py
import argparse
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# LangChain -> Google Gemini LLM wrapper + embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

def load_faiss(index_path="index"):
    # Must use the same embedding class used to create the index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return db

def answer_question(question: str, index_path="index", model_name="gemini-pro", k: int = 4):
    db = load_faiss(index_path)
    retriever = db.as_retriever(search_kwargs={"k": k})
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)

    # Create a RetrievalQA chain; return_source_documents=True gives you the chunks used
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    res = qa({"query": question})
    answer = res["result"]
    source_docs = res.get("source_documents", [])

    print("\n=== ANSWER ===\n")
    print(answer)
    print("\n=== SOURCES (top chunks) ===\n")
    for i, d in enumerate(source_docs):
        src = d.metadata.get("source", "unknown")
        snippet = d.page_content.strip().replace("\n", " ")[:300]
        print(f"{i+1}. {src} — {snippet}...")
    print()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question", help="Question to ask the PDF knowledge base (wrap in quotes).")
    p.add_argument("--index", default="index")
    p.add_argument("--model", default="gemini-pro", help="Gemini model (e.g., gemini-pro)")
    p.add_argument("--k", type=int, default=4, help="how many chunks to retrieve")
    args = p.parse_args()
    answer_question(args.question, args.index, args.model, args.k)


Notes:

We use ChatGoogleGenerativeAI(model="gemini-pro") so the chain uses your Gemini Pro model. langchain-google-genai provides this wrapper. 
LangChain

return_source_documents=True helps you show which chunks the LLM used (great for transparency).

6) Quick run (one-liners)

activate env, install:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


set API key(s):

export GEMINI_API_KEY="ya...YOUR_KEY..."
export GOOGLE_API_KEY="$GEMINI_API_KEY"


(If on Windows PowerShell: setx GEMINI_API_KEY "ya..."; setx GOOGLE_API_KEY "%GEMINI_API_KEY%")

ingest a PDF:

python ingest.py data/my_notes.pdf --index index


ask a question:

python query.py "Summarize the main idea of the PDF." --index index --model gemini-pro


You should see the answer plus the top chunks used as sources.

(If you prefer to avoid Google embeddings, change ingest.py to use HuggingFaceEmbeddings from langchain.embeddings — this keeps everything local but embeddings quality/behavior will differ.)

7) Helpful notes & best practices

Save the index folder (index/) — FAISS supports save_local / load_local so you don’t re-embed every time. 
LangChain

Chunk size: 500–1000 chars with overlap 50–150 is a good starting point.

Retrieval k: start with 3–5, later add MMR or hybrid search (BM25 + embedding) if you see irrelevant results.

Add a prompt wrapper that tells Gemini to cite or to answer concisely: supply a custom prompt template into the chain later for better control.

If you hit rate limits, check your Google quota (Gemini API). Google quickstart has details on keys and quotas. 
Google AI for Developers

8) Sources (docs I used to assemble this)

Gemini API quickstart & env var guidance. 
Google AI for Developers

LangChain Google GenAI integration + ChatGoogleGenerativeAI docs. 
LangChain

langchain-google-genai package (PyPI / docs). 
PyPI

LangChain FAISS vectorstore save/load docs. 
LangChain

LangChain Google embeddings (GoogleGenerativeAIEmbeddings) docs. 