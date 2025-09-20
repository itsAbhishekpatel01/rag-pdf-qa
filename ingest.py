# ingest.py - MongoDB implementation
import os
import argparse
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Fix HuggingFace tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/rag-pdf-qa")
DB_NAME = "rag_pdf_qa"
COLLECTION_NAME = "documents"

def build_index(pdf_path: str, collection_name: str = "documents"):
    """
    Build MongoDB index from PDF
    """
    print(f"Loading PDF: {pdf_path}")
    
    # 1) Load PDF -> documents
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDF")

    # 2) Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} text chunks")

    # 3) Setup embeddings
    print("Setting up HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4) Generate embeddings for each chunk
    print("Generating embeddings...")
    chunk_embeddings = []
    for chunk in chunks:
        embedding = embeddings.embed_query(chunk.page_content)
        chunk_embeddings.append(embedding)

    # 5) Store in MongoDB
    print("Storing in MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[collection_name]
    
    # Clear existing documents
    collection.delete_many({})
    
    # Store documents with embeddings
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        doc_data = {
            "chunk_id": i,
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
            "source": chunk.metadata.get("source", "unknown"),
            "embedding": embedding
        }
        collection.insert_one(doc_data)
    
    print(f"Successfully stored {len(chunks)} chunks in MongoDB collection '{collection_name}'")
    print(f"Database: {DB_NAME}")
    print(f"Collection: {collection_name}")
    print(f"Connection: {MONGO_URI}")
    
    client.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pdf", help="path to pdf (e.g., data/my_notes.pdf)")
    p.add_argument("--collection", default="documents", help="MongoDB collection name")
    args = p.parse_args()
    build_index(args.pdf, args.collection)
