# query.py - Groq LLM-powered RAG querying
import argparse
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Fix HuggingFace tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/rag-pdf-qa")
DB_NAME = "rag_pdf_qa"
COLLECTION_NAME = "documents"

# Groq API Key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

def answer_question(question: str, collection_name="documents", k: int = 4):
    """
    Answer question using Groq LLM with MongoDB retrieval
    """
    try:
        # Set Groq API key
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        
        # Load embeddings model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Get question embedding
        question_embedding = embeddings.embed_query(question)
        
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[collection_name]
        
        # Get all documents
        docs = list(collection.find())
        client.close()
        
        if not docs:
            print("No documents found in MongoDB collection.")
            return
        
        # Calculate similarities
        similarities = []
        for doc in docs:
            if 'embedding' in doc:
                # Calculate cosine similarity
                similarity = cosine_similarity([question_embedding], [doc['embedding']])[0][0]
                similarities.append((similarity, doc))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_docs = similarities[:k]
        
        if not top_docs:
            print("No relevant documents found.")
            return
        
        # Combine content from retrieved documents
        context = ""
        for similarity, doc in top_docs:
            content = doc.get('page_content', '').strip()
            context += content + "\n\n"
        
        # Initialize Groq LLM
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=1000
        )
        
        # Create prompt with context
        prompt = f"""Based on the following context from the document, please answer the question.

Context:
{context}

Question: {question}

Please provide a clear and comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please say so."""

        # Get answer from Groq
        response = llm.invoke(prompt)
        
        # Print clean answer
        print(response.content)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure MongoDB is running and the collection exists.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question", help="Question to ask the PDF knowledge base (wrap in quotes).")
    p.add_argument("--collection", default="documents", help="MongoDB collection name")
    p.add_argument("--k", type=int, default=4, help="how many chunks to retrieve")
    args = p.parse_args()
    answer_question(args.question, args.collection, args.k)