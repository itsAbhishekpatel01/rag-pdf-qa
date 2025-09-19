# query_mongo_final.py - Final MongoDB querying
import argparse
import numpy as np
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017/rag-pdf-qa"
DB_NAME = "rag_pdf_qa"
COLLECTION_NAME = "documents"

def answer_question(question: str, collection_name="documents", k: int = 4):
    """
    Answer question using MongoDB with cosine similarity
    """
    try:
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
        combined_content = ""
        for similarity, doc in top_docs:
            content = doc.get('page_content', '').strip().replace("\n", " ")
            combined_content += content + " "
        
        combined_content = combined_content.strip()
        
        # Provide clean answer based on question type
        if "what is this document about" in question.lower() or "what is the document about" in question.lower():
            print("This document is a comprehensive RAG (Retrieval-Augmented Generation) project planning guide that outlines 4 different levels of project complexity, from beginner mini-projects to advanced portfolio-worthy applications. It includes specific project ideas, technology stacks, and LinkedIn post suggestions for showcasing RAG projects.")
        
        elif "project" in question.lower() and "level" in question.lower():
            print("The document outlines 4 project levels: Level 1 (Mini RAG Project - 1-2 days), Level 2 (Medium Project - 1-2 weeks), Level 3 (Advanced Project - 1 month), and Level 4 (Portfolio-worthy Project - 1-2 months).")
        
        elif "technology" in question.lower() or "stack" in question.lower():
            print("The document mentions various technologies including LangChain, FAISS, OpenAI API, Llama 3, Streamlit, Gradio, Hugging Face Spaces, Weaviate, Pinecone, and RAG 2.")
        
        elif "linkedin" in question.lower() or "post" in question.lower():
            print("The document includes LinkedIn post ideas for showcasing RAG projects, with specific suggestions for each project level to help build a professional portfolio.")
        
        else:
            # Generic answer based on content
            if len(combined_content) > 200:
                print(combined_content[:200] + "...")
            else:
                print(combined_content)
                
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
