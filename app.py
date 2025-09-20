# app.py - Streamlit UI for RAG PDF QA System
import streamlit as st
import os
import tempfile
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import time

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

# Page configuration
st.set_page_config(
    page_title="RAG PDF QA System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #e8f4fd;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def load_documents_from_mongo(collection_name="documents"):
    """Load documents from MongoDB"""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[collection_name]
        docs = list(collection.find())
        client.close()
        return docs
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return []

def get_answer(question: str, collection_name="documents", k: int = 4):
    """Get answer using Groq LLM with MongoDB retrieval"""
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
            return "No documents found in MongoDB collection."
        
        # Calculate similarities
        similarities = []
        for doc in docs:
            if 'embedding' in doc:
                similarity = cosine_similarity([question_embedding], [doc['embedding']])[0][0]
                similarities.append((similarity, doc))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_docs = similarities[:k]
        
        if not top_docs:
            return "No relevant documents found."
        
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
        return response.content
        
    except Exception as e:
        return f"Error: {e}"

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 RAG PDF QA System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # MongoDB Status
        st.subheader("Database Status")
        try:
            client = MongoClient(MONGO_URI)
            client.admin.command('ping')
            st.success("✅ MongoDB Connected")
            client.close()
        except Exception as e:
            st.error(f"❌ MongoDB Error: {e}")
        
        # Groq API Status
        st.subheader("API Status")
        if GROQ_API_KEY:
            st.success("✅ Groq API Key Loaded")
        else:
            st.error("❌ Groq API Key Missing")
        
        # Collection selector
        st.subheader("Collection Settings")
        collection_name = st.text_input("Collection Name", value="documents")
        k_chunks = st.slider("Number of chunks to retrieve", 1, 10, 4)
        
        # Document count
        try:
            client = MongoClient(MONGO_URI)
            db = client[DB_NAME]
            collection = db[collection_name]
            doc_count = collection.count_documents({})
            st.info(f"📄 Documents in collection: {doc_count}")
            client.close()
        except:
            st.warning("Could not retrieve document count")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat with your PDF")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your PDF..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_answer(prompt, collection_name, k_chunks)
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.header("📊 System Info")
        
        # Model information
        st.subheader("🤖 AI Model")
        st.info("**Groq Llama 3.1-8b-instant**")
        st.caption("Fast inference with high quality responses")
        
        # Embeddings info
        st.subheader("🔍 Embeddings")
        st.info("**HuggingFace sentence-transformers**")
        st.caption("all-MiniLM-L6-v2 model")
        
        # Database info
        st.subheader("🗄️ Database")
        st.info("**MongoDB**")
        st.caption("Scalable document storage")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
        # Example questions
        st.subheader("💡 Example Questions")
        example_questions = [
            "What is this document about?",
            "What are the main topics?",
            "What skills are mentioned?",
            "What projects are described?",
            "What technologies are used?"
        ]
        
        for question in example_questions:
            if st.button(f"❓ {question}", key=f"example_{question}"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()

if __name__ == "__main__":
    main()
