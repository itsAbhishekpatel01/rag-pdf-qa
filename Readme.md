# RAG PDF QA System

A modern Retrieval-Augmented Generation (RAG) system with a beautiful Streamlit UI that allows you to chat with your PDF documents using MongoDB for storage, HuggingFace embeddings for retrieval, and Groq LLM for generation.

## ✨ Features

- 🎨 **Beautiful Streamlit UI**: Modern, responsive web interface
- 📄 **PDF Upload & Processing**: Drag-and-drop PDF upload with automatic processing
- 💬 **Interactive Chat**: Real-time conversation with your documents
- 🔍 **Semantic Search**: Advanced vector similarity search using HuggingFace embeddings
- 🤖 **Groq LLM**: Fast, high-quality responses with Llama 3.1-8b-instant
- 🗄️ **MongoDB Storage**: Scalable document and vector storage
- ⚙️ **Configurable**: Customizable chunk sizes, collections, and retrieval parameters

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to project
cd rag-pdf-qa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/rag-pdf-qa
DB_NAME=rag_pdf_qa
COLLECTION_NAME=documents

# Groq API Key (get from https://console.groq.com/)
GROQ_API_KEY=your_groq_api_key_here

```

### 3. Start MongoDB

**macOS (Homebrew):**
```bash
brew tap mongodb/brew
brew install mongodb-community@6.0
brew services start mongodb-community@6.0
```

**Ubuntu/Debian:**
```bash
sudo apt-get install mongodb
sudo systemctl start mongodb
```

**Windows:** Download from [mongodb.com](https://www.mongodb.com/try/download/community)

### 4. Run the Application

```bash
# Start the Streamlit app
streamlit run app.py
```

Open your browser to `http://localhost:8501` and start chatting with your PDFs!

## 📖 Usage

### Web Interface (Recommended)

1. **Upload Documents**: Use the file uploader to add PDF documents
2. **Chat Interface**: Ask questions about your uploaded documents
3. **Configuration**: Adjust settings in the sidebar (chunk size, collection, etc.)

### Command Line Interface

```bash
# Ingest PDF documents
python ingest.py data/your-document.pdf --collection documents

# Query documents
python query.py "What is this document about?" --collection documents --k 4
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   MongoDB        │────│   Groq LLM      │
│   (Frontend)    │    │   (Vector Store) │    │   (Generation)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Upload    │    │   HuggingFace    │    │   Chat Interface │
│   & Processing  │    │   Embeddings    │    │   & Responses    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGO_URI` | MongoDB connection string | `mongodb://localhost:27017/rag-pdf-qa` |
| `DB_NAME` | Database name | `rag_pdf_qa` |
| `COLLECTION_NAME` | Collection name | `documents` |
| `GROQ_API_KEY` | Groq API key for LLM | Required |

### Model Configuration

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: `llama-3.1-8b-instant` (Groq)
- **Chunk Size**: 800 characters (configurable)
- **Chunk Overlap**: 120 characters
- **Retrieval**: Top 4 most similar chunks

## 🚀 Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Connect your GitHub repo to [Streamlit Cloud](https://share.streamlit.io/)
3. Add secrets in Streamlit Cloud dashboard:
   ```
   MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
   GROQ_API_KEY=your_groq_key
   ```

### Local Production

```bash
# Install production dependencies
pip install -r requirements.txt

# Run with production settings
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 🔧 Troubleshooting

### Common Issues

**MongoDB Connection Failed:**
```bash
# Check MongoDB status
brew services list | grep mongodb
# Start if needed
brew services start mongodb-community@6.0
```

**Groq API Key Missing:**
- Get API key from [console.groq.com](https://console.groq.com/)
- Add to `.env` file or Streamlit secrets

**Memory Issues:**
- Reduce chunk size in `ingest.py`
- Use fewer retrieved chunks (`--k` parameter)
- Ensure sufficient RAM for embeddings

**Import Errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
rag-pdf-qa/
├── app.py                 # Streamlit web application
├── ingest.py              # PDF ingestion script
├── query.py               # Command-line query script
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (local)
├── .streamlit/
│   └── secrets.toml       # Streamlit secrets (production)
├── data/                  # PDF documents
└── venv/                  # Virtual environment
```

## 🛠️ Development

### Adding New Features

1. **New Embedding Models**: Update `HuggingFaceEmbeddings` model name
2. **Different LLMs**: Replace `ChatGroq` with other LangChain LLMs
3. **Custom UI**: Modify Streamlit components in `app.py`
4. **Database Changes**: Update MongoDB schema in `ingest.py`

### Testing

```bash
# Test MongoDB connection
python -c "from pymongo import MongoClient; print(MongoClient().admin.command('ping'))"

# Test Groq API
python -c "from langchain_groq import ChatGroq; print(ChatGroq().invoke('Hello'))"
```

## 📚 Dependencies

- **Core**: `streamlit`, `langchain`, `langchain-groq`
- **Database**: `pymongo`
- **Embeddings**: `langchain-huggingface`, `sentence-transformers`
- **ML**: `scikit-learn`, `numpy`
- **PDF**: `pypdf2`
- **Utils**: `python-dotenv`

## 📄 License

MIT License - feel free to use this project for your own applications!

---

**Built with ❤️ using Streamlit, MongoDB, HuggingFace, and Groq**

