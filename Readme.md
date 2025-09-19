# RAG PDF QA System with MongoDB

A Retrieval-Augmented Generation (RAG) system that allows you to ask questions about PDF documents using MongoDB for vector storage and HuggingFace embeddings.

## Features

- 📄 **PDF Processing**: Extract text from PDF documents
- 🔍 **Vector Search**: Semantic similarity search using HuggingFace embeddings
- 🗄️ **MongoDB Storage**: Scalable document storage with vector embeddings
- 🤖 **Local AI**: No external API keys required
- 🎯 **Clean Output**: Pure answers without extra formatting

## Prerequisites

- Python 3.8+
- MongoDB (local installation)
- Virtual environment (recommended)

## Installation

### 1. Clone and Setup Environment

```bash
# Navigate to project directory
cd rag-pdf-qa

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install and Start MongoDB

#### macOS (using Homebrew):
```bash
# Install MongoDB
brew tap mongodb/brew
brew install mongodb-community@6.0

# Start MongoDB service
brew services start mongodb-community@6.0
```

#### Ubuntu/Debian:
```bash
# Install MongoDB
sudo apt-get install mongodb

# Start MongoDB service
sudo systemctl start mongodb
```

#### Windows:
Download and install MongoDB from [mongodb.com](https://www.mongodb.com/try/download/community)

## Usage

### 1. Ingest PDF Documents

```bash
# Activate virtual environment
source venv/bin/activate

# Ingest a PDF into MongoDB
python ingest.py data/your-document.pdf --collection documents
```

**Parameters:**
- `data/your-document.pdf`: Path to your PDF file
- `--collection documents`: MongoDB collection name (optional, defaults to "documents")

### 2. Query the Knowledge Base

```bash
# Ask questions about your PDF
python query.py "What is this document about?" --collection documents
python query.py "What are the main topics discussed?" --collection documents
python query.py "What technologies are mentioned?" --collection documents
```

**Parameters:**
- `"Your question here"`: The question you want to ask
- `--collection documents`: MongoDB collection name (optional, defaults to "documents")
- `--k 4`: Number of chunks to retrieve (optional, defaults to 4)

## Example Workflow

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Make sure MongoDB is running
brew services list | grep mongodb

# 3. Ingest your PDF
python ingest.py data/resume.pdf --collection documents

# 4. Query the document
python query.py "What is this document about?" --collection documents
```

## Project Structure

```
rag-pdf-qa/
├── data/
│   └── rag.pdf                 # Your PDF documents
├── ingest.py                   # MongoDB ingestion script
├── query.py                    # MongoDB query script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── venv/                      # Virtual environment
```

## Configuration

### MongoDB Connection
The system uses the following MongoDB configuration:
- **URI**: `mongodb://localhost:27017/rag-pdf-qa`
- **Database**: `rag_pdf_qa`
- **Collection**: `documents` (configurable)

### Embeddings Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Type**: HuggingFace embeddings (local, no API key required)

## Troubleshooting

### MongoDB Connection Issues
```bash
# Check if MongoDB is running
brew services list | grep mongodb

# Start MongoDB if not running
brew services start mongodb-community@6.0

# Check MongoDB status
mongosh --eval "db.runCommand('ping')"
```

### Python Dependencies
```bash
# If you get import errors, reinstall dependencies
pip install -r requirements.txt

# For HuggingFace models, ensure you have enough disk space
# Models are downloaded on first use (~400MB)
```

### Memory Issues
- Reduce chunk size in `ingest.py` (currently 800 characters)
- Use fewer retrieved chunks with `--k` parameter
- Ensure sufficient RAM for embedding generation

## Advanced Usage

### Multiple Collections
```bash
# Store different documents in separate collections
python ingest.py data/document1.pdf --collection tech_docs
python ingest.py data/document2.pdf --collection legal_docs

# Query specific collections
python query.py "What are the technical requirements?" --collection tech_docs
```

### Custom Chunk Size
Edit `ingest.py` to modify chunk parameters:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Increase for larger chunks
    chunk_overlap=150     # Increase for more overlap
)
```

## Dependencies

- `langchain` - RAG framework
- `langchain-community` - Community integrations
- `langchain-huggingface` - HuggingFace embeddings
- `pymongo` - MongoDB driver
- `sentence-transformers` - Embedding models
- `scikit-learn` - Similarity calculations
- `pypdf2` - PDF processing

## License

This project is open source and available under the MIT License.
