# Multimodal RAG with Gemini 2 & Pinecone

This system is an advanced **Multimodal RAG (Retrieval-Augmented Generation)** pipeline. It allows you to create an intelligent database that understands and correlates different media types: **Text, PDF, Images, and Video**.

Powered by Google's latest Gemini models (Embeddings v2 and Gemini 2.5), the system can answer complex questions by consulting technical documents, analyzing photos, or finding specific moments in videos.

## Purpose
- **Technical Search**: Index books and PDFs for rapid querying (e.g., "What is an Anti-Corruption Layer?").
- **Media Analysis**: Search through a gallery of photos or videos using natural language descriptions.
- **Intelligent Assistant**: A bot that answers questions based on a multimodal knowledge base.

## How to Run

### 1. Prerequisites
- Python 3.9+
- A [Google AI Studio](https://aistudio.google.com/) API Key
- A [Pinecone](https://www.pinecone.io/) Account

### 2. Configuration
Set your API keys in the `.env` file:
```env
GEMINI_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=multimodal-rag-index
```

### 3. Installation
Install the required libraries:
```bash
pip install -r requirements.txt
```

### 4. Index Initialization
Create the Pinecone index with the correct dimensions (3072):
```bash
python pinecone_init.py
```

### 5. Data Ingestion
Place your files (PDF, TXT, JPG, MP4, etc.) in the `data/` folder and run:
```bash
python ingest.py
```
*The system will automatically chunk PDFs and extract video frames.*

### 6. Querying
Ask questions about your data:
```bash
python query.py
```

## Project Structure
- `ingest.py`: The ingestion pipeline. Handles chunking and multimodal processing.
- `query.py`: The retrieval interface. Fetches context and generates answers.
- `pinecone_init.py`: Initial setup for the vector store.
- `data/`: The directory where you should place files for indexing.
