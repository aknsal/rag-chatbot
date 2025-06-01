# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers customer support questions based on AngelOne documentation.

## Features

- **Document Processing**: Processes PDF, TXT, and MD files from a folder
- **Vector Database**: Uses FAISS for efficient document retrieval
- **Gemini AI Integration**: Powered by Google's Gemini 1.5 Flash model
- **Web Interface**: Clean Streamlit interface for chatting
- **Source Attribution**: Shows sources for each response

## Setup

1. **Install Dependencies**:

   ```bash
   pip install streamlit google-generativeai faiss-cpu numpy pandas pdfplumber
   ```

2. **Set Environment Variable**:

   ```bash
   export GEMINI_API_KEY="your_gemini_api_key_here"
   ```

3. **Prepare Your Documents**:
   - Create a folder with your documents (PDF, TXT, MD files)
   - For AngelOne support, just run the scrape AngelOne support page button

## Usage

### Step 1: Process Documents

Run the document processing script to add documents to the vector database:

```bash
python process_documents.py --folder /path/to/your/documents
```

Options:

- `--folder`: Path to folder containing documents
- `--clear-existing`: Clear existing embeddings before adding new ones
- `--verify`: Verify setup without processing documents

### Step 2: Start the Chatbot

Run the Streamlit application:

```bash
streamlit run simple_app.py --server.port 5000
```

### Step 3: Chat

- Open your browser to the provided URL
- Ask questions about the documentation
- The chatbot will only answer based on the processed documents

## Document Processing Details

The script supports:

- **PDF files**: Extracts text and tables
- **Text files**: .txt and .md files
- **Chunking**: Automatically splits large documents into manageable chunks
- **Embeddings**: Generates 768-dimensional embeddings using Gemini

## File Structure

```
├── simple_app.py              # Main Streamlit application
├── process_documents.py       # Document processing script
├── vector_store.py           # FAISS vector database implementation
├── document_processor.py     # PDF and text processing utilities
├── utils.py                  # Helper functions
├── web_scraper.py           # Web scraping utilities (optional)
└── README.md                # This file
```

## Example Usage

1. **Process AngelOne Documentation**:

   ```bash
   # Download docs to a folder called 'test_documents'
   python process_documents.py --folder test_documents
   ```

2. **Start Chatbot**:

   ```bash
   streamlit run simple_app.py --server.port 5000
   ```

3. **Ask Questions**:
   - "What is family declaration?"
   - "What are the charges for Margin Pledge?"
   - "What is Unpledging of shares?"

## Important Notes

- The chatbot only answers based on processed documents
- If no relevant information is found, it responds with "I don't know"
- All responses include source references
- The system maintains conversation history during the session

## Troubleshooting

- **"No documents loaded"**: Run the process_documents.py script first
- **API errors**: Check your GEMINI_API_KEY environment variable
- **Empty responses**: Ensure your documents contain relevant content
