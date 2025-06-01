#!/usr/bin/env python3
"""
Document Processing Script for AngelOne RAG Chatbot

This script processes documents from a specified folder and updates the vector database
with embeddings for use in the RAG chatbot.

Usage:
    python process_documents.py --folder /path/to/documents --clear-existing

Arguments:
    --folder: Path to folder containing documents (PDFs and text files)
    --clear-existing: Clear existing embeddings before adding new ones (optional)
"""

import os
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import google.generativeai as genai
from document_processor import DocumentProcessor
from vector_store import VectorStore
from utils import chunk_text, clean_text, is_meaningful_content

class DocumentEmbeddingProcessor:
    def __init__(self):
        """Initialize the document processor with Gemini API."""
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            print("Error: GEMINI_API_KEY environment variable is required")
            sys.exit(1)
        
        genai.configure(api_key=api_key)
        self.embedding_model = 'models/text-embedding-004'
        self.vector_store = VectorStore(dimension=768)  # Gemini embeddings are 768-dimensional
        self.doc_processor = DocumentProcessor()
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Gemini."""
        embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            try:
                print(f"Generating embedding {i+1}/{total}")
                response = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                print(f"Error generating embedding for text {i+1}: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 768)
        
        return embeddings
    
    def process_folder(self, folder_path: str, clear_existing: bool = False) -> None:
        """Process all documents in the specified folder."""
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"Error: Folder {folder_path} does not exist")
            return
        
        if clear_existing:
            print("Clearing existing vector store...")
            self.vector_store.clear()
        
        # Find all supported files
        supported_extensions = {'.pdf', '.txt', '.md'}
        files = []
        
        for ext in supported_extensions:
            files.extend(folder.glob(f"**/*{ext}"))
        
        if not files:
            print(f"No supported files found in {folder_path}")
            print(f"Supported formats: {', '.join(supported_extensions)}")
            return
        
        print(f"Found {len(files)} files to process")
        
        all_documents = []
        
        # Process each file
        for file_path in files:
            print(f"\nProcessing: {file_path.name}")
            
            try:
                if file_path.suffix.lower() == '.pdf':
                    # Process PDF files
                    documents = self.doc_processor.process_pdf(str(file_path), file_path.name)
                else:
                    # Process text files
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    if not is_meaningful_content(content):
                        print(f"Skipping {file_path.name} - insufficient content")
                        continue
                    
                    # Clean and chunk the content
                    cleaned_content = clean_text(content)
                    chunks = chunk_text(cleaned_content, max_chunk_size=1000, overlap=200)
                    
                    documents = []
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            'content': chunk,
                            'source': file_path.name,
                            'title': file_path.stem,
                            'type': 'text_file',
                            'chunk_id': i,
                            'total_chunks': len(chunks)
                        })
                
                all_documents.extend(documents)
                print(f"Extracted {len(documents)} chunks from {file_path.name}")
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                continue
        
        if not all_documents:
            print("No documents to process")
            return
        
        print(f"\nProcessing {len(all_documents)} document chunks...")
        
        # Extract texts and metadata
        texts = []
        metadatas = []
        
        for doc in all_documents:
            texts.append(doc['content'])
            metadatas.append({
                'source': doc['source'],
                'title': doc['title'],
                'type': doc['type'],
                'chunk_id': doc.get('chunk_id', 0),
                'total_chunks': doc.get('total_chunks', 1)
            })
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        embeddings = self.generate_embeddings(texts)
        
        # Add to vector store
        print("Adding documents to vector store...")
        self.vector_store.add_documents(texts, embeddings, metadatas)
        
        # Print statistics
        stats = self.vector_store.get_stats()
        print(f"\nProcessing complete!")
        print(f"Total documents in vector store: {stats['total_documents']}")
        print(f"Vector dimensions: {stats['vector_dimensions']}")
        
    def verify_setup(self) -> bool:
        """Verify that the setup is working correctly."""
        try:
            # Test embedding generation
            test_response = genai.embed_content(
                model=self.embedding_model,
                content="Test embedding",
                task_type="retrieval_document"
            )
            print(f"Gemini API working. Embedding dimension: {len(test_response['embedding'])}")
            return True
        except Exception as e:
            print(f"Error testing Gemini API: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Process documents for RAG chatbot')
    parser.add_argument('--folder', required=True, help='Path to folder containing documents')
    parser.add_argument('--clear-existing', action='store_true', 
                       help='Clear existing embeddings before adding new ones')
    parser.add_argument('--verify', action='store_true', 
                       help='Verify setup without processing documents')
    
    args = parser.parse_args()
    
    # Initialize processor
    try:
        processor = DocumentEmbeddingProcessor()
    except Exception as e:
        print(f"Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Verify setup if requested
    if args.verify:
        if processor.verify_setup():
            print("Setup verification successful!")
        else:
            print("Setup verification failed!")
        return
    
    # Process documents
    processor.process_folder(args.folder, args.clear_existing)

if __name__ == "__main__":
    main()