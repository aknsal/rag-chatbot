import os
from typing import List, Tuple, Dict, Any
import google.generativeai as genai
from database_manager import DatabaseManager
from utils import chunk_text

class GeminiDatabaseRAGSystem:
    def __init__(self):
        """Initialize the RAG system with Gemini and PostgreSQL database."""
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise Exception("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.embedding_model = 'models/text-embedding-004'
        self.db_manager = DatabaseManager()
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the database with embeddings."""
        try:
            texts = []
            processed_docs = []
            
            for doc in documents:
                content = doc.get('content', '')
                if not content.strip():
                    continue
                
                chunks = chunk_text(content, max_chunk_size=1000, overlap=200)
                
                for i, chunk in enumerate(chunks):
                    texts.append(chunk)
                    processed_docs.append({
                        'content': chunk,
                        'source': doc.get('source', 'unknown'),
                        'title': doc.get('title', ''),
                        'type': doc.get('type', 'unknown'),
                        'chunk_id': i,
                        'total_chunks': len(chunks),
                        'metadata': {
                            'original_source': doc.get('source', 'unknown'),
                            'original_title': doc.get('title', ''),
                        }
                    })
            
            if texts:
                embeddings = self._generate_embeddings(texts)
                self.db_manager.add_documents_with_embeddings(
                    processed_docs, embeddings, self.embedding_model
                )
                
        except Exception as e:
            raise Exception(f"Failed to add documents to RAG system: {str(e)}")
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Gemini."""
        try:
            embeddings = []
            for text in texts:
                response = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(response['embedding'])
            return embeddings
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def query(self, question: str, k: int = 5) -> Tuple[str, List[str]]:
        """Query the RAG system and return response with sources."""
        try:
            question_response = genai.embed_content(
                model=self.embedding_model,
                content=question,
                task_type="retrieval_query"
            )
            question_embedding = question_response['embedding']
            
            results = self.db_manager.search_similar_documents(question_embedding, k=k)
            
            if not results:
                return "I don't know. I can only answer questions based on the AngelOne documentation that has been loaded.", []
            
            context_texts = []
            sources = []
            
            for result in results:
                context_texts.append(result['text'])
                source_info = f"{result['metadata'].get('title', 'Unknown')} - {result['metadata'].get('source', 'Unknown source')}"
                if source_info not in sources:
                    sources.append(source_info)
            
            context = "\n\n".join(context_texts)
            
            # Generate response using the context
            response = self._generate_response(question, context)
            
            return response, sources[:3]  # Return top 3 sources
            
        except Exception as e:
            return f"Error processing query: {str(e)}", []
    
    def _generate_response(self, question: str, context: str) -> str:
        """Generate a response using Gemini with the retrieved context."""
        try:
            prompt = f"""You are a helpful customer support assistant for AngelOne, a stock trading and investment platform. You can only answer questions based on the provided context from AngelOne's official documentation.

IMPORTANT RULES:
1. Only answer questions using information from the provided context
2. If the context doesn't contain relevant information to answer the question, respond with "I don't know"
3. Be helpful and provide clear, accurate answers
4. Don't make up information or provide general knowledge outside the context
5. Keep responses concise but comprehensive
6. If you mention specific procedures or steps, make sure they are exactly as described in the context

Context information:
{context}

Question: {question}

Please answer the question based only on the context provided above."""

            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500
                )
            )
            
            return response.text
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def clear_vector_store(self) -> None:
        """Clear all documents from the database."""
        self.db_manager.clear_all_documents()
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded documents."""
        stats = self.db_manager.get_document_stats()
        return {
            'total_documents': stats.get('total_documents', 0),
            'vector_dimensions': 768, 
            'unique_sources': stats.get('unique_sources', 0),
            'sources': stats.get('sources', [])
        }