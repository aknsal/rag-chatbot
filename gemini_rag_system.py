import os
from typing import List, Tuple, Dict, Any
import google.generativeai as genai
import google.genai as generalai
from vector_store import VectorStore
from utils import chunk_text
from icecream import ic

class GeminiRAGSystem:
    def __init__(self):
        """Initialize the RAG system with Gemini and vector store."""
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise Exception("GEMINI_API_KEY environment variable is required")
        
        self.client = generalai.Client(api_key=api_key)
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        self.embedding_model = 'models/text-embedding-004'
        self.vector_store = VectorStore(dimension=768)  # Gemini embeddings are 768-dimensional
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store."""
        try:
            # Extract text content and create embeddings
            texts = []
            metadatas = []

            count = 0
            
            for doc in documents:
                count += 1
                print(f"Processing document {count + 1}/{len(documents)}: {doc.get('title', 'Untitled')}")
                content = doc.get('content', '')
                if not content.strip():
                    continue
                
                # Chunk the document if it's too long
                chunks = chunk_text(content, max_chunk_size=1000, overlap=200)
                
                for i, chunk in enumerate(chunks):
                    texts.append(chunk)
                    metadatas.append({
                        'source': doc.get('source', 'unknown'),
                        'title': doc.get('title', ''),
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    })
            
            if texts:
                ic(f"Generating embeddings for {len(texts)} chunks")
                embeddings = self._generate_embeddings(texts)
                ic(f"Generated {len(embeddings)} embeddings")
                self.vector_store.add_documents(texts, embeddings, metadatas)
                
        except Exception as e:
            raise Exception(f"Failed to add documents to RAG system: {str(e)}")
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Gemini, using multithreading for batching."""
        import concurrent.futures

        def embed_single(text):
            try:
                response = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                return response['embedding']
            except Exception as e:
                raise Exception(f"Failed to generate embedding for text: {str(e)}")

        try:
            embeddings = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(embed_single, texts))
                embeddings.extend(results)
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
            
            results = self.vector_store.similarity_search(question_embedding, k=k)
            

            context_texts = []
            sources = []
            
            for result in results:
                context_texts.append(result['text'])
                source_info = f"{result['metadata'].get('title', 'Unknown')} - {result['metadata'].get('source', 'Unknown source')}"
                if source_info not in sources:
                    sources.append(source_info)
            
            context = "\n\n".join(context_texts)

            ic(context)
            
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
2. If users asks something like what can you do or what is your purpose, tell that you are a customer support assistant for AngelOne and can answer questions to your knowledge.
2. If user asks a specific question and the context doesn't contain relevant information to answer the question, respond with "I don't know"
3. Be helpful and provide clear, accurate answers
4. Don't make up information or provide general knowledge outside the context
5. Keep responses concise but comprehensive
6. If you mention specific procedures or steps, make sure they are exactly as described in the context
7. Don't mention that there is some kind of context given to you, the cusotmer should not get any technical details about the system.

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
        """Clear all documents from the vector store."""
        self.vector_store.clear()
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded documents."""
        return self.vector_store.get_stats()