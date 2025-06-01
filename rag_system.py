import os
from typing import List, Tuple, Dict, Any
from openai import OpenAI
from vector_store import VectorStore
from utils import chunk_text

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with OpenAI and vector store."""
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "")
        )
        self.vector_store = VectorStore()
        self.model = "gpt-4o"
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store."""
        try:
            texts = []
            metadatas = []
            
            for doc in documents:
                content = doc.get('content', '')
                if not content.strip():
                    continue
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
                embeddings = self._generate_embeddings(texts)
                self.vector_store.add_documents(texts, embeddings, metadatas)
                
        except Exception as e:
            raise Exception(f"Failed to add documents to RAG system: {str(e)}")
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def query(self, question: str, k: int = 5) -> Tuple[str, List[str]]:
        """Query the RAG system and return response with sources."""
        try:
            question_embedding = self._generate_embeddings([question])[0]
            results = self.vector_store.similarity_search(question_embedding, k=k)
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
            response = self._generate_response(question, context)
            return response, sources[:3]
        except Exception as e:
            return f"Error processing query: {str(e)}", []
    
    def _generate_response(self, question: str, context: str) -> str:
        """Generate a response using OpenAI with the retrieved context."""
        try:
            system_prompt = """You are a helpful customer support assistant for AngelOne, a stock trading and investment platform. You can only answer questions based on the provided context from AngelOne's official documentation.

IMPORTANT RULES:
1. Only answer questions using information from the provided context
2. If the context doesn't contain relevant information to answer the question, respond with "I don't know"
3. Be helpful and provide clear, accurate answers
4. Don't make up information or provide general knowledge outside the context
5. Keep responses concise but comprehensive
6. If you mention specific procedures or steps, make sure they are exactly as described in the context

Context information:
{context}

Please answer the following question based only on the context provided above."""
            user_prompt = f"Question: {question}"
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt.format(context=context)},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def clear_vector_store(self) -> None:
        """Clear all documents from the vector store."""
        self.vector_store.clear()
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded documents."""
        return self.vector_store.get_stats()
