import streamlit as st
import os
from typing import List, Tuple, Dict, Any
import google.generativeai as genai
from vector_store import VectorStore

st.set_page_config(
    page_title="Support Chatbot",
    page_icon="💬",
    layout="wide"
)

class SimpleRAGSystem:
    def __init__(self):
        """Initialize the RAG system with Gemini and vector store."""
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise Exception("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.embedding_model = 'models/text-embedding-004'
        self.vector_store = VectorStore(dimension=768)
        
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
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded documents."""
        return self.vector_store.get_stats()

if 'rag_system' not in st.session_state:
    try:
        st.session_state.rag_system = SimpleRAGSystem()
        st.session_state.rag_error = None
    except Exception as e:
        st.session_state.rag_system = None
        st.session_state.rag_error = str(e)
    st.session_state.messages = []

st.title("AngelOne Support Chatbot")
st.markdown("Ask questions about AngelOne services based on official documentation and support materials.")

if hasattr(st.session_state, 'rag_error') and st.session_state.rag_error:
    st.error(f"Configuration Error: {st.session_state.rag_error}")
    st.info("Please provide a valid GEMINI_API_KEY to use this chatbot.")
    st.stop()

with st.sidebar:
    st.header("Document Management")
    
    st.subheader("Process Documents")
    st.info("To add documents to the chatbot, use the standalone script:")
    st.code("python process_documents.py --folder /path/to/documents", language="bash")
    st.markdown("**Supported formats:** PDF, TXT, MD files")
    
    if st.button("Reload Vector Store"):
        if st.session_state.rag_system:
            try:
                st.session_state.rag_system.vector_store._load_index()
                stats = st.session_state.rag_system.get_document_stats()
                if stats['total_documents'] > 0:
                    st.success(f"Loaded {stats['total_documents']} documents")
                else:
                    st.info("No documents found in vector store")
                st.rerun()
            except Exception as e:
                st.error(f"Error reloading: {str(e)}")
    
    if st.session_state.rag_system:
        st.subheader("Document Stats")
        stats = st.session_state.rag_system.get_document_stats()
        st.metric("Total Documents", stats.get('total_documents', 0))
        st.metric("Vector Dimensions", stats.get('vector_dimensions', 0))

st.header("Chat Interface")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:** {source}")

if prompt := st.chat_input("Ask a question about AngelOne services..."):
    if not st.session_state.rag_system:
        st.error("System not initialized. Please check configuration.")
    else:
        # Check if there are documents loaded
        stats = st.session_state.rag_system.get_document_stats()
        if stats.get('total_documents', 0) == 0:
            st.error("No documents loaded. Please process documents first using the script.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response, sources = st.session_state.rag_system.query(prompt)
                        st.markdown(response)
                        
                        # Display sources if available
                        if sources:
                            with st.expander("Sources"):
                                for i, source in enumerate(sources, 1):
                                    st.markdown(f"**Source {i}:** {source}")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })

if st.session_state.messages:
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.markdown("---")
st.markdown(
    "This chatbot is powered by RAG (Retrieval-Augmented Generation) "
    "and only answers questions based on AngelOne's official documentation."
)