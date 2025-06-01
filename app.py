import streamlit as st
import os
from gemini_rag_system import GeminiRAGSystem
from document_processor import DocumentProcessor
from web_scraper import AngelOneScraper
from process_documents import DocumentEmbeddingProcessor
import tempfile
from icecream import ic

# Page configuration
st.set_page_config(
    page_title="AngelOne Support Chatbot",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    try:
        st.session_state.rag_system = GeminiRAGSystem()
        st.session_state.rag_error = None
    except Exception as e:
        st.session_state.rag_system = None
        st.session_state.rag_error = str(e)
    st.session_state.doc_processor = DocumentProcessor()
    st.session_state.scraper = AngelOneScraper()
    st.session_state.process_docs = DocumentEmbeddingProcessor()
    st.session_state.messages = []
    st.session_state.documents_loaded = False

# Title and description
st.title("Support Chatbot")

# Check for API key error
if hasattr(st.session_state, 'rag_error') and st.session_state.rag_error:
    st.error(f"Configuration Error: {st.session_state.rag_error}")
    st.info("Please provide a valid GEMINI_API_KEY to use this chatbot.")
    st.stop()

# Sidebar for document management
with st.sidebar:
    st.header("📚 Document Management")
    
    # Check if documents are already loaded
    
    if st.button("🔄 Clear DB"):
        with st.spinner("Reloading documents..."):
            st.session_state.rag_system.clear_vector_store()
            st.session_state.documents_loaded = False
            st.rerun()
    
    st.subheader("🌐 Web Scraping")
    if st.button("📥 Scrape AngelOne Support Pages", disabled=st.session_state.documents_loaded):
        with st.spinner("Scraping AngelOne support pages..."):
            try:
                success, message = st.session_state.scraper.scrape_support_pages()
                if success:
                    # Load scraped documents into RAG system
                    documents = st.session_state.doc_processor.load_scraped_documents()
                    ic("completed loading documents")
                    if documents:
                        st.session_state.rag_system.add_documents(documents)
                        st.session_state.documents_loaded = True
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error("❌ No documents found after scraping")
                else:
                    st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"❌ Error during scraping: {str(e)}")
    
    st.subheader("Update PDFs to DB")
    
    
    if st.button("📤 Process PDFs"):
        with st.spinner("Processing PDF documents..."):
            try:
                st.session_state.process_docs.process_folder('./test_documents', True)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error processing PDFs: {str(e)}")
    
    # Document statistics
    if st.session_state.rag_system:
        st.subheader("📊 Document Stats")
        stats = st.session_state.rag_system.get_document_stats()
        st.metric("Total Documents", stats.get('total_documents', 0))
        if stats.get('total_chunks', 0) > 0:
            st.session_state.documents_loaded = True
        st.metric("Vector Dimensions", stats.get('vector_dimensions', 0))
        if stats.get('unique_sources', 0) > 0:
            st.metric("Unique Sources", stats.get('unique_sources', 0))
            with st.expander("View Sources"):
                for source in stats.get('sources', []):
                    st.text(source)

# Main chat interface
st.header("💭 Chat Interface")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:** {source}")

# Chat input
if prompt := st.chat_input("Ask a question"):
    if 1==2:
        st.error("I do not have any context plase contact the dev.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response, sources = st.session_state.rag_system.query(prompt)
                    st.markdown(response)
                    
                    # Display sources if available
                    if sources:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:** {source}")
                    
                    # Add assistant message to chat history
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

# Clear chat button
if st.session_state.messages:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


