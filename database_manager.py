import os
import json
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import numpy as np

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    source = Column(String(500), nullable=False)
    title = Column(String(500), nullable=False)
    doc_type = Column(String(100), nullable=False)
    chunk_id = Column(Integer, default=0)
    total_chunks = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    doc_metadata = Column(JSON)

class DocumentEmbedding(Base):
    __tablename__ = 'document_embeddings'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, nullable=False)
    embedding = Column(Text, nullable=False)  # JSON string of embedding vector
    embedding_model = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    def __init__(self):
        """Initialize the database manager."""
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            raise Exception("DATABASE_URL environment variable is required")
        
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()
    
    def add_documents_with_embeddings(self, documents: List[Dict[str, Any]], 
                                    embeddings: List[List[float]], 
                                    embedding_model: str = "text-embedding-004") -> None:
        """Add documents and their embeddings to the database."""
        session = self.get_session()
        try:
            for doc, embedding in zip(documents, embeddings):
                db_document = Document(
                    content=doc.get('content', ''),
                    source=doc.get('source', 'unknown'),
                    title=doc.get('title', ''),
                    doc_type=doc.get('type', 'unknown'),
                    chunk_id=doc.get('chunk_id', 0),
                    total_chunks=doc.get('total_chunks', 1),
                    doc_metadata=doc.get('metadata', {})
                )
                session.add(db_document)
                session.flush()
                
                db_embedding = DocumentEmbedding(
                    document_id=db_document.id,
                    embedding=json.dumps(embedding),
                    embedding_model=embedding_model
                )
                session.add(db_embedding)
            
            session.commit()
        except Exception as e:
            session.rollback()
            raise Exception(f"Error adding documents to database: {str(e)}")
        finally:
            session.close()
    
    def search_similar_documents(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity."""
        session = self.get_session()
        try:
            results = session.query(Document, DocumentEmbedding).join(
                DocumentEmbedding, Document.id == DocumentEmbedding.document_id
            ).all()
            
            similarities = []
            query_embedding = np.array(query_embedding)
            
            for doc, embedding in results:
                stored_embedding = np.array(json.loads(embedding.embedding))
                
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )
                
                similarities.append({
                    'text': doc.content,
                    'metadata': {
                        'source': doc.source,
                        'title': doc.title,
                        'type': doc.doc_type,
                        'chunk_id': doc.chunk_id,
                        'total_chunks': doc.total_chunks
                    },
                    'score': float(similarity),
                    'document_id': doc.id
                })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x['score'], reverse=True)
            return similarities[:k]
            
        except Exception as e:
            raise Exception(f"Error searching documents: {str(e)}")
        finally:
            session.close()
    
    def clear_all_documents(self) -> None:
        """Clear all documents and embeddings from the database."""
        session = self.get_session()
        try:
            session.query(DocumentEmbedding).delete()
            session.query(Document).delete()
            session.commit()
        except Exception as e:
            session.rollback()
            raise Exception(f"Error clearing documents: {str(e)}")
        finally:
            session.close()
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about stored documents."""
        session = self.get_session()
        try:
            doc_count = session.query(Document).count()
            embedding_count = session.query(DocumentEmbedding).count()
            
            sources = session.query(Document.source).distinct().all()
            unique_sources = [source[0] for source in sources]
            
            return {
                'total_documents': doc_count,
                'total_embeddings': embedding_count,
                'unique_sources': len(unique_sources),
                'sources': unique_sources
            }
        except Exception as e:
            raise Exception(f"Error getting document stats: {str(e)}")
        finally:
            session.close()
    
    def get_documents_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get all documents from a specific source."""
        session = self.get_session()
        try:
            documents = session.query(Document).filter(Document.source.like(f'%{source}%')).all()
            return [
                {
                    'id': doc.id,
                    'content': doc.content,
                    'source': doc.source,
                    'title': doc.title,
                    'type': doc.doc_type,
                    'created_at': doc.created_at.isoformat()
                }
                for doc in documents
            ]
        except Exception as e:
            raise Exception(f"Error getting documents by source: {str(e)}")
        finally:
            session.close()