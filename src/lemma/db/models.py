"""SQLite schema for lemma paper manager."""
from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    Boolean,
    ForeignKey,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Paper(Base):
    """Core paper metadata table."""

    __tablename__ = "papers"

    id = Column(Integer, primary_key=True)
    file_path = Column(String(1024), nullable=False, unique=True)
    file_hash = Column(String(64), nullable=False, unique=True, index=True)  # sha256
    file_size = Column(Integer, nullable=False)

    # Metadata (extracted locally via regex or PDF metadata)
    title = Column(String(512))
    authors = Column(Text)  # JSON array or comma-separated
    year = Column(Integer, index=True)
    publication = Column(String(256))
    doi = Column(String(128), index=True)
    arxiv_id = Column(String(32), index=True)
    abstract = Column(Text)

    # Processing metadata
    indexed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_accessed = Column(DateTime)
    embedding_status = Column(
        String(32), default="pending"
    )  # pending, processing, completed, failed

    # Relationships
    embeddings = relationship(
        "Embedding", back_populates="paper", cascade="all, delete-orphan"
    )
    citations = relationship(
        "Citation", back_populates="paper", cascade="all, delete-orphan"
    )
    processing_logs = relationship(
        "ProcessingLog", back_populates="paper", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Paper(id={self.id}, title='{self.title[:50]}...', year={self.year})>"


class Embedding(Base):
    """Vector embeddings for semantic search."""

    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False, index=True)
    chunk_index = Column(Integer, default=0)  # For chunked embeddings
    text_content = Column(Text)  # Original text that was embedded
    embedding_vector = Column(Text)  # JSON-serialized vector (for backup/portability)
    model_name = Column(String(128), default="all-MiniLM-L6-v2")
    created_at = Column(DateTime, default=datetime.utcnow)

    paper = relationship("Paper", back_populates="embeddings")

    __table_args__ = (Index("idx_paper_chunk", "paper_id", "chunk_index"),)


class Citation(Base):
    """Extracted citations from papers."""

    __tablename__ = "citations"

    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False, index=True)
    cited_title = Column(String(512))
    cited_authors = Column(Text)
    cited_year = Column(Integer)
    citation_context = Column(Text)  # Surrounding text where citation appears
    confidence = Column(Float, default=1.0)  # Extraction confidence (0-1)
    extracted_at = Column(DateTime, default=datetime.utcnow)

    paper = relationship("Paper", back_populates="citations")


class LLMCache(Base):
    """Cache for LLM responses to avoid redundant API calls."""

    __tablename__ = "llm_cache"

    id = Column(Integer, primary_key=True)
    prompt_hash = Column(String(64), unique=True, nullable=False, index=True)  # sha256
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    provider = Column(String(32))  # groq, gemini, openrouter
    model = Column(String(64))
    tokens_used = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime)  # Optional TTL for cache invalidation
    hit_count = Column(Integer, default=0)  # Track cache usage

    __table_args__ = (Index("idx_expires_at", "expires_at"),)


class ProcessingLog(Base):
    """Audit log for file operations and processing."""

    __tablename__ = "processing_logs"

    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), index=True)
    operation = Column(String(64), nullable=False)  # scan, extract, embed, rename, etc.
    status = Column(String(32), nullable=False)  # success, failed, skipped
    details = Column(Text)  # JSON with operation-specific details
    error_message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    paper = relationship("Paper", back_populates="processing_logs")


class FileOperation(Base):
    """Rollback log for file organization operations."""

    __tablename__ = "file_operations"

    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False, index=True)
    operation_type = Column(String(32), nullable=False)  # rename, move, delete
    original_path = Column(String(1024), nullable=False)
    new_path = Column(String(1024))
    can_rollback = Column(Boolean, default=True)
    rolled_back = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    rollback_timestamp = Column(DateTime)


class Config(Base):
    """Key-value configuration store."""

    __tablename__ = "config"

    key = Column(String(128), primary_key=True)
    value = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Note(Base):
    """Saved Q&A notes for literature review."""

    __tablename__ = "notes"

    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    formatted_note = Column(Text)  # LLM-formatted note for literature review
    paper_ids = Column(Text, nullable=False)  # JSON array of paper IDs used
    sources = Column(Text)  # JSON array of source strings
    provider = Column(String(32))  # LLM provider used (groq, gemini, cache)
    model = Column(String(64))  # Model name
    tokens_used = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def __repr__(self):
        return f"<Note(id={self.id}, question='{self.question[:50]}...', created_at={self.created_at})>"
