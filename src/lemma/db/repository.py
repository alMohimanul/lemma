"""Database repository for CRUD operations."""
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy import create_engine, desc, or_
from sqlalchemy.orm import sessionmaker, Session

from .models import (
    Base,
    Paper,
    Embedding,
    Citation,
    LLMCache,
    ProcessingLog,
    FileOperation,
    Config,
)


class Repository:
    """Database access layer for lemma."""

    def __init__(self, db_path: str = "~/.lemma/lemma.db"):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        db_path = Path(db_path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.Session()

    def close(self) -> None:
        """Close all database connections and dispose of the engine."""
        self.engine.dispose()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False

    # Paper operations
    def add_paper(
        self,
        file_path: str,
        file_hash: str,
        file_size: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Paper:
        """Add a new paper to the database.

        Args:
            file_path: Absolute path to PDF file
            file_hash: SHA256 hash of file content
            file_size: File size in bytes
            metadata: Optional metadata dict (title, authors, year, etc.)

        Returns:
            Created Paper object
        """
        with self.get_session() as session:
            paper = Paper(
                file_path=file_path,
                file_hash=file_hash,
                file_size=file_size,
                **(metadata or {}),
            )
            session.add(paper)
            session.commit()
            session.refresh(paper)
            return paper

    def get_paper_by_hash(self, file_hash: str) -> Optional[Paper]:
        """Find paper by content hash."""
        with self.get_session() as session:
            return session.query(Paper).filter(Paper.file_hash == file_hash).first()

    def get_paper_by_id(self, paper_id: int) -> Optional[Paper]:
        """Find paper by ID."""
        with self.get_session() as session:
            return session.query(Paper).filter(Paper.id == paper_id).first()

    def list_papers(
        self, limit: int = 100, offset: int = 0, sort_by: str = "indexed_at"
    ) -> List[Paper]:
        """List papers with pagination.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            sort_by: Field to sort by (indexed_at, year, title)

        Returns:
            List of Paper objects
        """
        with self.get_session() as session:
            query = session.query(Paper)

            if sort_by == "year":
                query = query.order_by(desc(Paper.year))
            elif sort_by == "title":
                query = query.order_by(Paper.title)
            else:
                query = query.order_by(desc(Paper.indexed_at))

            return query.limit(limit).offset(offset).all()

    def search_papers(self, query: str) -> List[Paper]:
        """Search papers by keyword (title, authors, abstract).

        Args:
            query: Search query string

        Returns:
            List of matching Paper objects
        """
        with self.get_session() as session:
            search_pattern = f"%{query}%"
            return (
                session.query(Paper)
                .filter(
                    or_(
                        Paper.title.like(search_pattern),
                        Paper.authors.like(search_pattern),
                        Paper.abstract.like(search_pattern),
                    )
                )
                .all()
            )

    def update_paper_metadata(self, paper_id: int, metadata: Dict[str, Any]) -> None:
        """Update paper metadata fields."""
        with self.get_session() as session:
            session.query(Paper).filter(Paper.id == paper_id).update(metadata)
            session.commit()

    def delete_paper(self, paper_id: int) -> None:
        """Delete a paper and all its related data (embeddings, citations, etc.).

        Args:
            paper_id: ID of the paper to delete
        """
        with self.get_session() as session:
            paper = session.query(Paper).filter(Paper.id == paper_id).first()
            if paper:
                session.delete(paper)  # Cascades to embeddings, citations, logs
                session.commit()

    def get_papers_for_embedding(self, force: bool = False) -> List[Paper]:
        """Get papers that need embeddings generated.

        Args:
            force: If True, return all papers regardless of embedding status

        Returns:
            List of Paper objects needing embeddings
        """
        with self.get_session() as session:
            query = session.query(Paper)

            if not force:
                # Only get papers that haven't been embedded yet
                query = query.filter(
                    or_(
                        Paper.embedding_status == "pending",
                        Paper.embedding_status == "failed",
                        Paper.embedding_status.is_(None),
                    )
                )

            return query.all()

    def get_papers_by_ids(self, paper_ids: List[int]) -> List[Paper]:
        """Get multiple papers by their IDs.

        Args:
            paper_ids: List of paper IDs to retrieve

        Returns:
            List of Paper objects (may be shorter if some IDs don't exist)
        """
        with self.get_session() as session:
            return session.query(Paper).filter(Paper.id.in_(paper_ids)).all()

    # Embedding operations
    def add_embedding(
        self,
        paper_id: int,
        text_content: str,
        embedding_vector: List[float],
        chunk_index: int = 0,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> Embedding:
        """Store embedding for a paper chunk."""
        with self.get_session() as session:
            embedding = Embedding(
                paper_id=paper_id,
                chunk_index=chunk_index,
                text_content=text_content,
                embedding_vector=json.dumps(embedding_vector),
                model_name=model_name,
            )
            session.add(embedding)
            session.commit()
            session.refresh(embedding)
        return embedding

    def get_embeddings_by_paper(self, paper_id: int) -> List[Embedding]:
        """Get all embeddings for a paper."""
        with self.get_session() as session:
            return (
                session.query(Embedding)
                .filter(Embedding.paper_id == paper_id)
                .order_by(Embedding.chunk_index)
                .all()
            )

    # Citation operations
    def add_citation(
        self,
        paper_id: int,
        cited_title: str,
        cited_authors: Optional[str] = None,
        cited_year: Optional[int] = None,
        citation_context: Optional[str] = None,
        confidence: float = 1.0,
    ) -> Citation:
        """Add a citation extracted from a paper."""
        with self.get_session() as session:
            citation = Citation(
                paper_id=paper_id,
                cited_title=cited_title,
                cited_authors=cited_authors,
                cited_year=cited_year,
                citation_context=citation_context,
                confidence=confidence,
            )
            session.add(citation)
            session.commit()
            session.refresh(citation)
            return citation

    # LLM cache operations
    def get_cached_response(self, prompt: str) -> Optional[str]:
        """Get cached LLM response if available and not expired.

        Args:
            prompt: The exact prompt to look up

        Returns:
            Cached response text or None
        """
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

        with self.get_session() as session:
            cache_entry = (
                session.query(LLMCache)
                .filter(LLMCache.prompt_hash == prompt_hash)
                .first()
            )

            if cache_entry:
                # Check expiration
                if (
                    cache_entry.expires_at
                    and cache_entry.expires_at < datetime.utcnow()
                ):
                    return None

                # Update hit count
                cache_entry.hit_count += 1
                session.commit()
                return cache_entry.response

            return None

    def cache_response(
        self,
        prompt: str,
        response: str,
        provider: str,
        model: str,
        tokens_used: int = 0,
        ttl_days: int = 30,
    ) -> None:
        """Cache an LLM response.

        Args:
            prompt: The prompt that was sent
            response: The LLM's response
            provider: Provider name (groq, gemini, etc.)
            model: Model identifier
            tokens_used: Number of tokens consumed
            ttl_days: Cache expiration in days
        """
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        expires_at = datetime.utcnow() + timedelta(days=ttl_days)

        with self.get_session() as session:
            # Upsert (replace if exists)
            existing = (
                session.query(LLMCache)
                .filter(LLMCache.prompt_hash == prompt_hash)
                .first()
            )

            if existing:
                existing.response = response
                existing.provider = provider
                existing.model = model
                existing.tokens_used = tokens_used
                existing.created_at = datetime.utcnow()
                existing.expires_at = expires_at
            else:
                cache_entry = LLMCache(
                    prompt_hash=prompt_hash,
                    prompt=prompt,
                    response=response,
                    provider=provider,
                    model=model,
                    tokens_used=tokens_used,
                    expires_at=expires_at,
                )
                session.add(cache_entry)

            session.commit()

    # Processing log operations
    def log_operation(
        self,
        operation: str,
        status: str,
        paper_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> ProcessingLog:
        """Log a processing operation."""
        with self.get_session() as session:
            log_entry = ProcessingLog(
                paper_id=paper_id,
                operation=operation,
                status=status,
                details=json.dumps(details) if details else None,
                error_message=error_message,
            )
            session.add(log_entry)
            session.commit()
            session.refresh(log_entry)
            return log_entry

    # File operation operations (for rollback)
    def log_file_operation(
        self,
        paper_id: int,
        operation_type: str,
        original_path: str,
        new_path: Optional[str] = None,
    ) -> FileOperation:
        """Log a file operation for potential rollback."""
        with self.get_session() as session:
            file_op = FileOperation(
                paper_id=paper_id,
                operation_type=operation_type,
                original_path=original_path,
                new_path=new_path,
            )
            session.add(file_op)
            session.commit()
            session.refresh(file_op)
            return file_op

    def get_rollback_operations(
        self, since: Optional[datetime] = None
    ) -> List[FileOperation]:
        """Get file operations that can be rolled back.

        Args:
            since: Only return operations after this timestamp

        Returns:
            List of FileOperation objects
        """
        with self.get_session() as session:
            query = session.query(FileOperation).filter(
                FileOperation.can_rollback.is_(True),
                FileOperation.rolled_back.is_(False),
            )

            if since:
                query = query.filter(FileOperation.timestamp >= since)

            return query.order_by(desc(FileOperation.timestamp)).all()

    # Config operations
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        with self.get_session() as session:
            config = session.query(Config).filter(Config.key == key).first()
            if config:
                try:
                    return json.loads(config.value)
                except json.JSONDecodeError:
                    return config.value
            return default

    def set_config(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        with self.get_session() as session:
            config = session.query(Config).filter(Config.key == key).first()

            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            elif not isinstance(value, str):
                value = str(value)

            if config:
                config.value = value
                config.updated_at = datetime.utcnow()
            else:
                config = Config(key=key, value=value)
                session.add(config)

            session.commit()
