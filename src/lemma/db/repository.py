"""Database repository for CRUD operations."""
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
from datetime import datetime, timedelta

from sqlalchemy import create_engine, desc, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from .models import (
    Base,
    Paper,
    Embedding,
    Citation,
    LLMCache,
    ProcessingLog,
    FileOperation,
    Config,
    Note,
    PaperComparison,
    ArxivQueryCache,
)
from ..utils.logger import get_logger, log_exception
from ..integrations.arxiv_client import normalize_arxiv_id

logger = get_logger(__name__)


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
    ) -> Optional[Paper]:
        """Add a new paper to the database.

        Args:
            file_path: Absolute path to PDF file
            file_hash: SHA256 hash of file content
            file_size: File size in bytes
            metadata: Optional metadata dict (title, authors, year, etc.)

        Returns:
            Created Paper object, or None if operation failed
        """
        with self.get_session() as session:
            try:
                paper = Paper(
                    file_path=file_path,
                    file_hash=file_hash,
                    file_size=file_size,
                    **(metadata or {}),
                )
                session.add(paper)
                session.commit()
                session.refresh(paper)
                logger.info(f"Added paper: {paper.title or file_path}")
                return paper
            except IntegrityError as e:
                session.rollback()
                logger.warning(f"Duplicate paper detected (hash: {file_hash}): {e}")
                return None
            except SQLAlchemyError as e:
                session.rollback()
                log_exception(logger, f"Failed to add paper {file_path}", e)
                return None

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

    def update_paper_metadata(self, paper_id: int, metadata: Dict[str, Any]) -> bool:
        """Update paper metadata fields.

        Args:
            paper_id: ID of paper to update
            metadata: Dictionary of fields to update

        Returns:
            True if update successful, False otherwise
        """
        with self.get_session() as session:
            try:
                session.query(Paper).filter(Paper.id == paper_id).update(metadata)
                session.commit()
                logger.info(f"Updated metadata for paper ID {paper_id}")
                return True
            except SQLAlchemyError as e:
                session.rollback()
                log_exception(logger, f"Failed to update paper {paper_id}", e)
                return False

    def delete_paper(self, paper_id: int) -> bool:
        """Delete a paper and all its related data (embeddings, citations, etc.).

        Args:
            paper_id: ID of the paper to delete

        Returns:
            True if deletion successful, False otherwise
        """
        with self.get_session() as session:
            try:
                paper = session.query(Paper).filter(Paper.id == paper_id).first()
                if paper:
                    session.delete(paper)  # Cascades to embeddings, citations, logs
                    session.commit()
                    logger.info(f"Deleted paper ID {paper_id}")
                    return True
                else:
                    logger.warning(f"Paper ID {paper_id} not found for deletion")
                    return False
            except SQLAlchemyError as e:
                session.rollback()
                log_exception(logger, f"Failed to delete paper {paper_id}", e)
                return False

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
        chunk_hash: Optional[str] = None,
        chunk_type: str = "paragraph",
        section_name: Optional[str] = None,
        importance_score: float = 0.5,
        content_version: int = 1,
    ) -> Embedding:
        """Store embedding for a paper chunk.

        Args:
            paper_id: ID of the paper this embedding belongs to
            text_content: Text that was embedded
            embedding_vector: The embedding vector as a list of floats
            chunk_index: Index of this chunk in the paper
            model_name: Name of the embedding model used
            chunk_hash: SHA256 hash of the chunk text (for incremental updates)
            chunk_type: Type of chunk (title, abstract, section, paragraph, etc.)
            section_name: Name of the section this chunk belongs to
            importance_score: Relevance weight for retrieval (0-1)
            content_version: Paper content version when this embedding was created

        Returns:
            Created Embedding object
        """
        with self.get_session() as session:
            embedding = Embedding(
                paper_id=paper_id,
                chunk_index=chunk_index,
                text_content=text_content,
                embedding_vector=json.dumps(embedding_vector),
                model_name=model_name,
                chunk_hash=chunk_hash,
                chunk_type=chunk_type,
                section_name=section_name,
                importance_score=importance_score,
                content_version=content_version,
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

    def get_section_embeddings(
        self, paper_id: int, section_name: str
    ) -> List[Embedding]:
        """Get embeddings for a specific section of a paper.

        Args:
            paper_id: Paper ID
            section_name: Section name (supports fuzzy matching via LIKE)

        Returns:
            List of Embedding objects for the specified section
        """
        with self.get_session() as session:
            # Use LIKE for fuzzy matching (e.g., "method" matches "Methods", "Methodology")
            return (
                session.query(Embedding)
                .filter(
                    Embedding.paper_id == paper_id,
                    Embedding.section_name.like(f"%{section_name}%"),
                    Embedding.is_valid.is_(True),
                )
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

    # Note operations
    def add_note(
        self,
        question: str,
        answer: str,
        paper_ids: str,
        sources: Optional[str] = None,
        formatted_note: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tokens_used: Optional[int] = None,
    ) -> Optional[Note]:
        """Add a new note to the database.

        Args:
            question: Question text
            answer: Answer text
            paper_ids: JSON string of paper IDs
            sources: Optional JSON string of sources
            formatted_note: Optional LLM-formatted note
            provider: LLM provider name
            model: Model name
            tokens_used: Token count

        Returns:
            Created Note object, or None on error
        """
        import logging
        from sqlalchemy.exc import SQLAlchemyError

        logger = logging.getLogger(__name__)

        with self.get_session() as session:
            try:
                note = Note(
                    question=question,
                    answer=answer,
                    paper_ids=paper_ids,
                    sources=sources,
                    formatted_note=formatted_note,
                    provider=provider,
                    model=model,
                    tokens_used=tokens_used,
                )
                session.add(note)
                session.commit()
                session.refresh(note)
                logger.info(f"Saved note {note.id}: {question[:50]}...")
                return note
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to save note: {e}", exc_info=True)
                return None

    def get_note_by_id(self, note_id: int) -> Optional[Note]:
        """Get a note by ID.

        Args:
            note_id: Note ID

        Returns:
            Note object or None if not found
        """
        with self.get_session() as session:
            return session.query(Note).filter(Note.id == note_id).first()

    def list_notes(self, limit: int = 100, offset: int = 0) -> List[Note]:
        """List notes with pagination, ordered by creation date (newest first).

        Args:
            limit: Maximum number of notes to return
            offset: Number of notes to skip

        Returns:
            List of Note objects
        """
        with self.get_session() as session:
            return (
                session.query(Note)
                .order_by(desc(Note.created_at))
                .limit(limit)
                .offset(offset)
                .all()
            )

    def delete_note(self, note_id: int) -> bool:
        """Delete a note by ID.

        Args:
            note_id: Note ID

        Returns:
            True if deleted, False if not found
        """
        import logging
        from sqlalchemy.exc import SQLAlchemyError

        logger = logging.getLogger(__name__)

        with self.get_session() as session:
            try:
                note = session.query(Note).filter(Note.id == note_id).first()
                if note:
                    session.delete(note)
                    session.commit()
                    logger.info(f"Deleted note {note_id}")
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to delete note {note_id}: {e}", exc_info=True)
                return False

    # Incremental embedding and versioning operations
    def update_embedding_index(self, embedding_id: int, new_index: int) -> bool:
        """Update the chunk_index of an existing embedding.

        Args:
            embedding_id: ID of the embedding to update
            new_index: New chunk index value

        Returns:
            True if update successful
        """
        with self.get_session() as session:
            try:
                session.query(Embedding).filter(Embedding.id == embedding_id).update(
                    {"chunk_index": new_index}
                )
                session.commit()
                return True
            except SQLAlchemyError as e:
                session.rollback()
                log_exception(
                    logger, f"Failed to update embedding index {embedding_id}", e
                )
                return False

    def invalidate_orphaned_embeddings(self, paper_id: int, valid_hashes: set) -> int:
        """Mark embeddings as invalid if their chunk_hash is not in valid_hashes.

        Args:
            paper_id: ID of the paper
            valid_hashes: Set of chunk hashes that are still valid

        Returns:
            Number of embeddings invalidated
        """
        with self.get_session() as session:
            try:
                # Get all embeddings for this paper
                embeddings = (
                    session.query(Embedding)
                    .filter(Embedding.paper_id == paper_id)
                    .all()
                )

                invalidated_count = 0
                for embedding in embeddings:
                    if (
                        embedding.chunk_hash
                        and embedding.chunk_hash not in valid_hashes
                    ):
                        embedding.is_valid = False
                        invalidated_count += 1

                session.commit()
                logger.info(
                    f"Invalidated {invalidated_count} orphaned embeddings for paper {paper_id}"
                )
                return invalidated_count

            except SQLAlchemyError as e:
                session.rollback()
                log_exception(
                    logger, f"Failed to invalidate embeddings for paper {paper_id}", e
                )
                return 0

    def update_paper_embedding_metadata(
        self,
        paper_id: int,
        content_hash: str,
        content_version: int,
        embedding_status: str = "completed",
    ) -> bool:
        """Update paper's embedding-related metadata after successful embedding.

        Args:
            paper_id: ID of the paper
            content_hash: SHA256 hash of the content
            content_version: New content version number
            embedding_status: New embedding status

        Returns:
            True if update successful
        """
        with self.get_session() as session:
            try:
                from datetime import datetime

                session.query(Paper).filter(Paper.id == paper_id).update(
                    {
                        "content_hash": content_hash,
                        "content_version": content_version,
                        "last_embedded_version": content_version,
                        "last_embedded_at": datetime.utcnow(),
                        "embedding_status": embedding_status,
                    }
                )
                session.commit()
                logger.info(f"Updated embedding metadata for paper {paper_id}")
                return True

            except SQLAlchemyError as e:
                session.rollback()
                log_exception(
                    logger,
                    f"Failed to update embedding metadata for paper {paper_id}",
                    e,
                )
                return False

    def cleanup_invalid_embeddings(self, older_than_days: int = 30) -> int:
        """Delete invalid embeddings that are older than specified days.

        Args:
            older_than_days: Only delete invalid embeddings older than this many days

        Returns:
            Number of embeddings deleted
        """
        with self.get_session() as session:
            try:
                from datetime import datetime, timedelta

                cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)

                result = (
                    session.query(Embedding)
                    .filter(
                        Embedding.is_valid.is_(False),
                        Embedding.created_at < cutoff_date,
                    )
                    .delete()
                )

                session.commit()
                logger.info(f"Cleaned up {result} invalid embeddings")
                return result

            except SQLAlchemyError as e:
                session.rollback()
                log_exception(logger, "Failed to cleanup invalid embeddings", e)
                return 0

    def get_papers_needing_update(self, force: bool = False) -> List[Paper]:
        """Get papers that need embedding updates based on version tracking.

        Args:
            force: If True, return all papers regardless of version

        Returns:
            List of Paper objects needing updates
        """
        with self.get_session() as session:
            if force:
                return session.query(Paper).all()

            # Papers where content_version > last_embedded_version
            # or where no embeddings exist
            return (
                session.query(Paper)
                .filter(
                    or_(
                        Paper.embedding_status == "pending",
                        Paper.embedding_status == "failed",
                        Paper.embedding_status.is_(None),
                        Paper.content_version > Paper.last_embedded_version,
                        Paper.last_embedded_version.is_(None),
                    )
                )
                .all()
            )

    def get_embedding_coverage_stats(self) -> Dict[str, Any]:
        """Get statistics about embedding coverage across all papers.

        Returns:
            Dictionary with coverage statistics
        """
        with self.get_session() as session:
            total_papers = session.query(Paper).count()
            embedded_papers = (
                session.query(Paper)
                .filter(Paper.embedding_status == "completed")
                .count()
            )
            pending_papers = (
                session.query(Paper)
                .filter(
                    or_(
                        Paper.embedding_status == "pending",
                        Paper.embedding_status.is_(None),
                    )
                )
                .count()
            )
            failed_papers = (
                session.query(Paper).filter(Paper.embedding_status == "failed").count()
            )

            # Papers needing updates (version mismatch)
            outdated_papers = (
                session.query(Paper)
                .filter(
                    Paper.content_version > Paper.last_embedded_version,
                    Paper.embedding_status == "completed",
                )
                .count()
            )

            total_embeddings = session.query(Embedding).count()
            valid_embeddings = (
                session.query(Embedding).filter(Embedding.is_valid.is_(True)).count()
            )

            return {
                "total_papers": total_papers,
                "embedded_papers": embedded_papers,
                "pending_papers": pending_papers,
                "failed_papers": failed_papers,
                "outdated_papers": outdated_papers,
                "total_embeddings": total_embeddings,
                "valid_embeddings": valid_embeddings,
                "invalid_embeddings": total_embeddings - valid_embeddings,
                "coverage_pct": (embedded_papers / total_papers * 100)
                if total_papers > 0
                else 0,
            }

    # Sync and watch directory operations
    def add_watched_directory(self, directory: str) -> bool:
        """Add a directory to the list of watched directories.

        Args:
            directory: Directory path to watch

        Returns:
            True if added successfully
        """
        try:
            watched = self.get_config("watched_directories", [])

            if isinstance(watched, str):
                watched = [watched]

            if directory not in watched:
                watched.append(directory)
                self.set_config("watched_directories", watched)
                logger.info(f"Added watched directory: {directory}")

            return True

        except Exception as e:
            log_exception(logger, f"Failed to add watched directory {directory}", e)
            return False

    def remove_watched_directory(self, directory: str) -> bool:
        """Remove a directory from the list of watched directories.

        Args:
            directory: Directory path to remove

        Returns:
            True if removed successfully
        """
        try:
            watched = self.get_config("watched_directories", [])

            if isinstance(watched, str):
                watched = [watched]

            if directory in watched:
                watched.remove(directory)
                self.set_config("watched_directories", watched)
                logger.info(f"Removed watched directory: {directory}")

            return True

        except Exception as e:
            log_exception(logger, f"Failed to remove watched directory {directory}", e)
            return False

    def get_watched_directories(self) -> List[str]:
        """Get list of all watched directories.

        Returns:
            List of directory paths
        """
        watched = self.get_config("watched_directories", [])

        if isinstance(watched, str):
            return [watched]

        return watched if isinstance(watched, list) else []

    def update_sync_stats(self, directory: str, stats: Dict[str, Any]) -> bool:
        """Update sync statistics for a directory.

        Args:
            directory: Directory that was synced
            stats: Statistics dictionary

        Returns:
            True if updated successfully
        """
        try:
            from datetime import datetime

            sync_data = {
                "last_sync": datetime.utcnow().isoformat(),
                "directory": directory,
                **stats,
            }

            self.set_config("last_sync_stats", sync_data)
            logger.info(f"Updated sync stats for {directory}")

            return True

        except Exception as e:
            log_exception(logger, "Failed to update sync stats", e)
            return False

    def get_sync_stats(self) -> Optional[Dict[str, Any]]:
        """Get the most recent sync statistics.

        Returns:
            Dictionary with sync stats or None
        """
        return self.get_config("last_sync_stats")

    # Paper comparison cache operations
    def add_paper_comparison(
        self,
        paper_ids: List[int],
        comparison_hash: str,
        comparison_result: Dict[str, Any],
        summary: str,
        section_name: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tokens_used: int = 0,
    ) -> Optional[PaperComparison]:
        """Store a paper comparison result in the cache.

        Args:
            paper_ids: List of paper IDs being compared (will be sorted)
            comparison_hash: Unique hash for this comparison
            comparison_result: Full comparison data as dict
            summary: Short summary of the comparison
            section_name: Section being compared (None for whole-paper)
            provider: LLM provider used
            model: Model name
            tokens_used: Total tokens consumed

        Returns:
            Created PaperComparison object, or None on error
        """
        with self.get_session() as session:
            try:
                # Sort paper IDs for consistency
                sorted_ids = sorted(paper_ids)

                comparison = PaperComparison(
                    paper_ids=json.dumps(sorted_ids),
                    comparison_hash=comparison_hash,
                    section_name=section_name,
                    comparison_result=json.dumps(comparison_result),
                    summary=summary,
                    provider=provider,
                    model=model,
                    tokens_used=tokens_used,
                )
                session.add(comparison)
                session.commit()
                session.refresh(comparison)
                logger.info(
                    f"Cached comparison for papers {sorted_ids}, section: {section_name or 'whole paper'}"
                )
                return comparison
            except IntegrityError:
                session.rollback()
                logger.warning(
                    f"Comparison already cached (hash: {comparison_hash[:16]}...)"
                )
                return None
            except SQLAlchemyError as e:
                session.rollback()
                log_exception(logger, "Failed to cache comparison", e)
                return None

    def get_paper_comparison(self, comparison_hash: str) -> Optional[PaperComparison]:
        """Retrieve a cached paper comparison by hash.

        Args:
            comparison_hash: Unique hash for the comparison

        Returns:
            PaperComparison object or None if not found
        """
        with self.get_session() as session:
            return (
                session.query(PaperComparison)
                .filter(PaperComparison.comparison_hash == comparison_hash)
                .first()
            )

    def get_comparisons_by_papers(
        self, paper_ids: List[int], section_name: Optional[str] = None
    ) -> List[PaperComparison]:
        """Get all cached comparisons involving specific papers.

        Args:
            paper_ids: List of paper IDs
            section_name: Optional filter by section

        Returns:
            List of PaperComparison objects
        """
        with self.get_session() as session:
            sorted_ids = sorted(paper_ids)
            query = session.query(PaperComparison).filter(
                PaperComparison.paper_ids == json.dumps(sorted_ids)
            )

            if section_name:
                query = query.filter(PaperComparison.section_name == section_name)

            return query.order_by(desc(PaperComparison.created_at)).all()

    def delete_comparisons_involving_paper(self, paper_id: int) -> int:
        """Delete all cached comparisons involving a specific paper.

        This is called when a paper is re-embedded to invalidate stale comparisons.

        Args:
            paper_id: Paper ID

        Returns:
            Number of comparisons deleted
        """
        with self.get_session() as session:
            try:
                # Find all comparisons that include this paper ID
                comparisons = session.query(PaperComparison).all()
                deleted_count = 0

                for comp in comparisons:
                    try:
                        paper_ids = json.loads(comp.paper_ids)
                        if paper_id in paper_ids:
                            session.delete(comp)
                            deleted_count += 1
                    except json.JSONDecodeError:
                        continue

                session.commit()
                if deleted_count > 0:
                    logger.info(
                        f"Invalidated {deleted_count} cached comparisons involving paper {paper_id}"
                    )
                return deleted_count
            except SQLAlchemyError as e:
                session.rollback()
                log_exception(
                    logger, f"Failed to delete comparisons for paper {paper_id}", e
                )
                return 0

    def get_all_section_names_for_papers(
        self, paper_ids: List[int]
    ) -> Dict[int, List[str]]:
        """Get all section names for given papers from their embeddings.

        Args:
            paper_ids: List of paper IDs

        Returns:
            Dict mapping paper_id to list of section names
        """
        with self.get_session() as session:
            result = {}

            for paper_id in paper_ids:
                sections = (
                    session.query(Embedding.section_name)
                    .filter(
                        Embedding.paper_id == paper_id,
                        Embedding.section_name.isnot(None),
                        Embedding.is_valid.is_(True),
                    )
                    .distinct()
                    .all()
                )

                result[paper_id] = [s[0] for s in sections if s[0]]

            return result

    def get_local_arxiv_and_doi_sets(self) -> Tuple[Set[str], Set[str]]:
        """Return normalized arXiv id set and lowercase DOI set for deduplication."""
        with self.get_session() as session:
            rows = session.query(Paper.arxiv_id, Paper.doi).all()

        arxiv_ids: Set[str] = set()
        dois: Set[str] = set()
        for aid, doi in rows:
            if aid and str(aid).strip():
                arxiv_ids.add(normalize_arxiv_id(str(aid).strip()))
            if doi and str(doi).strip():
                dois.add(str(doi).strip().lower())
        return arxiv_ids, dois

    def get_arxiv_query_cache(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Return cached arXiv search result list if not expired."""
        with self.get_session() as session:
            row = (
                session.query(ArxivQueryCache)
                .filter(ArxivQueryCache.query_hash == query_hash)
                .first()
            )
            if not row:
                return None
            if row.expires_at and row.expires_at < datetime.utcnow():
                session.delete(row)
                session.commit()
                return None
            try:
                data = json.loads(row.results_json)
                return data if isinstance(data, list) else None
            except json.JSONDecodeError:
                return None

    def set_arxiv_query_cache(
        self,
        query_hash: str,
        search_query: str,
        max_results: int,
        results: List[Dict[str, Any]],
        ttl_hours: int = 24,
    ) -> None:
        """Store arXiv API results with TTL."""
        with self.get_session() as session:
            try:
                session.query(ArxivQueryCache).filter(
                    ArxivQueryCache.query_hash == query_hash
                ).delete()
                row = ArxivQueryCache(
                    query_hash=query_hash,
                    search_query=search_query,
                    max_results=max_results,
                    results_json=json.dumps(results, ensure_ascii=False),
                    expires_at=datetime.utcnow() + timedelta(hours=ttl_hours),
                )
                session.add(row)
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                log_exception(logger, "Failed to cache arXiv query", e)
