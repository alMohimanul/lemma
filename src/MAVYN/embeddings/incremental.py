"""Incremental embedding system with change detection and smart updates.

This module handles intelligent re-embedding of papers by detecting changes
and reusing existing embeddings when possible.
"""
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
from pathlib import Path

from ..utils.logger import get_logger
from .chunking import PaperChunker
from .encoder import EmbeddingEncoder

logger = get_logger(__name__)


class ChangeType(Enum):
    """Types of changes that can occur to a paper."""

    NO_CHANGE = "no_change"  # No changes detected
    METADATA_ONLY = "metadata_only"  # Only metadata changed (no re-embedding needed)
    CONTENT_CHANGED = "content_changed"  # Text content changed (incremental update)
    NEW_PAPER = "new_paper"  # Paper has no embeddings yet
    MODEL_UPGRADE = "model_upgrade"  # Embedding model changed
    FORCE_UPDATE = "force_update"  # User requested full re-embedding


@dataclass
class EmbedResult:
    """Result of an embedding operation."""

    paper_id: int
    change_type: ChangeType
    total_chunks: int
    reused_chunks: int
    new_chunks: int
    deleted_chunks: int
    time_saved_pct: float = 0.0
    success: bool = True
    error: Optional[str] = None


class IncrementalEmbedder:
    """Manages incremental embedding updates with change detection."""

    def __init__(self, chunker: PaperChunker, encoder: EmbeddingEncoder, repository):
        """Initialize incremental embedder.

        Args:
            chunker: PaperChunker instance for text chunking
            encoder: EmbeddingEncoder instance for generating embeddings
            repository: Database repository for storage
        """
        self.chunker = chunker
        self.encoder = encoder
        self.repo = repository

    def compute_content_hash(self, text: str) -> str:
        """Compute SHA256 hash of text content.

        Args:
            text: Full text content

        Returns:
            Hex-encoded SHA256 hash
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def detect_changes(self, paper, full_text: str) -> ChangeType:
        """Detect what type of changes occurred to a paper.

        Args:
            paper: Paper database model instance
            full_text: Current extracted text from PDF

        Returns:
            ChangeType indicating what changed
        """
        # Check if paper has any embeddings
        existing_embeddings = self.repo.get_embeddings_by_paper(paper.id)

        if not existing_embeddings:
            return ChangeType.NEW_PAPER

        # Compute current content hash
        current_hash = self.compute_content_hash(full_text)

        # Check if content hash is stored
        if not paper.content_hash:
            # Legacy paper without hash - assume content changed
            logger.info(
                f"Paper {paper.id} has no content hash, assuming content changed"
            )
            return ChangeType.CONTENT_CHANGED

        # Compare hashes
        if current_hash == paper.content_hash:
            # Content hasn't changed
            # Check if model changed
            current_model = self.encoder.model_name
            embedded_model = (
                existing_embeddings[0].model_name if existing_embeddings else None
            )

            if current_model != embedded_model:
                return ChangeType.MODEL_UPGRADE

            return ChangeType.NO_CHANGE
        else:
            # Content has changed
            return ChangeType.CONTENT_CHANGED

    def needs_re_embedding(self, paper, change_type: ChangeType) -> bool:
        """Determine if a paper needs re-embedding.

        Args:
            paper: Paper database model instance
            change_type: Detected change type

        Returns:
            True if re-embedding is needed
        """
        if change_type in [
            ChangeType.NEW_PAPER,
            ChangeType.CONTENT_CHANGED,
            ChangeType.MODEL_UPGRADE,
            ChangeType.FORCE_UPDATE,
        ]:
            return True
        return False

    def incremental_embed(
        self,
        paper,
        full_text: str,
        extractor,
        force: bool = False,
        chunks: Optional[List] = None,
    ) -> EmbedResult:
        """Perform incremental embedding with chunk reuse.

        Args:
            paper: Paper database model instance
            full_text: Current extracted text from PDF (used for content-hash
                change detection; when ``chunks`` is provided this may be the
                joined chunk text rather than raw PDF text).
            extractor: MetadataExtractor instance for text extraction
            force: If True, force full re-embedding
            chunks: Optional pre-computed list of Chunk objects (e.g. from
                DoclingPaperChunker).  When supplied the internal
                ``self.chunker.chunk(full_text)`` step is skipped entirely.

        Returns:
            EmbedResult with statistics
        """
        try:
            # Detect changes
            if force:
                change_type = ChangeType.FORCE_UPDATE
            else:
                change_type = self.detect_changes(paper, full_text)

            # If no re-embedding needed, return early
            if not self.needs_re_embedding(paper, change_type):
                existing_count = len(self.repo.get_embeddings_by_paper(paper.id))
                return EmbedResult(
                    paper_id=paper.id,
                    change_type=change_type,
                    total_chunks=existing_count,
                    reused_chunks=existing_count,
                    new_chunks=0,
                    deleted_chunks=0,
                    time_saved_pct=100.0,
                )

            # Chunk the text — use caller-supplied chunks when available
            # (e.g. from Docling), otherwise fall back to the legacy chunker.
            if chunks is None:
                chunks = self.chunker.chunk(full_text)

            if not chunks:
                raise ValueError("No chunks generated from text")

            # Get existing embeddings if this is an incremental update
            from typing import Any as AnyType

            existing_embeddings_map: Dict[str, AnyType] = {}
            if change_type == ChangeType.CONTENT_CHANGED:
                existing = self.repo.get_embeddings_by_paper(paper.id)
                # Map chunk_hash -> embedding for quick lookup
                for emb in existing:
                    if emb.chunk_hash:
                        existing_embeddings_map[emb.chunk_hash] = emb

            # Process chunks and determine which to reuse vs regenerate
            reused_count = 0
            new_count = 0
            new_embeddings = []

            for idx, chunk in enumerate(chunks):
                chunk_hash = chunk.hash

                # Check if we can reuse existing embedding
                if chunk_hash in existing_embeddings_map and not force:
                    # Reuse existing embedding - just update chunk_index
                    existing_emb = existing_embeddings_map[chunk_hash]
                    if hasattr(existing_emb, "id"):
                        self.repo.update_embedding_index(existing_emb.id, idx)
                    reused_count += 1
                    logger.debug(
                        f"Reusing embedding for chunk {idx} (hash: {chunk_hash[:8]}...)"
                    )
                else:
                    # Generate new embedding
                    embedding_vector = self.encoder.encode(chunk.text)
                    new_embeddings.append((idx, chunk, embedding_vector))
                    new_count += 1

            # Batch insert new embeddings
            if new_embeddings:
                # Compute new content version
                new_version = (paper.content_version or 0) + 1

                for idx, chunk, embedding_vector in new_embeddings:
                    self.repo.add_embedding(
                        paper_id=paper.id,
                        chunk_index=idx,
                        text_content=chunk.text[:1000],  # Limit stored text
                        embedding_vector=embedding_vector.tolist(),
                        chunk_hash=chunk.hash,
                        chunk_type=chunk.metadata.chunk_type.value,
                        section_name=chunk.metadata.section_name,
                        importance_score=chunk.metadata.importance_score,
                        content_version=new_version,
                        model_name=self.encoder.model_name,
                    )
                    logger.debug(f"Created new embedding for chunk {idx}")

            # Invalidate orphaned embeddings (chunks that no longer exist)
            current_hashes = {chunk.hash for chunk in chunks}
            deleted_count = self.repo.invalidate_orphaned_embeddings(
                paper.id, valid_hashes=current_hashes
            )

            # Update paper metadata
            content_hash = self.compute_content_hash(full_text)
            new_version = (paper.content_version or 0) + 1

            self.repo.update_paper_embedding_metadata(
                paper_id=paper.id,
                content_hash=content_hash,
                content_version=new_version,
                embedding_status="completed",
            )

            # Calculate time saved percentage
            total_chunks = len(chunks)
            time_saved_pct = (
                (reused_count / total_chunks * 100) if total_chunks > 0 else 0
            )

            logger.info(
                f"Paper {paper.id}: Reused {reused_count}/{total_chunks} chunks "
                f"({time_saved_pct:.1f}% time saved)"
            )

            return EmbedResult(
                paper_id=paper.id,
                change_type=change_type,
                total_chunks=total_chunks,
                reused_chunks=reused_count,
                new_chunks=new_count,
                deleted_chunks=deleted_count,
                time_saved_pct=time_saved_pct,
            )

        except Exception as e:
            logger.error(f"Failed to embed paper {paper.id}: {e}", exc_info=True)

            # Mark as failed
            self.repo.update_paper_metadata(paper.id, {"embedding_status": "failed"})

            return EmbedResult(
                paper_id=paper.id,
                change_type=change_type
                if "change_type" in locals()
                else ChangeType.NEW_PAPER,
                total_chunks=0,
                reused_chunks=0,
                new_chunks=0,
                deleted_chunks=0,
                success=False,
                error=str(e),
            )

    def process_papers(
        self, papers: List, extractor, force: bool = False
    ) -> List[EmbedResult]:
        """Process multiple papers with incremental embedding.

        Args:
            papers: List of Paper instances to process
            extractor: MetadataExtractor instance
            force: If True, force full re-embedding for all

        Returns:
            List of EmbedResult objects
        """
        results = []

        for paper in papers:
            try:
                # Extract full text
                paper_path = Path(paper.file_path)
                if not paper_path.exists():
                    logger.warning(f"Paper file not found: {paper_path}")
                    results.append(
                        EmbedResult(
                            paper_id=paper.id,
                            change_type=ChangeType.NO_CHANGE,
                            total_chunks=0,
                            reused_chunks=0,
                            new_chunks=0,
                            deleted_chunks=0,
                            success=False,
                            error="File not found",
                        )
                    )
                    continue

                # Update status to processing
                self.repo.update_paper_metadata(
                    paper.id, {"embedding_status": "processing"}
                )

                full_text = extractor.extract_full_text(paper_path)

                if not full_text or len(full_text.strip()) < 100:
                    raise ValueError("Insufficient text extracted from PDF")

                # Perform incremental embedding
                result = self.incremental_embed(
                    paper, full_text, extractor, force=force
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing paper {paper.id}: {e}", exc_info=True)
                results.append(
                    EmbedResult(
                        paper_id=paper.id,
                        change_type=ChangeType.NEW_PAPER,
                        total_chunks=0,
                        reused_chunks=0,
                        new_chunks=0,
                        deleted_chunks=0,
                        success=False,
                        error=str(e),
                    )
                )

        return results

    def get_embedding_stats(self, paper) -> Dict:
        """Get embedding statistics for a paper.

        Args:
            paper: Paper database model instance

        Returns:
            Dictionary with embedding statistics
        """
        embeddings = self.repo.get_embeddings_by_paper(paper.id)
        valid_embeddings = [e for e in embeddings if e.is_valid]

        return {
            "total_embeddings": len(embeddings),
            "valid_embeddings": len(valid_embeddings),
            "invalid_embeddings": len(embeddings) - len(valid_embeddings),
            "content_version": paper.content_version or 1,
            "last_embedded_version": paper.last_embedded_version,
            "needs_update": (paper.content_version or 1)
            > (paper.last_embedded_version or 0),
            "has_content_hash": paper.content_hash is not None,
        }
