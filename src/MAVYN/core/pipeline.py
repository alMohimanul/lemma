"""Automated paper processing pipeline."""
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .scanner import PDFScanner
from .extractor import MetadataExtractor
from .organizer import FileOrganizer
from ..db.repository import Repository
from ..embeddings.encoder import EmbeddingEncoder
from ..embeddings.search import SemanticSearchIndex
from ..embeddings.chunking import PaperChunker, ChunkingStrategy
from ..embeddings.incremental import IncrementalEmbedder
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Result of processing a single paper through the pipeline."""

    success: bool
    paper_id: Optional[int] = None
    file_path: Optional[str] = None
    original_path: Optional[str] = None
    renamed: bool = False
    embedded: bool = False
    error: Optional[str] = None
    stage: str = "not_started"  # scan, metadata, rename, embed, completed


class PaperProcessingPipeline:
    """Orchestrates the complete paper processing workflow."""

    def __init__(
        self,
        repo: Repository,
        auto_rename: bool = True,
        rename_pattern: str = "{year}_{first_author}_{short_title}.pdf",
        embed_immediately: bool = True,
        chunking_strategy: str = "hybrid",
        index_path: str = "~/.MAVYN/search.index",
    ):
        """Initialize the processing pipeline.

        Args:
            repo: Database repository
            auto_rename: Whether to automatically rename files
            rename_pattern: Pattern for renaming files
            embed_immediately: Whether to embed papers immediately
            chunking_strategy: Strategy for chunking (hybrid, structure, sentence, simple)
            index_path: Path to FAISS index
        """
        self.repo = repo
        self.auto_rename = auto_rename
        self.rename_pattern = rename_pattern
        self.embed_immediately = embed_immediately
        self.chunking_strategy = chunking_strategy
        self.index_path = Path(index_path).expanduser()

        # Initialize components
        self.scanner = PDFScanner()
        self.extractor = MetadataExtractor()
        self.organizer = FileOrganizer(dry_run=False)

        # Initialize embedding components (lazy loaded)
        self._encoder: Optional[EmbeddingEncoder] = None
        self._search_index: Optional[SemanticSearchIndex] = None
        self._chunker: Optional[PaperChunker] = None
        self._embedder: Optional[IncrementalEmbedder] = None

    @property
    def encoder(self) -> EmbeddingEncoder:
        """Lazy load embedding encoder."""
        if self._encoder is None:
            self._encoder = EmbeddingEncoder()
            logger.info(f"Loaded embedding model: {self._encoder.model_name}")
        return self._encoder

    @property
    def search_index(self) -> SemanticSearchIndex:
        """Lazy load search index."""
        if self._search_index is None:
            self._search_index = SemanticSearchIndex(
                embedding_dim=self.encoder.embedding_dim, index_path=self.index_path
            )
            logger.info(f"Loaded search index with {self._search_index.size()} vectors")
        return self._search_index

    @property
    def chunker(self) -> PaperChunker:
        """Lazy load chunker."""
        if self._chunker is None:
            strategy_map = {
                "structure": ChunkingStrategy.STRUCTURE_AWARE,
                "sentence": ChunkingStrategy.SENTENCE_BASED,
                "hybrid": ChunkingStrategy.HYBRID,
                "simple": ChunkingStrategy.SIMPLE,
            }
            self._chunker = PaperChunker(strategy=strategy_map[self.chunking_strategy])
            logger.info(f"Using {self.chunking_strategy} chunking strategy")
        return self._chunker

    @property
    def embedder(self) -> IncrementalEmbedder:
        """Lazy load incremental embedder."""
        if self._embedder is None:
            self._embedder = IncrementalEmbedder(self.chunker, self.encoder, self.repo)
        return self._embedder

    def process_file(self, file_path: Path) -> PipelineResult:
        """Process a single PDF file through the complete pipeline.

        Args:
            file_path: Path to PDF file

        Returns:
            PipelineResult with processing status
        """
        result = PipelineResult(success=False, original_path=str(file_path))

        try:
            # Stage 1: Scan and hash
            result.stage = "scan"
            scanned = self.scanner.scan_file(file_path)

            # Check if already in database
            existing = self.repo.get_paper_by_hash(scanned.file_hash)
            if existing:
                logger.info(f"Paper already indexed: {file_path.name}")
                result.success = True
                result.paper_id = existing.id
                result.file_path = existing.file_path
                result.stage = "duplicate"
                return result

            # Stage 2: Extract metadata
            result.stage = "metadata"
            metadata = self.extractor.extract(scanned.path)

            # Stage 3: Add to database
            paper = self.repo.add_paper(
                file_path=str(scanned.path),
                file_hash=scanned.file_hash,
                file_size=scanned.file_size,
                metadata=metadata.to_dict(),
            )

            if not paper:
                result.error = "Failed to add paper to database"
                return result

            result.paper_id = paper.id
            result.file_path = str(scanned.path)

            # Stage 4: Rename (if enabled)
            if self.auto_rename:
                result.stage = "rename"
                renamed_path = self._rename_paper(paper, scanned.path)
                if renamed_path:
                    result.file_path = str(renamed_path)
                    result.renamed = True
                    # Reload paper to get updated file_path
                    reloaded_paper = self.repo.get_paper_by_id(paper.id)
                    if reloaded_paper:
                        paper = reloaded_paper

            # Stage 5: Embed (if enabled)
            if self.embed_immediately:
                result.stage = "embed"
                embed_success = self._embed_paper(paper)
                result.embedded = embed_success

            # Mark as completed
            result.stage = "completed"
            result.success = True

            # Log success
            self.repo.log_operation(
                operation="pipeline",
                status="success",
                paper_id=paper.id,
                details={
                    "original_path": str(file_path),
                    "final_path": result.file_path,
                    "renamed": result.renamed,
                    "embedded": result.embedded,
                },
            )

            logger.info(
                f"Successfully processed: {Path(result.file_path).name} "
                f"(renamed={result.renamed}, embedded={result.embedded})"
            )

        except Exception as e:
            result.error = str(e)
            logger.error(f"Pipeline failed for {file_path}: {e}", exc_info=True)

            # Log failure
            if result.paper_id:
                self.repo.log_operation(
                    operation="pipeline",
                    status="failed",
                    paper_id=result.paper_id,
                    error_message=str(e),
                    details={"stage": result.stage},
                )

        return result

    def _rename_paper(self, paper, file_path: Path) -> Optional[Path]:
        """Rename a paper file based on metadata.

        Args:
            paper: Paper database object
            file_path: Current file path

        Returns:
            New file path if renamed, None otherwise
        """
        try:
            # Build paper dict for organizer
            paper_dict = {
                "id": paper.id,
                "file_path": str(file_path),
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
            }

            # Preview rename
            preview = self.organizer.preview_renames(
                [paper_dict], pattern=self.rename_pattern
            )

            if not preview or not preview[0].get("changed"):
                logger.info(f"No rename needed for: {file_path.name}")
                return None

            # Perform rename
            new_path = self.organizer.rename_file(
                original_path=file_path, new_filename=Path(preview[0]["new"]).name
            )

            if new_path:
                # Update database
                self.repo.update_paper_metadata(paper.id, {"file_path": str(new_path)})

                # Log file operation
                self.repo.log_file_operation(
                    paper_id=paper.id,
                    operation_type="rename",
                    original_path=str(file_path),
                    new_path=str(new_path),
                )

                logger.info(f"Renamed: {file_path.name} → {new_path.name}")
                return new_path

        except Exception as e:
            logger.warning(f"Failed to rename {file_path.name}: {e}")

        return None

    def _embed_paper(self, paper) -> bool:
        """Generate embeddings for a paper.

        Args:
            paper: Paper database object

        Returns:
            True if embedding successful
        """
        try:
            paper_path = Path(paper.file_path)
            if not paper_path.exists():
                logger.error(f"PDF not found: {paper_path}")
                return False

            # ── Stage 1: Extract text + chunks ───────────────────────────────
            # Try Docling first (AI layout model — correctly detects numbered
            # section headings like "3. Results" and two-column layouts).
            # Fall back to plain-text extraction if Docling is unavailable or
            # the conversion fails.
            docling_chunks = None
            full_text = None

            try:
                from ..embeddings.docling_chunker import chunk_pdf_with_docling

                docling_chunks = chunk_pdf_with_docling(paper_path)
                # Synthesise full_text from chunk text for content-hash tracking
                full_text = " ".join(c.text for c in docling_chunks)
                logger.info(
                    f"Docling extraction succeeded for {paper_path.name} "
                    f"({len(docling_chunks)} chunks)"
                )
            except ImportError:
                logger.debug("docling not installed — using legacy text extraction")
            except Exception as exc:
                logger.warning(
                    f"Docling failed for {paper_path.name} ({exc}); "
                    "falling back to plain-text extraction"
                )

            if full_text is None:
                full_text = self.extractor.extract_full_text(paper_path)

            if not full_text or len(full_text.strip()) < 100:
                logger.warning(f"Insufficient text extracted from {paper_path.name}")
                return False

            # ── Stage 2: Embed ────────────────────────────────────────────────
            # Pass pre-computed Docling chunks when available so the embedder
            # skips its internal chunker (which uses regex-based detection).
            embed_result = self.embedder.incremental_embed(
                paper, full_text, self.extractor, force=False, chunks=docling_chunks
            )

            if not embed_result.success:
                logger.error(f"Embedding failed: {embed_result.error}")
                return False

            # Add embeddings to FAISS index
            valid_embeddings = [
                e for e in self.repo.get_embeddings_by_paper(paper.id) if e.is_valid
            ]

            if valid_embeddings:
                import json
                import numpy as np

                # Extract embeddings and add to FAISS
                embedding_vectors = []
                chunk_indices = []

                for emb in valid_embeddings:
                    vec = json.loads(emb.embedding_vector)
                    embedding_vectors.append(vec)
                    chunk_indices.append(emb.chunk_index)

                embeddings_array = np.array(embedding_vectors, dtype=np.float32)

                self.search_index.add(
                    embeddings=embeddings_array,
                    paper_id=paper.id,
                    chunk_indices=chunk_indices,
                )

                # Save index
                self.search_index.save(self.index_path)

            logger.info(
                f"Embedded paper {paper.id}: {embed_result.total_chunks} chunks "
                f"({embed_result.reused_chunks} reused, {embed_result.new_chunks} new)"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to embed paper {paper.id}: {e}", exc_info=True)

            # Update status to failed
            self.repo.update_paper_metadata(paper.id, {"embedding_status": "failed"})

            return False

    def cleanup_missing_files(self) -> int:
        """Remove papers from database whose files no longer exist.

        Returns:
            Number of papers removed
        """
        logger.info("Checking for missing files in database...")

        # Get all papers
        papers = self.repo.list_papers(limit=100000)
        removed_count = 0

        for paper in papers:
            file_path = Path(paper.file_path)
            if not file_path.exists():
                logger.info(f"Removing missing paper: {paper.file_path}")
                self.repo.delete_paper(paper.id)
                removed_count += 1

        if removed_count > 0:
            logger.info(f"Removed {removed_count} papers with missing files")

        return removed_count

    def process_directory(
        self, directory: Path, recursive: bool = True
    ) -> Dict[str, Any]:
        """Process all PDFs in a directory.

        Args:
            directory: Directory to process
            recursive: Whether to scan subdirectories

        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing directory: {directory}")

        # First, cleanup missing files from database
        removed_count = self.cleanup_missing_files()

        # Scan for PDFs
        scanned_files = self.scanner.scan_directory(directory, recursive=recursive)

        # Process each file
        results: Dict[str, Any] = {
            "total": len(scanned_files),
            "success": 0,
            "duplicate": 0,
            "failed": 0,
            "renamed": 0,
            "embedded": 0,
            "removed": removed_count,
            "errors": [],
        }

        for scanned in scanned_files:
            result = self.process_file(scanned.path)

            if result.success:
                if result.stage == "duplicate":
                    results["duplicate"] += 1
                else:
                    results["success"] += 1
                    if result.renamed:
                        results["renamed"] += 1
                    if result.embedded:
                        results["embedded"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(
                    {
                        "file": str(scanned.path),
                        "error": result.error,
                        "stage": result.stage,
                    }
                )

        return results
