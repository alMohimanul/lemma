"""Database migration utilities for adding incremental embedding support.

This module provides functions to safely migrate existing MAVYN databases
to support the new versioning and advanced chunking features.
"""
import hashlib
from pathlib import Path
from typing import Dict, Any
from sqlalchemy import text

from ..utils.logger import get_logger
from .repository import Repository
from .models import Paper, Embedding

logger = get_logger(__name__)


def migrate_to_versioning(db_path: str = "~/.MAVYN/MAVYN.db") -> bool:
    """Migrate existing database to support versioning and incremental embeddings.

    This migration:
    1. Checks if new columns exist (safe to run multiple times)
    2. Adds new columns to Paper table with defaults
    3. Adds new columns to Embedding table with defaults
    4. Computes initial content hashes for existing papers
    5. Sets version numbers based on existing embedding status

    Args:
        db_path: Path to the database file

    Returns:
        True if migration successful, False otherwise
    """
    logger.info("Starting database migration to versioning schema")

    try:
        repo = Repository(db_path)

        with repo.get_session() as session:
            # Check if migration is needed by looking for new columns
            try:
                result = session.execute(
                    text("SELECT content_hash FROM papers LIMIT 1")
                )
                result.fetchone()
                logger.info("Migration already applied - content_hash column exists")
                return True
            except Exception:
                # Column doesn't exist, need to migrate
                logger.info("Applying migration - adding new columns")

            # Add columns to papers table
            try:
                session.execute(
                    text("ALTER TABLE papers ADD COLUMN content_hash VARCHAR(64)")
                )
                logger.info("Added content_hash column to papers table")
            except Exception as e:
                logger.debug(f"content_hash column may already exist: {e}")

            try:
                session.execute(
                    text(
                        "ALTER TABLE papers ADD COLUMN content_version INTEGER DEFAULT 1 NOT NULL"
                    )
                )
                logger.info("Added content_version column to papers table")
            except Exception as e:
                logger.debug(f"content_version column may already exist: {e}")

            try:
                session.execute(
                    text("ALTER TABLE papers ADD COLUMN last_embedded_version INTEGER")
                )
                logger.info("Added last_embedded_version column to papers table")
            except Exception as e:
                logger.debug(f"last_embedded_version column may already exist: {e}")

            try:
                session.execute(
                    text("ALTER TABLE papers ADD COLUMN last_embedded_at DATETIME")
                )
                logger.info("Added last_embedded_at column to papers table")
            except Exception as e:
                logger.debug(f"last_embedded_at column may already exist: {e}")

            # Add columns to embeddings table
            try:
                session.execute(
                    text(
                        "ALTER TABLE embeddings ADD COLUMN content_version INTEGER DEFAULT 1 NOT NULL"
                    )
                )
                logger.info("Added content_version column to embeddings table")
            except Exception as e:
                logger.debug(f"content_version column may already exist: {e}")

            try:
                session.execute(
                    text("ALTER TABLE embeddings ADD COLUMN chunk_hash VARCHAR(64)")
                )
                logger.info("Added chunk_hash column to embeddings table")
            except Exception as e:
                logger.debug(f"chunk_hash column may already exist: {e}")

            try:
                session.execute(
                    text(
                        "ALTER TABLE embeddings ADD COLUMN chunk_type VARCHAR(32) DEFAULT 'paragraph'"
                    )
                )
                logger.info("Added chunk_type column to embeddings table")
            except Exception as e:
                logger.debug(f"chunk_type column may already exist: {e}")

            try:
                session.execute(
                    text("ALTER TABLE embeddings ADD COLUMN section_name VARCHAR(256)")
                )
                logger.info("Added section_name column to embeddings table")
            except Exception as e:
                logger.debug(f"section_name column may already exist: {e}")

            try:
                session.execute(
                    text(
                        "ALTER TABLE embeddings ADD COLUMN importance_score REAL DEFAULT 0.5"
                    )
                )
                logger.info("Added importance_score column to embeddings table")
            except Exception as e:
                logger.debug(f"importance_score column may already exist: {e}")

            try:
                session.execute(
                    text("ALTER TABLE embeddings ADD COLUMN is_valid BOOLEAN DEFAULT 1")
                )
                logger.info("Added is_valid column to embeddings table")
            except Exception as e:
                logger.debug(f"is_valid column may already exist: {e}")

            session.commit()

        # Initialize version data for existing papers
        logger.info("Initializing version data for existing papers")
        _initialize_paper_versions(repo)

        # Compute chunk hashes for existing embeddings
        logger.info("Computing chunk hashes for existing embeddings")
        _compute_chunk_hashes(repo)

        logger.info("Migration completed successfully")
        repo.close()
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return False


def _initialize_paper_versions(repo: Repository):
    """Initialize version numbers for existing papers.

    Args:
        repo: Repository instance
    """
    from ..core.extractor import MetadataExtractor

    extractor = MetadataExtractor()

    with repo.get_session() as session:
        papers = session.query(Paper).all()

        for paper in papers:
            # Set content_version to 1 for all existing papers
            paper.content_version = 1  # type: ignore[assignment]

            # If paper has embeddings, set last_embedded_version
            embeddings = repo.get_embeddings_by_paper(int(paper.id))
            if embeddings:
                paper.last_embedded_version = 1  # type: ignore[assignment]

                # Try to compute content hash
                try:
                    paper_path = Path(paper.file_path)
                    if paper_path.exists():
                        full_text = extractor.extract_full_text(paper_path)
                        content_hash = hashlib.sha256(
                            full_text.encode("utf-8")
                        ).hexdigest()
                        paper.content_hash = content_hash  # type: ignore[assignment]
                        logger.debug(f"Computed content hash for paper {paper.id}")
                except Exception as e:
                    logger.warning(
                        f"Could not compute content hash for paper {paper.id}: {e}"
                    )

        session.commit()
        logger.info(f"Initialized version data for {len(papers)} papers")


def _compute_chunk_hashes(repo: Repository):
    """Compute hashes for existing embedding chunks.

    Args:
        repo: Repository instance
    """
    with repo.get_session() as session:
        embeddings = session.query(Embedding).all()

        count = 0
        for embedding in embeddings:
            if embedding.text_content and not embedding.chunk_hash:
                chunk_hash = hashlib.sha256(
                    embedding.text_content.encode("utf-8")
                ).hexdigest()
                embedding.chunk_hash = chunk_hash  # type: ignore[assignment]
                count += 1

        session.commit()
        logger.info(f"Computed chunk hashes for {count} embeddings")


def check_migration_status(db_path: str = "~/.MAVYN/MAVYN.db") -> dict:
    """Check if database has been migrated to versioning schema.

    Args:
        db_path: Path to the database file

    Returns:
        Dictionary with migration status information
    """
    repo = Repository(db_path)

    status: Dict[str, Any] = {
        "needs_migration": False,
        "has_content_hash": False,
        "has_content_version": False,
        "has_chunk_hash": False,
        "papers_with_hashes": 0,
        "total_papers": 0,
    }

    try:
        with repo.get_session() as session:
            # Check for new columns
            try:
                session.execute(text("SELECT content_hash FROM papers LIMIT 1"))
                status["has_content_hash"] = True
            except Exception:
                status["needs_migration"] = True

            try:
                session.execute(text("SELECT content_version FROM papers LIMIT 1"))
                status["has_content_version"] = True
            except Exception:
                status["needs_migration"] = True

            try:
                session.execute(text("SELECT chunk_hash FROM embeddings LIMIT 1"))
                status["has_chunk_hash"] = True
            except Exception:
                status["needs_migration"] = True

            # Count papers with hashes
            if status["has_content_hash"]:
                result = session.execute(
                    text("SELECT COUNT(*) FROM papers WHERE content_hash IS NOT NULL")
                )
                count_row = result.fetchone()
                status["papers_with_hashes"] = int(count_row[0]) if count_row else 0

            result = session.execute(text("SELECT COUNT(*) FROM papers"))
            count_row = result.fetchone()
            status["total_papers"] = int(count_row[0]) if count_row else 0

    except Exception as e:
        logger.error(f"Error checking migration status: {e}")
        status["error"] = str(e)

    finally:
        repo.close()

    return status


if __name__ == "__main__":
    # Run migration when executed directly
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else "~/.MAVYN/MAVYN.db"

    print(f"Migrating database: {db_path}")
    success = migrate_to_versioning(db_path)

    if success:
        print("✓ Migration completed successfully")
        print("\nYou can now use:")
        print("  - lemma embed --incremental (for smart updates)")
        print("  - lemma embed --strategy hybrid (for better chunking)")
        print("  - lemma embed-status (to check status)")
        sys.exit(0)
    else:
        print("✗ Migration failed - check logs for details")
        sys.exit(1)
