"""Basic smoke tests for core functionality."""


def test_imports():
    """Test that core modules can be imported."""
    from lemma.core.scanner import PDFScanner
    from lemma.core.extractor import MetadataExtractor
    from lemma.core.organizer import FileOrganizer
    from lemma.db.repository import Repository
    from lemma.db.models import Paper
    from lemma.embeddings.encoder import EmbeddingEncoder
    from lemma.embeddings.search import SemanticSearchIndex

    assert PDFScanner is not None
    assert MetadataExtractor is not None
    assert FileOrganizer is not None
    assert Repository is not None
    assert Paper is not None
    assert EmbeddingEncoder is not None
    assert SemanticSearchIndex is not None


def test_scanner_basic(temp_dir):
    """Test basic scanner functionality."""
    from lemma.core.scanner import PDFScanner

    scanner = PDFScanner()
    results = scanner.scan_directory(temp_dir, recursive=False)
    assert isinstance(results, list)


def test_repository_basic(test_db_path):
    """Test basic repository initialization."""
    from lemma.db.repository import Repository

    with Repository(test_db_path) as repo:
        all_papers = repo.list_papers()
        assert isinstance(all_papers, list)


def test_encoder_basic():
    """Test basic encoder functionality."""
    from lemma.embeddings.encoder import EmbeddingEncoder
    import numpy as np

    encoder = EmbeddingEncoder()
    embedding = encoder.encode("Test sentence")
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 384
