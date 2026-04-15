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


def test_extract_paper_id_from_question():
    from lemma.llm.question_parser import extract_paper_ids

    assert 9 in extract_paper_ids("explain the methodology of paper 9")
    assert extract_paper_ids("compare [1] and [2]") == [1, 2]


def test_wants_similar_papers():
    from lemma.llm.question_parser import wants_similar_papers

    assert wants_similar_papers("Find similar papers on transformers")
    assert wants_similar_papers("What related work exists for GNNs?")
    assert wants_similar_papers("Papers like paper 5 on RL")
    assert not wants_similar_papers("What is the main contribution of paper 3?")


def test_arxiv_client_helpers():
    from lemma.integrations.arxiv_client import (
        normalize_arxiv_id,
        build_arxiv_search_query,
        parse_arxiv_atom,
    )

    assert normalize_arxiv_id("2301.00001v2") == "2301.00001"
    assert normalize_arxiv_id("arxiv:1234.5678") == "1234.5678"
    q = build_arxiv_search_query("Graph neural networks for molecules")
    assert q.startswith("all:")
    assert parse_arxiv_atom("<not>xml</not>") == []


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
