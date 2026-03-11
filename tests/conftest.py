"""Pytest configuration and fixtures."""
import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp = Path(tempfile.mkdtemp())
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def sample_pdf(temp_dir):
    """Create a sample PDF file for testing."""
    pdf_path = temp_dir / "test_paper.pdf"
    # Create a minimal valid PDF
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
>>
endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
190
%%EOF"""
    pdf_path.write_bytes(pdf_content)
    return pdf_path


@pytest.fixture
def test_db_path(temp_dir):
    """Create a temporary database path with proper cleanup."""
    db_path = str(temp_dir / "test_lemma.db")
    yield db_path

    # Ensure all SQLAlchemy connections are closed before cleanup
    # This prevents Windows file locking issues if we add Windows support later
    import gc

    gc.collect()  # Force garbage collection to close any lingering connections
