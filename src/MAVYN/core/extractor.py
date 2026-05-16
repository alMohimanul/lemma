"""Metadata extraction from PDF files using regex-first approach."""
import re
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None  # type: ignore[assignment]

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore[assignment]


@dataclass
class PaperMetadata:
    """Extracted paper metadata."""

    title: Optional[str] = None
    authors: Optional[str] = None  # Comma-separated
    year: Optional[int] = None
    publication: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    abstract: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class MetadataExtractor:
    """Extract metadata from PDF files using regex patterns and PDF metadata."""

    # Regex patterns for common metadata fields
    PATTERNS = {
        "doi": re.compile(r"10\.\d{4,}/[^\s]+", re.IGNORECASE),
        "arxiv": re.compile(r"arXiv:(\d{4}\.\d{4,5})", re.IGNORECASE),
        "year": re.compile(r"\b(19|20)\d{2}\b"),
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    }

    # Common patterns for title extraction (usually first large text)
    TITLE_PATTERNS = [
        re.compile(r"^([A-Z][^\n]{10,200})\n", re.MULTILINE),  # First capitalized line
        re.compile(
            r"^(.{10,200}?)\n\n", re.MULTILINE
        ),  # First line before double newline
    ]

    # Abstract markers
    ABSTRACT_PATTERNS = [
        re.compile(
            r"abstract[:\s]+(.{50,2000}?)(?:\n\n|keywords|introduction)",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"summary[:\s]+(.{50,2000}?)(?:\n\n|keywords|introduction)",
            re.IGNORECASE | re.DOTALL,
        ),
    ]

    # Year patterns ordered by reliability (most specific first)
    PUBLICATION_YEAR_PATTERNS = [
        re.compile(
            r"(?:published|accepted|received|submitted)[^\n]{0,40}?((?:19|20)\d{2})",
            re.IGNORECASE,
        ),
        re.compile(r"(?:©|\(c\)|copyright)\s*((?:19|20)\d{2})", re.IGNORECASE),
        re.compile(
            r"(?:proceedings?|conference|workshop|symposium)[^\n]{0,60}?((?:19|20)\d{2})",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:journal|volume|vol\.?)[^\n]{0,40}?((?:19|20)\d{2})", re.IGNORECASE
        ),
    ]

    def __init__(self):
        """Initialize extractor."""
        if PyPDF2 is None and pdfplumber is None:
            raise ImportError("Either PyPDF2 or pdfplumber must be installed")

    def extract(self, pdf_path: Path) -> PaperMetadata:
        """Extract metadata from a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PaperMetadata object with extracted fields
        """
        pdf_path = Path(pdf_path).expanduser().resolve()

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Try PDF metadata first (fastest)
        metadata = self._extract_pdf_metadata(pdf_path)

        # Extract text and apply regex patterns
        text = self._extract_text(pdf_path)
        if text:
            regex_metadata = self._extract_from_text(text)
            # Merge, preferring regex results for missing fields
            for key, value in regex_metadata.to_dict().items():
                if getattr(metadata, key) is None:
                    setattr(metadata, key, value)
            # ArXiv ID encodes submission year — always prefer it over PDF date
            if regex_metadata.arxiv_id and regex_metadata.year:
                metadata.year = regex_metadata.year

        return metadata

    def _extract_pdf_metadata(self, pdf_path: Path) -> PaperMetadata:
        """Extract metadata from PDF's internal metadata fields.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PaperMetadata with available PDF metadata
        """
        metadata = PaperMetadata()

        if PyPDF2:
            try:
                with open(pdf_path, "rb") as f:
                    pdf = PyPDF2.PdfReader(f)
                    info = pdf.metadata

                    if info:
                        if hasattr(info, "title") and info.title:
                            metadata.title = self._clean_text(info.title)
                        if hasattr(info, "author") and info.author:
                            metadata.authors = self._clean_text(info.author)
                        if hasattr(info, "subject") and info.subject:
                            metadata.publication = self._clean_text(info.subject)
                        # Extract year from PDF creation/modification date
                        for date_key in ("/CreationDate", "/ModDate"):
                            date_val = info.get(date_key)
                            if date_val:
                                date_match = re.search(
                                    r"((?:19|20)\d{2})", str(date_val)
                                )
                                if date_match:
                                    metadata.year = int(date_match.group(1))
                                    break
            except Exception:
                pass

        return metadata

    def _extract_text(self, pdf_path: Path, max_pages: int = 3) -> str:
        """Extract text from PDF using PyPDF2 with pdfplumber fallback.

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum pages to extract (first N pages for metadata)

        Returns:
            Extracted text content
        """
        text = ""

        # Try PyPDF2 first (faster)
        if PyPDF2:
            try:
                with open(pdf_path, "rb") as f:
                    pypdf_reader = PyPDF2.PdfReader(f)
                    for i in range(min(max_pages, len(pypdf_reader.pages))):
                        page = pypdf_reader.pages[i]
                        text += page.extract_text() or ""
                        text += "\n\n"

                # If we got good text, return it
                if len(text.strip()) > 100:
                    return text
            except Exception:
                pass  # Fall through to pdfplumber

        # Fallback to pdfplumber (more robust for complex PDFs)
        if pdfplumber:
            try:
                with pdfplumber.open(pdf_path) as plumber_pdf:
                    for i in range(min(max_pages, len(plumber_pdf.pages))):
                        plumber_page = plumber_pdf.pages[i]
                        text += plumber_page.extract_text() or ""
                        text += "\n\n"
            except Exception:
                pass

        return text

    def _extract_from_text(self, text: str) -> PaperMetadata:
        """Extract metadata from text using regex patterns.

        Args:
            text: Extracted text content

        Returns:
            PaperMetadata with extracted fields
        """
        metadata = PaperMetadata()

        # Extract DOI
        doi_match = self.PATTERNS["doi"].search(text)
        if doi_match:
            metadata.doi = doi_match.group(0)

        # Extract arXiv ID
        arxiv_match = self.PATTERNS["arxiv"].search(text)
        if arxiv_match:
            metadata.arxiv_id = arxiv_match.group(1)

        # Extract year — priority chain:
        # 1. ArXiv ID encodes the submission year (most reliable)
        if metadata.arxiv_id:
            metadata.year = self._year_from_arxiv_id(metadata.arxiv_id)

        # 2. Publication-context keywords near a year
        if not metadata.year:
            for pattern in self.PUBLICATION_YEAR_PATTERNS:
                m = pattern.search(text)
                if m:
                    try:
                        metadata.year = int(m.group(1))
                        break
                    except ValueError:
                        pass

        # 3. Fallback: first year in text
        if not metadata.year:
            year_match = self.PATTERNS["year"].search(text)
            if year_match:
                try:
                    metadata.year = int(year_match.group(0))
                except ValueError:
                    pass

        # Extract title (try multiple patterns)
        for pattern in self.TITLE_PATTERNS:
            title_match = pattern.search(text)
            if title_match:
                candidate = self._clean_text(title_match.group(1))
                # Basic validation (title should be reasonable length)
                if 10 <= len(candidate) <= 200 and not candidate.startswith("http"):
                    metadata.title = candidate
                    break

        # Extract abstract
        for pattern in self.ABSTRACT_PATTERNS:
            abstract_match = pattern.search(text)
            if abstract_match:
                metadata.abstract = self._clean_text(abstract_match.group(1))
                break

        # Extract authors (heuristic: names after title, before abstract)
        # This is tricky and often unreliable, so we keep it simple
        if metadata.title:
            # Look for text between title and abstract/introduction
            title_pos = text.find(metadata.title)
            if title_pos != -1:
                chunk = text[title_pos + len(metadata.title) : title_pos + 500]
                # Look for author-like patterns (capitalized names)
                author_pattern = re.compile(r"([A-Z][a-z]+\s+[A-Z][a-z]+)")
                authors = author_pattern.findall(chunk)
                if authors:
                    # Take first few matches, likely authors
                    metadata.authors = ", ".join(authors[:5])

        return metadata

    def _year_from_arxiv_id(self, arxiv_id: str) -> Optional[int]:
        """Derive publication year from an arXiv ID."""
        # New format: YYMM.NNNNN (e.g. 2301.12345 → 2023)
        m = re.match(r"^(\d{2})\d{2}\.\d+", arxiv_id)
        if m:
            yy = int(m.group(1))
            return 2000 + yy
        # Old format: category/YYMMNNN (e.g. math/9812345 → 1998)
        m = re.match(r"[a-z.-]+/(\d{2})\d{5}", arxiv_id, re.IGNORECASE)
        if m:
            yy = int(m.group(1))
            return (1900 + yy) if yy >= 91 else (2000 + yy)
        return None

    def _clean_text(self, text: str) -> str:
        """Clean extracted text (remove extra whitespace, etc.).

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_full_text(self, pdf_path: Path) -> str:
        """Extract full text from all pages of a PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Full text content
        """
        pdf_path = Path(pdf_path).expanduser().resolve()
        text = ""

        # Try PyPDF2 first
        if PyPDF2:
            try:
                with open(pdf_path, "rb") as f:
                    pypdf_reader = PyPDF2.PdfReader(f)
                    for page in pypdf_reader.pages:
                        text += page.extract_text() or ""
                        text += "\n\n"

                if len(text.strip()) > 100:
                    return text
            except Exception:
                pass

        # Fallback to pdfplumber
        if pdfplumber:
            try:
                with pdfplumber.open(pdf_path) as plumber_pdf:
                    for plumber_page in plumber_pdf.pages:
                        text += plumber_page.extract_text() or ""
                        text += "\n\n"
            except Exception:
                pass

        return text
