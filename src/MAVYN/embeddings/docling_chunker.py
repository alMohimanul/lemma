"""Docling-based PDF chunker for accurate section detection.

Replaces the regex-based section detection in PaperChunker with Docling's
AI layout model (RT-DETR trained on DocLayNet), which detects section headers
visually — so "3. Results" on a two-column page is correctly identified even
when plain text extraction merges it with surrounding content.

Usage:
    from .docling_chunker import chunk_pdf_with_docling
    chunks = chunk_pdf_with_docling(Path("paper.pdf"))

Fallback: if Docling is unavailable, raises ImportError so the caller can
fall back to the legacy PaperChunker.
"""
import re
from pathlib import Path
from typing import List, Optional, Tuple

from .chunking import Chunk, ChunkType, PaperChunker
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Heading levels to split on (# = H1, ## = H2, ### = H3)
_HEADER_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

# Strip markdown formatting noise from section text before chunking
_MD_NOISE_RE = re.compile(
    r"^\s*[-*]{3,}\s*$|"  # horizontal rules
    r"!\[.*?\]\(.*?\)|"  # image references
    r"\[([^\]]+)\]\([^\)]+\)",  # inline links → keep link text
    re.MULTILINE,
)


def chunk_pdf_with_docling(
    pdf_path: Path,
    target_chunk_size: int = 300,
    max_chunk_size: int = 400,
) -> List[Chunk]:
    """Convert a PDF with Docling and return structured Chunk objects.

    Docling uses an AI layout model that detects section headers at the
    bounding-box level, so it correctly handles:
      - Numbered headings ("3. Results") that aren't on their own line
      - Two-column layouts (reads columns in correct order)
      - Mixed single/double column pages (title + abstract single, body double)

    Args:
        pdf_path: Path to the PDF file.
        target_chunk_size: Approximate token count per chunk.
        max_chunk_size: Hard token limit per chunk.

    Returns:
        List of Chunk objects with section_name and chunk_type set from
        Docling's heading detection.

    Raises:
        ImportError: If docling is not installed.
        Exception: If Docling conversion fails (caller should fall back).
    """
    from docling.document_converter import DocumentConverter

    logger.info(f"Docling: converting {pdf_path.name}")
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    md_text = result.document.export_to_markdown()

    chunks = _chunk_markdown(md_text, target_chunk_size, max_chunk_size)
    logger.info(f"Docling: produced {len(chunks)} chunks from {pdf_path.name}")
    return chunks


def _parse_markdown_sections(md_text: str) -> List[Tuple[Optional[str], str]]:
    """Split Markdown text on headers into (section_name, body_text) pairs.

    Subsection headers (## / ###) become the section_name of their block.
    Text that precedes the first header is returned with section_name=None.
    """
    matches = list(_HEADER_RE.finditer(md_text))

    if not matches:
        return [(None, md_text)]

    sections: List[Tuple[Optional[str], str]] = []

    # Content before the first header
    pre = md_text[: matches[0].start()].strip()
    if pre:
        sections.append((None, pre))

    for i, m in enumerate(matches):
        name = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        body = md_text[start:end].strip()

        # Clean markdown noise from body text
        body = _MD_NOISE_RE.sub(r"\1", body).strip()

        sections.append((name, body))

    return sections


def _chunk_markdown(
    md_text: str,
    target_chunk_size: int,
    max_chunk_size: int,
) -> List[Chunk]:
    """Parse Markdown into Chunk objects, one section at a time.

    Each Markdown header becomes the `section_name` of the chunks it
    contains. The existing sentence-based chunker is reused so chunk sizing
    behaviour is identical to the legacy pipeline.
    """
    base_chunker = PaperChunker(
        target_chunk_size=target_chunk_size,
        max_chunk_size=max_chunk_size,
    )

    sections = _parse_markdown_sections(md_text)
    chunks: List[Chunk] = []

    for section_name, body in sections:
        if not body:
            continue

        chunk_type = (
            base_chunker._classify_section(section_name)
            if section_name
            else ChunkType.PARAGRAPH
        )

        section_chunks = base_chunker._chunk_text_by_sentences(
            body,
            section_name=section_name,
            chunk_type=chunk_type,
        )
        chunks.extend(section_chunks)

    # Docling always returns a non-empty document; guard against edge cases
    if not chunks and md_text.strip():
        chunks = base_chunker._chunk_text_by_sentences(
            md_text, section_name=None, chunk_type=ChunkType.PARAGRAPH
        )

    return chunks
