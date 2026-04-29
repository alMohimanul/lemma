"""Advanced chunking strategies for academic papers.

This module implements structure-aware and semantic chunking for research papers,
optimized for embedding generation and semantic search.
"""
import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

# Try to import nltk for sentence tokenization
try:
    import nltk
    from nltk.tokenize import sent_tokenize

    # Download required data if not already present
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class ChunkType(Enum):
    """Types of chunks in academic papers."""

    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    REFERENCE = "reference"
    FIGURE_CAPTION = "figure_caption"


class ChunkingStrategy(Enum):
    """Available chunking strategies."""

    STRUCTURE_AWARE = "structure"  # Parse sections and paragraphs
    SENTENCE_BASED = "sentence"  # Semantic sentence-based chunking
    HYBRID = "hybrid"  # Combine structure + sentences (recommended)
    SIMPLE = "simple"  # Legacy word-based chunking


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""

    chunk_type: ChunkType = ChunkType.PARAGRAPH
    section_name: Optional[str] = None
    page_number: Optional[int] = None
    importance_score: float = 0.5
    start_char: int = 0
    end_char: int = 0
    sentence_count: int = 0


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    text: str
    metadata: ChunkMetadata
    hash: Optional[str] = None

    def __post_init__(self):
        """Compute hash after initialization."""
        if self.hash is None:
            self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Compute SHA256 hash of chunk text."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass
class PaperStructure:
    """Parsed structure of an academic paper."""

    title: Optional[str] = None
    abstract: Optional[str] = None
    sections: List[Tuple[str, str]] = field(default_factory=list)  # (name, text)
    references: Optional[str] = None


class PaperChunker:
    """Advanced chunker for academic papers with multiple strategies."""

    # Regex patterns for section detection
    SECTION_PATTERNS = [
        # Numbered sections on their own line: "1. Introduction", "2.1 Background"
        re.compile(r"^\s*(\d+\.(?:\d+\.?)*)\s+([A-Z][^\n]{2,100})\s*$", re.MULTILINE),
        # Numbered sections at start of a paragraph (PDF extraction sometimes
        # merges the heading into the next line): "3. Results\nThe model..."
        re.compile(r"^\s*(\d+\.(?:\d+\.?)*)\s+([A-Z][^\n]{2,80})\n", re.MULTILINE),
        # Keyword-based sections
        re.compile(
            r"^\s*(Abstract|Introduction|Background|Related Work|Methods?|"
            r"Methodology|Results?|Discussion|Conclusion|References?|"
            r"Acknowledgements?|Appendix)\s*$",
            re.MULTILINE | re.IGNORECASE,
        ),
        # All-caps sections: "INTRODUCTION", "METHODS"
        re.compile(r"^\s*([A-Z][A-Z\s]{3,50})\s*$", re.MULTILINE),
    ]

    # Abstract detection patterns
    ABSTRACT_PATTERNS = [
        re.compile(
            r"\b(Abstract|ABSTRACT|Summary)\b[:\s]*\n(.{100,3000}?)\n\n", re.DOTALL
        ),
    ]

    # Importance scores for different chunk types
    IMPORTANCE_SCORES = {
        ChunkType.TITLE: 1.0,
        ChunkType.ABSTRACT: 1.0,
        ChunkType.INTRODUCTION: 0.9,
        ChunkType.METHODS: 0.7,
        ChunkType.RESULTS: 0.8,
        ChunkType.DISCUSSION: 0.9,
        ChunkType.CONCLUSION: 0.9,
        ChunkType.SECTION: 0.7,
        ChunkType.PARAGRAPH: 0.5,
        ChunkType.REFERENCE: 0.3,
        ChunkType.FIGURE_CAPTION: 0.6,
    }

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
        target_chunk_size: int = 300,  # tokens
        overlap_size: int = 50,  # tokens
        max_chunk_size: int = 400,  # tokens (hard limit)
    ):
        """Initialize chunker with strategy and parameters.

        Args:
            strategy: Chunking strategy to use
            target_chunk_size: Target number of tokens per chunk
            overlap_size: Number of tokens to overlap between chunks
            max_chunk_size: Maximum tokens per chunk
        """
        self.strategy = strategy
        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> List[Chunk]:
        """Chunk text using the configured strategy.

        Args:
            text: Full text to chunk

        Returns:
            List of Chunk objects with metadata
        """
        if self.strategy == ChunkingStrategy.STRUCTURE_AWARE:
            return self.chunk_by_structure(text)
        elif self.strategy == ChunkingStrategy.SENTENCE_BASED:
            return self.chunk_by_sentences(text)
        elif self.strategy == ChunkingStrategy.HYBRID:
            return self.chunk_hybrid(text)
        else:  # SIMPLE
            return self.chunk_simple(text)

    def chunk_hybrid(self, text: str) -> List[Chunk]:
        """Hybrid chunking: structure-aware + sentence-based.

        This is the recommended strategy for academic papers.

        Args:
            text: Full text to chunk

        Returns:
            List of Chunk objects
        """
        chunks = []

        # Parse paper structure
        structure = self.parse_structure(text)

        # Add title as single chunk if found
        if structure.title:
            chunks.append(
                Chunk(
                    text=structure.title,
                    metadata=ChunkMetadata(
                        chunk_type=ChunkType.TITLE,
                        importance_score=self.IMPORTANCE_SCORES[ChunkType.TITLE],
                        sentence_count=1,
                    ),
                )
            )

        # Add abstract as single chunk if found
        if structure.abstract:
            # Abstract might be long, chunk it if necessary
            if self._estimate_tokens(structure.abstract) > self.max_chunk_size:
                abstract_chunks = self._chunk_text_by_sentences(
                    structure.abstract,
                    section_name="Abstract",
                    chunk_type=ChunkType.ABSTRACT,
                )
                chunks.extend(abstract_chunks)
            else:
                chunks.append(
                    Chunk(
                        text=structure.abstract,
                        metadata=ChunkMetadata(
                            chunk_type=ChunkType.ABSTRACT,
                            section_name="Abstract",
                            importance_score=self.IMPORTANCE_SCORES[ChunkType.ABSTRACT],
                            sentence_count=len(
                                self._split_sentences(structure.abstract)
                            ),
                        ),
                    )
                )

        # Process each section
        for section_name, section_text in structure.sections:
            # Determine chunk type from section name
            chunk_type = self._classify_section(section_name)

            # Chunk section text by sentences
            section_chunks = self._chunk_text_by_sentences(
                section_text, section_name=section_name, chunk_type=chunk_type
            )
            chunks.extend(section_chunks)

        # Add references if present (low importance)
        if structure.references:
            # Just add as one chunk, references are less important for search
            ref_text = structure.references[:2000]  # Limit references length
            chunks.append(
                Chunk(
                    text=ref_text,
                    metadata=ChunkMetadata(
                        chunk_type=ChunkType.REFERENCE,
                        section_name="References",
                        importance_score=self.IMPORTANCE_SCORES[ChunkType.REFERENCE],
                    ),
                )
            )

        return chunks

    def chunk_by_structure(self, text: str) -> List[Chunk]:
        """Structure-aware chunking based on paper sections.

        Args:
            text: Full text to chunk

        Returns:
            List of Chunk objects
        """
        return self.chunk_hybrid(text)  # Use hybrid as default structure-aware

    def chunk_by_sentences(self, text: str) -> List[Chunk]:
        """Sentence-based semantic chunking.

        Args:
            text: Full text to chunk

        Returns:
            List of Chunk objects
        """
        return self._chunk_text_by_sentences(text)

    def chunk_simple(self, text: str) -> List[Chunk]:
        """Simple word-based chunking (legacy compatibility).

        Args:
            text: Full text to chunk

        Returns:
            List of Chunk objects
        """
        words = text.split()
        chunks = []

        if len(words) <= self.target_chunk_size:
            return [
                Chunk(
                    text=text,
                    metadata=ChunkMetadata(
                        chunk_type=ChunkType.PARAGRAPH,
                    ),
                )
            ]

        for i in range(0, len(words), self.target_chunk_size - self.overlap_size):
            chunk_words = words[i : i + self.target_chunk_size]
            chunk_text = " ".join(chunk_words)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        chunk_type=ChunkType.PARAGRAPH,
                    ),
                )
            )

            if i + self.target_chunk_size >= len(words):
                break

        return chunks

    def parse_structure(self, text: str) -> PaperStructure:
        """Parse paper structure to identify sections.

        Args:
            text: Full paper text

        Returns:
            PaperStructure with identified components
        """
        structure = PaperStructure()

        # Extract title (first non-empty line, often)
        lines = text.split("\n")
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                structure.title = line
                break

        # Extract abstract
        for pattern in self.ABSTRACT_PATTERNS:
            match = pattern.search(text)
            if match:
                structure.abstract = match.group(2).strip()
                break

        # Find section boundaries
        section_positions = []
        for pattern in self.SECTION_PATTERNS:
            for match in pattern.finditer(text):
                section_name = match.group(0).strip()
                start_pos = match.start()
                section_positions.append((start_pos, section_name))

        # Sort by position
        section_positions.sort(key=lambda x: x[0])

        # Extract section texts
        for i, (start_pos, section_name) in enumerate(section_positions):
            # Determine end position (start of next section or end of text)
            end_pos = (
                section_positions[i + 1][0]
                if i + 1 < len(section_positions)
                else len(text)
            )

            # Extract section text (skip the header line itself)
            section_text = text[start_pos:end_pos].strip()

            # Remove the header from the text
            section_text = "\n".join(section_text.split("\n")[1:]).strip()

            # Check if this is the references section
            if section_name.lower().startswith("reference"):
                structure.references = section_text
            else:
                structure.sections.append((section_name, section_text))

        # If no sections found, treat entire text as one section
        if not structure.sections and not structure.abstract:
            # Remove title if found
            main_text = text
            if structure.title:
                main_text = text.replace(structure.title, "", 1).strip()

            structure.sections.append(("Main Content", main_text))

        return structure

    def _chunk_text_by_sentences(
        self,
        text: str,
        section_name: Optional[str] = None,
        chunk_type: ChunkType = ChunkType.PARAGRAPH,
    ) -> List[Chunk]:
        """Chunk text by sentence boundaries.

        Args:
            text: Text to chunk
            section_name: Name of the section this text belongs to
            chunk_type: Type of chunk

        Returns:
            List of Chunk objects
        """
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk_sentences: List[str] = []
        current_token_count = 0

        importance_score = self.IMPORTANCE_SCORES.get(chunk_type, 0.5)

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            # Check if adding this sentence would exceed max size
            if (
                current_chunk_sentences
                and (current_token_count + sentence_tokens) > self.max_chunk_size
            ):
                # Save current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata=ChunkMetadata(
                            chunk_type=chunk_type,
                            section_name=section_name,
                            importance_score=importance_score,
                            sentence_count=len(current_chunk_sentences),
                        ),
                    )
                )

                # Start new chunk with overlap (last sentence)
                if self.overlap_size > 0 and len(current_chunk_sentences) > 0:
                    current_chunk_sentences = [current_chunk_sentences[-1]]
                    current_token_count = self._estimate_tokens(
                        current_chunk_sentences[0]
                    )
                else:
                    current_chunk_sentences = []
                    current_token_count = 0

            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens

        # Add remaining sentences as final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        chunk_type=chunk_type,
                        section_name=section_name,
                        importance_score=importance_score,
                        sentence_count=len(current_chunk_sentences),
                    ),
                )
            )

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        if NLTK_AVAILABLE:
            try:
                result: List[str] = sent_tokenize(text)
                return result
            except Exception:
                pass  # Fall back to simple splitting

        # Simple fallback: split on period, question mark, exclamation
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        This is a rough approximation. For accurate counts, we'd need
        the actual tokenizer from the embedding model.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Rough approximation: 1 token ≈ 4 characters for English
        # This is conservative; actual ratio is often better
        return len(text) // 4

    def _classify_section(self, section_name: str) -> ChunkType:
        """Classify a section by its name.

        Args:
            section_name: Name of the section

        Returns:
            Appropriate ChunkType
        """
        section_lower = section_name.lower()

        if "abstract" in section_lower:
            return ChunkType.ABSTRACT
        elif any(word in section_lower for word in ["introduction", "intro"]):
            return ChunkType.INTRODUCTION
        elif any(
            word in section_lower for word in ["method", "methodology", "approach"]
        ):
            return ChunkType.METHODS
        elif "result" in section_lower:
            return ChunkType.RESULTS
        elif "discussion" in section_lower:
            return ChunkType.DISCUSSION
        elif any(word in section_lower for word in ["conclusion", "summary"]):
            return ChunkType.CONCLUSION
        elif "reference" in section_lower:
            return ChunkType.REFERENCE
        else:
            return ChunkType.SECTION
