"""Text embedding generation using sentence-transformers."""
from typing import List
import numpy as np
import os
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set environment variables to reduce verbosity
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Set HF token if available in environment
if "HF_TOKEN" in os.environ:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    os.environ["HF_HUB_TOKEN"] = os.environ["HF_TOKEN"]

# Suppress all logging except critical errors
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]


class EmbeddingEncoder:
    """Generate embeddings using sentence-transformers models."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """Initialize encoder with a specific model.

        Args:
            model_name: Name of sentence-transformers model to use
                       Default: BAAI/bge-m3 (1024 dimensions, multilingual)
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        # Load model with reduced verbosity
        import logging

        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding vector.

        Args:
            text: Text to encode

        Returns:
            Numpy array of shape (embedding_dim,)
        """
        return np.array(self.model.encode(text, convert_to_numpy=True))

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode multiple texts into embedding vectors.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding

        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        return np.array(
            self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10,
            )
        )

    def chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> List[str]:
        """Split text into overlapping chunks for embedding.

        Args:
            text: Full text to chunk
            chunk_size: Approximate number of words per chunk
            overlap: Number of words to overlap between chunks

        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []

        if len(words) <= chunk_size:
            return [text]

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)

            # Stop if we've reached the end
            if i + chunk_size >= len(words):
                break

        return chunks

    def encode_with_chunking(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> List[np.ndarray]:
        """Encode long text by chunking it first.

        Args:
            text: Full text to encode
            chunk_size: Words per chunk
            overlap: Overlapping words between chunks

        Returns:
            List of embedding vectors, one per chunk
        """
        chunks = self.chunk_text(text, chunk_size, overlap)
        return [self.encode(chunk) for chunk in chunks]
