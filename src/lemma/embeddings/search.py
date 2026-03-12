"""FAISS-based semantic search for papers."""
from pathlib import Path
from typing import List, Tuple, Optional
import pickle
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from ..utils.logger import get_logger, log_exception

logger = get_logger(__name__)


class SemanticSearchIndex:
    """FAISS-based vector index for semantic paper search."""

    def __init__(self, embedding_dim: int = 384, index_path: Optional[Path] = None):
        """Initialize search index.

        Args:
            embedding_dim: Dimension of embedding vectors
            index_path: Path to saved index file (optional)
        """
        if faiss is None:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            )

        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path).expanduser() if index_path else None

        # Initialize FAISS index (flat L2 for simplicity and accuracy)
        self.index = faiss.IndexFlatL2(embedding_dim)

        # Metadata mapping: index position -> (paper_id, chunk_index)
        self.id_map: List[Tuple[int, int]] = []

        # Load existing index if path provided
        if self.index_path and self.index_path.with_suffix(".faiss").exists():
            self.load()

    def add(
        self,
        embeddings: np.ndarray,
        paper_id: int,
        chunk_indices: Optional[List[int]] = None,
    ):
        """Add embeddings to the index.

        Args:
            embeddings: Numpy array of shape (n_chunks, embedding_dim)
            paper_id: ID of the paper these embeddings belong to
            chunk_indices: Optional list of chunk indices (default: [0, 1, 2, ...])
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Validate dimensions
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        # Generate chunk indices if not provided
        if chunk_indices is None:
            chunk_indices = list(range(embeddings.shape[0]))

        # Validate that we have the right number of chunk indices
        if len(chunk_indices) != embeddings.shape[0]:
            raise ValueError(
                f"Chunk indices count mismatch: {len(chunk_indices)} indices "
                f"but {embeddings.shape[0]} embeddings"
            )

        # Build ID mapping entries first (before modifying index)
        new_entries = [(paper_id, chunk_idx) for chunk_idx in chunk_indices]

        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))

        # Update ID mapping (after successful FAISS add)
        self.id_map.extend(new_entries)

        # Verify sync: index size must match id_map size
        if self.index.ntotal != len(self.id_map):
            # Critical desync detected - log error
            logger.error(
                f"CRITICAL: Index desync detected! "
                f"FAISS index has {self.index.ntotal} vectors but "
                f"ID map has {len(self.id_map)} entries. "
                f"This will cause wrong papers to be returned in search results!"
            )
            raise RuntimeError(
                f"Index desync: {self.index.ntotal} vectors != {len(self.id_map)} mappings"
            )

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[int, int, float]]:
        """Search for similar papers.

        Args:
            query_embedding: Query vector of shape (embedding_dim,)
            top_k: Number of results to return

        Returns:
            List of tuples: (paper_id, chunk_index, distance)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.astype(np.float32), top_k
        )

        # Map indices back to paper IDs
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.id_map):  # Valid index
                paper_id, chunk_idx = self.id_map[idx]
                results.append((paper_id, chunk_idx, float(dist)))

        return results

    def get_top_papers(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Get top papers (aggregated by paper_id).

        Args:
            query_embedding: Query vector
            top_k: Number of papers to return

        Returns:
            List of tuples: (paper_id, best_distance)
        """
        # Get more results than needed (since we'll aggregate by paper)
        chunk_results = self.search(query_embedding, top_k=top_k * 3)

        # Aggregate by paper_id (take best distance per paper)
        paper_scores: dict[int, float] = {}
        for paper_id, chunk_idx, dist in chunk_results:
            if paper_id not in paper_scores or dist < paper_scores[paper_id]:
                paper_scores[paper_id] = dist

        # Sort by distance and take top_k
        sorted_papers = sorted(paper_scores.items(), key=lambda x: x[1])
        return sorted_papers[:top_k]

    def save(self, index_path: Optional[Path] = None) -> bool:
        """Save index to disk.

        Args:
            index_path: Path to save index (uses self.index_path if not provided)

        Returns:
            True if save successful, False otherwise
        """
        path_to_use = index_path if index_path is not None else self.index_path
        if path_to_use is None:
            raise ValueError("No index path provided")
        save_path = Path(path_to_use).expanduser()

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.index, str(save_path.with_suffix(".faiss")))

            # Save ID mapping
            with open(save_path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(self.id_map, f)

            logger.info(f"Saved search index to {save_path}")
            return True
        except (OSError, IOError, PermissionError) as e:
            log_exception(logger, f"Failed to save index to {save_path}", e)
            return False
        except Exception as e:
            log_exception(logger, f"Unexpected error saving index to {save_path}", e)
            return False

    def load(self, index_path: Optional[Path] = None) -> bool:
        """Load index from disk.

        Args:
            index_path: Path to load from (uses self.index_path if not provided)

        Returns:
            True if load successful, False otherwise
        """
        path_to_use = index_path if index_path is not None else self.index_path
        if path_to_use is None:
            raise ValueError("No index path provided")
        load_path = Path(path_to_use).expanduser()

        try:
            # Load FAISS index
            faiss_path = load_path.with_suffix(".faiss")
            if not faiss_path.exists():
                logger.warning(f"FAISS index not found: {faiss_path}")
                return False

            self.index = faiss.read_index(str(faiss_path))

            # Load ID mapping
            pkl_path = load_path.with_suffix(".pkl")
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    self.id_map = pickle.load(f)
            else:
                self.id_map = []
                logger.warning(f"ID mapping not found at {pkl_path}, using empty list")

            # Verify sync after loading
            if self.index.ntotal != len(self.id_map):
                logger.error(
                    f"CRITICAL: Index desync detected after loading! "
                    f"FAISS index has {self.index.ntotal} vectors but "
                    f"ID map has {len(self.id_map)} entries. "
                    f"Index file may be corrupted."
                )
                # Clear both to prevent wrong results
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.id_map = []
                return False

            logger.info(f"Loaded search index from {load_path} ({self.size()} vectors)")
            return True
        except (OSError, IOError, PermissionError) as e:
            log_exception(logger, f"Failed to load index from {load_path}", e)
            return False
        except Exception as e:
            log_exception(logger, f"Unexpected error loading index from {load_path}", e)
            return False

    def size(self) -> int:
        """Get number of vectors in the index."""
        return int(self.index.ntotal)

    def clear(self):
        """Clear the index."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.id_map = []

    def verify_integrity(self) -> Tuple[bool, str]:
        """Verify index and ID map are in sync.

        Returns:
            Tuple of (is_valid, message)
        """
        index_size = int(self.index.ntotal)
        map_size = len(self.id_map)

        if index_size != map_size:
            return (
                False,
                f"Index desync: {index_size} vectors in FAISS but {map_size} in ID map",
            )

        # Check for duplicate entries that might cause issues
        seen_positions = set()
        for i, (paper_id, chunk_idx) in enumerate(self.id_map):
            if i in seen_positions:
                return (False, f"Duplicate mapping at position {i}")
            seen_positions.add(i)

        return (True, f"Index is valid: {index_size} vectors, all mappings consistent")
