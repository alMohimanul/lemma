"""Comparison cache manager for storing and retrieving paper comparison results."""
import hashlib
import json
from typing import List, Optional, Dict, Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ComparisonCache:
    """Manages caching of paper comparison results."""

    def __init__(self, repository):
        """Initialize comparison cache manager.

        Args:
            repository: Database repository instance
        """
        self.repo = repository

    def compute_comparison_hash(
        self, paper_ids: List[int], section: Optional[str] = None
    ) -> str:
        """Generate unique hash for a comparison.

        The hash is based on sorted paper IDs and optional section name,
        ensuring the same comparison always produces the same hash.

        Args:
            paper_ids: List of paper IDs being compared
            section: Optional section name

        Returns:
            SHA256 hash as hex string
        """
        # Sort paper IDs for consistency (so [1,5] == [5,1])
        sorted_ids = sorted(paper_ids)

        # Create comparison string: "1,5,8" or "1,5,8:methodology"
        comparison_str = ",".join(str(pid) for pid in sorted_ids)
        if section:
            comparison_str += f":{section.lower()}"

        # Compute hash
        hash_obj = hashlib.sha256(comparison_str.encode("utf-8"))
        return hash_obj.hexdigest()

    def get_cached_comparison(
        self, paper_ids: List[int], section: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached comparison if it exists.

        Args:
            paper_ids: List of paper IDs
            section: Optional section name

        Returns:
            Comparison result dict or None if not cached
        """
        comparison_hash = self.compute_comparison_hash(paper_ids, section)

        cached = self.repo.get_paper_comparison(comparison_hash)

        if cached:
            logger.info(
                f"Cache HIT for comparison {comparison_hash[:16]}... "
                f"(papers: {paper_ids}, section: {section or 'whole paper'})"
            )

            try:
                result: Dict[str, Any] = json.loads(cached.comparison_result)
                result["cached"] = True
                result["cached_at"] = (
                    cached.created_at.isoformat() if cached.created_at else None
                )
                result["cache_provider"] = cached.provider
                result["cache_model"] = cached.model
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode cached comparison: {e}")
                return None
        else:
            logger.debug(
                f"Cache MISS for comparison {comparison_hash[:16]}... "
                f"(papers: {paper_ids}, section: {section or 'whole paper'})"
            )
            return None

    def store_comparison(
        self,
        paper_ids: List[int],
        section: Optional[str],
        result: Dict[str, Any],
        summary: str,
        provider: str,
        model: str,
        tokens_used: int = 0,
    ) -> bool:
        """Store comparison result in cache.

        Args:
            paper_ids: List of paper IDs
            section: Optional section name
            result: Full comparison result dict
            summary: Short summary of the comparison
            provider: LLM provider used
            model: Model name
            tokens_used: Total tokens consumed

        Returns:
            True if stored successfully, False otherwise
        """
        comparison_hash = self.compute_comparison_hash(paper_ids, section)

        # Remove any cached metadata from result before storing
        clean_result = {
            k: v
            for k, v in result.items()
            if k not in ["cached", "cached_at", "cache_provider", "cache_model"]
        }

        cached = self.repo.add_paper_comparison(
            paper_ids=paper_ids,
            comparison_hash=comparison_hash,
            comparison_result=clean_result,
            summary=summary,
            section_name=section,
            provider=provider,
            model=model,
            tokens_used=tokens_used,
        )

        if cached:
            logger.info(
                f"Stored comparison in cache: {comparison_hash[:16]}... "
                f"(papers: {paper_ids}, section: {section or 'whole paper'})"
            )
            return True
        else:
            logger.warning(
                f"Failed to store comparison in cache (may already exist): {comparison_hash[:16]}..."
            )
            return False

    def invalidate_comparisons_for_paper(self, paper_id: int) -> int:
        """Invalidate all cached comparisons involving a specific paper.

        This should be called when a paper is re-embedded to ensure
        comparisons use the latest embeddings.

        Args:
            paper_id: Paper ID to invalidate comparisons for

        Returns:
            Number of comparisons invalidated
        """
        deleted_count: int = self.repo.delete_comparisons_involving_paper(paper_id)

        if deleted_count > 0:
            logger.info(
                f"Invalidated {deleted_count} cached comparisons for paper {paper_id}"
            )

        return deleted_count

    def get_comparison_stats(self) -> Dict[str, Any]:
        """Get statistics about cached comparisons.

        Returns:
            Dict with cache statistics
        """
        # Note: This would require adding a count method to repository
        # For now, return placeholder
        return {
            "message": "Cache stats not yet implemented",
            "suggestion": "Use database queries to inspect paper_comparisons table",
        }
