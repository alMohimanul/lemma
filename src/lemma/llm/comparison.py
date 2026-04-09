"""Multi-paper comparison engine with incremental context management."""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..utils.logger import get_logger
from . import prompts
from .question_parser import find_common_sections

logger = get_logger(__name__)


@dataclass
class ComparisonResult:
    """Result of a paper comparison operation."""

    paper_ids: List[int]
    section_name: Optional[str]
    comparison_type: str  # "section" or "whole"
    summary: str
    details: Dict[str, Any]
    section_comparisons: Optional[Dict[str, str]] = None  # For whole-paper comparisons
    provider: str = "unknown"
    model: str = "unknown"
    tokens_used: int = 0
    from_cache: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching/display."""
        return {
            "paper_ids": self.paper_ids,
            "section_name": self.section_name,
            "comparison_type": self.comparison_type,
            "summary": self.summary,
            "details": self.details,
            "section_comparisons": self.section_comparisons,
            "provider": self.provider,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "from_cache": self.from_cache,
        }


class ComparisonEngine:
    """Engine for comparing multiple papers with caching and incremental processing."""

    def __init__(self, repository, llm_router, comparison_cache):
        """Initialize comparison engine.

        Args:
            repository: Database repository
            llm_router: LLM router for generating comparisons
            comparison_cache: Comparison cache manager
        """
        self.repo = repository
        self.llm_router = llm_router
        self.cache = comparison_cache

    def compare_section(
        self,
        paper_ids: List[int],
        section_name: str,
    ) -> ComparisonResult:
        """Compare a specific section across multiple papers.

        Args:
            paper_ids: List of paper IDs to compare
            section_name: Section name to compare

        Returns:
            ComparisonResult object
        """
        logger.info(f"Comparing section '{section_name}' across papers {paper_ids}")

        # Check cache first
        cached_result = self.cache.get_cached_comparison(paper_ids, section_name)
        if cached_result:
            return ComparisonResult(
                paper_ids=paper_ids,
                section_name=section_name,
                comparison_type="section",
                summary=cached_result.get("summary", ""),
                details=cached_result.get("details", {}),
                provider=cached_result.get("cache_provider", "cache"),
                model=cached_result.get("cache_model", "cached"),
                tokens_used=0,
                from_cache=True,
            )

        # Get papers metadata
        papers = self.repo.get_papers_by_ids(paper_ids)
        papers_metadata = [self._paper_to_dict(p) for p in papers]

        # Get section embeddings for each paper
        papers_content = []
        for paper in papers:
            section_embeddings = self.repo.get_section_embeddings(
                paper.id, section_name
            )

            if not section_embeddings:
                logger.warning(
                    f"No embeddings found for section '{section_name}' in paper {paper.id}"
                )
                section_text = f"[Section '{section_name}' not found in this paper]"
            else:
                # Combine embedding text chunks
                chunks = [
                    emb.text_content for emb in section_embeddings if emb.text_content
                ]
                section_text = " ".join(chunks[:10])  # Limit to first 10 chunks

            papers_content.append(
                {
                    "paper_id": paper.id,
                    "title": paper.title or "Untitled",
                    "authors": paper.authors or "Unknown",
                    "year": paper.year or "N/A",
                    "content": section_text[:1500],  # Limit to 1500 chars per paper
                }
            )

        # Generate comparison prompt
        prompt = prompts.build_multi_paper_section_comparison_prompt(
            section_name=section_name,
            papers_content=papers_content,
        )

        # Generate comparison with LLM
        try:
            response = self.llm_router.generate(prompt=prompt, max_tokens=2000)

            if not response:
                raise Exception("LLM returned empty response")

            # Extract summary (first paragraph or first 200 chars)
            summary_lines = response.text.split("\n\n")
            summary = summary_lines[0] if summary_lines else response.text[:200]

            result = ComparisonResult(
                paper_ids=paper_ids,
                section_name=section_name,
                comparison_type="section",
                summary=summary,
                details={
                    "full_comparison": response.text,
                    "papers": papers_metadata,
                    "section": section_name,
                },
                provider=response.provider,
                model=response.model,
                tokens_used=response.tokens_used,
                from_cache=False,
            )

            # Store in cache
            self.cache.store_comparison(
                paper_ids=paper_ids,
                section=section_name,
                result=result.to_dict(),
                summary=summary,
                provider=response.provider,
                model=response.model,
                tokens_used=response.tokens_used,
            )

            return result

        except Exception as e:
            logger.error(f"Failed to generate section comparison: {e}", exc_info=True)
            raise

    def compare_papers(
        self,
        paper_ids: List[int],
    ) -> ComparisonResult:
        """Compare multiple papers section-by-section with incremental context.

        Args:
            paper_ids: List of paper IDs to compare

        Returns:
            ComparisonResult object with full comparison
        """
        logger.info(f"Comparing {len(paper_ids)} papers: {paper_ids}")

        # Check cache first
        cached_result = self.cache.get_cached_comparison(paper_ids, section=None)
        if cached_result:
            return ComparisonResult(
                paper_ids=paper_ids,
                section_name=None,
                comparison_type="whole",
                summary=cached_result.get("summary", ""),
                details=cached_result.get("details", {}),
                section_comparisons=cached_result.get("section_comparisons"),
                provider=cached_result.get("cache_provider", "cache"),
                model=cached_result.get("cache_model", "cached"),
                tokens_used=0,
                from_cache=True,
            )

        # Get papers metadata
        papers = self.repo.get_papers_by_ids(paper_ids)
        papers_metadata = [self._paper_to_dict(p) for p in papers]

        # Get all sections for each paper
        sections_by_paper = self.repo.get_all_section_names_for_papers(paper_ids)

        # Find common sections
        common_sections = find_common_sections(sections_by_paper, fuzzy=True)

        if not common_sections:
            logger.warning(f"No common sections found across papers {paper_ids}")
            # Fall back to comparing abstracts/titles only
            common_sections = ["Abstract"]

        logger.info(f"Found {len(common_sections)} common sections: {common_sections}")

        # Process sections incrementally
        section_summaries: Dict[str, str] = {}
        total_tokens = 0

        for section in common_sections:
            logger.debug(f"Processing section: {section}")

            # Get section content for all papers
            papers_content = []
            for paper in papers:
                # Get section embeddings
                section_embeddings = self.repo.get_section_embeddings(paper.id, section)

                if section_embeddings:
                    chunks = [
                        emb.text_content
                        for emb in section_embeddings
                        if emb.text_content
                    ]
                    section_text = " ".join(chunks[:8])  # Limit chunks
                else:
                    section_text = "[Section not found]"

                papers_content.append(
                    {
                        "paper_id": paper.id,
                        "title": paper.title or "Untitled",
                        "authors": paper.authors or "Unknown",
                        "year": paper.year or "N/A",
                        "content": section_text[:1200],  # Limit per paper
                    }
                )

            # Build context from previous sections
            previous_context = self._format_previous_sections(section_summaries)

            # Generate comparison for this section
            prompt = prompts.build_multi_paper_section_comparison_prompt(
                section_name=section,
                papers_content=papers_content,
                previous_sections_context=previous_context,
            )

            try:
                response = self.llm_router.generate(prompt=prompt, max_tokens=1500)

                if response:
                    section_summaries[section] = response.text
                    total_tokens += response.tokens_used
                else:
                    section_summaries[section] = f"[Failed to compare {section}]"

            except Exception as e:
                logger.error(f"Failed to compare section '{section}': {e}")
                section_summaries[section] = f"[Error comparing {section}]"

        # Generate final synthesis
        final_prompt = prompts.build_multi_paper_synthesis_prompt(
            section_summaries=section_summaries,
            papers_metadata=papers_metadata,
        )

        try:
            final_response = self.llm_router.generate(
                prompt=final_prompt, max_tokens=2500
            )

            if not final_response:
                raise Exception("Failed to generate final synthesis")

            total_tokens += final_response.tokens_used

            # Extract summary
            summary_lines = final_response.text.split("\n\n")
            summary = summary_lines[0] if summary_lines else final_response.text[:250]

            result = ComparisonResult(
                paper_ids=paper_ids,
                section_name=None,
                comparison_type="whole",
                summary=summary,
                details={
                    "final_synthesis": final_response.text,
                    "papers": papers_metadata,
                    "compared_sections": list(section_summaries.keys()),
                },
                section_comparisons=section_summaries,
                provider=final_response.provider,
                model=final_response.model,
                tokens_used=total_tokens,
                from_cache=False,
            )

            # Store in cache
            self.cache.store_comparison(
                paper_ids=paper_ids,
                section=None,
                result=result.to_dict(),
                summary=summary,
                provider=final_response.provider,
                model=final_response.model,
                tokens_used=total_tokens,
            )

            return result

        except Exception as e:
            logger.error(f"Failed to generate final synthesis: {e}", exc_info=True)
            raise

    def _paper_to_dict(self, paper) -> Dict[str, Any]:
        """Convert paper model to dict for prompt building."""
        return {
            "id": paper.id,
            "title": paper.title or "Untitled",
            "authors": paper.authors or "Unknown",
            "year": paper.year or "N/A",
            "abstract": paper.abstract or "",
        }

    def _format_previous_sections(self, section_summaries: Dict[str, str]) -> str:
        """Format previous section summaries as context.

        Args:
            section_summaries: Dict mapping section name to comparison summary

        Returns:
            Formatted context string
        """
        if not section_summaries:
            return ""

        lines = ["Previous sections compared:"]
        for section, summary in section_summaries.items():
            # Limit each summary to 150 chars
            short_summary = summary[:150] + "..." if len(summary) > 150 else summary
            lines.append(f"- {section}: {short_summary}")

        return "\n".join(lines)
