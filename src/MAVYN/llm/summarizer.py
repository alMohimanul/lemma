"""Map-reduce paper summarization for Groq's bounded context windows.

Strategy:
  MAP  — summarize each section independently (fits in compound-beta's 8K window)
  REDUCE — combine section summaries into a full paper profile (~1-2K tokens total)

This gives complete paper coverage without requiring a large-context model.
"""
import logging
from typing import Dict, Optional, Callable

from ..embeddings.retrieval import extract_sections_full, estimate_tokens
from . import prompts

logger = logging.getLogger(__name__)

# Max characters sent per section (~6K tokens, safe for compound-beta 8K window)
_SECTION_CHAR_LIMIT = 24_000


class MapReduceSummarizer:
    """Summarize a full paper using map-reduce over its stored sections.

    Works entirely within compound-beta's 8K context window by processing
    each section independently (map), then combining section summaries
    into a full paper profile (reduce).
    """

    def __init__(self, repo, llm_router):
        self.repo = repo
        self.llm_router = llm_router

    def summarize_paper(
        self,
        paper_id: int,
        notify: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, str]:
        """MAP phase: summarize each section independently.

        Returns an ordered dict of section_name → summary, preserving
        document order. Returns empty dict if paper has no embedded chunks.
        """
        sections = extract_sections_full(paper_id, self.repo)
        if not sections:
            return {}

        summaries: Dict[str, str] = {}
        for section_name, text in sections.items():
            if not text.strip():
                continue

            # Very short sections don't need LLM summarization
            if estimate_tokens(text) < 40:
                summaries[section_name] = text.strip()
                continue

            prompt = prompts.build_map_section_prompt(
                section_name, text[:_SECTION_CHAR_LIMIT]
            )
            response = self.llm_router.generate(
                prompt=prompt, max_tokens=250, tier="light"
            )
            if response and response.text.strip():
                summaries[section_name] = response.text.strip()
            else:
                summaries[section_name] = text[:600].strip()

        return summaries

    def build_context(self, section_summaries: Dict[str, str]) -> str:
        """Format section summaries into a context string for the reduce step.

        Preserves natural document order — no canonical section name assumptions.
        """
        parts = [
            f"[{name.upper()}]\n{text}"
            for name, text in section_summaries.items()
            if text.strip()
        ]
        return "\n\n".join(parts)

    def generate_profile(
        self,
        paper,
        notify: Optional[Callable[[str], None]] = None,
    ) -> Optional[dict]:
        """Full map-reduce pipeline: sections → summaries → profile dict.

        Returns a dict ready for repo.upsert_paper_profile(), or None on failure.
        """
        if notify:
            notify(f"Reading '{paper.title or 'Untitled'}'...")

        section_summaries = self.summarize_paper(int(paper.id), notify=notify)
        if not section_summaries:
            return None

        context_str = self.build_context(section_summaries)

        if notify:
            notify("Generating profile...")

        response = self.llm_router.generate(
            prompt=prompts.build_paper_profile_prompt(
                title=paper.title or "Untitled",
                authors=paper.authors or "Unknown",
                year=str(paper.year or "n.d."),
                context=context_str,
            ),
            max_tokens=800,
            tier="light",
        )
        if not response:
            return None

        parsed = prompts.parse_paper_profile(response.text)
        full_summary = parsed.get("summary", "").strip() or response.text.strip()[:600]

        return {
            "problem_statement": parsed.get("problem", ""),
            "methodology_summary": parsed.get("methodology", ""),
            "key_findings": parsed.get("findings", ""),
            "contributions": parsed.get("contributions", ""),
            "limitations": parsed.get("limitations", ""),
            "full_summary": full_summary,
            "provider": response.provider,
            "model": response.model,
            "content_version": int(paper.content_version)
            if paper.content_version
            else 1,
        }
