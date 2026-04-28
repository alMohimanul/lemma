"""Literature review generation engine."""
import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable

from . import prompts

logger = logging.getLogger(__name__)


@dataclass
class LitReviewResult:
    topic: str
    papers: List[dict]
    paper_summaries: List[dict]  # [{id, title, authors, year, summary}]
    themes: List[dict]  # [{name, description, paper_ids}]
    theme_sections: Dict[str, str]  # theme_name -> section text
    introduction: str
    gaps: str
    conclusion: str


class LiteratureReviewEngine:
    """Generates a full literature review via sequential, input-controlled LLM calls."""

    def __init__(self, repo, llm_router):
        self.repo = repo
        self.llm_router = llm_router

    def generate(
        self,
        papers: list,
        topic: str,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> LitReviewResult:
        def notify(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)

        # ── Step 1: Per-paper summaries (light tier) ──────────────────────
        paper_summaries = []
        for i, paper in enumerate(papers, 1):
            notify(
                f"Summarizing paper {i}/{len(papers)}: {paper.title or 'Untitled'}..."
            )
            abstract = self._get_abstract(paper)
            prompt = prompts.build_litreview_paper_summary_prompt(
                title=paper.title or "Untitled",
                authors=paper.authors or "Unknown",
                year=str(paper.year or "n.d."),
                abstract=abstract,
            )
            response = self.llm_router.generate(
                prompt=prompt, max_tokens=800, tier="light"
            )
            paper_summaries.append(
                {
                    "id": paper.id,
                    "title": paper.title or "Untitled",
                    "authors": paper.authors or "Unknown",
                    "year": str(paper.year or "n.d."),
                    "summary": response.text.strip()
                    if response
                    else "[Summary unavailable]",
                }
            )

        # ── Step 2: Theme identification (heavy tier) ──────────────────────
        notify("Identifying research themes...")
        summaries_block = self._format_summaries_block(paper_summaries)
        theme_prompt = prompts.build_litreview_theme_identification_prompt(
            paper_summaries=summaries_block, topic=topic
        )
        theme_response = self.llm_router.generate(
            prompt=theme_prompt, max_tokens=1000, tier="heavy"
        )
        themes = self._parse_themes(
            theme_response.text if theme_response else "", paper_summaries
        )

        # ── Step 3: Per-theme sections (heavy tier) ─────────────────────
        theme_sections: Dict[str, str] = {}
        for i, theme in enumerate(themes, 1):
            notify(f"Writing section {i}/{len(themes)}: {theme['name']}...")
            relevant = [s for s in paper_summaries if s["id"] in theme["paper_ids"]]
            if not relevant:
                relevant = paper_summaries
            relevant_block = self._format_summaries_block(relevant)
            section_prompt = prompts.build_litreview_theme_section_prompt(
                theme_name=theme["name"],
                theme_desc=theme["description"],
                relevant_summaries=relevant_block,
            )
            section_response = self.llm_router.generate(
                prompt=section_prompt, max_tokens=1500, tier="heavy"
            )
            theme_sections[theme["name"]] = (
                section_response.text.strip()
                if section_response
                else "[Section unavailable]"
            )

        # ── Step 4: Introduction (heavy tier) ─────────────────────────────
        notify("Writing introduction...")
        paper_list_block = "\n".join(
            f"[{s['id']}] {s['authors']} ({s['year']}). {s['title']}."
            for s in paper_summaries
        )
        theme_names_block = "\n".join(f"- {t['name']}" for t in themes)
        intro_prompt = prompts.build_litreview_introduction_prompt(
            topic=topic,
            paper_list=paper_list_block,
            theme_names=theme_names_block,
        )
        intro_response = self.llm_router.generate(
            prompt=intro_prompt, max_tokens=1200, tier="heavy"
        )
        introduction = intro_response.text.strip() if intro_response else ""

        # ── Step 5: Research gaps (heavy tier) ────────────────────────────
        notify("Identifying research gaps...")
        # Keep gaps input bounded: summaries block + first 300 chars per theme section
        theme_snippets = "\n\n".join(
            f"[{name}]\n{text[:300]}..." for name, text in theme_sections.items()
        )
        gaps_prompt = prompts.build_litreview_gaps_prompt(
            paper_summaries=summaries_block,
            theme_sections=theme_snippets,
        )
        gaps_response = self.llm_router.generate(
            prompt=gaps_prompt, max_tokens=1200, tier="heavy"
        )
        gaps = gaps_response.text.strip() if gaps_response else ""

        # ── Step 6: Conclusion (heavy tier) ───────────────────────────────
        notify("Writing conclusion...")
        conclusion_prompt = prompts.build_litreview_conclusion_prompt(
            topic=topic,
            theme_names=theme_names_block,
            gaps_text=gaps[:600],  # pass first part of gaps to keep input small
        )
        conclusion_response = self.llm_router.generate(
            prompt=conclusion_prompt, max_tokens=800, tier="heavy"
        )
        conclusion = conclusion_response.text.strip() if conclusion_response else ""

        return LitReviewResult(
            topic=topic,
            papers=[
                {"id": p.id, "title": p.title, "authors": p.authors, "year": p.year}
                for p in papers
            ],
            paper_summaries=paper_summaries,
            themes=themes,
            theme_sections=theme_sections,
            introduction=introduction,
            gaps=gaps,
            conclusion=conclusion,
        )

    def _get_abstract(self, paper) -> str:
        if paper.abstract and paper.abstract.strip():
            return str(paper.abstract.strip())
        # Fallback: fetch abstract/intro chunks from embeddings
        try:
            chunks = self.repo.get_chunks_by_type(
                paper.id, ["abstract", "introduction"], limit=2
            )
            if chunks:
                return " ".join(c.text_content for c in chunks if c.text_content)[:1200]
        except Exception:
            pass
        return "Abstract not available."

    def _format_summaries_block(self, summaries: list) -> str:
        parts = []
        for s in summaries:
            parts.append(
                f"[Paper {s['id']}] {s['title']} — {s['authors']} ({s['year']})\n{s['summary']}"
            )
        return "\n\n".join(parts)

    def _parse_themes(self, raw: str, paper_summaries: list) -> list:
        """Parse THEME/DESCRIPTION/PAPERS blocks from LLM output."""
        themes = []
        all_ids = {s["id"] for s in paper_summaries}

        blocks = re.split(r"-{3,}", raw)
        for block in blocks:
            name_m = re.search(r"THEME:\s*(.+)", block)
            desc_m = re.search(r"DESCRIPTION:\s*(.+)", block)
            papers_m = re.search(r"PAPERS:\s*(.+)", block)
            if not name_m:
                continue
            name = name_m.group(1).strip()
            desc = desc_m.group(1).strip() if desc_m else ""
            paper_ids: list = []
            if papers_m:
                paper_ids = [
                    int(x)
                    for x in re.findall(r"\d+", papers_m.group(1))
                    if int(x) in all_ids
                ]
            if name:
                themes.append(
                    {"name": name, "description": desc, "paper_ids": paper_ids}
                )

        # Fallback: if parsing failed, put all papers in one theme
        if not themes:
            logger.warning("Theme parsing failed — using single fallback theme")
            themes = [
                {
                    "name": "Overview of the Literature",
                    "description": "A synthesis of all reviewed papers.",
                    "paper_ids": list(all_ids),
                }
            ]

        return themes
