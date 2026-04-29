"""Question parser for detecting comparison intent and extracting paper references."""
import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ComparisonRequest:
    """Structured comparison request extracted from a question."""

    paper_ids: List[int]
    section_name: Optional[str] = None
    comparison_type: str = "whole"  # "whole" or "section"


# Comparison keyword patterns
COMPARISON_KEYWORDS = [
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bdifference[s]?\b",
    r"\bcontrast\b",
    r"\bsimilarit(?:y|ies)\b",
    r"\bhow\s+(?:do|does)\s+(?:.*?)\s+differ\b",
    r"\bwhat(?:'s| is)\s+the\s+difference\b",
]

# Paper ID extraction patterns
PAPER_ID_PATTERNS = [
    r"\[(\d+)\]",  # [1], [5]
    r"\bpaper\s+(?:id\s*)?(\d+)",  # paper 5, paper ID 12, paper ID12
    r"\b#(\d+)\b",  # #5
    r"\bID\s*(\d+)",  # ID 42, ID42
    r"\bid\s*(\d+)",  # id 42, id42 (lowercase)
]

# Section name patterns
SECTION_PATTERNS = {
    "abstract": [r"\babstract[s]?\b"],
    "introduction": [r"\bintroduction\b", r"\bintro\b"],
    "methodology": [
        r"\bmethodolog(?:y|ies)\b",
        r"\bmethod[s]?\b",
        r"\bapproach(?:es)?\b",
    ],
    "results": [r"\bresult[s]?\b", r"\bfinding[s]?\b"],
    "discussion": [r"\bdiscussion[s]?\b"],
    "conclusion": [r"\bconclusion[s]?\b", r"\bsummar(?:y|ies)\b"],
    "references": [
        r"\breference[s]?\b",
        r"\bcitation[s]?\b",
        r"\bbibliograph(?:y|ies)\b",
    ],
}

# Similar-papers discovery (local semantic neighbors + optional arXiv)
SIMILAR_PAPERS_PATTERNS = [
    r"\bsimilar\s+papers?\b",
    r"\brelated\s+work\b",
    r"\brelated\s+papers?\b",
    r"\bpapers?\s+similar\b",
    r"\bpapers?\s+like\b",
    r"\bmore\s+papers?\s+(?:on|about|like)\b",
    r"\bfind\s+similar\b",
    r"\bsuggestions?\s+for\s+related\b",
    r"\bwhat\s+else\s+(?:to\s+)?read\b",
    r"\barxiv\s+(?:for\s+)?(?:similar|related)\b",
    r"\bother\s+papers?\s+(?:on|about)\b",
]


def wants_similar_papers(question: str) -> bool:
    """True when the user wants suggestions for related / similar papers.

    Used by ``MAVYN ask`` (without a separate command). Example phrases:
    "similar papers on X", "related work for transformers", "papers like paper 5",
    "what else should I read on this topic".
    """
    q = question.lower()
    return any(re.search(p, q) for p in SIMILAR_PAPERS_PATTERNS)


_LIST_SECTIONS_PATTERNS = [
    r"\blist\s+(?:the\s+)?sections?\b",
    r"\bshow\s+(?:me\s+)?(?:the\s+)?sections?\b",
    r"\bwhat\s+sections?\s+(?:does|do|are\s+in|are\s+there)\b",
    r"\bwhich\s+sections?\s+(?:does|do|are\s+in|are\s+there)\b",
    r"\bsections?\s+(?:does|do)\s+(?:this|the)\s+paper\s+have\b",
    r"\bwhat(?:'s| is| are)\s+(?:the\s+)?sections?\s+(?:in|of)\b",
    r"\bsections?\s+(?:in|of)\s+(?:this|the)\s+paper\b",
    r"\btable\s+of\s+contents?\b",
    r"\bpaper\s+structure\b",
    r"\bhow\s+is\s+(?:this\s+|the\s+)?paper\s+(?:organized|structured|divided)\b",
]


def wants_list_sections(question: str) -> bool:
    """True when the user wants to enumerate which sections a paper has.

    Only matches *listing* intent, not *content* intent — so
    "summarize the results section" returns False.
    """
    q = question.lower()
    return any(re.search(p, q) for p in _LIST_SECTIONS_PATTERNS)


def extract_seed_paper_ids_for_similar(question: str) -> List[int]:
    """Paper IDs named as seeds for similarity search (e.g. 'like paper 7').

    Returns all extracted IDs; the caller excludes them from neighbor results.
    """
    return extract_paper_ids(question)


def detect_comparison_intent(question: str) -> Tuple[bool, Dict[str, Any]]:
    """Detect if the question is requesting a paper comparison.

    Args:
        question: User's question text

    Returns:
        Tuple of (is_comparison, comparison_info)
        comparison_info contains: {"type": "comparison", "raw_question": question}
    """
    question_lower = question.lower()

    # Check for comparison keywords
    for pattern in COMPARISON_KEYWORDS:
        if re.search(pattern, question_lower, re.IGNORECASE):
            logger.debug(f"Detected comparison intent via pattern: {pattern}")
            return True, {"type": "comparison", "raw_question": question}

    return False, {}


def extract_paper_ids(question: str) -> List[int]:
    """Extract paper IDs from a question.

    Args:
        question: User's question text

    Returns:
        List of paper IDs found in the question
    """
    paper_ids = []

    for pattern in PAPER_ID_PATTERNS:
        matches = re.findall(pattern, question, re.IGNORECASE)
        for match in matches:
            try:
                paper_id = int(match)
                if paper_id not in paper_ids:
                    paper_ids.append(paper_id)
            except ValueError:
                continue

    logger.debug(f"Extracted paper IDs from question: {paper_ids}")
    return paper_ids


def extract_section_name(question: str) -> Optional[str]:
    """Extract section name from a question if specified.

    Args:
        question: User's question text

    Returns:
        Normalized section name or None if not specified
    """
    question_lower = question.lower()

    for section_name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, question_lower):
                logger.debug(
                    f"Detected section in question: {section_name} (pattern: {pattern})"
                )
                return section_name

    return None


def parse_comparison_request(
    question: str, available_paper_ids: Optional[List[int]] = None
) -> Optional[ComparisonRequest]:
    """Parse a question into a structured comparison request.

    Args:
        question: User's question text
        available_paper_ids: Optional list of available paper IDs for validation

    Returns:
        ComparisonRequest object or None if not a valid comparison request
    """
    # First, check if this is a comparison question
    is_comparison, _ = detect_comparison_intent(question)

    if not is_comparison:
        return None

    # Extract paper IDs
    paper_ids = extract_paper_ids(question)

    if len(paper_ids) < 2:
        logger.warning(
            f"Comparison request detected but found {len(paper_ids)} paper IDs (need at least 2)"
        )
        return None

    # Optionally validate against available papers
    if available_paper_ids is not None:
        valid_ids = [pid for pid in paper_ids if pid in available_paper_ids]
        if len(valid_ids) < 2:
            logger.warning(
                f"Only {len(valid_ids)} of {len(paper_ids)} extracted IDs are valid"
            )
            return None
        paper_ids = valid_ids

    # Extract section name if specified
    section_name = extract_section_name(question)

    # Determine comparison type
    comparison_type = "section" if section_name else "whole"

    return ComparisonRequest(
        paper_ids=paper_ids, section_name=section_name, comparison_type=comparison_type
    )


def fuzzy_match_section(
    section_name: str, available_sections: List[str]
) -> Optional[str]:
    """Fuzzy match a section name against available sections.

    This helps match user queries like "methodology" to actual section names like "Methods"
    or "Methodology and Approach".

    Args:
        section_name: Normalized section name from user query
        available_sections: List of actual section names from papers

    Returns:
        Best matching section name or None
    """
    section_lower = section_name.lower()

    # Direct substring matches (highest priority)
    for avail_section in available_sections:
        if section_lower in avail_section.lower():
            return avail_section
        if avail_section.lower() in section_lower:
            return avail_section

    # Keyword-based matching
    section_keywords = {
        "abstract": ["abstract", "summary"],
        "introduction": ["introduction", "intro", "background"],
        "methodology": ["method", "methodology", "approach", "technique"],
        "results": ["result", "finding", "outcome", "observation"],
        "discussion": ["discussion", "analysis", "interpretation"],
        "conclusion": ["conclusion", "summary", "final"],
    }

    keywords = section_keywords.get(section_lower, [section_lower])

    for avail_section in available_sections:
        avail_lower = avail_section.lower()
        for keyword in keywords:
            if keyword in avail_lower:
                return avail_section

    return None


def find_common_sections(
    sections_by_paper: Dict[int, List[str]], fuzzy: bool = True
) -> List[str]:
    """Find sections that exist in all papers.

    Args:
        sections_by_paper: Dict mapping paper_id to list of section names
        fuzzy: If True, use fuzzy matching to align similar section names

    Returns:
        List of common section names (using names from first paper as reference)
    """
    if not sections_by_paper:
        return []

    # Get the first paper's sections as reference
    paper_ids = list(sections_by_paper.keys())
    if not paper_ids:
        return []

    reference_sections = sections_by_paper[paper_ids[0]]
    common_sections = []

    # For each section in the reference paper, check if it exists in all other papers
    for ref_section in reference_sections:
        found_in_all = True

        for other_paper_id in paper_ids[1:]:
            other_sections = sections_by_paper[other_paper_id]

            if fuzzy:
                # Try fuzzy matching
                match = fuzzy_match_section(ref_section, other_sections)
                if not match:
                    found_in_all = False
                    break
            else:
                # Exact match only
                if ref_section not in other_sections:
                    found_in_all = False
                    break

        if found_in_all:
            common_sections.append(ref_section)

    logger.debug(
        f"Found {len(common_sections)} common sections across {len(paper_ids)} papers: {common_sections}"
    )
    return common_sections
