"""Reusable prompt templates for LLM interactions."""
from typing import List, Dict, Any


def build_qa_prompt(question: str, context_papers: List[Dict[str, Any]]) -> str:
    """Build a prompt for question answering across papers.

    Args:
        question: User's question
        context_papers: List of paper dictionaries with metadata and text

    Returns:
        Formatted prompt string
    """
    context_parts = []

    for i, paper in enumerate(context_papers, 1):
        title = paper.get("title", "Untitled")
        authors = paper.get("authors", "Unknown")
        year = paper.get("year", "N/A")
        lemma_id = paper.get("id", "")
        text = paper.get("text", "")

        context_parts.append(
            f"[Paper {i} | Lemma id: {lemma_id}]\n"
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Year: {year}\n"
            f"Content: {text[:6000]}\n"  # Limit to avoid token overflow
        )

    context = "\n\n".join(context_parts)

    prompt = f"""You are a research assistant helping to answer questions about academic papers.

Based on the following papers, answer the question below. When the user names a lemma id (e.g. "paper 9"), use the block with that Lemma id. Cite specific papers when relevant.

Papers:
{context}

Question: {question}

Provide a clear, concise answer based on the information in the papers above. If the papers don't contain enough information to answer the question, say so."""

    return prompt


def build_similar_papers_prompt(
    question: str,
    local_candidates_text: str,
    arxiv_candidates_text: str,
) -> str:
    """Prompt for similar/related paper discovery (local + optional arXiv)."""
    arxiv_section = ""
    if arxiv_candidates_text.strip():
        arxiv_section = f"""
ARXIV_CANDIDATES (from arXiv.org API only — these rows are authoritative):
{arxiv_candidates_text}

When mentioning arXiv items, use only the arxiv_id and abs_url from this block. Do not invent arXiv IDs or URLs.
"""
    else:
        arxiv_section = """
No arXiv candidates were fetched for this request (local library only or arXiv disabled).
"""

    return f"""You are helping a researcher find related academic work.

The user asked:
{question}

LOCAL_LIBRARY_CANDIDATES (papers already in the user's Lemma library; Lemma paper id is authoritative):
{local_candidates_text}
{arxiv_section}

Instructions:
1. Summarize which local papers are most related to the question and why (use Lemma paper ids from the block).
2. If ARXIV_CANDIDATES is non-empty, briefly describe how each suggested arXiv preprint relates; treat arXiv results as keyword-based suggestions, not guaranteed semantic neighbors.
3. Never fabricate arXiv identifiers, DOIs, or URLs — only use those explicitly listed in ARXIV_CANDIDATES.
4. Clearly separate "In your library" from "On arXiv (not in library)" in your answer.

Answer:"""


def build_summary_prompt(paper: Dict[str, Any], max_length: int = 200) -> str:
    """Build a prompt for summarizing a paper.

    Args:
        paper: Paper dictionary with metadata and text
        max_length: Maximum words in summary

    Returns:
        Formatted prompt string
    """
    title = paper.get("title", "Untitled")
    text = paper.get("text", "")

    prompt = f"""Summarize the following academic paper in approximately {max_length} words.

Title: {title}

Content:
{text[:5000]}

Provide a concise summary covering:
1. Main contribution/finding
2. Methodology (if applicable)
3. Key results or implications

Summary:"""

    return prompt


def build_citation_extraction_prompt(text: str) -> str:
    """Build a prompt for extracting citations from paper text.

    Args:
        text: Paper text content

    Returns:
        Formatted prompt string
    """
    prompt = f"""Extract all academic citations from the following text.

For each citation, provide:
- Title (if available)
- Authors (if available)
- Year (if available)

Text:
{text[:4000]}

Return citations in a structured format like:
1. Title: "..." | Authors: "..." | Year: XXXX
2. Title: "..." | Authors: "..." | Year: XXXX

Citations:"""

    return prompt


def build_comparison_prompt(paper1: Dict[str, Any], paper2: Dict[str, Any]) -> str:
    """Build a prompt for comparing two papers.

    Args:
        paper1: First paper dictionary
        paper2: Second paper dictionary

    Returns:
        Formatted prompt string
    """

    def format_paper(paper: Dict[str, Any]) -> str:
        return f"""Title: {paper.get('title', 'Untitled')}
Authors: {paper.get('authors', 'Unknown')}
Year: {paper.get('year', 'N/A')}
Abstract: {paper.get('abstract', 'N/A')}
Content: {paper.get('text', '')[:1500]}"""

    prompt = f"""Compare the following two academic papers. Focus on:
1. Research objectives and questions
2. Methodologies
3. Key findings
4. Similarities and differences
5. Complementary insights

Paper 1:
{format_paper(paper1)}

Paper 2:
{format_paper(paper2)}

Provide a structured comparison highlighting similarities, differences, and how they relate to each other.

Comparison:"""

    return prompt


def build_collection_summary_prompt(papers: List[Dict[str, Any]]) -> str:
    """Build a prompt for summarizing a collection of papers.

    Args:
        papers: List of paper dictionaries

    Returns:
        Formatted prompt string
    """
    paper_summaries = []
    for i, paper in enumerate(papers, 1):
        paper_summaries.append(
            f"{i}. {paper.get('title', 'Untitled')} "
            f"({paper.get('year', 'N/A')}) - "
            f"{paper.get('abstract', 'No abstract')[:200]}"
        )

    papers_text = "\n".join(paper_summaries)

    prompt = f"""Summarize the key themes and findings across this collection of {len(papers)} academic papers.

Papers:
{papers_text}

Provide:
1. Common themes and research areas
2. Notable findings or contributions
3. Chronological trends (if apparent)
4. Gaps or future directions

Summary:"""

    return prompt


def build_note_formatting_prompt(
    question: str, answer: str, context_papers: List[Dict[str, Any]]
) -> str:
    """Build a prompt for formatting Q&A into a literature review note.

    Args:
        question: Original question
        answer: LLM-generated answer
        context_papers: List of papers used to answer

    Returns:
        Formatted prompt string
    """
    # Format papers list
    papers_list = []
    for i, paper in enumerate(context_papers, 1):
        papers_list.append(
            f"[{paper.get('id')}] {paper.get('title', 'Untitled')} "
            f"({paper.get('authors', 'Unknown')}, {paper.get('year', 'N/A')})"
        )
    papers_text = "\n".join(papers_list)

    prompt = f"""You are helping a researcher prepare literature review notes. Transform the following Q&A into a well-structured note suitable for academic writing.

ORIGINAL QUESTION:
{question}

ANSWER:
{answer}

PAPERS REFERENCED:
{papers_text}

Format the note with these sections:

## Key Findings
- Main insights from the answer (2-4 bullet points)

## Relevant Papers
- List papers with their key contributions to this question

## Methodology Insights (if applicable)
- Methods or approaches mentioned

## Direct Evidence
- Important quotes or specific findings worth citing
- Include paper IDs like [1] for citations

Keep the note concise (200-300 words), focused on actionable insights for writing a literature review. Use academic tone but remain clear and direct.

FORMATTED NOTE:"""

    return prompt


def build_multi_paper_section_comparison_prompt(
    section_name: str,
    papers_content: List[Dict[str, Any]],
    previous_sections_context: str = "",
) -> str:
    """Build a prompt for comparing a specific section across multiple papers.

    Args:
        section_name: Name of the section being compared
        papers_content: List of dicts with paper metadata and section content
        previous_sections_context: Context from previously compared sections

    Returns:
        Formatted prompt string
    """
    # Format each paper's content
    papers_text = []
    for i, paper in enumerate(papers_content, 1):
        papers_text.append(
            f"""Paper {i} - [{paper.get('paper_id')}] {paper.get('title', 'Untitled')}
Authors: {paper.get('authors', 'Unknown')}
Year: {paper.get('year', 'N/A')}

{section_name} Content:
{paper.get('content', '[No content available]')}
"""
        )

    papers_section = "\n" + "=" * 80 + "\n".join(papers_text)

    context_section = ""
    if previous_sections_context:
        context_section = f"""
CONTEXT FROM PREVIOUS SECTIONS:
{previous_sections_context}

Use this context to build a coherent comparison across sections.
"""

    prompt = f"""You are a research assistant comparing academic papers. Compare the "{section_name}" section across {len(papers_content)} papers.
{context_section}
PAPERS TO COMPARE:
{papers_section}

Provide a structured comparison with:

1. **Overview**: Brief summary of how each paper approaches this section

2. **Key Similarities**: Common themes, methods, or findings across papers

3. **Key Differences**: Distinct approaches, methodologies, or perspectives

4. **Notable Insights**: Important points from each paper that stand out

Keep the comparison concise (200-300 words), focused on actionable insights. Use paper IDs like [1], [2] to reference specific papers.

COMPARISON:"""

    return prompt


def build_multi_paper_synthesis_prompt(
    section_summaries: Dict[str, str], papers_metadata: List[Dict[str, Any]]
) -> str:
    """Build a prompt to synthesize section-by-section comparisons into final summary.

    Args:
        section_summaries: Dict mapping section names to their comparison summaries
        papers_metadata: List of paper metadata dicts

    Returns:
        Formatted prompt string
    """
    # Format papers being compared
    papers_list = []
    for i, paper in enumerate(papers_metadata, 1):
        papers_list.append(
            f"[{paper.get('id')}] {paper.get('title', 'Untitled')} "
            f"({paper.get('authors', 'Unknown')}, {paper.get('year', 'N/A')})"
        )
    papers_text = "\n".join(papers_list)

    # Format section summaries
    sections_text = []
    for section, summary in section_summaries.items():
        sections_text.append(
            f"""### {section}
{summary}
"""
        )

    sections_section = "\n".join(sections_text)

    prompt = f"""You are synthesizing a comprehensive comparison of {len(papers_metadata)} academic papers.

PAPERS COMPARED:
{papers_text}

SECTION-BY-SECTION COMPARISONS:
{sections_section}

Based on these section comparisons, provide a comprehensive synthesis covering:

## Overall Assessment
Holistic view of how these papers relate to each other

## Research Approaches
How methodologies and approaches differ or complement each other

## Key Contributions
Most significant findings or contributions from each paper

## Complementary Insights
How these papers together advance understanding of the topic

## Research Gaps & Future Directions
Gaps identified when viewing these papers together

## Practical Implications
What researchers should take away from comparing these works

Keep the synthesis comprehensive but concise (400-500 words). Use paper IDs [1], [2], etc. for citations.

COMPREHENSIVE COMPARISON:"""

    return prompt
