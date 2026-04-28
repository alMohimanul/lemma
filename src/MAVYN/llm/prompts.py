"""Reusable prompt templates for LLM interactions."""
from typing import List, Dict, Any, Optional


def _name_line(user_name: Optional[str]) -> str:
    return (
        f"You are assisting {user_name} with their research.\n\n" if user_name else ""
    )


def build_general_qa_prompt(question: str, user_name: Optional[str] = None) -> str:
    """Prompt for questions not tied to any paper in the library."""
    return f"""{_name_line(user_name)}You are MAVYN, an AI research assistant with broad knowledge.

Answer the following question clearly and concisely from your general knowledge.
If relevant, mention that the user can add papers to their MAVYN library for deeper, source-grounded answers.

Question: {question}

Answer:"""


def build_qa_prompt(
    question: str, context: str, user_name: Optional[str] = None
) -> str:
    """Build a Q&A prompt from pre-formatted context excerpts.

    Args:
        question: User's question
        context: Pre-formatted excerpt string produced by the retrieval pipeline.
                 Each excerpt is labeled [Paper N | Title | Section].

    Returns:
        Formatted prompt string
    """
    return f"""{_name_line(user_name)}You are a research assistant answering questions about academic papers.

Use ONLY the excerpts below to answer. Each excerpt is labeled with its paper number and section.

{context}

Question: {question}

Instructions:
- Answer directly from the excerpts. Do not use outside knowledge.
- Cite inline as [Paper N] each time you draw from a specific excerpt.
- If no excerpt is relevant to the question, say so clearly.
- In your References at the end, list ONLY papers you actually cited inline — not every paper shown above.

Answer:"""


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

LOCAL_LIBRARY_CANDIDATES (papers already in the user's MAVYN library; MAVYN paper id is authoritative):
{local_candidates_text}
{arxiv_section}

Instructions:
1. Summarize which local papers are most related to the question and why (use MAVYN paper ids from the block).
2. If ARXIV_CANDIDATES is non-empty, briefly describe how each suggested arXiv preprint relates; treat arXiv results as keyword-based suggestions, not guaranteed semantic neighbors.
3. Never fabricate arXiv identifiers, DOIs, or URLs — only use those explicitly listed in ARXIV_CANDIDATES.
4. Clearly separate "In your library" from "On arXiv (not in library)" in your answer.

Answer:"""


def build_section_missing_prompt(context: str, title: str, section_name: str) -> str:
    """Prompt used when the requested section was not found in stored chunks.

    Asks the LLM to acknowledge the missing section and still summarise the
    full paper, flagging any content that touches on the requested section.

    Args:
        context: Full-paper excerpt from StructuredExtractor.
        title: Paper title.
        section_name: The section the user asked for but was not found.

    Returns:
        Formatted prompt string
    """
    display = section_name.title()
    return f"""The user asked for the {display} section of "{title}", but that specific section could not be located in the stored content.

Below are structured excerpts from the rest of the paper:

{context}

Please respond in two parts:
1. **Section not found**: One sentence noting the {display} section was not found in this paper's stored content.
2. **Paper summary**: Provide a concise summary (200–300 words) of the full paper based on the excerpts above, and highlight any content that touches on {display.lower()}-related information if present.

Response:"""


def build_section_summary_prompt(context: str, title: str, section_name: str) -> str:
    """Build a prompt to summarise one specific section of a paper.

    Args:
        context: Pre-formatted excerpt from the target section (AlignedExtractor).
        title: Paper title.
        section_name: Canonical section key (e.g. 'results', 'methodology').

    Returns:
        Formatted prompt string
    """
    display = section_name.title()
    return f"""Summarise the {display} section of "{title}" using the excerpt below.

{context}

Write a focused summary (150–250 words) covering only this section.
Highlight the key points, findings, or arguments it presents.
Use only information from the excerpt. Do not invent details.

Summary:"""


def build_summary_prompt(
    context: str, title: str, user_name: Optional[str] = None
) -> str:
    """Build a summarisation prompt from pre-formatted section excerpts.

    Args:
        context: Pre-formatted section excerpt string from StructuredExtractor.
        title: Paper title (used in the instruction header).

    Returns:
        Formatted prompt string
    """
    return f"""{_name_line(user_name)}Summarise the paper "{title}" using the structured section excerpts below.
Each excerpt is labeled with its section name.

{context}

Write a coherent summary (250–350 words) covering:
1. Problem and motivation
2. Proposed approach or method
3. Key results and findings
4. Conclusions and implications

Use only information from the excerpts. Do not invent details.

Summary:"""


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


# ── Literature Review prompts ────────────────────────────────────────────────


def build_litreview_paper_summary_prompt(
    title: str, authors: str, year: str, abstract: str
) -> str:
    return f"""You are preparing material for an academic literature review.

Paper: "{title}"
Authors: {authors} ({year})
Abstract: {abstract}

Write a structured summary covering:
1. Research question / objective
2. Methodology / approach
3. Key findings and results
4. Contribution to the field

Be thorough and precise. Preserve specific technical details, numbers, and named methods — these matter in a literature review.

SUMMARY:"""


def build_litreview_theme_identification_prompt(
    paper_summaries: str, topic: str
) -> str:
    return f"""You are organizing an academic literature review on: "{topic}"

Here are summaries of the papers to be reviewed:
{paper_summaries}

Identify 3 to 5 major research themes that cut across these papers. For each theme:
- Give it a concise academic name (e.g. "Attention Mechanisms in Transformers")
- Write 1–2 sentences describing what the theme covers
- List the paper IDs (numbers) whose work falls under this theme

Respond in this exact format for each theme:
THEME: <name>
DESCRIPTION: <1-2 sentences>
PAPERS: <comma-separated IDs>
---

Identify only themes genuinely supported by multiple papers. Every paper must appear in at least one theme.

THEMES:"""


def build_litreview_theme_section_prompt(
    theme_name: str, theme_desc: str, relevant_summaries: str
) -> str:
    return f"""You are writing the "{theme_name}" section of an academic literature review.

Theme description: {theme_desc}

Relevant papers:
{relevant_summaries}

Write a cohesive academic section (400–600 words) that:
- Introduces the theme and its significance
- Synthesizes how the papers approach this theme
- Highlights agreements, tensions, and complementary findings
- Uses in-text citations as [Author Year] or [PaperID]

Write in formal academic prose. Do not use bullet points. Be substantive.

SECTION:"""


def build_litreview_introduction_prompt(
    topic: str, paper_list: str, theme_names: str
) -> str:
    return f"""You are writing the Introduction section of an academic literature review on: "{topic}"

Papers included in this review:
{paper_list}

The review is structured around these themes:
{theme_names}

Write an Introduction (350–500 words) that:
- Establishes the importance and context of the research area
- States the scope and objectives of this literature review
- Briefly describes how the review is organized (mention the themes)
- Does NOT summarize findings — save that for the body sections

Write in formal academic prose.

INTRODUCTION:"""


def build_litreview_gaps_prompt(paper_summaries: str, theme_sections: str) -> str:
    return f"""You are writing the "Research Gaps and Future Directions" section of an academic literature review.

Paper summaries:
{paper_summaries}

Synthesized theme sections:
{theme_sections}

Write a substantive section (350–500 words) that:
- Identifies what questions remain unanswered across these papers
- Highlights methodological limitations present in the body of work
- Points to underexplored areas or contradictions between papers
- Suggests concrete directions for future research

Be specific — cite which papers have which gaps. Write in formal academic prose.

RESEARCH GAPS AND FUTURE DIRECTIONS:"""


def build_litreview_conclusion_prompt(
    topic: str, theme_names: str, gaps_text: str
) -> str:
    return f"""You are writing the Conclusion of an academic literature review on: "{topic}"

The review covered these themes:
{theme_names}

The identified research gaps were:
{gaps_text}

Write a Conclusion (250–350 words) that:
- Synthesizes the overall state of knowledge in this area
- Restates the most important collective insights from the papers
- Connects back to the gaps and what they mean for the field
- Ends with a forward-looking statement

Write in formal academic prose. Do not introduce new citations.

CONCLUSION:"""
