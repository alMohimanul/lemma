"""Smart context retrieval: hybrid dense+keyword search with sentence-level precision.

Three strategies, one per task type:
  Q&A       → HybridRetriever   (FAISS dense + SQL keyword, RRF, sentence extraction, MMR)
  Summarize → StructuredExtractor (section-proportional coverage, no vector search)
  Compare   → AlignedExtractor  (section-aligned parallel extraction per paper)
"""

import math
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any

from ..utils.logger import get_logger

logger = get_logger(__name__)

# ── Token budget constants ────────────────────────────────────────────────────

TOKEN_BUDGET_QA = 2500
TOKEN_BUDGET_SUMMARY = 2200
TOKEN_BUDGET_COMPARE_PER = 650  # per paper per section

# ── Stopwords (for keyword extraction) ───────────────────────────────────────

_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "i",
    "we",
    "you",
    "he",
    "she",
    "it",
    "they",
    "me",
    "us",
    "him",
    "her",
    "them",
    "my",
    "our",
    "your",
    "his",
    "its",
    "their",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "up",
    "about",
    "into",
    "through",
    "before",
    "after",
    "above",
    "below",
    "between",
    "out",
    "over",
    "under",
    "then",
    "how",
    "when",
    "where",
    "why",
    "all",
    "any",
    "both",
    "each",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "not",
    "only",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "as",
    "or",
    "and",
    "but",
    "if",
    "while",
    "paper",
    "papers",
    "study",
    "work",
    "works",
    "show",
    "shows",
    "also",
    "however",
    "therefore",
    "thus",
    "hence",
    "although",
    "though",
    "based",
    "using",
    "used",
    "proposed",
    "approach",
    "given",
    "well",
    "two",
    "one",
    "new",
    "use",
    "can",
    "per",
    "get",
    "via",
    "i.e",
    "e.g",
}

# Section priority order and token budgets for summarisation
_SUMMARY_ORDER = [
    "abstract",
    "introduction",
    "conclusion",
    "results",
    "discussion",
    "methods",
    "methodology",
    "experiment",
    "related",
    "background",
]
_SUMMARY_BUDGETS: Dict[str, int] = {
    "abstract": 350,
    "introduction": 200,
    "conclusion": 200,
    "results": 220,
    "discussion": 150,
    "methods": 150,
    "methodology": 150,
    "experiment": 150,
    "related": 80,
    "background": 80,
    "_other": 60,
}

# ── Text helpers ──────────────────────────────────────────────────────────────


def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def extract_keywords(text: str, min_len: int = 3) -> List[str]:
    """Return meaningful, non-stopword terms from text."""
    return [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) >= min_len]


def estimate_tokens(text: str) -> int:
    """1 token ≈ 4 chars — rough but fast and consistent."""
    return max(1, len(text) // 4)


def _split_sentences(text: str) -> List[str]:
    try:
        from nltk.tokenize import sent_tokenize

        result: List[str] = sent_tokenize(text)
        return result
    except Exception:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p]


def extract_sentences(
    chunk_text: str, question: str, n_best: int = 3, ctx: int = 1
) -> str:
    """Return the most question-relevant sentences from a chunk (grep-style).

    Scores each sentence by keyword overlap with the question, picks the top
    n_best, then expands by ±ctx context sentences on each side.
    """
    sentences = _split_sentences(chunk_text)
    if len(sentences) <= n_best + 2 * ctx:
        return chunk_text

    q_terms = set(extract_keywords(question))
    if not q_terms:
        return chunk_text

    scored = [(i, len(q_terms & set(_tokenize(s)))) for i, s in enumerate(sentences)]
    top = sorted(i for i, _ in sorted(scored, key=lambda x: -x[1])[:n_best])

    expanded: set = set()
    for idx in top:
        for off in range(-ctx, ctx + 1):
            expanded.add(max(0, min(len(sentences) - 1, idx + off)))

    return " ".join(sentences[i] for i in sorted(expanded))


# ── BM25 (pure Python, no deps) ───────────────────────────────────────────────


class BM25:
    """Lightweight in-memory BM25 scorer."""

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(corpus)
        self.tok = [_tokenize(d) for d in corpus]
        self.dl = [len(d) for d in self.tok]
        self.avg_dl = sum(self.dl) / max(self.n, 1)
        self.df: Dict[str, int] = {}
        for doc in self.tok:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    def score(self, query: str, idx: int) -> float:
        tf = Counter(self.tok[idx])
        dl = self.dl[idx]
        s = 0.0
        for t in _tokenize(query):
            if t not in self.df:
                continue
            idf = math.log((self.n - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1)
            tf_n = (tf[t] * (self.k1 + 1)) / (
                tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            )
            s += idf * tf_n
        return s

    def top_n(self, query: str, n: int = 10) -> List[Tuple[int, float]]:
        return sorted(
            ((i, self.score(query, i)) for i in range(self.n)),
            key=lambda x: -x[1],
        )[:n]


# ── Context packer (shared by all strategies) ─────────────────────────────────


def pack_context(
    excerpts: List[Dict[str, Any]],
    paper_meta: Dict[int, Dict],
    token_budget: int,
) -> Tuple[str, List[int]]:
    """Assemble ranked excerpts into a context string within the token budget.

    Each excerpt: {'paper_id': int, 'text': str, 'section': str, 'score': float}
    Returns (context_string, list_of_included_paper_ids).
    """
    # Use actual lemma IDs as display numbers so the LLM labels match what the
    # user types ("paper 3" → [Paper 3 | …] in context, not "Paper 2").
    nums: Dict[int, int] = {}
    for exc in excerpts:
        pid = exc["paper_id"]
        if pid not in nums:
            nums[pid] = pid

    blocks: List[str] = []
    used = 0
    included: set = set()

    for exc in sorted(excerpts, key=lambda x: -x.get("score", 0)):
        text = exc.get("text", "").strip()
        if not text:
            continue
        cost = estimate_tokens(text) + 15
        if used + cost > token_budget:
            continue
        pid = exc["paper_id"]
        meta = paper_meta.get(pid, {})
        title = meta.get("title", f"Paper {pid}")
        short = title[:55] + "…" if len(title) > 55 else title
        section = exc.get("section", "Content")
        blocks.append(f"[Paper {nums[pid]} | {short} | {section}]\n{text}")
        used += cost
        included.add(pid)

    legend_lines = []
    for pid, num in sorted(nums.items(), key=lambda x: x[1]):
        if pid in included:
            m = paper_meta.get(pid, {})
            legend_lines.append(
                f"  Paper {num}: [id {pid}] {m.get('title', '?')} "
                f"— {m.get('authors', '')}, {m.get('year', '')}"
            )

    header = "Papers in context:\n" + "\n".join(legend_lines)
    return header + "\n\n" + "\n\n".join(blocks), list(included)


# ── Hybrid Retriever (Q&A) ─────────────────────────────────────────────────────


class HybridRetriever:
    """FAISS dense + SQL keyword search fused with RRF, then sentence-level extraction."""

    def __init__(self, search_index, repo):
        self.index = search_index
        self.repo = repo

    def retrieve(
        self,
        question: str,
        query_vector,
        top_k: int = 8,
        pinned_paper_ids: Optional[List[int]] = None,
    ) -> Tuple[str, List[int]]:
        """Return (context_string, included_paper_ids)."""
        # ── Stage 1: Dense retrieval ──────────────────────────────────────────
        dense_raw = self.index.search(query_vector, top_k=top_k * 3)
        dense_rank = {(pid, cidx): r for r, (pid, cidx, _) in enumerate(dense_raw)}

        # ── Stage 2: Keyword SQL retrieval ────────────────────────────────────
        keywords = extract_keywords(question)
        kw_chunks = (
            self.repo.search_chunks_by_keywords(keywords, limit=top_k * 3)
            if keywords
            else []
        )
        kw_rank = {(c.paper_id, c.chunk_index): r for r, c in enumerate(kw_chunks)}

        # ── Stage 3: RRF fusion ───────────────────────────────────────────────
        all_keys = set(dense_rank) | set(kw_rank)
        K = 60
        rrf: Dict = {}
        nd, nk = len(dense_raw) * 2, len(kw_chunks) * 2
        for key in all_keys:
            rrf[key] = 1 / (K + dense_rank.get(key, nd)) + 1 / (
                K + kw_rank.get(key, nk)
            )
        sorted_keys = sorted(rrf.items(), key=lambda x: -x[1])

        # ── Stage 4: Fetch chunk texts ────────────────────────────────────────
        chunk_ids = [(pid, cidx) for (pid, cidx), _ in sorted_keys]
        fetched = self.repo.get_chunks_by_ids(chunk_ids)
        chunk_map = {(c.paper_id, c.chunk_index): c for c in fetched}
        for c in kw_chunks:  # keyword chunks may not overlap with dense
            chunk_map.setdefault((c.paper_id, c.chunk_index), c)

        # ── Stage 5: Sentence extraction + MMR (max 3 chunks per paper) ───────
        excerpts: List[Dict] = []
        paper_count: Dict[int, int] = {}

        for (pid, cidx), score in sorted_keys:
            if len(excerpts) >= top_k:
                break
            if paper_count.get(pid, 0) >= 3:
                continue
            chunk = chunk_map.get((pid, cidx))
            if not chunk or not chunk.text_content:
                continue
            text = extract_sentences(chunk.text_content, question)
            excerpts.append(
                {
                    "paper_id": pid,
                    "section": chunk.section_name or chunk.chunk_type or "Content",
                    "text": text,
                    "score": score * (chunk.importance_score or 0.5),
                }
            )
            paper_count[pid] = paper_count.get(pid, 0) + 1

        # ── Stage 6: Guarantee pinned papers appear ───────────────────────────
        if pinned_paper_ids:
            in_ctx = {e["paper_id"] for e in excerpts}
            for pid in pinned_paper_ids:
                if pid not in in_ctx:
                    fallback = self.repo.get_chunks_by_type(
                        pid, ["abstract", "introduction", "paragraph"], limit=1
                    )
                    if fallback:
                        c = fallback[0]
                        excerpts.append(
                            {
                                "paper_id": pid,
                                "section": c.section_name or "Content",
                                "text": extract_sentences(c.text_content, question),
                                "score": 0.05,
                            }
                        )

        if not excerpts:
            return "", []

        # ── Stage 7: Fetch paper metadata and pack ────────────────────────────
        unique_ids = list({e["paper_id"] for e in excerpts})
        papers = self.repo.get_papers_by_ids(unique_ids)
        paper_meta = {
            p.id: {
                "title": p.title or "Untitled",
                "authors": p.authors or "",
                "year": p.year or "",
            }
            for p in papers
        }
        return pack_context(excerpts, paper_meta, TOKEN_BUDGET_QA)


# ── Structured Extractor (Summarize) ──────────────────────────────────────────


class StructuredExtractor:
    """Section-proportional extraction for single-paper summarisation.

    Uses stored chunk_type / section_name metadata rather than vector search,
    so every major section is covered regardless of keyword overlap.
    """

    def __init__(self, repo):
        self.repo = repo

    def _section_key(self, chunk) -> str:
        raw = (chunk.section_name or chunk.chunk_type or "").lower()
        for key in _SUMMARY_ORDER:
            if key in raw:
                return key
        return "_other"

    def extract(self, paper_id: int, paper_obj) -> Tuple[str, List[int]]:
        """Return (context_string, [paper_id])."""
        chunks = self.repo.get_embeddings_by_paper(paper_id)
        valid = [c for c in chunks if c.is_valid and c.text_content]

        if not valid:
            return "", [paper_id]

        # Group by section key
        by_section: Dict[str, List] = {}
        for c in valid:
            key = self._section_key(c)
            by_section.setdefault(key, []).append(c)

        title = getattr(paper_obj, "title", None) or "Unknown"
        authors = getattr(paper_obj, "authors", None) or ""
        year = getattr(paper_obj, "year", None) or ""

        header = f"[Paper 1 | {title} | {authors}, {year}]"
        parts: List[str] = []
        used = estimate_tokens(header)

        for section_key in _SUMMARY_ORDER + ["_other"]:
            section_chunks = by_section.get(section_key, [])
            if not section_chunks:
                continue
            # Highest importance first, then original order
            section_chunks.sort(key=lambda c: (-c.importance_score, c.chunk_index))

            budget_chars = _SUMMARY_BUDGETS.get(section_key, 60) * 4
            text = ""
            for c in section_chunks:
                if len(text) >= budget_chars:
                    break
                text += c.text_content + " "
            text = text.strip()[:budget_chars]
            if not text:
                continue

            cost = estimate_tokens(text) + 5
            if used + cost > TOKEN_BUDGET_SUMMARY:
                break

            display = section_chunks[0].section_name or section_key.title()
            parts.append(f"[{display}]\n{text}")
            used += cost

        context_str = header + "\n\n" + "\n\n".join(parts)
        return context_str, [paper_id]


# ── Aligned Extractor (Compare) ───────────────────────────────────────────────


class AlignedExtractor:
    """Section-aligned parallel extraction for multi-paper comparison.

    Ensures the LLM sees apple-to-apple content: Paper A's Methods next to
    Paper B's Methods, rather than a blob of mixed content.
    """

    def __init__(self, repo):
        self.repo = repo

    def extract_section_text(
        self,
        paper_id: int,
        section_name: str,
        token_budget: int = TOKEN_BUDGET_COMPARE_PER,
    ) -> str:
        """Best-effort section text for one paper, importance-sorted, budget-capped.

        Falls back to BM25 over all paper chunks when the section_name label is
        absent (e.g. because PDF extraction collapsed section headings into a
        surrounding block and they were never stored as a distinct section).
        """
        chunks = self.repo.get_section_embeddings(paper_id, section_name)
        valid = sorted(
            [c for c in chunks if c.text_content],
            key=lambda c: -c.importance_score,
        )

        # BM25 fallback: section heading not found in stored labels — rank all
        # paper chunks by keyword relevance and take the top scorers.
        if not valid:
            all_chunks = [
                c
                for c in self.repo.get_embeddings_by_paper(paper_id)
                if c.is_valid and c.text_content
            ]
            if all_chunks:
                corpus = [c.text_content for c in all_chunks]
                bm25 = BM25(corpus)
                ranked = bm25.top_n(section_name, n=5)
                valid = [all_chunks[i] for i, score in ranked if score > 0]

        char_budget = token_budget * 4
        text = ""
        for c in valid:
            if len(text) >= char_budget:
                break
            text += c.text_content + " "
        return text.strip()[:char_budget]

    def extract(
        self,
        paper_ids: List[int],
        paper_metas: Dict[int, Dict],
        section_name: str,
        token_budget_per_paper: int = TOKEN_BUDGET_COMPARE_PER,
    ) -> Tuple[str, List[int]]:
        """Build a side-by-side context string for a section across all papers."""
        parts: List[str] = []
        included: List[int] = []

        for n, pid in enumerate(paper_ids, 1):
            meta = paper_metas.get(pid, {})
            title = meta.get("title", f"Paper {pid}")
            short = title[:55] + "…" if len(title) > 55 else title
            text = self.extract_section_text(pid, section_name, token_budget_per_paper)
            if not text:
                text = f"[Section '{section_name}' not found]"
            parts.append(f"[Paper {n} | {short} | {section_name}]\n{text}")
            included.append(pid)

        legend = [
            f"  Paper {n}: [id {pid}] {paper_metas.get(pid, {}).get('title', '?')} "
            f"({paper_metas.get(pid, {}).get('authors', '')}, "
            f"{paper_metas.get(pid, {}).get('year', '')})"
            for n, pid in enumerate(paper_ids, 1)
        ]
        ctx = "Papers:\n" + "\n".join(legend) + "\n\n" + "\n\n".join(parts)
        return ctx, included or paper_ids


# ── Task Router ───────────────────────────────────────────────────────────────

_SUMMARIZE_RE = re.compile(
    r"\b(summari[sz]e?|summarise|give\s+(me\s+)?(a\s+)?(summary|overview)|"
    r"what\s+is\s+(this|the)\s+paper|explain\s+(this|the|paper)|"
    r"describe\s+(this|the)\s+paper|tldr|tl;?dr)\b",
    re.IGNORECASE,
)

# Maps canonical section key → regex that detects it in a question
_SECTION_PATTERNS: Dict[str, str] = {
    "abstract": r"\babstract\b",
    "introduction": r"\bintro(?:duction)?\b",
    "methodology": r"\bmethodolog(?:y|ies)\b|\bmethods?\b",
    "results": r"\bresults?\b|\bfindings?\b",
    "discussion": r"\bdiscussion\b",
    "conclusion": r"\bconclusions?\b",
    "related": r"\brelated\s+work\b",
    "background": r"\bbackground\b",
    "experiment": r"\bexperiments?\b",
}


def _detect_section(question: str) -> Optional[str]:
    """Return the canonical section name mentioned in the question, or None."""
    q = question.lower()
    for section, pattern in _SECTION_PATTERNS.items():
        if re.search(pattern, q):
            return section
    return None


class TaskRouter:
    """Detects task type and any targeted section from the question.

    Returns a (task_type, section_name) tuple:
      ('summarize',         None)      — whole-paper summary
      ('summarize_section', 'results') — section-specific summary
      ('qa',                None)      — general Q&A
    """

    def detect(
        self, question: str, explicit_paper_ids: List[int]
    ) -> Tuple[str, Optional[str]]:
        if len(explicit_paper_ids) == 1 and _SUMMARIZE_RE.search(question):
            section = _detect_section(question)
            if section:
                return "summarize_section", section
            return "summarize", None
        return "qa", None
