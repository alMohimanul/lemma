"""arXiv Atom API client for similar-papers suggestions."""
import hashlib
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import urlencode

import httpx

from ..utils.logger import get_logger

logger = get_logger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_URI = "http://arxiv.org/schemas/atom"


def _arxiv_tag(local: str) -> str:
    return f"{{{ARXIV_URI}}}{local}"


MAX_RESULTS_CAP = 30
DEFAULT_USER_AGENT = (
    "Lemma/1.0 (+https://github.com/alMohimanul/lemma; similar-papers feature)"
)

_STOPWORDS = frozenset(
    """
    the a an and or for of in on at to from with by as is are was were be been
    being this that these those it its we our your their what which who how
    when where why can could should would may might must shall will do does did
    has have had having not no yes all any some such same other into than then
    too very just also only about over after before between both each few more
    most much such than through while paper papers work related similar find
    """.split()
)


def normalize_arxiv_id(raw: str) -> str:
    """Strip version suffix and arxiv: prefix for comparison."""
    s = raw.strip().lower()
    if s.startswith("arxiv:"):
        s = s[6:].strip()
    # remove version e.g. v2
    s = re.sub(r"v\d+$", "", s, flags=re.IGNORECASE)
    return s


def _arxiv_id_from_entry_id_url(id_url: str) -> str:
    if not id_url:
        return ""
    m = re.search(r"arxiv\.org/abs/([^?\s#]+)", id_url, re.IGNORECASE)
    if m:
        return normalize_arxiv_id(m.group(1))
    return ""


def _tokenize_for_query(text: str, max_terms: int = 14) -> List[str]:
    """Extract lowercase alphanumeric tokens for arXiv `all:` search."""
    if not text or not text.strip():
        return []
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+\-]{1,}", text.lower())
    out: List[str] = []
    for w in words:
        if w in _STOPWORDS or len(w) < 2:
            continue
        if w not in out:
            out.append(w)
        if len(out) >= max_terms:
            break
    return out


def build_arxiv_search_query(seed_text: str) -> str:
    """Build a conservative `search_query` value for the arXiv API."""
    terms = _tokenize_for_query(seed_text)
    if not terms:
        return "all:learning"
    return "all:" + "+".join(terms)


def arxiv_cache_key(search_query: str, max_results: int) -> str:
    payload = f"{search_query}\n{max_results}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_arxiv_atom(xml_text: str) -> List[Dict[str, Any]]:
    """Parse arXiv Atom API response into plain dicts."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning(f"Failed to parse arXiv Atom XML: {e}")
        return []

    entries: List[Dict[str, Any]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        id_el = entry.find("atom:id", ATOM_NS)
        title_el = entry.find("atom:title", ATOM_NS)
        summary_el = entry.find("atom:summary", ATOM_NS)
        published_el = entry.find("atom:published", ATOM_NS)

        id_url = (id_el.text or "").strip() if id_el is not None else ""
        title = " ".join((title_el.text or "").split()) if title_el is not None else ""
        summary = (
            " ".join((summary_el.text or "").split()) if summary_el is not None else ""
        )
        published = (
            (published_el.text or "").strip() if published_el is not None else ""
        )

        authors: List[str] = []
        for a in entry.findall("atom:author", ATOM_NS):
            name_el = a.find("atom:name", ATOM_NS)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        cat_el = entry.find(_arxiv_tag("primary_category"))
        primary = cat_el.get("term", "") if cat_el is not None else ""

        doi_el = entry.find(_arxiv_tag("doi"))
        doi = (doi_el.text or "").strip().lower() if doi_el is not None else ""

        aid = _arxiv_id_from_entry_id_url(id_url)
        if not aid:
            continue

        abs_url = f"https://arxiv.org/abs/{aid}"

        entries.append(
            {
                "arxiv_id": aid,
                "arxiv_id_norm": normalize_arxiv_id(aid) if aid else "",
                "title": title,
                "authors": ", ".join(authors) if authors else "Unknown",
                "summary": summary[:2000],
                "primary_category": primary,
                "published": published,
                "abs_url": abs_url,
                "doi": doi,
            }
        )

    return entries


def fetch_arxiv_search(
    search_query: str,
    max_results: int = 10,
    timeout: float = 45.0,
) -> List[Dict[str, Any]]:
    """Query export.arxiv.org and return parsed entries (network I/O)."""
    mr = max(1, min(int(max_results), MAX_RESULTS_CAP))
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": mr,
        "sortBy": "relevance",
    }
    url = f"{ARXIV_API}?{urlencode(params)}"
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(url, headers=headers)
            r.raise_for_status()
    except httpx.HTTPError as e:
        logger.error(f"arXiv API request failed: {e}")
        return []

    return parse_arxiv_atom(r.text)


def dedupe_against_library(
    entries: List[Dict[str, Any]],
    local_arxiv_norm: Set[str],
    local_dois_lower: Set[str],
) -> List[Dict[str, Any]]:
    """Drop arXiv hits that match papers already in the local library."""
    out: List[Dict[str, Any]] = []
    for e in entries:
        an = e.get("arxiv_id_norm") or ""
        if an and an in local_arxiv_norm:
            continue
        doi = (e.get("doi") or "").strip().lower()
        if doi and doi in local_dois_lower:
            continue
        out.append(e)
    return out


def rerank_by_embedding_similarity(
    entries: List[Dict[str, Any]],
    query_embedding,
    encoder,
) -> List[Dict[str, Any]]:
    """Re-rank arXiv hits by L2 distance to the query vector (same model as FAISS)."""
    import numpy as np

    if not entries:
        return entries

    q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for e in entries:
        text = f"{e.get('title', '')}\n{e.get('summary', '')}"[:2500]
        try:
            vec = encoder.encode(text)
            v = np.asarray(vec, dtype=np.float32).reshape(1, -1)
            dist = float(np.sum((q - v) ** 2))
        except Exception as ex:
            logger.debug(f"Skip re-rank for entry: {ex}")
            dist = 1e9
        scored.append((dist, e))

    scored.sort(key=lambda x: x[0])
    return [x[1] for x in scored]
