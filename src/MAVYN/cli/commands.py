"""CLI commands for lemma paper manager."""
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import click
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import box
from dotenv import load_dotenv

from ..core.scanner import PDFScanner
from ..core.extractor import MetadataExtractor
from ..db.repository import Repository
from . import output
from .setup_wizard import run_setup_wizard, is_first_run

# Load environment variables from .env file and ~/.MAVYN/.env
load_dotenv()  # Load from current directory
load_dotenv(Path.home() / ".MAVYN" / ".env")  # Load from config directory


def _extract_cited_paper_ids(answer_text: str) -> Set[int]:
    """Return the set of lemma paper IDs cited as [Paper N] in an LLM answer."""
    return {int(m) for m in re.findall(r"\[Paper\s+(\d+)", answer_text)}


def _display_sections_list(
    sections_by_paper: Dict[int, List[str]],
    repo,
) -> None:
    """Print a Rich Panel per paper listing its stored section names."""
    from rich.panel import Panel
    from rich import box as rbox

    for paper_id, section_names in sections_by_paper.items():
        paper_obj = repo.get_paper_by_id(paper_id)
        title = (paper_obj.title or "Untitled") if paper_obj else f"Paper {paper_id}"

        if not section_names:
            output.console.print(
                Panel(
                    "[dim]No sections indexed. Run [bold]lemma embed[/bold] first.[/dim]",
                    title=f"[bold cyan][{paper_id}] {title}[/bold cyan]",
                    border_style="yellow",
                    box=rbox.ROUNDED,
                )
            )
            continue

        body = "\n".join(
            f"  [cyan]{i:>2}.[/cyan] {name}" for i, name in enumerate(section_names, 1)
        )
        output.console.print(
            Panel(
                body,
                title=f"[bold cyan][{paper_id}] {title}[/bold cyan]",
                border_style="cyan",
                box=rbox.ROUNDED,
            )
        )

    example_section = next(
        (s for snames in sections_by_paper.values() for s in snames), None
    )
    first_pid = next(iter(sections_by_paper), None)
    if example_section and first_pid is not None:
        output.console.print(
            f'\n[dim]Tip: Ask [bold]"summarize the {example_section} section of '
            f'paper {first_pid}"[/bold] to dive into a section.[/dim]'
        )
    else:
        output.console.print(
            "[dim]Tip: Run [bold]lemma embed[/bold] to index paper sections.[/dim]"
        )


@click.group()
@click.version_option(version="2.0.0")
def cli():
    """MAVYN - Local-first paper manager.

    Manage your research papers with local semantic search and cloud LLM reasoning.
    """
    pass


@cli.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--recursive/--no-recursive", default=True, help="Scan subdirectories")
@click.option("--db", default="~/.MAVYN/MAVYN.db", help="Database path")
def scan(directory: Path, recursive: bool, db: str):
    """Scan a directory for PDF files and add them to the database.

    DIRECTORY: Path to folder containing PDFs
    """
    with Repository(db) as repo:
        scanner = PDFScanner()
        extractor = MetadataExtractor()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            # Scan for PDFs
            task = progress.add_task("Scanning for PDFs...", total=None)
            scanned_files = scanner.scan_directory(directory, recursive=recursive)
            progress.update(task, completed=True)

            # Process each PDF
            total = len(scanned_files)
            new_count = 0
            duplicate_count = 0
            error_count = 0

            task = progress.add_task(f"Processing {total} PDFs...", total=total)

            for scanned in scanned_files:
                try:
                    # Check if already indexed
                    existing = repo.get_paper_by_hash(scanned.file_hash)
                    if existing:
                        duplicate_count += 1
                        progress.advance(task)
                        continue

                    # Extract metadata
                    metadata = extractor.extract(scanned.path)

                    # Add to database
                    repo.add_paper(
                        file_path=str(scanned.path),
                        file_hash=scanned.file_hash,
                        file_size=scanned.file_size,
                        metadata=metadata.to_dict(),
                    )

                    # Log success
                    repo.log_operation(
                        operation="scan",
                        status="success",
                        details={"file": str(scanned.path)},
                    )

                    new_count += 1

                except Exception as e:
                    error_count += 1
                    repo.log_operation(
                        operation="scan",
                        status="failed",
                        details={"file": str(scanned.path)},
                        error_message=str(e),
                    )

                progress.advance(task)

        # Print results
        output.print_scan_results(
            total=total,
            new=new_count,
            duplicates=duplicate_count,
            errors=error_count,
        )


@cli.command(name="list")
@click.option("--limit", type=int, default=50, help="Maximum number of papers to show")
@click.option("--offset", type=int, default=0, help="Number of papers to skip")
@click.option(
    "--sort-by",
    type=click.Choice(["indexed_at", "year", "title"]),
    default="indexed_at",
    help="Sort by field",
)
@click.option("--db", default="~/.MAVYN/MAVYN.db", help="Database path")
def list_papers(limit: int, offset: int, sort_by: str, db: str):
    """List indexed papers."""
    with Repository(db) as repo:
        papers = repo.list_papers(limit=limit, offset=offset, sort_by=sort_by)

        # Convert to dicts for display
        paper_dicts = []
        for paper in papers:
            paper_dicts.append(
                {
                    "id": paper.id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.year,
                    "embedding_status": paper.embedding_status,
                }
            )

        output.print_paper_table(paper_dicts)


@cli.command()
@click.argument("query")
@click.option("--db", default="~/.MAVYN/MAVYN.db", help="Database path")
def search(query: str, db: str):
    """Search papers by keyword (title, authors, abstract).

    QUERY: Search terms
    """
    with Repository(db) as repo:
        papers = repo.search_papers(query)

        # Convert to dicts for display
        paper_dicts = []
        for paper in papers:
            paper_dicts.append(
                {
                    "id": paper.id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.year,
                    "embedding_status": paper.embedding_status,
                }
            )

        output.print_search_results(paper_dicts, query)


@cli.command(name="show")
@click.argument("item_id", type=int, required=False)
@click.option("--paper", "-p", type=int, help="Show paper by ID")
@click.option("--note", "-n", type=int, help="Show note by ID")
@click.option("--db", default="~/.MAVYN/MAVYN.db", help="Database path")
def show(item_id: int, paper: int, note: int, db: str):
    """Show detailed information about a paper or note.

    ITEM_ID: ID to display (defaults to paper)

    Examples:
      lemma show 5        # Show paper 5
      lemma show -n 3     # Show note 3
      lemma show --note 3 # Show note 3
    """
    from ..core.notes import NoteManager

    with Repository(db) as repo:
        # Determine what to show
        if note is not None:
            # Show note
            note_manager = NoteManager()

            # Validate note ID
            try:
                note_manager.validate_note_id(note)
            except ValueError as e:
                output.print_error(str(e))
                return

            # Fetch note
            try:
                note_obj = repo.get_note_by_id(note)
            except Exception as e:
                output.print_error(f"Failed to fetch note: {e}")
                return

            if not note_obj:
                output.print_error(f"Note with ID {note} not found")
                output.print_info("Use 'lemma notes' to see available notes")
                return

            # Convert note to dict for formatting
            note_dict = {
                "id": note_obj.id,
                "question": note_obj.question,
                "answer": note_obj.answer,
                "formatted_note": note_obj.formatted_note,
                "sources": note_obj.sources,
                "provider": note_obj.provider,
                "model": note_obj.model,
                "tokens_used": note_obj.tokens_used,
                "created_at": note_obj.created_at,
            }

            # Format and display
            formatted_output = note_manager.format_note_display(note_dict)

            output.console.print(
                Panel(
                    formatted_output,
                    title=f"[bold cyan]Note #{note}[/bold cyan]",
                    border_style="cyan",
                    box=box.ROUNDED,
                )
            )

        elif paper is not None or item_id is not None:
            # Show paper
            paper_id = paper if paper is not None else item_id

            paper_obj = repo.get_paper_by_id(paper_id)

            if not paper_obj:
                output.print_error(f"Paper with ID {paper_id} not found")
                return

            # Convert to dict for display
            paper_dict = {
                "id": paper_obj.id,
                "title": paper_obj.title,
                "authors": paper_obj.authors,
                "year": paper_obj.year,
                "publication": paper_obj.publication,
                "doi": paper_obj.doi,
                "arxiv_id": paper_obj.arxiv_id,
                "abstract": paper_obj.abstract,
                "file_path": paper_obj.file_path,
                "file_size": paper_obj.file_size,
                "indexed_at": paper_obj.indexed_at.strftime("%Y-%m-%d %H:%M:%S")
                if paper_obj.indexed_at
                else None,
                "embedding_status": paper_obj.embedding_status,
            }

            output.print_paper_details(paper_dict)

        else:
            output.print_error("Please provide an ID to show")
            output.print_info("Usage: lemma show <id>  OR  lemma show -n <note_id>")


# Keep 'info' as alias for backwards compatibility
@cli.command(name="info", hidden=True)
@click.argument("paper_id", type=int)
@click.option("--db", default="~/.MAVYN/MAVYN.db", help="Database path")
def info(paper_id: int, db: str):
    """Show detailed information about a paper (alias for 'show')."""
    with Repository(db) as repo:
        paper = repo.get_paper_by_id(paper_id)

        if not paper:
            output.print_error(f"Paper with ID {paper_id} not found")
            return

        # Convert to dict for display
        paper_dict = {
            "id": paper.id,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "publication": paper.publication,
            "doi": paper.doi,
            "arxiv_id": paper.arxiv_id,
            "abstract": paper.abstract,
            "file_path": paper.file_path,
            "file_size": paper.file_size,
            "indexed_at": paper.indexed_at.strftime("%Y-%m-%d %H:%M:%S")
            if paper.indexed_at
            else None,
            "embedding_status": paper.embedding_status,
        }

        output.print_paper_details(paper_dict)


@cli.command()
@click.argument("question")
@click.option("--db", default="~/.MAVYN/MAVYN.db", help="Database path")
@click.option("--top-k", type=int, default=5, help="Number of papers to retrieve")
@click.option("--index-path", default="~/.MAVYN/search.index", help="FAISS index path")
@click.option("--save", "-s", is_flag=True, help="Automatically save answer as a note")
@click.option(
    "--arxiv",
    "arxiv_cli",
    is_flag=True,
    default=False,
    help="Include related papers from the arXiv API (network; search query sent to arxiv.org).",
)
@click.option(
    "--no-arxiv",
    "no_arxiv_cli",
    is_flag=True,
    default=False,
    help="Do not query arXiv even if MAVYN_ARXIV_RELATED=1 is set.",
)
def ask(
    question: str,
    db: str,
    top_k: int,
    index_path: str,
    save: bool,
    arxiv_cli: bool,
    no_arxiv_cli: bool,
):
    """Ask a question across all papers (semantic search + LLM) or compare papers.

    QUESTION: Your question about the papers

    Examples:
      lemma ask "What are the main findings on topic X?"
      lemma ask "Compare papers 1 and 5"
      lemma ask "Compare the methodology in papers [2], [7], and [12]"
      lemma ask "Similar papers on graph neural networks"
      lemma ask "Related work like paper 3" --arxiv

    Note: Requires embeddings to be generated first. Run 'lemma embed' if needed.

    Use --save or -s to automatically save the answer as a formatted note.
    """
    from ..embeddings.encoder import EmbeddingEncoder
    from ..embeddings.search import SemanticSearchIndex
    from ..llm.providers import LLMRouter
    from ..llm.rate_limits import RateLimitStore
    from ..llm.cache import LLMCache
    from ..llm import prompts
    from ..llm.question_parser import (
        parse_comparison_request,
        wants_similar_papers,
        wants_list_sections,
        extract_seed_paper_ids_for_similar,
        extract_paper_ids,
    )
    from ..integrations import arxiv_client
    from ..llm.comparison_cache import ComparisonCache
    from ..llm.comparison import ComparisonEngine
    from pathlib import Path

    with Repository(db) as repo:
        extractor = MetadataExtractor()

        # NEW: Detect if this is a comparison request
        comparison_request = parse_comparison_request(question)

        if comparison_request:
            # This is a comparison request - handle it differently
            output.print_info(
                f"Detected comparison request for {len(comparison_request.paper_ids)} papers"
            )

            # Validate papers exist and have embeddings
            papers = repo.get_papers_by_ids(comparison_request.paper_ids)

            if len(papers) != len(comparison_request.paper_ids):
                output.print_error(
                    f"Some papers not found. Requested: {comparison_request.paper_ids}, "
                    f"Found: {[p.id for p in papers]}"
                )
                return

            # Check embeddings status
            for paper in papers:
                if paper.embedding_status != "completed":
                    output.print_error(
                        f"Paper {paper.id} ({paper.title or 'Untitled'}) doesn't have embeddings. "
                        f"Run 'lemma embed' first."
                    )
                    return

            # Initialize LLM router
            llm_router = LLMRouter(rate_store=RateLimitStore(), cache_enabled=True)

            if not llm_router.is_available():
                output.print_error(
                    "No LLM providers available. Please set up API keys:\n"
                    "  GROQ_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY\n"
                    "  — or run a local Ollama server: ollama serve"
                )
                return

            # Initialize comparison components
            comp_cache = ComparisonCache(repo)
            comp_engine = ComparisonEngine(repo, llm_router, comp_cache)

            # Perform comparison
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True,
                ) as progress:
                    if comparison_request.comparison_type == "section":
                        task = progress.add_task(
                            f"Comparing {comparison_request.section_name} section...",
                            total=None,
                        )
                        result = comp_engine.compare_section(
                            paper_ids=comparison_request.paper_ids,
                            section_name=comparison_request.section_name,
                        )
                    else:
                        task = progress.add_task(
                            f"Comparing {len(papers)} papers...", total=None
                        )
                        result = comp_engine.compare_papers(
                            paper_ids=comparison_request.paper_ids
                        )

                    progress.update(task, completed=True)

                # Display results
                output.print_comparison_results(result.to_dict(), papers)

                # Note: Comparison saving functionality could be added here if needed

                return

            except Exception as e:
                output.print_error(f"Comparison failed: {e}")
                import traceback

                output.print_error(f"Traceback: {traceback.format_exc()}")
                return

        use_arxiv = (
            arxiv_cli or os.environ.get("MAVYN_ARXIV_RELATED", "").strip() == "1"
        ) and not no_arxiv_cli

        if wants_similar_papers(question):
            index_file = Path(index_path).expanduser()
            faiss_path = index_file.with_suffix(".faiss")
            has_index = faiss_path.exists()

            encoder = None
            search_index = None
            if has_index:
                try:
                    encoder = EmbeddingEncoder()
                    search_index = SemanticSearchIndex(
                        embedding_dim=encoder.embedding_dim, index_path=index_file
                    )
                    if search_index.size() == 0:
                        search_index = None
                except Exception as e:
                    output.print_warning(f"Could not load search index: {e}")
                    encoder = None
                    search_index = None

            if use_arxiv and encoder is None:
                try:
                    encoder = EmbeddingEncoder()
                except Exception as e:
                    output.print_error(f"Failed to load embedding model: {e}")
                    return

            if not has_index and not use_arxiv:
                output.print_error(
                    "Similar-papers questions need a FAISS index (run 'lemma embed') "
                    "or enable arXiv with --arxiv or MAVYN_ARXIV_RELATED=1."
                )
                return

            seed_ids = extract_seed_paper_ids_for_similar(question)
            exclude: set = set()
            seed_chunks: List[str] = []
            for sid in seed_ids:
                sp = repo.get_paper_by_id(sid)
                if sp:
                    exclude.add(sid)
                    seed_chunks.append(
                        ((sp.title or "") + "\n" + (sp.abstract or ""))[:4000]
                    )
            seed_blob = "\n\n".join(seed_chunks)

            query_vector = None
            local_top: List[Tuple[int, float]] = []
            if encoder is not None and search_index is not None:
                try:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        transient=True,
                    ) as progress:
                        task = progress.add_task(
                            "Encoding for similarity...", total=None
                        )
                        qtext = seed_blob.strip() if seed_blob.strip() else question
                        query_vector = encoder.encode(qtext[:8000])
                        progress.update(task, completed=True)
                        task = progress.add_task("Searching library...", total=None)
                        raw = search_index.get_top_papers(
                            query_vector, top_k=max(top_k * 4, 20)
                        )
                        progress.update(task, completed=True)
                    local_top = [
                        (pid, dist) for pid, dist in raw if pid not in exclude
                    ][:top_k]
                except Exception as e:
                    output.print_error(f"Similar-papers search failed: {e}")
                    return
            elif encoder is not None and use_arxiv:
                qtext = seed_blob.strip() if seed_blob.strip() else question
                query_vector = encoder.encode(qtext[:8000])

            paper_ids = [pid for pid, _ in local_top]
            papers = repo.get_papers_by_ids(paper_ids) if paper_ids else []
            order = {pid: i for i, pid in enumerate(paper_ids)}
            papers.sort(key=lambda p: order.get(p.id, 9999))

            context_papers: List[Dict[str, Any]] = []
            for paper in papers:
                if paper.abstract and len(paper.abstract) > 100:
                    paper_text = paper.abstract
                else:
                    try:
                        paper_path = Path(paper.file_path)
                        if paper_path.exists():
                            full_text = extractor.extract_full_text(paper_path)
                            paper_text = (
                                full_text[:2000] if full_text else "No text available"
                            )
                        else:
                            paper_text = "PDF file not found"
                    except Exception:
                        paper_text = "Error extracting text"
                context_papers.append(
                    {
                        "id": paper.id,
                        "title": paper.title or "Untitled",
                        "authors": paper.authors or "Unknown",
                        "year": paper.year or "N/A",
                        "doi": paper.doi or "",
                        "arxiv_id": paper.arxiv_id or "",
                        "publication": paper.publication or "",
                        "text": paper_text,
                    }
                )

            arxiv_max = min(12, max(top_k, 8))
            arxiv_entries: List[Dict[str, Any]] = []
            if use_arxiv:
                output.print_info(
                    "Querying arXiv API (network): a search query derived from your "
                    "question or seed paper is sent to export.arxiv.org."
                )
                seed_for_arxiv = seed_blob.strip() if seed_blob.strip() else question
                sq = arxiv_client.build_arxiv_search_query(seed_for_arxiv)
                qh = arxiv_client.arxiv_cache_key(sq, arxiv_max)
                cached = repo.get_arxiv_query_cache(qh)
                if cached is not None:
                    arxiv_entries = list(cached)
                else:
                    try:
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            transient=True,
                        ) as progress:
                            task = progress.add_task("Fetching arXiv...", total=None)
                            arxiv_entries = arxiv_client.fetch_arxiv_search(
                                sq, max_results=arxiv_max
                            )
                            progress.update(task, completed=True)
                    except Exception as e:
                        output.print_warning(f"arXiv request failed: {e}")
                        arxiv_entries = []
                    if arxiv_entries:
                        repo.set_arxiv_query_cache(qh, sq, arxiv_max, arxiv_entries)

                la, ld = repo.get_local_arxiv_and_doi_sets()
                arxiv_entries = arxiv_client.dedupe_against_library(
                    arxiv_entries, la, ld
                )
                if query_vector is not None and encoder is not None and arxiv_entries:
                    arxiv_entries = arxiv_client.rerank_by_embedding_similarity(
                        arxiv_entries, query_vector, encoder
                    )
                arxiv_entries = arxiv_entries[:arxiv_max]

            local_lines: List[str] = []
            for p in context_papers:
                local_lines.append(
                    f"- Lemma id {p['id']}: {p['title']} ({p['year']}) | "
                    f"authors: {p['authors']} | doi: {p.get('doi') or 'N/A'} | "
                    f"arxiv: {p.get('arxiv_id') or 'N/A'}"
                )
            local_block = (
                "\n".join(local_lines)
                if local_lines
                else "(No local library matches from the semantic index for this query.)"
            )

            arxiv_lines: List[str] = []
            for e in arxiv_entries:
                arxiv_lines.append(
                    f"- arxiv_id={e.get('arxiv_id', '')} | abs_url={e.get('abs_url', '')} | "
                    f"title={e.get('title', '')} | category={e.get('primary_category', '')} | "
                    f"published={e.get('published', '')} | authors={e.get('authors', '')} | "
                    f"summary={e.get('summary', '')[:400]}"
                )
            arxiv_block = "\n".join(arxiv_lines) if arxiv_lines else ""

            llm_router = LLMRouter(rate_store=RateLimitStore(), cache_enabled=True)
            llm_cache = LLMCache(repo)

            if not llm_router.is_available():
                output.print_warning(
                    "No LLM configured; listing similarity candidates only."
                )
                if context_papers:
                    output.print_paper_table(
                        [
                            {
                                "id": p["id"],
                                "title": p["title"],
                                "authors": p["authors"],
                                "year": p["year"],
                                "embedding_status": "completed",
                            }
                            for p in context_papers
                        ]
                    )
                if arxiv_entries:
                    output.print_arxiv_related(arxiv_entries)
                return

            try:
                prompt = prompts.build_similar_papers_prompt(
                    question, local_block, arxiv_block
                )
                with output.thinking_spinner():
                    response = llm_router.generate(
                        prompt=prompt,
                        cache_lookup=llm_cache.get,
                        cache_store=llm_cache.store,
                    )

                if not response:
                    output.print_error("Failed to generate answer from LLM")
                    return

                sources = [
                    f"[{p['id']}] {p['title']} ({p['year']})" for p in context_papers
                ]
                output.print_answer(question, response.text, sources)
                if arxiv_entries:
                    output.print_arxiv_related(arxiv_entries)

                if response.provider != "cache":
                    output.print_info(
                        f"\nProvider: {response.provider} | Model: {response.model} | "
                        f"Tokens: {response.tokens_used}"
                    )
                else:
                    output.print_info("\n[Cached response]")

                session_data = {
                    "question": question,
                    "answer": response.text,
                    "paper_ids": [p["id"] for p in context_papers],
                    "sources": sources,
                    "context_papers": context_papers,
                    "provider": response.provider,
                    "model": response.model,
                    "tokens_used": response.tokens_used,
                    "arxiv_related": arxiv_entries,
                }
                repo.set_config("last_qa_session", session_data)

                if save:
                    _save_note_from_session(repo, session_data)
                else:
                    output.print_info(
                        "\n💡 Tip: Answers are automatically saved in your session"
                    )
            except Exception as e:
                output.print_error(f"Failed to generate answer: {e}")
            return

        # ── List Sections — no-LLM early return ─────────────────────────────
        if wants_list_sections(question):
            _ls_ids: List[int] = []
            for pid in extract_paper_ids(question):
                if repo.get_paper_by_id(pid):
                    _ls_ids.append(pid)
                else:
                    output.print_warning(f"Paper id {pid} not found in library.")

            if not _ls_ids:
                output.print_error(
                    "No paper specified. Mention a paper (e.g. 'paper 3' or '[3]'), "
                    "or first ask a question about a paper to set session context."
                )
                return

            sections_by_paper = repo.get_all_section_names_for_papers(_ls_ids)
            _display_sections_list(sections_by_paper, repo)
            return

        # Q&A / Summarize flow — task-aware retrieval

        from ..embeddings.retrieval import (
            HybridRetriever,
            StructuredExtractor,
            TaskRouter,
        )

        # Detect task type before touching FAISS (summarize skips it entirely)
        _preview_ids = extract_paper_ids(question)
        task_type, task_section = TaskRouter().detect(question, _preview_ids)

        context_str: str
        included_ids: List[int]

        if task_type == "summarize_section" and len(_preview_ids) == 1:
            # Section-targeted summarise: fetch only that section
            from ..embeddings.retrieval import AlignedExtractor

            _pid = _preview_ids[0]
            _paper_obj = repo.get_paper_by_id(_pid)
            if not _paper_obj:
                output.print_error(f"Paper id {_pid} not found in library.")
                return

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"Extracting {task_section} section...", total=None
                )
                section_text = AlignedExtractor(repo).extract_section_text(
                    _pid, task_section, token_budget=1000
                )
                progress.update(task, completed=True)

            _title = _paper_obj.title or "Untitled"
            if not section_text:
                # Section missing: build full-paper context so the LLM can
                # acknowledge the gap and still provide a useful summary
                context_str, included_ids = StructuredExtractor(repo).extract(
                    _pid, _paper_obj
                )
                task_section = task_section + ":missing"  # signal to prompt dispatch
                output.print_info(
                    f"'{task_section.split(':')[0]}' section not found — "
                    f"summarising full paper: {_title}"
                )
            else:
                context_str = (
                    f"[Paper 1 | {_title} | {task_section.title()}]\n{section_text}"
                )
                included_ids = [_pid]
                output.print_info(f"Summarising {task_section} section of: {_title}")

        elif task_type == "summarize" and len(_preview_ids) == 1:
            # Single-paper structural extraction — no FAISS needed
            _pid = _preview_ids[0]
            _paper_obj = repo.get_paper_by_id(_pid)
            if not _paper_obj:
                output.print_error(f"Paper id {_pid} not found in library.")
                return

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("Extracting paper structure...", total=None)
                context_str, included_ids = StructuredExtractor(repo).extract(
                    _pid, _paper_obj
                )
                progress.update(task, completed=True)

            output.print_info(f"Summarising: {_paper_obj.title or 'Untitled'}")

            if not context_str:
                output.print_error(
                    f"No embedded chunks found for paper {_pid}. "
                    "Run 'lemma embed' first."
                )
                return

        else:
            # General Q&A — FAISS + hybrid retrieval
            index_file = Path(index_path).expanduser()
            if not index_file.with_suffix(".faiss").exists():
                output.print_error(
                    "No embeddings found. Please run 'lemma embed' first."
                )
                return

            try:
                encoder = EmbeddingEncoder()
                search_index = SemanticSearchIndex(
                    embedding_dim=encoder.embedding_dim, index_path=index_file
                )
            except Exception as e:
                output.print_error(f"Failed to load search index: {e}")
                return

            if search_index.size() == 0:
                output.print_error(
                    "Search index is empty. Please run 'lemma embed' first."
                )
                return

            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True,
                ) as progress:
                    task = progress.add_task("Encoding question...", total=None)
                    query_vector = encoder.encode(question)
                    progress.update(task, completed=True)
            except Exception as e:
                output.print_error(f"Failed to encode question: {e}")
                return

            explicit_ids = extract_paper_ids(question)
            valid_explicit: List[int] = []
            for pid in explicit_ids:
                p = repo.get_paper_by_id(pid)
                if p:
                    valid_explicit.append(pid)
                else:
                    output.print_warning(f"Paper id {pid} not found in library.")

            if valid_explicit:
                output.print_info(
                    f"Including lemma id(s) from your question: {valid_explicit}"
                )

            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True,
                ) as progress:
                    task = progress.add_task("Searching papers...", total=None)
                    context_str, included_ids = HybridRetriever(
                        search_index, repo
                    ).retrieve(
                        question=question,
                        query_vector=query_vector,
                        top_k=top_k,
                        pinned_paper_ids=valid_explicit if valid_explicit else None,
                    )
                    progress.update(task, completed=True)
            except Exception as e:
                output.print_error(f"Search failed: {e}")
                return

            if not included_ids:
                output.print_warning("No relevant papers found for your question.")
                return

            output.print_info(f"Using {len(included_ids)} paper(s) for context")

        # Build minimal context_papers for downstream (note saving, display)
        included_papers = repo.get_papers_by_ids(included_ids)
        context_papers = [
            {
                "id": p.id,
                "title": p.title or "Untitled",
                "authors": p.authors or "Unknown",
                "year": p.year or "N/A",
            }
            for p in included_papers
        ]

        # Initialize LLM router and cache
        llm_router = LLMRouter(rate_store=RateLimitStore(), cache_enabled=True)
        llm_cache = LLMCache(repo)

        # Check if LLM is available
        if not llm_router.is_available():
            output.print_error(
                "No LLM providers available. Please set up API keys:\n"
                "  GROQ_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY\n"
                "  — or run a local Ollama server: ollama serve"
            )
            output.print_info("\nShowing relevant papers instead:")
            output.print_paper_table(
                [
                    {
                        "id": p["id"],
                        "title": p["title"],
                        "authors": p["authors"],
                        "year": p["year"],
                        "embedding_status": "completed",
                    }
                    for p in context_papers
                ]
            )
            return

        # Build prompt and generate answer
        try:
            # Build prompt
            _title = context_papers[0]["title"] if context_papers else "the paper"
            if (
                task_type == "summarize_section"
                and task_section
                and len(included_ids) == 1
            ):
                _raw_section = task_section.replace(":missing", "")
                if task_section.endswith(":missing"):
                    prompt = prompts.build_section_missing_prompt(
                        context_str, _title, _raw_section
                    )
                else:
                    prompt = prompts.build_section_summary_prompt(
                        context_str, _title, _raw_section
                    )
            elif task_type == "summarize" and len(included_ids) == 1:
                prompt = prompts.build_summary_prompt(context_str, _title)
            else:
                prompt = prompts.build_qa_prompt(question, context_str)

            with output.thinking_spinner():
                response = llm_router.generate(
                    prompt=prompt,
                    cache_lookup=llm_cache.get,
                    cache_store=llm_cache.store,
                )

            if not response:
                output.print_error("Failed to generate answer from LLM")
                return

            # Display answer — only show papers the LLM actually cited
            cited_ids = _extract_cited_paper_ids(response.text)
            cited_papers = (
                [p for p in context_papers if p["id"] in cited_ids]
                if cited_ids
                else context_papers
            )
            sources = [f"[{p['id']}] {p['title']} ({p['year']})" for p in cited_papers]
            output.print_answer(question, response.text, sources)

            # Show provider info
            if response.provider != "cache":
                output.print_info(
                    f"\nProvider: {response.provider} | Model: {response.model} | Tokens: {response.tokens_used}"
                )
            else:
                output.print_info("\n[Cached response]")

            # Store session context for note saving
            session_data = {
                "question": question,
                "answer": response.text,
                "paper_ids": [p["id"] for p in context_papers],
                "sources": sources,
                "context_papers": context_papers,
                "provider": response.provider,
                "model": response.model,
                "tokens_used": response.tokens_used,
            }
            repo.set_config("last_qa_session", session_data)

            # Auto-save note if --save flag is set
            if save:
                _save_note_from_session(repo, session_data)
            else:
                output.print_info(
                    "\n💡 Tip: Answers are automatically saved in your session"
                )

        except Exception as e:
            output.print_error(f"Failed to generate answer: {e}")
            output.print_info("\nShowing relevant papers:")
            output.print_paper_table(
                [
                    {
                        "id": p["id"],
                        "title": p["title"],
                        "authors": p["authors"],
                        "year": p["year"],
                        "embedding_status": "completed",
                    }
                    for p in context_papers
                ]
            )


@cli.command(name="notes")
@click.option("--db", default="~/.MAVYN/MAVYN.db", help="Database path")
@click.option("--limit", type=int, default=20, help="Number of notes to show")
def notes_list(db: str, limit: int):
    """List all saved notes (shortcut for note list)."""
    with Repository(db) as repo:
        # Fetch notes
        try:
            notes = repo.list_notes(limit=limit)
        except Exception as e:
            output.print_error(f"Failed to fetch notes: {e}")
            return

        if not notes:
            output.print_info(
                "No notes found. Save your first note with 'lemma ask ... --save'"
            )
            return

        _display_notes_table(notes, repo, "\nView full note: lemma show -n <id>")


@cli.command()
@click.option("--db", default="~/.MAVYN/MAVYN.db", help="Database path")
@click.option(
    "--force", is_flag=True, help="Re-embed papers that already have embeddings"
)
@click.option("--index-path", default="~/.MAVYN/search.index", help="FAISS index path")
@click.option(
    "--checkpoint-every",
    type=int,
    default=10,
    help="Save index checkpoint every N papers (default: 10)",
)
@click.option(
    "--incremental/--no-incremental",
    default=True,
    help="Use incremental updates (reuse unchanged chunks)",
)
@click.option(
    "--strategy",
    type=click.Choice(["structure", "sentence", "hybrid", "simple"]),
    default="hybrid",
    help="Chunking strategy (hybrid recommended)",
)
def embed(
    db: str,
    force: bool,
    index_path: str,
    checkpoint_every: int,
    incremental: bool,
    strategy: str,
):
    """Generate embeddings for all papers with smart chunking and incremental updates.

    This enables semantic search and the 'ask' command.

    Features:
    - Smart chunking: Structure-aware chunking respects paper sections
    - Incremental updates: Reuses unchanged chunks (70-90% time savings)
    - Automatic resume: Only processes papers not yet embedded
    - Checkpointing: Saves progress every N papers (default: 10)
    - Crash recovery: Re-run command to continue from last checkpoint

    Chunking strategies:
      - hybrid: Structure + sentence-based (recommended for papers)
      - structure: Section-aware chunking
      - sentence: Semantic sentence boundaries
      - simple: Legacy word-based chunking
    """
    from ..embeddings.encoder import EmbeddingEncoder
    from ..embeddings.search import SemanticSearchIndex
    from ..embeddings.chunking import PaperChunker, ChunkingStrategy
    from ..embeddings.incremental import IncrementalEmbedder
    from ..utils.logger import get_logger
    from pathlib import Path

    logger = get_logger(__name__)

    with Repository(db) as repo:
        extractor = MetadataExtractor()

        # Get papers that need embeddings
        papers = repo.get_papers_for_embedding(force=force)

        if not papers:
            if force:
                output.print_info("No papers found in database")
            else:
                output.print_success("All papers already have embeddings!")
                output.print_info("Use --force to regenerate embeddings")
            return

        output.print_info(
            f"Found {len(papers)} papers to embed "
            f"(incremental: {incremental}, strategy: {strategy})"
        )

        # Initialize encoder and search index
        try:
            encoder = EmbeddingEncoder()
            output.print_success(
                f"Loaded embedding model: {encoder.model_name} ({encoder.embedding_dim}D)"
            )
        except Exception as e:
            output.print_error(f"Failed to load embedding model: {e}")
            return

        # Initialize chunker with selected strategy
        try:
            strategy_map = {
                "structure": ChunkingStrategy.STRUCTURE_AWARE,
                "sentence": ChunkingStrategy.SENTENCE_BASED,
                "hybrid": ChunkingStrategy.HYBRID,
                "simple": ChunkingStrategy.SIMPLE,
            }
            chunker = PaperChunker(strategy=strategy_map[strategy])
            output.print_info(f"Using {strategy} chunking strategy")
        except Exception as e:
            output.print_error(f"Failed to initialize chunker: {e}")
            return

        # Load or create FAISS index
        index_file = Path(index_path).expanduser()
        try:
            if force or not index_file.with_suffix(".faiss").exists():
                # Create new index
                search_index = SemanticSearchIndex(
                    embedding_dim=encoder.embedding_dim, index_path=index_file
                )
                output.print_info("Created new FAISS index")
            else:
                # Load existing index
                search_index = SemanticSearchIndex(
                    embedding_dim=encoder.embedding_dim, index_path=index_file
                )
                output.print_success(
                    f"Loaded existing FAISS index ({search_index.size()} vectors)"
                )
        except Exception as e:
            output.print_error(f"Failed to initialize FAISS index: {e}")
            return

        # Initialize incremental embedder
        embedder = IncrementalEmbedder(chunker, encoder, repo)

        # Process each paper
        success_count = 0
        error_count = 0
        total_reused = 0
        total_new = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            task = progress.add_task(
                f"Embedding {len(papers)} papers...", total=len(papers)
            )

            for paper in papers:
                try:
                    paper_path = Path(paper.file_path)
                    if not paper_path.exists():
                        raise FileNotFoundError(f"PDF not found: {paper_path}")

                    # Try Docling first; fall back to plain-text on failure
                    docling_chunks = None
                    full_text = None
                    try:
                        from ..embeddings.docling_chunker import chunk_pdf_with_docling

                        docling_chunks = chunk_pdf_with_docling(paper_path)
                        full_text = " ".join(c.text for c in docling_chunks)
                    except ImportError:
                        pass
                    except Exception as exc:
                        logger.warning(
                            f"Docling failed for {paper_path.name} ({exc}); "
                            "using plain-text fallback"
                        )

                    if full_text is None:
                        full_text = extractor.extract_full_text(paper_path)

                    if not full_text or len(full_text.strip()) < 100:
                        raise ValueError("Insufficient text extracted from PDF")

                    # Use incremental embedder if enabled, otherwise use legacy
                    if incremental and not force:
                        result = embedder.incremental_embed(
                            paper,
                            full_text,
                            extractor,
                            force=False,
                            chunks=docling_chunks,
                        )
                    else:
                        result = embedder.incremental_embed(
                            paper,
                            full_text,
                            extractor,
                            force=True,
                            chunks=docling_chunks,
                        )

                    if not result.success:
                        raise Exception(result.error or "Embedding failed")

                    # Track statistics
                    total_reused += result.reused_chunks
                    total_new += result.new_chunks

                    # Get embeddings and add to FAISS index
                    # Note: We rebuild FAISS from database for simplicity
                    # TODO: Optimize to only add new embeddings to FAISS
                    valid_embeddings = [
                        e for e in repo.get_embeddings_by_paper(paper.id) if e.is_valid
                    ]

                    if valid_embeddings:
                        import json
                        import numpy as np

                        # Extract embeddings and add to FAISS
                        embedding_vectors = []
                        chunk_indices = []

                        for emb in valid_embeddings:
                            vec = json.loads(emb.embedding_vector)
                            embedding_vectors.append(vec)
                            chunk_indices.append(emb.chunk_index)

                        embeddings_array = np.array(embedding_vectors, dtype=np.float32)

                        search_index.add(
                            embeddings=embeddings_array,
                            paper_id=paper.id,
                            chunk_indices=chunk_indices,
                        )

                    # Log success
                    repo.log_operation(
                        operation="embed",
                        status="success",
                        paper_id=paper.id,
                        details={
                            "total_chunks": result.total_chunks,
                            "reused_chunks": result.reused_chunks,
                            "new_chunks": result.new_chunks,
                            "model": encoder.model_name,
                            "strategy": strategy,
                        },
                    )

                    success_count += 1

                    # Checkpoint: Save index periodically
                    if checkpoint_every > 0 and success_count % checkpoint_every == 0:
                        try:
                            search_index.save(index_file)
                            logger.info(
                                f"Checkpoint: Saved index after {success_count} papers"
                            )
                        except Exception as checkpoint_error:
                            logger.warning(
                                f"Failed to save checkpoint: {checkpoint_error}"
                            )
                            # Don't fail the entire process for checkpoint errors

                except Exception as e:
                    error_count += 1

                    # Print full traceback for debugging
                    import traceback

                    output.print_error(f"Error embedding paper {paper.id}: {e}")
                    output.print_error(f"Traceback: {traceback.format_exc()}")

                    # Update status to failed
                    repo.update_paper_metadata(paper.id, {"embedding_status": "failed"})

                    # Log error
                    repo.log_operation(
                        operation="embed",
                        status="failed",
                        paper_id=paper.id,
                        error_message=str(e),
                    )

                progress.advance(task)

        # Save FAISS index to disk
        try:
            search_index.save(index_file)
            output.print_success(f"Saved FAISS index to {index_file}")
        except Exception as e:
            output.print_error(f"Failed to save FAISS index: {e}")

        # Print summary with reuse statistics
        total_chunks = total_reused + total_new
        reuse_pct = (total_reused / total_chunks * 100) if total_chunks > 0 else 0

        output.print_info(
            f"\nEmbedding complete: {success_count} succeeded, {error_count} failed"
        )

        if incremental and total_reused > 0:
            output.print_success(
                f"Chunk reuse: {total_reused}/{total_chunks} chunks reused "
                f"({reuse_pct:.1f}% time saved)"
            )

        if success_count > 0:
            output.print_success(
                "Embeddings complete! You can now ask questions about your papers."
            )


@cli.command(name="embed-status")
@click.option("--db", default="~/.MAVYN/MAVYN.db", help="Database path")
def embed_status(db: str):
    """Show embedding coverage and version status for all papers.

    Displays statistics about:
    - Embedding coverage (how many papers are embedded)
    - Papers needing updates (content version mismatch)
    - Chunk reuse potential
    - Model consistency
    """
    from rich.table import Table

    with Repository(db) as repo:
        # Get overall statistics
        stats = repo.get_embedding_coverage_stats()

        # Display summary panel
        output.console.print()
        output.console.print(
            Panel(
                f"[cyan]Total Papers:[/cyan] {stats['total_papers']}\n"
                f"[green]Embedded:[/green] {stats['embedded_papers']} "
                f"({stats['coverage_pct']:.1f}%)\n"
                f"[yellow]Pending:[/yellow] {stats['pending_papers']}\n"
                f"[red]Failed:[/red] {stats['failed_papers']}\n"
                f"[magenta]Outdated:[/magenta] {stats['outdated_papers']} "
                f"(need incremental update)",
                title="[bold]Embedding Status[/bold]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

        # Display chunk statistics
        output.console.print()
        output.console.print(
            Panel(
                f"[cyan]Total Chunks:[/cyan] {stats['total_embeddings']}\n"
                f"[green]Valid:[/green] {stats['valid_embeddings']}\n"
                f"[red]Invalid/Orphaned:[/red] {stats['invalid_embeddings']}",
                title="[bold]Chunk Statistics[/bold]",
                border_style="green",
                box=box.ROUNDED,
            )
        )

        # Show papers needing updates
        outdated_papers = repo.get_papers_needing_update()

        if outdated_papers:
            output.console.print()
            table = Table(
                title=f"Papers Needing Updates ({len(outdated_papers)} shown)",
                box=box.ROUNDED,
            )
            table.add_column("ID", justify="right", style="cyan")
            table.add_column("Title", style="white", max_width=40)
            table.add_column("Content Ver", justify="center", style="yellow")
            table.add_column("Embedded Ver", justify="center", style="green")
            table.add_column("Status", style="magenta")

            for paper in outdated_papers[:20]:  # Show first 20
                title = (paper.title or "Untitled")[:40]
                content_ver = str(paper.content_version or 1)
                embedded_ver = str(paper.last_embedded_version or 0)
                status = paper.embedding_status or "pending"

                table.add_row(
                    str(paper.id),
                    title,
                    content_ver,
                    embedded_ver,
                    status,
                )

            output.console.print(table)

            if len(outdated_papers) > 20:
                output.print_info(f"\n... and {len(outdated_papers) - 20} more papers")

            output.print_info(
                "\nRun 'lemma embed --incremental' to update these papers efficiently"
            )
        else:
            output.print_success("\nAll embedded papers are up to date!")

        # Show cleanup recommendation if there are invalid embeddings
        if stats["invalid_embeddings"] > 0:
            output.print_info(
                f"\n💡 Tip: Run 'lemma embed --incremental' or use the cleanup command "
                f"to remove {stats['invalid_embeddings']} orphaned chunks"
            )


@cli.command()
@click.option("--db", default="~/.MAVYN/MAVYN.db", help="Database path")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be renamed without doing it"
)
@click.option(
    "--pattern",
    default="{year}_{first_author}_{short_title}.pdf",
    help="Rename pattern",
)
def organize(db: str, dry_run: bool, pattern: str):
    """Organize PDF files with smart renaming.

    Renames files based on metadata with rollback capability.
    """
    from ..core.organizer import FileOrganizer
    from rich.table import Table

    with Repository(db) as repo:
        organizer = FileOrganizer(dry_run=dry_run)

        # Get all papers
        papers = repo.list_papers(limit=10000)  # Get all papers

        if not papers:
            output.print_info("No papers found in database")
            return

        # Convert to dicts for organizer, filtering out missing files
        paper_dicts = []
        missing_count = 0
        for paper in papers:
            # Check if file exists on disk
            from pathlib import Path

            if not Path(paper.file_path).exists():
                missing_count += 1
                continue

            paper_dicts.append(
                {
                    "id": paper.id,
                    "file_path": paper.file_path,
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.year,
                    "doi": paper.doi,
                    "arxiv_id": paper.arxiv_id,
                }
            )

        if missing_count > 0:
            output.print_warning(
                f"Skipping {missing_count} paper(s) with missing files. "
                f"Run 'lemma verify' to clean up the database."
            )

        # Preview renames
        previews = organizer.preview_renames(paper_dicts, pattern=pattern)

        # Filter to only show files that would change
        changes = [p for p in previews if p["changed"]]

        if not changes:
            output.print_success("All files are already organized!")
            return

        # Display preview table
        table = Table(
            title=f"{'[DRY RUN] ' if dry_run else ''}File Organization Preview"
        )
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Title", style="white", max_width=40)
        table.add_column("Current Name", style="yellow", max_width=30)
        table.add_column("New Name", style="green", max_width=30)

        for preview in changes[:20]:  # Show first 20
            paper = next(
                (p for p in paper_dicts if p["file_path"] == preview["original"]), None
            )
            if paper:
                from pathlib import Path

                current = Path(preview["original"]).name
                new = Path(preview["new"]).name

                table.add_row(
                    str(paper["id"]),
                    preview["title"][:40] if preview["title"] else "Untitled",
                    current[:30],
                    new[:30],
                )

        output.console.print(table)

        if len(changes) > 20:
            output.print_info(f"\n... and {len(changes) - 20} more files")

        output.print_info(f"\nTotal files to rename: {len(changes)}")

        if dry_run:
            output.print_warning(
                "\nThis is a dry run. Use without --dry-run to apply changes."
            )
            return

        # Confirm before proceeding
        if not click.confirm("\nProceed with renaming?"):
            output.print_info("Operation cancelled")
            return

        # Perform renames
        success_count = 0
        error_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            task = progress.add_task(
                f"Renaming {len(changes)} files...", total=len(changes)
            )

            for preview in changes:
                try:
                    paper = next(
                        (
                            p
                            for p in paper_dicts
                            if p["file_path"] == preview["original"]
                        ),
                        None,
                    )
                    if not paper:
                        continue

                    original_path = Path(preview["original"])
                    new_path = Path(preview["new"])

                    # Perform rename
                    result = organizer.rename_file(
                        original_path=original_path,
                        new_filename=new_path.name,
                    )

                    if result:
                        # Update database
                        repo.update_paper_metadata(
                            paper["id"], {"file_path": str(result)}
                        )

                        # Log the operation
                        repo.log_file_operation(
                            paper_id=paper["id"],
                            operation_type="rename",
                            original_path=str(original_path),
                            new_path=str(result),
                        )

                        repo.log_operation(
                            operation="organize",
                            status="success",
                            paper_id=paper["id"],
                            details={"from": str(original_path), "to": str(result)},
                        )

                        success_count += 1

                except Exception as e:
                    error_count += 1
                    repo.log_operation(
                        operation="organize",
                        status="failed",
                        paper_id=paper.get("id"),
                        error_message=str(e),
                    )
                    output.print_error(f"Error renaming {original_path.name}: {e}")

                progress.advance(task)

        # Print summary
        output.print_info(
            f"\nRenaming complete: {success_count} succeeded, {error_count} failed"
        )

        if success_count > 0:
            output.print_success("Files have been organized!")
            output.print_info("Use the database logs to rollback if needed")


@cli.command()
def setup():
    """Run the setup wizard to configure API keys.

    Configure your LLM provider API keys for AI-powered features.
    """
    run_setup_wizard(skip_if_configured=False)


@cli.command()
@click.option("--db", default="~/.MAVYN/MAVYN.db", help="Database path")
def migrate(db: str):
    """Migrate database to support incremental embeddings and advanced chunking.

    This command is safe to run multiple times. It will:
    - Add new database columns for versioning
    - Compute content hashes for existing papers
    - Set up chunk hashes for existing embeddings
    - Preserve all existing data

    Run this after upgrading to enable new features.
    """
    from ..db.migrate import migrate_to_versioning, check_migration_status

    output.print_info("Checking migration status...")

    status = check_migration_status(db)

    if not status["needs_migration"]:
        output.print_success("✓ Database is already migrated!")
        output.print_info(
            f"  Papers with content hashes: {status['papers_with_hashes']}/{status['total_papers']}"
        )
        return

    output.print_info("Migration needed - starting migration process...")
    output.print_info("This may take a few minutes for large libraries...")

    success = migrate_to_versioning(db)

    if success:
        output.print_success("\n✓ Migration completed successfully!")
        output.print_info("\nNew features now available:")
        output.print_info(
            "  • lemma embed --incremental (smart updates, 70-90% faster)"
        )
        output.print_info(
            "  • lemma embed --strategy hybrid (better chunking for papers)"
        )
        output.print_info("  • lemma embed-status (monitor embedding status)")
    else:
        output.print_error("\n✗ Migration failed - please check the logs")
        output.print_info(
            "You can still use Lemma, but new features won't be available"
        )


@cli.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=False,
)
@click.option("--db", default="~/.MAVYN/MAVYN.db", help="Database path")
@click.option(
    "--watch", is_flag=True, help="Continuously monitor directory for new papers"
)
@click.option(
    "--set-default", is_flag=True, help="Set this directory as default for future syncs"
)
@click.option("--no-rename", is_flag=True, help="Skip automatic file renaming")
@click.option(
    "--rename-pattern",
    default="{year}_{first_author}_{short_title}.pdf",
    help="Pattern for renaming files",
)
@click.option(
    "--no-embed",
    is_flag=True,
    help="Skip automatic embedding (embed later with 'lemma embed')",
)
@click.option(
    "--strategy",
    type=click.Choice(["structure", "sentence", "hybrid", "simple"]),
    default="hybrid",
    help="Chunking strategy for embeddings",
)
@click.option("--index-path", default="~/.MAVYN/search.index", help="FAISS index path")
@click.option(
    "--scan-interval",
    type=int,
    default=60,
    help="Scan interval in seconds (for periodic mode)",
)
@click.option("--recursive/--no-recursive", default=True, help="Scan subdirectories")
def sync(
    directory: Optional[Path],
    db: str,
    watch: bool,
    set_default: bool,
    no_rename: bool,
    rename_pattern: str,
    no_embed: bool,
    strategy: str,
    index_path: str,
    scan_interval: int,
    recursive: bool,
):
    """Automatically process papers: scan → rename → embed.

    DIRECTORY: Path to folder containing PDFs (optional if default is set)

    This command provides automatic paper processing with a complete pipeline:

    • Scans directory for new PDFs
    • Extracts metadata (title, authors, year, etc.)
    • Renames files using metadata (optional, configurable)
    • Generates embeddings for semantic search (optional)
    • Auto-migrates database if needed

    First-time setup:
      lemma sync ~/Papers              # Set your papers folder and sync
      lemma sync                       # Future syncs use the same folder

    Examples:
      lemma sync ~/Papers              # One-time sync
      lemma sync ~/Papers --watch      # Continuous monitoring
      lemma sync                       # Use default/last directory
      lemma sync --no-rename           # Skip renaming
      lemma sync --no-embed            # Only scan, embed later
    """
    from ..core.sync import SyncOrchestrator, setup_signal_handlers
    import time

    with Repository(db) as repo:
        # Determine directory
        if not directory:
            # Try to get default/last synced directory
            default_dir = repo.get_config("default_papers_directory")

            if default_dir:
                directory = Path(default_dir)
                output.print_info(f"📂 Using default papers directory: {directory}")
            else:
                # Fall back to last synced directory
                last_stats = repo.get_sync_stats()
                if last_stats and "directory" in last_stats:
                    directory = Path(last_stats["directory"])
                    output.print_info(f"📂 Using last synced directory: {directory}")
                    output.print_info(
                        "💡 Tip: Use /sync to process papers from this directory"
                    )
                else:
                    output.print_error("❌ No directory specified")
                    output.print_info("\nFirst-time setup:")
                    output.print_info(
                        "  /sync ~/Papers              # Process your papers"
                    )
                    output.print_info("\nThen use /list and ask questions naturally!")
                    return

        directory = Path(directory).expanduser().resolve()

        if not directory.exists():
            output.print_error(f"Directory not found: {directory}")
            return

        # Set as default if requested
        if set_default:
            repo.set_config("default_papers_directory", str(directory))
            output.console.print(
                f"[green]Set default papers directory:[/green] {directory}"
            )
            output.console.print(
                "[blue]You can now use /sync to add new papers from this directory[/blue]\n"
            )

        # Initialize orchestrator
        try:
            orchestrator = SyncOrchestrator(
                repo=repo,
                auto_rename=not no_rename,
                rename_pattern=rename_pattern,
                embed_immediately=not no_embed,
                chunking_strategy=strategy,
                index_path=index_path,
                use_watchdog=True,  # Try watchdog first, fall back to periodic
                scan_interval=scan_interval,
            )
        except Exception as e:
            output.print_error(f"Failed to initialize sync: {e}")
            return

        # Track stats
        stats = {
            "total": 0,
            "new": 0,
            "duplicates": 0,
            "failed": 0,
            "renamed": 0,
            "embedded": 0,
            "processing": [],
        }

        def update_progress(event_type: str, data: dict):
            """Callback for progress updates."""
            if event_type == "scanning":
                output.print_info(f"📂 Scanning: {data.get('directory')}")
            elif event_type == "processing":
                file_name = Path(data.get("file", "")).name
                stats["processing"].append(file_name)
            elif event_type == "file_completed":
                file_name = Path(data.get("file", "")).name
                if file_name in stats["processing"]:
                    stats["processing"].remove(file_name)
                stats["new"] += 1
                if data.get("renamed"):
                    stats["renamed"] += 1
                if data.get("embedded"):
                    stats["embedded"] += 1

                status = "✓ " + file_name
                if data.get("renamed"):
                    status += " [renamed]"
                if data.get("embedded"):
                    status += " [embedded]"
                output.print_success(status)
            elif event_type == "file_failed":
                file_name = Path(data.get("file", "")).name
                if file_name in stats["processing"]:
                    stats["processing"].remove(file_name)
                stats["failed"] += 1
                output.print_error(f"✗ {file_name}: {data.get('error')}")
            elif event_type == "completed":
                stats["total"] = data.get("total", 0)
                stats["new"] = data.get("success", 0)
                stats["duplicates"] = data.get("duplicate", 0)
                stats["failed"] = data.get("failed", 0)
                stats["renamed"] = data.get("renamed", 0)
                stats["embedded"] = data.get("embedded", 0)

        if watch:
            # Continuous monitoring mode
            output.console.print(
                Panel(
                    f"[bold cyan]Starting continuous monitoring[/bold cyan]\n"
                    f"Directory: {directory}\n"
                    f"Auto-rename: {'Yes' if not no_rename else 'No'}\n"
                    f"Auto-embed: {'Yes' if not no_embed else 'No'}\n"
                    f"Strategy: {strategy}\n"
                    f"\nPress Ctrl+C to stop",
                    border_style="cyan",
                    box=box.ROUNDED,
                )
            )

            # Setup signal handlers for graceful shutdown
            setup_signal_handlers(orchestrator)

            try:
                orchestrator.start_watching(
                    directory=directory,
                    recursive=recursive,
                    progress_callback=update_progress,
                )

                output.print_success("\n👀 Watching for new papers...")
                output.print_info(
                    "Add PDFs to the directory and they'll be processed automatically\n"
                )

                # Keep running until interrupted
                while orchestrator.is_watching():
                    time.sleep(1)

            except KeyboardInterrupt:
                output.print_info("\n\nStopping watcher...")
                orchestrator.stop_watching()
                output.print_success("Sync stopped")

            except Exception as e:
                output.print_error(f"Watch mode failed: {e}")
                orchestrator.stop_watching()

        else:
            # One-time sync mode
            output.console.print(
                Panel(
                    f"[bold cyan]Syncing directory[/bold cyan]\n"
                    f"Directory: {directory}\n"
                    f"Auto-rename: {'Yes' if not no_rename else 'No'}\n"
                    f"Auto-embed: {'Yes' if not no_embed else 'No'}\n"
                    f"Strategy: {strategy}",
                    border_style="cyan",
                    box=box.ROUNDED,
                )
            )

            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=False,
                ) as progress:
                    task = progress.add_task("Processing papers...", total=None)

                    results = orchestrator.sync_directory_once(
                        directory=directory,
                        recursive=recursive,
                        progress_callback=update_progress,
                    )

                    progress.update(task, completed=True)

                # Print summary
                output.console.print()
                summary_lines = [
                    f"[cyan]Total files:[/cyan] {results['total']}",
                    f"[green]New papers:[/green] {results['success']}",
                    f"[yellow]Duplicates:[/yellow] {results['duplicate']}",
                    f"[red]Failed:[/red] {results['failed']}",
                    f"[blue]Renamed:[/blue] {results['renamed']}",
                    f"[magenta]Embedded:[/magenta] {results['embedded']}",
                ]

                # Add removed count if any files were cleaned up
                if results.get("removed", 0) > 0:
                    summary_lines.append(
                        f"[red]Removed (missing):[/red] {results['removed']}"
                    )

                output.console.print(
                    Panel(
                        "\n".join(summary_lines),
                        title="[bold]Sync Summary[/bold]",
                        border_style="cyan",
                        box=box.ROUNDED,
                    )
                )

                # Show cleanup message if files were removed
                if results.get("removed", 0) > 0:
                    output.console.print(
                        f"[yellow]Cleaned up {results['removed']} paper(s) with missing files[/yellow]"
                    )

                if results["success"] > 0:
                    if results["embedded"] > 0:
                        output.console.print(
                            "[green]Ready! Type /list to see your papers, then ask questions naturally[/green]"
                        )
                    elif no_embed:
                        output.console.print(
                            "[yellow]Papers indexed but not embedded - questions may be limited[/yellow]"
                        )

                if results["failed"] > 0:
                    output.console.print(
                        f"[yellow]⚠ {results['failed']} paper(s) failed to process[/yellow]"
                    )

                # Suggest helpful next steps
                if results["success"] > 0 or results["duplicate"] > 0:
                    # Check if directory is set as default
                    default_dir = repo.get_config("default_papers_directory")

                    output.console.print()
                    if not default_dir or str(directory) != default_dir:
                        output.console.print(
                            "[blue]💡 Tip: Use /sync again to add new papers[/blue]"
                        )
                    else:
                        output.console.print("[blue]💡 Quick commands:[/blue]")
                        output.console.print(
                            "  [cyan]/list[/cyan]                   # View your papers"
                        )
                        output.console.print(
                            "  [cyan]/sync --watch[/cyan]           # Auto-process new papers"
                        )
                        output.console.print(
                            "  [cyan]Ask naturally![/cyan]          # e.g., 'tell me about paper 5'"
                        )

            except Exception as e:
                output.print_error(f"Sync failed: {e}")


@cli.command()
@click.option("--db", default="~/.MAVYN/MAVYN.db", help="Database path")
@click.option(
    "--remove", is_flag=True, help="Remove stale entries (default: just report)"
)
def verify(db: str, remove: bool):
    """Verify database integrity and clean up stale entries.

    Checks all papers in the database to see if their files still exist on disk.
    Use --remove to delete stale entries.
    """
    from rich.table import Table

    with Repository(db) as repo:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Checking files...", total=None)
            papers = repo.list_papers(limit=10000)
            progress.update(task, completed=True)

        if not papers:
            output.print_info("No papers found in database")
            return

        # Check each paper
        missing_papers = []
        for paper in papers:
            from pathlib import Path

            if not Path(paper.file_path).exists():
                missing_papers.append(paper)

        if not missing_papers:
            output.print_success(f"✓ All {len(papers)} papers have valid file paths!")
            return

        # Display missing papers
        table = Table(title=f"Missing Files ({len(missing_papers)} found)")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Title", style="white", max_width=40)
        table.add_column("File Path", style="red", max_width=50)

        for paper in missing_papers[:20]:  # Show first 20
            table.add_row(
                str(paper.id),
                (paper.title or "Untitled")[:40],
                str(paper.file_path)[-50:],
            )

        output.console.print(table)

        if len(missing_papers) > 20:
            output.print_info(
                f"\n... and {len(missing_papers) - 20} more missing files"
            )

        output.print_warning(
            f"\nFound {len(missing_papers)} stale database entries "
            f"(out of {len(papers)} total papers)"
        )

        # Remove if requested
        if remove:
            if not click.confirm("\nRemove these stale entries from the database?"):
                output.print_info("Operation cancelled")
                return

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=False,
            ) as progress:
                task = progress.add_task(
                    f"Removing {len(missing_papers)} stale entries...",
                    total=len(missing_papers),
                )

                for paper in missing_papers:
                    repo.delete_paper(paper.id)
                    progress.advance(task)

            output.print_success(
                f"✓ Removed {len(missing_papers)} stale entries from database"
            )
        else:
            output.print_info("\nRun with --remove to delete these entries")


def _display_notes_table(notes: list, repo: Repository, tip_message: str) -> None:
    """Helper function to display notes in a table.

    Args:
        notes: List of Note objects from database
        repo: Repository instance (unused, kept for consistency)
        tip_message: Tip message to show at bottom
    """
    from ..core.notes import NoteManager
    from rich.table import Table

    note_manager = NoteManager()

    # Display notes table
    table = Table(title=f"Saved Notes ({len(notes)} shown)", box=box.ROUNDED)
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Question", style="white", max_width=40)
    table.add_column("Papers", justify="center", style="yellow", no_wrap=True)
    table.add_column("Created", style="green", max_width=20)
    table.add_column("Provider", style="magenta", no_wrap=True)

    for note in notes:
        # Convert note to dict for formatting
        note_dict = {
            "id": note.id,
            "question": note.question,
            "answer": note.answer,
            "paper_ids": note.paper_ids,
            "created_at": note.created_at,
            "provider": note.provider,
        }

        preview = note_manager.format_note_preview(note_dict)

        # Format created_at
        created_str = (
            note.created_at.strftime("%Y-%m-%d %H:%M") if note.created_at else "N/A"
        )

        table.add_row(
            str(note.id),
            preview["question_preview"],
            str(preview["paper_count"]),
            created_str,
            preview["provider"] or "unknown",
        )

    output.console.print(table)
    output.print_info(tip_message)


def _save_note_from_session(repo: Repository, session_data: dict) -> None:
    """Helper function to save a note from session data.

    Args:
        repo: Database repository
        session_data: Session data dictionary with question, answer, papers, etc.
    """
    from ..core.notes import NoteManager
    from ..llm.providers import LLMRouter
    from ..llm.rate_limits import RateLimitStore
    from ..llm import prompts
    from ..llm.cache import LLMCache

    note_manager = NoteManager()

    try:
        # Extract session data
        question = session_data["question"]
        answer = session_data["answer"]
        paper_ids = session_data["paper_ids"]
        sources = session_data["sources"]
        context_papers = session_data["context_papers"]
        provider = session_data.get("provider", "unknown")
        model = session_data.get("model", "unknown")
        tokens_used = session_data.get("tokens_used", 0)

        # Validate note data
        try:
            note_manager.validate_note_data(question, answer, paper_ids)
        except ValueError as e:
            output.print_error(f"Invalid note data: {e}")
            return

        # Format note with LLM
        output.print_info("\n📝 Formatting and saving note...")

        llm_router = LLMRouter(rate_store=RateLimitStore(), cache_enabled=True)
        llm_cache = LLMCache(repo)

        if not llm_router.is_available():
            formatted_note = None
        else:
            try:
                # Build formatting prompt
                format_prompt = prompts.build_note_formatting_prompt(
                    question, answer, context_papers
                )

                # Generate formatted note (quietly, no progress bar for auto-save)
                format_response = llm_router.generate(
                    prompt=format_prompt,
                    max_tokens=800,
                    cache_lookup=llm_cache.get,
                    cache_store=llm_cache.store,
                )

                formatted_note = format_response.text if format_response else None

            except Exception:
                formatted_note = None

        # Prepare note data
        try:
            note_data = note_manager.prepare_note_data(
                question=question,
                answer=answer,
                paper_ids=paper_ids,
                sources=sources,
                formatted_note=formatted_note,
                provider=provider,
                model=model,
                tokens_used=tokens_used,
            )
        except (ValueError, IOError) as e:
            output.print_error(f"Failed to prepare note: {e}")
            return

        # Save to database
        note = repo.add_note(**note_data)

        if note:
            output.print_success(f"✓ Note saved successfully (ID: {note.id})")
            output.print_info(f"View with: lemma show -n {note.id}")
        else:
            output.print_error("Failed to save note to database")

    except KeyError as e:
        output.print_error(f"Invalid session data format: missing {e}")
    except Exception as e:
        output.print_error(f"Failed to save note: {e}")


@cli.group()
def note():
    """Manage Q&A notes for literature review."""
    pass


@note.command(name="save")
@click.option("--db", default="~/.MAVYN/MAVYN.db", help="Database path")
def note_save(db: str):
    """Save the last Q&A session as a formatted note.

    This command saves the most recent answer from 'lemma ask' as a
    literature review note with LLM formatting.
    """
    from ..core.notes import NoteManager
    from ..llm.providers import LLMRouter
    from ..llm.rate_limits import RateLimitStore
    from ..llm import prompts
    from ..llm.cache import LLMCache

    with Repository(db) as repo:
        note_manager = NoteManager()

        # Retrieve last Q&A session
        session_data = repo.get_config("last_qa_session")

        if not session_data:
            output.print_error(
                "No recent Q&A session found. Please run 'lemma ask <question>' first."
            )
            return

        try:
            # Extract session data
            question = session_data["question"]
            answer = session_data["answer"]
            paper_ids = session_data["paper_ids"]
            sources = session_data["sources"]
            context_papers = session_data["context_papers"]
            provider = session_data.get("provider", "unknown")
            model = session_data.get("model", "unknown")
            tokens_used = session_data.get("tokens_used", 0)

            # Validate note data
            try:
                note_manager.validate_note_data(question, answer, paper_ids)
            except ValueError as e:
                output.print_error(f"Invalid note data: {e}")
                return

            # Format note with LLM
            output.print_info("Formatting note with LLM...")

            llm_router = LLMRouter(rate_store=RateLimitStore(), cache_enabled=True)
            llm_cache = LLMCache(repo)

            if not llm_router.is_available():
                output.print_warning(
                    "No LLM available for formatting. Saving raw note without formatting."
                )
                formatted_note = None
            else:
                try:
                    # Build formatting prompt
                    format_prompt = prompts.build_note_formatting_prompt(
                        question, answer, context_papers
                    )

                    # Generate formatted note
                    with output.thinking_spinner():
                        format_response = llm_router.generate(
                            prompt=format_prompt,
                            max_tokens=800,
                            cache_lookup=llm_cache.get,
                            cache_store=llm_cache.store,
                        )

                    formatted_note = format_response.text if format_response else None

                except Exception as e:
                    output.print_warning(f"LLM formatting failed: {e}")
                    output.print_info("Saving raw note without formatting.")
                    formatted_note = None

            # Prepare note data
            try:
                note_data = note_manager.prepare_note_data(
                    question=question,
                    answer=answer,
                    paper_ids=paper_ids,
                    sources=sources,
                    formatted_note=formatted_note,
                    provider=provider,
                    model=model,
                    tokens_used=tokens_used,
                )
            except (ValueError, IOError) as e:
                output.print_error(f"Failed to prepare note: {e}")
                return

            # Save to database
            note = repo.add_note(**note_data)

            if note:
                output.print_success(f"✓ Note saved successfully (ID: {note.id})")
                output.print_info(f"View with: lemma note show {note.id}")
            else:
                output.print_error("Failed to save note to database")

        except KeyError as e:
            output.print_error(f"Invalid session data format: missing {e}")
        except Exception as e:
            output.print_error(f"Failed to save note: {e}")


@note.command(name="list")
@click.option("--db", default="~/.MAVYN/MAVYN.db", help="Database path")
@click.option("--limit", type=int, default=20, help="Number of notes to show")
def note_list(db: str, limit: int):
    """List all saved notes."""
    with Repository(db) as repo:
        # Fetch notes
        try:
            notes = repo.list_notes(limit=limit)
        except Exception as e:
            output.print_error(f"Failed to fetch notes: {e}")
            return

        if not notes:
            output.print_info(
                "No notes found. Save your first note with 'lemma note save'"
            )
            return

        _display_notes_table(notes, repo, "\nView full note: lemma note show <id>")


@note.command(name="show")
@click.argument("note_id", type=int)
@click.option("--db", default="~/.MAVYN/MAVYN.db", help="Database path")
def note_show(note_id: int, db: str):
    """Show a specific note by ID.

    NOTE_ID: The ID of the note to display
    """
    from ..core.notes import NoteManager

    with Repository(db) as repo:
        note_manager = NoteManager()

        # Validate note ID
        try:
            note_manager.validate_note_id(note_id)
        except ValueError as e:
            output.print_error(str(e))
            return

        # Fetch note
        try:
            note = repo.get_note_by_id(note_id)
        except Exception as e:
            output.print_error(f"Failed to fetch note: {e}")
            return

        if not note:
            output.print_error(f"Note with ID {note_id} not found")
            output.print_info("Use 'lemma note list' to see available notes")
            return

        # Convert note to dict for formatting
        note_dict = {
            "id": note.id,
            "question": note.question,
            "answer": note.answer,
            "formatted_note": note.formatted_note,
            "sources": note.sources,
            "provider": note.provider,
            "model": note.model,
            "tokens_used": note.tokens_used,
            "created_at": note.created_at,
        }

        # Format and display
        formatted_output = note_manager.format_note_display(note_dict)

        output.console.print(
            Panel(
                formatted_output,
                title=f"[bold cyan]Note #{note_id}[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )


# Wrapper functions for REPL use
def sync_command(**kwargs):
    """Wrapper for sync command to be called from REPL."""
    # Build arguments
    args = []
    if kwargs.get("directory"):
        args.append(str(kwargs["directory"]))
    if kwargs.get("watch"):
        args.append("--watch")
    if kwargs.get("set_default"):
        args.append("--set-default")
    if kwargs.get("no_rename"):
        args.append("--no-rename")
    if kwargs.get("no_embed"):
        args.append("--no-embed")
    # Call sync directly
    sync.callback(**kwargs)


def list_papers_command(**kwargs):
    """Wrapper for list command to be called from REPL."""
    list_papers.callback(**kwargs)


def ask_command(**kwargs):
    """Wrapper for ask command to be called from REPL."""
    ask.callback(**kwargs)


def main():
    """Entry point for the CLI."""
    import sys

    # Run first-time setup wizard if needed
    if is_first_run():
        run_setup_wizard(skip_if_configured=True)

    # If no arguments provided, start REPL
    if len(sys.argv) == 1:
        from .repl import start_repl

        start_repl(db_path="~/.MAVYN/MAVYN.db")
    else:
        cli()


if __name__ == "__main__":
    main()
