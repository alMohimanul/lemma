"""CLI commands for lemma paper manager."""
from pathlib import Path

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

# Load environment variables from .env file and ~/.lemma/.env
load_dotenv()  # Load from current directory
load_dotenv(Path.home() / ".lemma" / ".env")  # Load from config directory


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """lemma - Local-first paper manager.

    Manage your research papers with local semantic search and cloud LLM reasoning.
    """
    pass


@cli.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--recursive/--no-recursive", default=True, help="Scan subdirectories")
@click.option("--db", default="~/.lemma/lemma.db", help="Database path")
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
@click.option("--db", default="~/.lemma/lemma.db", help="Database path")
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
@click.option("--db", default="~/.lemma/lemma.db", help="Database path")
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
@click.option("--db", default="~/.lemma/lemma.db", help="Database path")
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
@click.option("--db", default="~/.lemma/lemma.db", help="Database path")
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
@click.option("--db", default="~/.lemma/lemma.db", help="Database path")
@click.option("--top-k", type=int, default=5, help="Number of papers to retrieve")
@click.option("--index-path", default="~/.lemma/search.index", help="FAISS index path")
@click.option("--save", "-s", is_flag=True, help="Automatically save answer as a note")
def ask(question: str, db: str, top_k: int, index_path: str, save: bool):
    """Ask a question across all papers (semantic search + LLM).

    QUESTION: Your question about the papers

    Note: Requires embeddings to be generated first. Run 'lemma embed' if needed.

    Use --save or -s to automatically save the answer as a formatted note.
    """
    from ..embeddings.encoder import EmbeddingEncoder
    from ..embeddings.search import SemanticSearchIndex
    from ..llm.providers import LLMRouter
    from ..llm.cache import LLMCache
    from ..llm import prompts
    from pathlib import Path

    with Repository(db) as repo:
        extractor = MetadataExtractor()

        # Check if FAISS index exists
        index_file = Path(index_path).expanduser()
        if not index_file.with_suffix(".faiss").exists():
            output.print_error("No embeddings found. Please run 'lemma embed' first.")
            return

        # Initialize components
        try:
            encoder = EmbeddingEncoder()
            search_index = SemanticSearchIndex(
                embedding_dim=encoder.embedding_dim, index_path=index_file
            )
            output.print_info(f"Loaded search index with {search_index.size()} vectors")
        except Exception as e:
            output.print_error(f"Failed to load search index: {e}")
            return

        # Check if index is empty
        if search_index.size() == 0:
            output.print_error("Search index is empty. Please run 'lemma embed' first.")
            return

        # Encode the question
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

        # Search for relevant papers
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("Searching papers...", total=None)
                top_papers = search_index.get_top_papers(query_vector, top_k=top_k)
                progress.update(task, completed=True)

            if not top_papers:
                output.print_warning("No relevant papers found for your question.")
                return

            paper_ids = [paper_id for paper_id, _ in top_papers]
            output.print_info(f"Found {len(paper_ids)} relevant papers")

        except Exception as e:
            output.print_error(f"Search failed: {e}")
            return

        # Retrieve paper details
        try:
            papers = repo.get_papers_by_ids(paper_ids)

            # Build context for LLM
            context_papers = []
            for paper in papers:
                # Extract text (use abstract if available, otherwise extract from PDF)
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
                        "text": paper_text,
                    }
                )

        except Exception as e:
            output.print_error(f"Failed to retrieve papers: {e}")
            return

        # Initialize LLM router and cache
        llm_router = LLMRouter(cache_enabled=True)
        llm_cache = LLMCache(repo)

        # Check if LLM is available
        if not llm_router.is_available():
            output.print_error(
                "No LLM providers available. Please set up API keys:\n"
                "  GROQ_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY"
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
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("Generating answer...", total=None)

                # Build prompt
                prompt = prompts.build_qa_prompt(question, context_papers)

                # Generate response with cache
                response = llm_router.generate(
                    prompt=prompt,
                    cache_lookup=llm_cache.get,
                    cache_store=llm_cache.store,
                )

                progress.update(task, completed=True)

            if not response:
                output.print_error("Failed to generate answer from LLM")
                return

            # Display answer
            sources = [
                f"[{p['id']}] {p['title']} ({p['year']})" for p in context_papers
            ]
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
                    "\n💡 Tip: Save this answer as a note with 'lemma ask ... --save' or 'lemma note save'"
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
@click.option("--db", default="~/.lemma/lemma.db", help="Database path")
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
@click.option("--db", default="~/.lemma/lemma.db", help="Database path")
@click.option(
    "--force", is_flag=True, help="Re-embed papers that already have embeddings"
)
@click.option("--index-path", default="~/.lemma/search.index", help="FAISS index path")
@click.option(
    "--checkpoint-every",
    type=int,
    default=10,
    help="Save index checkpoint every N papers (default: 10)",
)
def embed(db: str, force: bool, index_path: str, checkpoint_every: int):
    """Generate embeddings for all papers.

    This enables semantic search and the 'ask' command.

    Features:
    - Automatic resume: Only processes papers not yet embedded
    - Checkpointing: Saves progress every N papers (default: 10)
    - Crash recovery: Re-run command to continue from last checkpoint
    """
    from ..embeddings.encoder import EmbeddingEncoder
    from ..embeddings.search import SemanticSearchIndex
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
            f"(resumes from previous run, skips completed papers)"
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

        # Process each paper
        success_count = 0
        error_count = 0

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
                    # Update status to processing
                    repo.update_paper_metadata(
                        paper.id, {"embedding_status": "processing"}
                    )

                    # Extract full text
                    paper_path = Path(paper.file_path)
                    if not paper_path.exists():
                        raise FileNotFoundError(f"PDF not found: {paper_path}")

                    full_text = extractor.extract_full_text(paper_path)

                    if not full_text or len(full_text.strip()) < 100:
                        raise ValueError("Insufficient text extracted from PDF")

                    # Chunk and encode text
                    chunks = encoder.chunk_text(full_text, chunk_size=500, overlap=50)

                    if not chunks:
                        raise ValueError("No chunks generated from text")

                    # Encode all chunks in batch
                    chunk_embeddings = encoder.encode_batch(chunks, batch_size=32)

                    # Ensure chunk_embeddings is a 2D array
                    if chunk_embeddings.ndim == 1:
                        chunk_embeddings = chunk_embeddings.reshape(1, -1)

                    num_chunks = len(chunks)
                    num_embeddings = len(chunk_embeddings)

                    if num_chunks != num_embeddings:
                        raise ValueError(
                            f"Chunk count mismatch: {num_chunks} chunks but {num_embeddings} embeddings"
                        )

                    # Store embeddings in database
                    for i in range(num_chunks):
                        chunk_text = chunks[i]
                        embedding_vec = chunk_embeddings[i]

                        repo.add_embedding(
                            paper_id=paper.id,
                            text_content=chunk_text[:1000],  # Limit stored text
                            embedding_vector=embedding_vec.tolist(),
                            chunk_index=i,
                            model_name=encoder.model_name,
                        )

                    # Add to FAISS index
                    search_index.add(
                        embeddings=chunk_embeddings,
                        paper_id=paper.id,
                        chunk_indices=list(range(num_chunks)),
                    )

                    # Update status to completed
                    repo.update_paper_metadata(
                        paper.id, {"embedding_status": "completed"}
                    )

                    # Log success
                    repo.log_operation(
                        operation="embed",
                        status="success",
                        paper_id=paper.id,
                        details={"chunks": num_chunks, "model": encoder.model_name},
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

        # Print summary
        output.print_info(
            f"\nEmbedding complete: {success_count} succeeded, {error_count} failed"
        )
        if success_count > 0:
            output.print_success("You can now use 'lemma ask' to query your papers!")


@cli.command()
@click.option("--db", default="~/.lemma/lemma.db", help="Database path")
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
@click.option("--db", default="~/.lemma/lemma.db", help="Database path")
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

        llm_router = LLMRouter(cache_enabled=True)
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
@click.option("--db", default="~/.lemma/lemma.db", help="Database path")
def note_save(db: str):
    """Save the last Q&A session as a formatted note.

    This command saves the most recent answer from 'lemma ask' as a
    literature review note with LLM formatting.
    """
    from ..core.notes import NoteManager
    from ..llm.providers import LLMRouter
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

            llm_router = LLMRouter(cache_enabled=True)
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
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        transient=True,
                    ) as progress:
                        task = progress.add_task(
                            "Generating formatted note...", total=None
                        )
                        format_response = llm_router.generate(
                            prompt=format_prompt,
                            max_tokens=800,
                            cache_lookup=llm_cache.get,
                            cache_store=llm_cache.store,
                        )
                        progress.update(task, completed=True)

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
@click.option("--db", default="~/.lemma/lemma.db", help="Database path")
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
@click.option("--db", default="~/.lemma/lemma.db", help="Database path")
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


def main():
    """Entry point for the CLI."""
    # Run first-time setup wizard if needed
    if is_first_run():
        run_setup_wizard(skip_if_configured=True)

    cli()


if __name__ == "__main__":
    main()
