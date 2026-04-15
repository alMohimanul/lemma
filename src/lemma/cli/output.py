"""Output formatting helpers for CLI."""
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[red]✗[/red] {message}", style="red")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}", style="yellow")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


def print_paper_table(papers: List[dict]) -> None:
    """Print papers in a formatted table.

    Args:
        papers: List of paper dictionaries with id, title, authors, year, etc.
    """
    if not papers:
        print_info("No papers found")
        return

    table = Table(title="Papers", box=box.ROUNDED, show_lines=False)

    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Title", style="white", max_width=50)
    table.add_column("Authors", style="green", max_width=30)
    table.add_column("Year", justify="center", style="yellow", no_wrap=True)
    table.add_column("Status", justify="center", style="magenta", no_wrap=True)

    for paper in papers:
        table.add_row(
            str(paper.get("id", "")),
            paper.get("title", "Untitled")[:50],
            paper.get("authors", "Unknown")[:30] if paper.get("authors") else "Unknown",
            str(paper.get("year", "-")),
            paper.get("embedding_status", "pending"),
        )

    console.print(table)


def print_paper_details(paper: dict) -> None:
    """Print detailed information about a single paper.

    Args:
        paper: Paper dictionary with metadata
    """
    lines = []

    lines.append(f"[cyan bold]ID:[/cyan bold] {paper.get('id', 'N/A')}")

    if title := paper.get("title"):
        lines.append(f"[cyan bold]Title:[/cyan bold] {title}")

    if authors := paper.get("authors"):
        lines.append(f"[cyan bold]Authors:[/cyan bold] {authors}")

    if year := paper.get("year"):
        lines.append(f"[cyan bold]Year:[/cyan bold] {year}")

    if publication := paper.get("publication"):
        lines.append(f"[cyan bold]Publication:[/cyan bold] {publication}")

    if doi := paper.get("doi"):
        lines.append(f"[cyan bold]DOI:[/cyan bold] {doi}")

    if arxiv_id := paper.get("arxiv_id"):
        lines.append(f"[cyan bold]arXiv:[/cyan bold] {arxiv_id}")

    if file_path := paper.get("file_path"):
        lines.append(f"[cyan bold]File:[/cyan bold] {file_path}")

    if file_size := paper.get("file_size"):
        size_mb = file_size / (1024 * 1024)
        lines.append(f"[cyan bold]Size:[/cyan bold] {size_mb:.2f} MB")

    if indexed_at := paper.get("indexed_at"):
        lines.append(f"[cyan bold]Indexed:[/cyan bold] {indexed_at}")

    if abstract := paper.get("abstract"):
        lines.append(f"\n[cyan bold]Abstract:[/cyan bold]\n{abstract}")

    panel = Panel(
        "\n".join(lines),
        title="[bold]Paper Details[/bold]",
        border_style="blue",
        box=box.ROUNDED,
    )

    console.print(panel)


def print_scan_results(total: int, new: int, duplicates: int, errors: int) -> None:
    """Print scan results summary.

    Args:
        total: Total PDFs found
        new: New papers added
        duplicates: Duplicate files skipped
        errors: Number of errors
    """
    lines = [
        f"Total PDFs found: [cyan]{total}[/cyan]",
        f"New papers added: [green]{new}[/green]",
        f"Duplicates skipped: [yellow]{duplicates}[/yellow]",
    ]

    if errors > 0:
        lines.append(f"Errors: [red]{errors}[/red]")

    panel = Panel(
        "\n".join(lines),
        title="[bold]Scan Results[/bold]",
        border_style="green" if errors == 0 else "yellow",
        box=box.ROUNDED,
    )

    console.print(panel)


def print_search_results(papers: List[dict], query: str) -> None:
    """Print search results.

    Args:
        papers: List of matching papers
        query: Search query
    """
    if not papers:
        print_info(f'No papers found matching "{query}"')
        return

    console.print(f'\n[bold]Search results for:[/bold] "{query}"\n')
    print_paper_table(papers)


def print_answer(
    question: str, answer: str, sources: Optional[List[str]] = None
) -> None:
    """Print an answer to a question with optional sources.

    Args:
        question: The question asked
        answer: The answer
        sources: Optional list of source paper IDs/titles
    """
    # Print question
    console.print(
        Panel(
            question,
            title="[bold cyan]Question[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    # Print answer
    console.print(
        Panel(
            answer,
            title="[bold green]Answer[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        )
    )

    # Print sources if provided
    if sources:
        console.print("\n[bold]Sources:[/bold]")
        for source in sources:
            console.print(f"  • {source}")


def print_arxiv_related(entries: List[dict]) -> None:
    """Print arXiv suggestions as a copy-friendly table (API-grounded links)."""
    if not entries:
        return

    table = Table(
        title="[bold]arXiv suggestions (API results)[/bold]",
        box=box.ROUNDED,
        show_lines=False,
    )
    table.add_column("ID", style="cyan", no_wrap=True, max_width=14)
    table.add_column("Title", style="white", max_width=42)
    table.add_column("Link", style="blue", max_width=36)

    for e in entries:
        aid = e.get("arxiv_id") or e.get("arxiv_id_norm") or "—"
        title = (e.get("title") or "Untitled")[:80]
        link = e.get("abs_url") or ""
        table.add_row(str(aid), title, link)

    console.print()
    console.print(table)


def print_comparison_results(result_dict: dict, papers: list) -> None:
    """Display paper comparison results in rich terminal format.

    Args:
        result_dict: Comparison result dictionary from ComparisonResult.to_dict()
        papers: List of Paper objects being compared
    """
    from rich.table import Table
    from datetime import datetime

    # Header with papers being compared
    paper_list_text = "\n".join(
        [
            f"[{p.id}] {p.title or 'Untitled'} ({p.authors or 'Unknown'}, {p.year or 'N/A'})"
            for p in papers
        ]
    )

    # Check if from cache
    cache_indicator = ""
    if result_dict.get("from_cache"):
        cached_at = result_dict.get("cached_at")
        if cached_at:
            try:
                dt = datetime.fromisoformat(cached_at)
                cache_indicator = f"\n\n💾 [dim]Using cached comparison from {dt.strftime('%Y-%m-%d %H:%M')}[/dim]"
            except Exception:
                cache_indicator = "\n\n💾 [dim]Using cached comparison[/dim]"

    header_text = f"""[bold cyan]Papers Compared ({len(papers)} total):[/bold cyan]
{paper_list_text}{cache_indicator}"""

    console.print(
        Panel(
            header_text,
            title="[bold]Paper Comparison[/bold]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    # Comparison type indicator
    comp_type = result_dict.get("comparison_type", "whole")
    section_name = result_dict.get("section_name")

    if comp_type == "section" and section_name:
        console.print(
            f"\n[bold magenta]Section Comparison:[/bold magenta] {section_name}\n"
        )
    else:
        console.print("\n[bold magenta]Whole Paper Comparison[/bold magenta]\n")

    # Overall summary
    summary = result_dict.get("summary", "No summary available")
    console.print(
        Panel(
            summary,
            title="[bold green]Summary[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        )
    )

    # Section-by-section comparisons (for whole-paper comparisons)
    section_comparisons = result_dict.get("section_comparisons")
    if section_comparisons:
        console.print("\n[bold yellow]Section-by-Section Analysis:[/bold yellow]\n")

        for section, comparison_text in section_comparisons.items():
            # Create table for side-by-side view
            table = Table(title=f"📄 {section}", box=box.ROUNDED, show_header=True)
            table.add_column("Section", style="cyan", width=15)
            table.add_column("Comparison", style="white", width=70)

            table.add_row(
                section,
                comparison_text[:500] + "..."
                if len(comparison_text) > 500
                else comparison_text,
            )

            console.print(table)
            console.print()

    # Full comparison details (for section comparisons)
    details = result_dict.get("details", {})
    if comp_type == "section" and "full_comparison" in details:
        full_text = details["full_comparison"]
        console.print(
            Panel(
                full_text,
                title="[bold blue]Detailed Comparison[/bold blue]",
                border_style="blue",
                box=box.ROUNDED,
            )
        )
    elif comp_type == "whole" and "final_synthesis" in details:
        # For whole-paper, show the synthesis
        synthesis = details["final_synthesis"]
        console.print(
            Panel(
                synthesis,
                title="[bold blue]Comprehensive Synthesis[/bold blue]",
                border_style="blue",
                box=box.ROUNDED,
            )
        )

    # Provider info
    provider = result_dict.get("provider", "unknown")
    model = result_dict.get("model", "unknown")
    tokens = result_dict.get("tokens_used", 0)

    if provider != "cache" and not result_dict.get("from_cache"):
        console.print(
            f"\n[dim]Provider: {provider} | Model: {model} | Tokens: {tokens}[/dim]"
        )

    # Tip for saving
    if not result_dict.get("from_cache"):
        console.print(
            "\n[dim]💡 This comparison is now cached for future reference[/dim]"
        )
