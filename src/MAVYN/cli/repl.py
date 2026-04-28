"""Interactive REPL for MAVYN - Natural Language Research Assistant."""
from pathlib import Path
from typing import List, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich import box
from rich.markdown import Markdown

_HISTORY_FILE = Path.home() / ".MAVYN" / ".history"


def _setup_readline() -> bool:
    """Enable arrow-key navigation and persistent history via readline.

    Returns True if readline is available (Unix/macOS), False on Windows.
    """
    try:
        import readline

        if _HISTORY_FILE.exists():
            readline.read_history_file(str(_HISTORY_FILE))
        readline.set_history_length(500)
        return True
    except (ImportError, OSError):
        return False


def _save_readline_history() -> None:
    try:
        import readline

        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(str(_HISTORY_FILE))
    except (ImportError, OSError):
        pass


console = Console()


class MAVYNRepl:
    """Interactive REPL for MAVYN."""

    def __init__(self, db_path: str = "~/.MAVYN/MAVYN.db"):
        """Initialize the REPL.

        Args:
            db_path: Path to the database file
        """
        self.db_path = db_path
        self.running = False
        self.history: List[str] = []
        self.session_id: Optional[str] = None
        self.conversation_context: List[dict[str, Any]] = []  # Recent Q&A for context

    def print_welcome(self):
        """Print welcome message."""
        welcome_text = """[bold cyan]Welcome to MAVYN[/bold cyan] 🤖

[dim]Commands:[/dim]  /sync ~/Papers  |  /list  |  /clear  |  /help  |  /exit
[dim]Example:[/dim]   tell me about paper 5

Type [cyan]/help[/cyan] for more information."""
        console.print(
            Panel(welcome_text, border_style="cyan", box=box.ROUNDED, padding=(1, 2))
        )

    def print_help(self):
        """Print help message."""
        help_text = """
# MAVYN Commands

## Setup & Management
- `/sync [directory]` - Scan, rename, and embed papers from a directory
- `/sync --watch` - Continuously monitor directory for new papers
- `/list` - Show all indexed papers with IDs
- `/clear` - Clear the terminal screen

## Asking Questions
No special command needed! Just type naturally:
- "tell me about paper 5"
- "summarize the methodology of paper 3"
- "compare papers 1, 4, and 7"
- "find papers similar to paper 2"
- "what are the key contributions in paper 6?"

## Other Commands
- `/help` - Show this help message
- `/model` - Show LLM model status and rate limit cooldowns
- `/exit` or Ctrl+D - Exit MAVYN

## Tips
- Papers are automatically numbered when you use `/list`
- Reference papers by their ID number in your questions
- All processing happens locally - your papers never leave your machine
- Set up API keys for AI features with environment variables (GROQ_API_KEY, etc.)
        """
        console.print(
            Panel(
                Markdown(help_text),
                border_style="blue",
                box=box.ROUNDED,
                title="[bold blue]Help[/bold blue]",
            )
        )

    def handle_slash_command(self, command: str) -> bool:
        """Handle slash commands.

        Args:
            command: The command string (without leading /)

        Returns:
            True if should continue REPL, False if should exit
        """
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "exit" or cmd == "quit":
            console.print("[yellow]Goodbye![/yellow]")
            return False

        elif cmd == "help":
            self.print_help()
            return True

        elif cmd == "clear":
            console.clear()
            return True

        elif cmd == "sync":
            self.handle_sync(args)
            return True

        elif cmd == "list":
            self.handle_list()
            return True

        elif cmd == "model":
            self.handle_model()
            return True

        else:
            console.print(f"[red]Unknown command: /{cmd}[/red]")
            console.print("[yellow]Type /help to see available commands[/yellow]")
            return True

    def handle_sync(self, args: str):
        """Handle /sync command."""
        from .commands import sync_command

        # Parse arguments
        args_list = args.split() if args else []
        directory = Path(args_list[0]).expanduser() if args_list else None
        watch = "--watch" in args_list

        try:
            sync_command(
                directory=directory,
                db=self.db_path,
                watch=watch,
                set_default=False,
                no_rename=False,
                rename_pattern="{year}_{first_author}_{short_title}.pdf",
                no_embed=False,
                strategy="hybrid",
                index_path="~/.MAVYN/search.index",
                scan_interval=60,
                recursive=True,
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Sync interrupted[/yellow]")
        except Exception as e:
            console.print(f"[red]Sync failed: {e}[/red]")

    def handle_list(self):
        """Handle /list command."""
        from .commands import list_papers_command

        try:
            list_papers_command(
                limit=50, offset=0, sort_by="indexed_at", db=self.db_path
            )
        except Exception as e:
            console.print(f"[red]Failed to list papers: {e}[/red]")

    def handle_model(self):
        """Handle /model command — show Groq model status and rate limit cooldowns."""
        from rich.table import Table
        from rich import box as rbox
        from ..llm.rate_limits import RateLimitStore
        from ..llm.providers import LLMRouter

        store = RateLimitStore()
        all_models = LLMRouter.HEAVY_MODELS

        table = Table(box=rbox.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Model", style="white")
        table.add_column("Tier", style="dim")
        table.add_column("Status", justify="center")
        table.add_column("Cooldown", justify="left")

        for model in all_models:
            tier = "heavy + light" if model == "openai/gpt-oss-120b" else "light"
            display = model if len(model) <= 38 else model[:35] + "..."
            cooldown = store.cooldown_display(model)
            if cooldown == "Available":
                status = "[green]✓ Available[/green]"
                cd_text = "[dim]—[/dim]"
            elif cooldown.startswith("RPM"):
                status = "[yellow]⏳ RPM[/yellow]"
                cd_text = f"[yellow]{cooldown}[/yellow]"
            else:
                status = "[red]✗ Daily[/red]"
                cd_text = f"[red]{cooldown}[/red]"
            table.add_row(display, tier, status, cd_text)

        console.print(table)

    def handle_natural_language(self, query: str):
        """Handle natural language query.

        Args:
            query: The user's natural language question
        """
        from .commands import ask_command
        import re

        try:
            # Extract paper IDs from question (e.g., "paper 5", "papers 3 and 7")
            paper_ids = re.findall(r"paper\s+(\d+)", query.lower())
            paper_ids = [int(pid) for pid in paper_ids]

            # TODO: Get the actual answer from ask_command
            # For now, ask_command doesn't return the answer, it just prints it
            # We'll need to modify ask_command to return the response
            ask_command(
                question=query,
                db=self.db_path,
                top_k=5,
                index_path="~/.MAVYN/search.index",
                save=False,
                arxiv_cli=False,
                no_arxiv_cli=False,
            )

            # TODO: Save conversation turn to database
            # This requires modifying ask_command to return response data
            # For now, we'll implement this in Phase 2

        except KeyboardInterrupt:
            console.print("\n[yellow]Query interrupted[/yellow]")
        except Exception as e:
            console.print(f"[red]Query failed: {e}[/red]")

    def run(self):
        """Run the REPL loop."""
        from ..db.repository import Repository

        self.running = True

        # Create a new session
        with Repository(self.db_path) as repo:
            self.session_id = repo.create_session()
            if not self.session_id:
                console.print(
                    "[yellow]Warning: Could not create session. Context tracking disabled.[/yellow]"
                )

        _setup_readline()
        self.print_welcome()

        try:
            while self.running:
                try:
                    # Get user input.
                    # Use plain input() with readline-safe ANSI codes (\001/\002 wrap
                    # invisible sequences so readline measures the prompt width correctly,
                    # preventing backspace from eating into the prompt text).
                    _prompt = "\n\001\033[1;36m\002MAVYN>\001\033[0m\002 "
                    user_input = input(_prompt).strip()

                    # Skip empty input
                    if not user_input:
                        continue

                    # Add to history
                    self.history.append(user_input)

                    # Handle commands
                    if user_input.startswith("/"):
                        command = user_input[1:]
                        should_continue = self.handle_slash_command(command)
                        if not should_continue:
                            break
                    else:
                        # Handle as natural language query
                        self.handle_natural_language(user_input)

                except EOFError:
                    # Ctrl+D pressed
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break

                except KeyboardInterrupt:
                    # Ctrl+C pressed
                    console.print("\n[yellow]Use /exit to quit[/yellow]")
                    continue

        finally:
            self.running = False
            _save_readline_history()
            if self.session_id:
                from ..db.repository import Repository

                with Repository(self.db_path) as repo:
                    repo.end_session(self.session_id)


def start_repl(db_path: str = "~/.MAVYN/MAVYN.db"):
    """Start the MAVYN REPL.

    Args:
        db_path: Path to the database file
    """
    repl = MAVYNRepl(db_path=db_path)
    repl.run()
