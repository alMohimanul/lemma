"""First-run setup wizard for lemma."""
import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

console = Console()


def get_config_dir() -> Path:
    """Get the lemma configuration directory."""
    config_dir = Path.home() / ".lemma"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_env_file() -> Path:
    """Get the path to the .env file."""
    return get_config_dir() / ".env"


def is_first_run() -> bool:
    """Check if this is the first time lemma is run."""
    env_file = get_env_file()
    return not env_file.exists()


def has_api_keys() -> bool:
    """Check if API keys are configured (env vars or .env file)."""
    # Check environment variables
    if os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return True

    # Check .env file
    env_file = get_env_file()
    if env_file.exists():
        content = env_file.read_text()
        if "GROQ_API_KEY=" in content or "GEMINI_API_KEY=" in content:
            # Check if keys have actual values (not placeholders)
            for line in content.split("\n"):
                if line.startswith("GROQ_API_KEY=") or line.startswith(
                    "GEMINI_API_KEY="
                ):
                    key_value = line.split("=", 1)[1].strip()
                    if key_value and not key_value.startswith("your_"):
                        return True

    return False


def save_api_key(provider: str, api_key: str) -> None:
    """Save API key to ~/.lemma/.env file."""
    env_file = get_env_file()

    # Read existing content if file exists
    existing_content = []
    if env_file.exists():
        existing_content = env_file.read_text().split("\n")

    # Update or add the key
    key_name = f"{provider.upper()}_API_KEY"
    key_line = f"{key_name}={api_key}"

    # Check if key already exists
    updated = False
    for i, line in enumerate(existing_content):
        if line.startswith(f"{key_name}="):
            existing_content[i] = key_line
            updated = True
            break

    if not updated:
        existing_content.append(key_line)

    # Write back
    env_file.write_text("\n".join(existing_content))
    console.print(f"✓ API key saved to {env_file}", style="green")


def run_setup_wizard(skip_if_configured: bool = True) -> None:
    """Run the interactive setup wizard."""
    # Skip if already configured (unless explicitly requested)
    if skip_if_configured and has_api_keys():
        return

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Welcome to Lemma! 📚[/bold cyan]\n\n"
            "Lemma is a local-first paper manager with AI-powered features.\n\n"
            "[dim]Let's set up your API keys for AI features (optional).[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # Ask if user wants to configure API keys
    if not Confirm.ask(
        "Would you like to set up API keys for AI features now?", default=True
    ):
        console.print("\n[yellow]Skipping API key setup.[/yellow]")
        console.print(
            "You can configure API keys later by running: [cyan]lemma setup[/cyan]"
        )
        console.print("or by setting environment variables manually.\n")

        # Create empty .env file to mark as configured
        get_env_file().touch()
        return

    console.print()
    console.print("[bold]Choose your LLM provider:[/bold]")
    console.print("  1. [green]Groq[/green] (recommended - fast & generous free tier)")
    console.print("     Get your key: https://console.groq.com/")
    console.print("  2. [blue]Google Gemini[/blue] (alternative option)")
    console.print("     Get your key: https://makersuite.google.com/")
    console.print("  3. [dim]Skip for now[/dim]")
    console.print()

    choice = Prompt.ask("Enter choice", choices=["1", "2", "3"], default="1")

    if choice == "3":
        console.print("\n[yellow]Skipping API key setup.[/yellow]")
        console.print(
            "You can add API keys later by running: [cyan]lemma setup[/cyan]\n"
        )
        get_env_file().touch()
        return

    # Get API key
    provider = "groq" if choice == "1" else "gemini"
    provider_name = "Groq" if choice == "1" else "Google Gemini"

    console.print()
    api_key = Prompt.ask(f"Enter your {provider_name} API key", password=True)

    if not api_key or api_key.strip() == "":
        console.print("\n[red]No API key provided. Setup cancelled.[/red]\n")
        get_env_file().touch()
        return

    # Save the key
    save_api_key(provider, api_key.strip())

    # Ask if they want to add a backup provider
    console.print()
    if Confirm.ask("Would you like to add a backup provider?", default=False):
        backup_provider = "gemini" if provider == "groq" else "groq"
        backup_name = "Google Gemini" if backup_provider == "gemini" else "Groq"

        backup_key = Prompt.ask(f"Enter your {backup_name} API key", password=True)
        if backup_key and backup_key.strip():
            save_api_key(backup_provider, backup_key.strip())

    console.print()
    console.print("[bold green]✓ Setup complete![/bold green]")
    console.print(f"Your API keys are saved in: [cyan]{get_env_file()}[/cyan]")
    console.print()


def show_setup_help() -> None:
    """Show help for manual API key setup."""
    console.print()
    console.print(
        Panel.fit(
            "[bold]How to set up API keys manually:[/bold]\n\n"
            "1. Get a free API key from:\n"
            "   • Groq: https://console.groq.com/\n"
            "   • Gemini: https://makersuite.google.com/\n\n"
            f"2. Add to [cyan]{get_env_file()}[/cyan]:\n"
            "   GROQ_API_KEY=your_key_here\n\n"
            "3. Or set environment variable:\n"
            "   export GROQ_API_KEY=your_key_here",
            border_style="blue",
        )
    )
    console.print()
