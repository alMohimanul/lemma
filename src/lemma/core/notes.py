"""Note management for Q&A sessions with safety checks."""
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class NoteManager:
    """Manage saved Q&A notes with validation and safety checks."""

    MIN_FREE_SPACE_MB = 10  # Require at least 10MB free space
    MAX_NOTE_SIZE_MB = 5  # Maximum note size

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize note manager.

        Args:
            data_dir: Directory for lemma data (default: ~/.lemma)
        """
        if data_dir is None:
            data_dir = Path.home() / ".lemma"

        self.data_dir = Path(data_dir).expanduser().resolve()

        # Ensure data directory exists
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            logger.error(
                f"Failed to create data directory {self.data_dir}: {e}", exc_info=True
            )
            raise IOError(f"Cannot create data directory: {e}") from e

    def validate_note_data(
        self, question: str, answer: str, paper_ids: List[int]
    ) -> bool:
        """Validate note data before saving.

        Args:
            question: Question text
            answer: Answer text
            paper_ids: List of paper IDs

        Returns:
            True if valid, raises ValueError otherwise
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty")

        if not paper_ids:
            raise ValueError("At least one paper ID is required")

        if len(question) > 10000:
            raise ValueError("Question is too long (max 10,000 characters)")

        if len(answer) > 50000:
            raise ValueError("Answer is too long (max 50,000 characters)")

        # Check total size
        total_size = len(question) + len(answer)
        max_size_bytes = self.MAX_NOTE_SIZE_MB * 1024 * 1024
        if total_size > max_size_bytes:
            raise ValueError(
                f"Note content too large ({total_size / 1024 / 1024:.1f}MB), max {self.MAX_NOTE_SIZE_MB}MB"
            )

        return True

    def check_disk_space(self) -> bool:
        """Check if there's enough disk space.

        Returns:
            True if sufficient space available

        Raises:
            IOError: If insufficient disk space
        """
        try:
            stat = shutil.disk_usage(self.data_dir)
            free_mb = stat.free / (1024 * 1024)

            if free_mb < self.MIN_FREE_SPACE_MB:
                raise IOError(
                    f"Insufficient disk space: {free_mb:.1f}MB free, "
                    f"need at least {self.MIN_FREE_SPACE_MB}MB"
                )

            logger.debug(f"Disk space check passed: {free_mb:.1f}MB free")
            return True

        except OSError as e:
            logger.error(f"Failed to check disk space: {e}", exc_info=True)
            # Don't fail the operation, just warn
            logger.warning("Could not verify disk space, proceeding anyway")
            return True

    def validate_note_id(self, note_id: int) -> bool:
        """Validate note ID.

        Args:
            note_id: Note ID to validate

        Returns:
            True if valid

        Raises:
            ValueError: If invalid note ID
        """
        if not isinstance(note_id, int) or note_id <= 0:
            raise ValueError(f"Invalid note ID: {note_id}. Must be a positive integer.")
        return True

    def prepare_note_data(
        self,
        question: str,
        answer: str,
        paper_ids: List[int],
        sources: Optional[List[str]] = None,
        formatted_note: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tokens_used: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Prepare and validate note data for storage.

        Args:
            question: Question text
            answer: Answer text
            paper_ids: List of paper IDs used
            sources: Optional list of source strings
            formatted_note: Optional LLM-formatted note
            provider: LLM provider name
            model: Model name
            tokens_used: Token count

        Returns:
            Validated note data dictionary

        Raises:
            ValueError: If validation fails
            IOError: If insufficient disk space
        """
        # Validate inputs
        self.validate_note_data(question, answer, paper_ids)

        # Check disk space
        self.check_disk_space()

        # Prepare data
        note_data = {
            "question": question.strip(),
            "answer": answer.strip(),
            "paper_ids": json.dumps(paper_ids),
            "sources": json.dumps(sources) if sources else None,
            "formatted_note": formatted_note.strip() if formatted_note else None,
            "provider": provider,
            "model": model,
            "tokens_used": tokens_used,
        }

        logger.debug(
            f"Prepared note data: {len(paper_ids)} papers, "
            f"{len(question)} char question, {len(answer)} char answer"
        )

        return note_data

    def format_note_preview(
        self, note: Dict[str, Any], max_length: int = 100
    ) -> Dict[str, Any]:
        """Format a note for preview display.

        Args:
            note: Note dictionary
            max_length: Maximum length for preview text

        Returns:
            Formatted note dictionary with preview fields
        """
        question = note.get("question", "")
        answer = note.get("answer", "")

        # Parse paper_ids from JSON if string
        paper_ids_raw = note.get("paper_ids", "[]")
        if isinstance(paper_ids_raw, str):
            try:
                paper_ids = json.loads(paper_ids_raw)
            except json.JSONDecodeError:
                paper_ids = []
        else:
            paper_ids = paper_ids_raw

        return {
            "id": note.get("id"),
            "question_preview": (
                question[:max_length] + "..."
                if len(question) > max_length
                else question
            ),
            "answer_preview": (
                answer[:max_length] + "..." if len(answer) > max_length else answer
            ),
            "paper_count": len(paper_ids),
            "created_at": note.get("created_at"),
            "provider": note.get("provider"),
        }

    def format_note_display(self, note: Dict[str, Any]) -> str:
        """Format a note for full display.

        Args:
            note: Note dictionary

        Returns:
            Formatted string for display
        """
        lines = []

        lines.append(f"Note ID: {note.get('id')}")
        lines.append(f"Created: {note.get('created_at')}")
        lines.append("")

        lines.append("Question:")
        lines.append(note.get("question", ""))
        lines.append("")

        # Show formatted note if available, otherwise raw answer
        if formatted_note := note.get("formatted_note"):
            lines.append("Literature Review Note:")
            lines.append(formatted_note)
        else:
            lines.append("Answer:")
            lines.append(note.get("answer", ""))

        lines.append("")

        # Parse and display sources
        sources_raw = note.get("sources")
        if sources_raw:
            try:
                sources = (
                    json.loads(sources_raw)
                    if isinstance(sources_raw, str)
                    else sources_raw
                )
                lines.append("Sources:")
                for source in sources:
                    lines.append(f"  • {source}")
                lines.append("")
            except json.JSONDecodeError:
                pass

        # Metadata
        if provider := note.get("provider"):
            model = note.get("model", "")
            tokens = note.get("tokens_used", "")
            lines.append(f"Generated by: {provider} {model} ({tokens} tokens)")

        return "\n".join(lines)
