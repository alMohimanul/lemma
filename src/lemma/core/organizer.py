"""File organization with smart renaming and rollback capability."""
import re
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

from ..utils.logger import get_logger, log_exception

logger = get_logger(__name__)


class FileOrganizer:
    """Organize PDF files with pattern-based renaming and rollback."""

    DEFAULT_PATTERN = "{year}_{first_author}_{short_title}.pdf"

    def __init__(self, dry_run: bool = False):
        """Initialize organizer.

        Args:
            dry_run: If True, simulate operations without actually moving files
        """
        self.dry_run = dry_run

    def generate_filename(
        self, metadata: Dict[str, Any], pattern: str = DEFAULT_PATTERN
    ) -> str:
        """Generate a filename from paper metadata using a pattern.

        Args:
            metadata: Paper metadata dictionary
            pattern: Filename pattern with placeholders

        Returns:
            Generated filename

        Supported placeholders:
            {year} - Publication year
            {first_author} - First author's last name
            {authors} - All authors (shortened)
            {title} - Full title (sanitized)
            {short_title} - Title truncated to first 50 chars
            {doi} - DOI (sanitized)
            {arxiv_id} - arXiv ID
        """
        values = {
            "year": str(metadata.get("year", "unknown")),
            "first_author": self._extract_first_author(metadata.get("authors", "")),
            "authors": self._sanitize_authors(metadata.get("authors", "")),
            "title": self._sanitize_filename(metadata.get("title", "untitled")),
            "short_title": self._sanitize_filename(
                metadata.get("title", "untitled")[:50]
            ),
            "doi": self._sanitize_filename(metadata.get("doi", "")),
            "arxiv_id": (metadata.get("arxiv_id") or "").replace(".", "_"),
        }

        # Replace placeholders
        filename = pattern
        for key, value in values.items():
            placeholder = f"{{{key}}}"
            if placeholder in filename:
                filename = filename.replace(placeholder, value or "unknown")

        # Ensure .pdf extension
        if not filename.endswith(".pdf"):
            filename += ".pdf"

        return filename

    def rename_file(
        self,
        original_path: Path,
        new_filename: str,
        target_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Rename a file, optionally moving to a different directory.

        Args:
            original_path: Current file path
            new_filename: New filename
            target_dir: Optional target directory (default: same directory)

        Returns:
            New file path, or None if operation was skipped
        """
        original_path = Path(original_path).expanduser().resolve()

        if not original_path.exists():
            raise FileNotFoundError(f"File not found: {original_path}")

        # Determine target directory
        try:
            if target_dir:
                target_dir = Path(target_dir).expanduser().resolve()
                target_dir.mkdir(parents=True, exist_ok=True)
            else:
                target_dir = original_path.parent
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to create target directory {target_dir}: {e}")
            raise IOError(f"Cannot create directory {target_dir}: {e}") from e

        # Build new path
        new_path = target_dir / new_filename

        # Check if file already exists
        if new_path.exists() and new_path != original_path:
            # Generate unique filename
            new_path = self._make_unique_path(new_path)

        # Skip if same path
        if new_path == original_path:
            return None

        if self.dry_run:
            print(f"[DRY RUN] Would rename: {original_path} -> {new_path}")
            return new_path

        # Perform rename/move
        try:
            shutil.move(str(original_path), str(new_path))
            logger.info(f"Renamed file: {original_path.name} -> {new_path.name}")
            return new_path
        except (OSError, PermissionError, shutil.Error) as e:
            log_exception(logger, f"Failed to move {original_path} to {new_path}", e)
            raise IOError(f"Cannot move file {original_path} to {new_path}: {e}") from e

    def rollback_operation(self, original_path: Path, new_path: Path) -> bool:
        """Rollback a rename operation.

        Args:
            original_path: Original file path
            new_path: Current file path (after rename)

        Returns:
            True if rollback successful, False otherwise
        """
        new_path = Path(new_path).expanduser().resolve()
        original_path = Path(original_path).expanduser().resolve()

        if not new_path.exists():
            raise FileNotFoundError(f"File not found: {new_path}")

        if original_path.exists():
            # Original path already exists, cannot rollback
            return False

        if self.dry_run:
            print(f"[DRY RUN] Would rollback: {new_path} -> {original_path}")
            return True

        try:
            # Ensure parent directory exists
            original_path.parent.mkdir(parents=True, exist_ok=True)

            # Perform rollback
            shutil.move(str(new_path), str(original_path))
            logger.info(f"Rolled back file: {new_path.name} -> {original_path.name}")
            return True
        except (OSError, PermissionError, shutil.Error) as e:
            log_exception(
                logger, f"Failed to rollback {new_path} to {original_path}", e
            )
            return False

    def _extract_first_author(self, authors: str) -> str:
        """Extract first author's last name from author string.

        Args:
            authors: Author string (e.g., "Smith, J., Jones, K.")

        Returns:
            First author's last name
        """
        if not authors:
            return "unknown"

        # Try to extract last name from various formats
        # Format: "LastName, FirstName" or "FirstName LastName"
        parts = authors.split(",")[0].strip().split()

        if len(parts) >= 2:
            # Assume "FirstName LastName" format
            return parts[-1]
        elif len(parts) == 1:
            return parts[0]

        return "unknown"

    def _sanitize_authors(self, authors: str, max_authors: int = 2) -> str:
        """Sanitize authors string for filename.

        Args:
            authors: Author string
            max_authors: Maximum number of authors to include

        Returns:
            Sanitized author string
        """
        if not authors:
            return "unknown"

        # Split by comma, take first max_authors
        author_list = [a.strip() for a in authors.split(",")]
        selected = author_list[:max_authors]

        # Extract last names
        last_names = [self._extract_first_author(a) for a in selected]

        # Join with underscore
        result = "_".join(last_names)

        if len(author_list) > max_authors:
            result += "_et_al"

        return self._sanitize_filename(result)

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use in filenames.

        Args:
            name: String to sanitize

        Returns:
            Sanitized string safe for filenames
        """
        # Handle None or empty values
        if not name:
            return "unnamed"

        # Convert to string if not already
        name = str(name)

        # Remove or replace problematic characters
        name = re.sub(r'[<>:"/\\|?*]', "", name)  # Remove invalid chars
        name = re.sub(r"\s+", "_", name)  # Replace whitespace with underscore
        name = re.sub(
            r"[^\w\-_.]", "", name
        )  # Keep only alphanumeric, dash, underscore, dot
        name = re.sub(r"_+", "_", name)  # Collapse multiple underscores
        name = name.strip("._")  # Remove leading/trailing dots and underscores

        # Limit length (max filename length is usually 255)
        if len(name) > 200:
            name = name[:200]

        return name or "unnamed"

    def _make_unique_path(self, path: Path) -> Path:
        """Generate a unique file path by appending a counter.

        Args:
            path: Original path

        Returns:
            Unique path that doesn't exist
        """
        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix
        parent = path.parent

        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

    def preview_renames(
        self, papers: List[Dict[str, Any]], pattern: str = DEFAULT_PATTERN
    ) -> List[Dict[str, str]]:
        """Preview what files would be renamed to.

        Args:
            papers: List of paper dictionaries with metadata and file_path
            pattern: Filename pattern

        Returns:
            List of dicts with 'original', 'new', 'metadata' keys
        """
        previews = []

        for paper in papers:
            original_path = Path(paper["file_path"])
            new_filename = self.generate_filename(paper, pattern)
            new_path = original_path.parent / new_filename

            previews.append(
                {
                    "original": str(original_path),
                    "new": str(new_path),
                    "changed": original_path.name != new_filename,
                    "title": paper.get("title", "Untitled"),
                }
            )

        return previews
