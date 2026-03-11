"""PDF file scanner with deduplication."""
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScannedFile:
    """Represents a scanned PDF file."""

    path: Path
    file_hash: str
    file_size: int
    exists: bool = True


class PDFScanner:
    """Scans directories for PDF files and computes content hashes."""

    CHUNK_SIZE = 8192  # 8KB chunks for hashing

    def __init__(self, ignore_patterns: Optional[List[str]] = None):
        """Initialize scanner.

        Args:
            ignore_patterns: List of glob patterns to ignore (e.g., ['*_draft.pdf'])
        """
        self.ignore_patterns = ignore_patterns or []

    def scan_directory(
        self, directory: Path, recursive: bool = True
    ) -> List[ScannedFile]:
        """Scan a directory for PDF files.

        Args:
            directory: Directory path to scan
            recursive: Whether to scan subdirectories

        Returns:
            List of ScannedFile objects
        """
        directory = Path(directory).expanduser().resolve()

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        # Find all PDFs
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(directory.glob(pattern))

        # Filter out ignored patterns
        pdf_files = [f for f in pdf_files if not self._should_ignore(f)]

        # Scan each file
        scanned = []
        for pdf_path in pdf_files:
            try:
                scanned_file = self.scan_file(pdf_path)
                scanned.append(scanned_file)
            except Exception as e:
                # Log error but continue scanning
                print(f"Error scanning {pdf_path}: {e}")
                continue

        return scanned

    def scan_file(self, file_path: Path) -> ScannedFile:
        """Scan a single PDF file and compute its hash.

        Args:
            file_path: Path to PDF file

        Returns:
            ScannedFile object with hash and metadata
        """
        file_path = Path(file_path).expanduser().resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        # Compute SHA256 hash
        file_hash = self.compute_hash(file_path)
        file_size = file_path.stat().st_size

        return ScannedFile(
            path=file_path,
            file_hash=file_hash,
            file_size=file_size,
        )

    def compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            Hex-encoded SHA256 hash

        Raises:
            IOError: If file cannot be read
        """
        sha256 = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(self.CHUNK_SIZE):
                    sha256.update(chunk)
        except (OSError, IOError, PermissionError) as e:
            logger.error(f"Failed to compute hash for {file_path}: {e}")
            raise IOError(f"Cannot read file {file_path}: {e}") from e

        return sha256.hexdigest()

    def _should_ignore(self, file_path: Path) -> bool:
        """Check if a file matches any ignore patterns.

        Args:
            file_path: Path to check

        Returns:
            True if file should be ignored
        """
        for pattern in self.ignore_patterns:
            if file_path.match(pattern):
                return True
        return False

    def find_duplicates(
        self, scanned_files: List[ScannedFile]
    ) -> dict[str, List[Path]]:
        """Find duplicate files by content hash.

        Args:
            scanned_files: List of scanned files

        Returns:
            Dictionary mapping hash to list of file paths with that hash
        """
        hash_map: dict[str, List[Path]] = {}

        for scanned in scanned_files:
            if scanned.file_hash not in hash_map:
                hash_map[scanned.file_hash] = []
            hash_map[scanned.file_hash].append(scanned.path)

        # Return only duplicates (hash with >1 file)
        return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


def scan_and_deduplicate(
    directory: Path, recursive: bool = True
) -> Tuple[List[ScannedFile], dict[str, List[Path]]]:
    """Convenience function to scan directory and find duplicates.

    Args:
        directory: Directory to scan
        recursive: Whether to scan subdirectories

    Returns:
        Tuple of (all scanned files, duplicates dictionary)
    """
    scanner = PDFScanner()
    scanned = scanner.scan_directory(directory, recursive=recursive)
    duplicates = scanner.find_duplicates(scanned)

    return scanned, duplicates
