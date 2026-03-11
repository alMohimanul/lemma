"""Input validation utilities for lemma."""
from pathlib import Path
from typing import List, Optional
import click


def validate_paper_id(paper_id: int) -> int:
    """Validate paper ID is positive.

    Args:
        paper_id: Paper ID to validate

    Returns:
        Validated paper ID

    Raises:
        click.BadParameter: If paper ID is invalid
    """
    if paper_id <= 0:
        raise click.BadParameter(f"Paper ID must be positive, got: {paper_id}")
    return paper_id


def validate_top_k(top_k: int, min_val: int = 1, max_val: int = 100) -> int:
    """Validate top_k parameter is within reasonable bounds.

    Args:
        top_k: Number of results to return
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Validated top_k value

    Raises:
        click.BadParameter: If top_k is out of bounds
    """
    if top_k < min_val or top_k > max_val:
        raise click.BadParameter(
            f"top_k must be between {min_val} and {max_val}, got: {top_k}"
        )
    return top_k


def validate_sort_by(sort_by: str, allowed_values: Optional[List[str]] = None) -> str:
    """Validate sort_by parameter.

    Args:
        sort_by: Sort field name
        allowed_values: List of allowed values (default: indexed_at, year, title)

    Returns:
        Validated sort_by value

    Raises:
        click.BadParameter: If sort_by is invalid
    """
    if allowed_values is None:
        allowed_values = ["indexed_at", "year", "title"]

    if sort_by not in allowed_values:
        raise click.BadParameter(
            f"sort_by must be one of {allowed_values}, got: {sort_by}"
        )
    return sort_by


def validate_file_path(file_path: str, must_exist: bool = True) -> Path:
    """Validate and normalize file path.

    Args:
        file_path: Path to validate
        must_exist: Whether file must exist

    Returns:
        Validated Path object

    Raises:
        click.BadParameter: If path is invalid
    """
    try:
        path = Path(file_path).expanduser().resolve()

        if must_exist and not path.exists():
            raise click.BadParameter(f"File does not exist: {file_path}")

        return path
    except (ValueError, OSError) as e:
        raise click.BadParameter(f"Invalid file path: {file_path} ({e})")


def validate_query_string(
    query: str, min_length: int = 1, max_length: int = 500
) -> str:
    """Validate search query string.

    Args:
        query: Search query
        min_length: Minimum query length
        max_length: Maximum query length

    Returns:
        Validated query string

    Raises:
        click.BadParameter: If query is invalid
    """
    query = query.strip()

    if len(query) < min_length:
        raise click.BadParameter(f"Query must be at least {min_length} characters")

    if len(query) > max_length:
        raise click.BadParameter(f"Query must be at most {max_length} characters")

    return query


def validate_limit_offset(limit: int, offset: int) -> tuple[int, int]:
    """Validate pagination parameters.

    Args:
        limit: Maximum number of results
        offset: Number of results to skip

    Returns:
        Tuple of (validated_limit, validated_offset)

    Raises:
        click.BadParameter: If parameters are invalid
    """
    if limit < 1:
        raise click.BadParameter(f"limit must be at least 1, got: {limit}")

    if limit > 10000:
        raise click.BadParameter(f"limit must be at most 10000, got: {limit}")

    if offset < 0:
        raise click.BadParameter(f"offset cannot be negative, got: {offset}")

    return limit, offset
