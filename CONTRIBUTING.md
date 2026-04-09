# Contributing to Lemma

Thank you for your interest in contributing to Lemma! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, Lemma version)
- **Relevant logs or error messages**

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- **Clear use case** - Why is this feature useful?
- **Proposed solution** - How should it work?
- **Alternatives considered** - What other approaches did you think about?

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** with clear, logical commits
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Ensure tests pass** (`pytest tests/`)
6. **Follow code style** (`ruff check src/`)
7. **Submit the PR** with a clear description

## 🛠️ Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/lemma.git
cd lemma

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_scanner.py::test_scan_directory
```

### Code Quality

```bash
# Lint code
ruff check src/

# Format code
ruff format src/

# Type checking
mypy src/ --ignore-missing-imports
```

## 📝 Code Style

- Follow [PEP 8](https://pep8.org/)
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Keep functions focused and single-purpose
- Prefer composition over inheritance

### Example

```python
def process_paper(paper_path: Path, extract_metadata: bool = True) -> Paper:
    """Process a PDF paper and extract metadata.

    Args:
        paper_path: Path to the PDF file
        extract_metadata: Whether to extract metadata (default: True)

    Returns:
        Paper object with extracted information

    Raises:
        FileNotFoundError: If paper_path doesn't exist
        ValueError: If PDF is corrupted or unreadable
    """
    if not paper_path.exists():
        raise FileNotFoundError(f"Paper not found: {paper_path}")

    # Implementation...
```

## 🏗️ Project Structure

```
lemma/
├── src/lemma/
│   ├── cli/          # Command-line interface
│   ├── core/         # Core functionality (scanner, extractor, organizer)
│   ├── db/           # Database models and repository
│   ├── embeddings/   # Embedding generation and search
│   └── llm/          # LLM provider integration
├── tests/            # Test suite
├── .github/          # GitHub Actions workflows
└── docs/             # Documentation (future)
```

## 🧪 Testing Guidelines

- Write tests for new features
- Maintain or improve code coverage
- Test edge cases and error handling
- Use fixtures for common setup
- Mock external API calls

### Test Structure

```python
def test_scan_directory_with_valid_pdfs(tmp_path):
    """Test scanning a directory containing valid PDF files."""
    # Arrange
    pdf_path = tmp_path / "test.pdf"
    create_test_pdf(pdf_path)

    # Act
    scanner = PDFScanner()
    results = scanner.scan_directory(tmp_path)

    # Assert
    assert len(results) == 1
    assert results[0].path == pdf_path
```

## 📦 Adding Dependencies

When adding new dependencies:

1. Add to `setup.py` in the appropriate section (`install_requires` or `extras_require`)
2. Document why the dependency is needed
3. Consider the impact on installation size and time
4. Prefer well-maintained, popular libraries

## 🔄 Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add citation extraction
fix: resolve duplicate detection bug
docs: update README with new commands
refactor: simplify embedding encoder
test: add tests for organizer
chore: update dependencies
```

## 🌿 Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Refactoring

## 📋 Pull Request Process

1. **Update the README** if you change functionality
2. **Add tests** for new features
3. **Update CHANGELOG.md** (if it exists)
4. **Ensure CI passes** - all tests and linters must pass
5. **Request review** from maintainers
6. **Address feedback** promptly and professionally

## 🎯 Areas for Contribution

### High Priority

- [ ] Additional metadata extractors (CrossRef, PubMed, etc.)
- [ ] More LLM providers (Anthropic, OpenAI, local models)
- [ ] Citation graph visualization
- [x] Paper comparison features (v2.0.0)
- [ ] Collection summaries
- [ ] Export to bibliography formats (BibTeX, EndNote)

### Good First Issues

- [ ] Add more file naming patterns
- [ ] Improve error messages
- [ ] Add shell completions
- [ ] Write additional tests
- [ ] Improve documentation

### Future Features

- [ ] Web UI
- [ ] Topic auto-classification
- [ ] Annotation support
- [ ] Enhanced comparison visualizations
- [ ] Collection/folder-based comparisons

## 🚫 What NOT to Include

- Large binary files
- API keys or secrets
- Personal data or papers
- Copyrighted content
- Breaking changes without discussion

## 📞 Getting Help

- **GitHub Discussions**: Ask questions, share ideas
- **GitHub Issues**: Report bugs, request features
- **Code Review**: Request feedback on your PR

## 🎖️ Recognition

Contributors will be:
- Listed in the README
- Credited in release notes
- Thanked in the community

## 📜 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Give constructive feedback
- Focus on what's best for the project

---

**Thank you for contributing to Lemma!** Your efforts help researchers worldwide manage their papers more effectively. 🚀📚
