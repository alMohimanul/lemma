# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-04-09

### Added - Major: Paper Comparison System 🎉
- **Multi-Paper Comparison**: Compare 2+ papers simultaneously with intelligent analysis
  - Section-specific comparisons (Introduction, Methods, Results, Discussion, etc.)
  - Whole-paper comparisons with incremental section-by-section processing
  - Maintains context across sections for coherent analysis
  - Memory-efficient in-memory processing (no temporary files)
- **Smart Question Parser**: Auto-detects comparison intent from natural language
  - Supports multiple ID formats: `[1]`, `paper 5`, `ID 8`, `id8`, `#3`
  - Extracts section names from queries
  - Fuzzy matching for section names ("Methodology" matches "Methods")
- **Intelligent Caching System**: Database-backed comparison cache
  - Hash-based lookup for instant repeated queries
  - Automatic invalidation when papers are re-embedded
  - Stores full comparison results with metadata
  - Shows cache timestamp in output
- **Rich Terminal Output**: Beautiful formatted comparison display
  - Side-by-side comparison tables
  - Section-by-section analysis
  - Comprehensive synthesis for whole-paper comparisons
  - Cache indicators and provider information
- **LLM Prompts for Comparison**: Optimized prompts for multi-paper analysis
  - Section comparison prompts with contextual awareness
  - Synthesis prompts for final comprehensive analysis
  - Token-optimized for free tier LLMs (Groq)

### Added - Additional Features
- Incremental embedding system with chunk-level change detection
- Advanced chunking strategies (structure-aware, sentence-based, hybrid)
- Embedding version tracking for smart updates
- LLM response caching to reduce API costs
- Note management system for literature review
- Formatted note generation for academic writing

### Changed
- Enhanced `ask` command to detect and handle comparison requests
- Updated command help text with comparison examples
- Improved documentation with comparison usage
- Better error messages for comparison failures

### Fixed
- Paper ID extraction now handles formats without spaces (`ID8`, `id42`)
- Improved regex patterns for more flexible ID matching

### Technical
- New database table: `PaperComparison` for caching comparison results
- New modules:
  - `src/lemma/llm/question_parser.py` - Comparison detection and parsing
  - `src/lemma/llm/comparison.py` - Multi-paper comparison engine
  - `src/lemma/llm/comparison_cache.py` - Cache management
- Extended `Repository` with comparison CRUD methods:
  - `add_paper_comparison()`, `get_paper_comparison()`
  - `get_section_embeddings()`, `get_all_section_names_for_papers()`
  - `delete_comparisons_involving_paper()`
- New prompt templates in `prompts.py`:
  - `build_multi_paper_section_comparison_prompt()`
  - `build_multi_paper_synthesis_prompt()`
- Added `print_comparison_results()` to output.py

## [1.0.0] - 2025-03-11

### Added
- Initial release of Lemma
- PDF scanning and hash-based deduplication
- Metadata extraction from PDFs
- Semantic search using sentence-transformers
- FAISS-based vector search index
- LLM integration (Groq Compound Mini, Google Gemini)
- Smart paper organization with customizable naming patterns
- Command-line interface with multiple commands:
  - `scan`: Scan directories for PDF papers
  - `embed`: Generate embeddings for semantic search
  - `search`: Semantic search across papers
  - `ask`: Interactive LLM-powered Q&A
  - `list`: List all papers in library
  - `organize`: Smart file organization
  - `export`: Export library metadata
- SQLite database for efficient paper management
- Incremental updates (only new papers are processed)
- Dry-run mode for safe testing
- Comprehensive test suite
- CI/CD pipeline with GitHub Actions
- Pre-commit hooks for code quality
- Full documentation and contribution guidelines

### Features
- **Local-First**: All data stored locally, full privacy control
- **Semantic Search**: Natural language queries across your paper library
- **LLM Reasoning**: Ask questions about papers with context-aware responses
- **Smart Organization**: Automatic file naming based on metadata
- **Incremental Processing**: Efficient updates when adding new papers
- **Multiple LLM Providers**: Support for Groq and Gemini with easy extensibility
- **Hash-Based Deduplication**: Prevents duplicate papers using SHA256
- **Batch Processing**: Efficient handling of large paper collections

## [0.1.0] - 2024-02-25

### Added
- Initial project structure
- Core scanning, extraction, and embedding functionality
- Basic CLI commands
- Database models and repository pattern
- LLM provider abstraction
- Documentation and setup files

---

## Version History

- `0.1.0` - Initial alpha release with core functionality
- Future versions will be documented here

## How to Read This Changelog

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes

For the complete list of changes in each release, see the [GitHub Releases](https://github.com/yourusername/lemma/releases) page.
