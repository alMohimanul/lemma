# 📚 Lemma - Local-First Research Paper Manager

> A privacy-first research paper manager with local semantic search and optional AI-powered insights.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🔒 **Privacy-First**: All papers stored locally, no cloud uploads
- 🚀 **Fast Semantic Search**: Local vector search across all papers
- 🤖 **AI Q&A (Optional)**: Ask questions using cloud LLMs
- 📊 **Smart Organization**: Automatic metadata extraction and file naming
- 🔄 **Incremental Updates**: Only processes new papers
- 📂 **Duplicate Detection**: SHA256-based deduplication

## Installation

```bash
# Install via pip
pip install lemma-ai

# Or with pipx (recommended for CLI tools)
pipx install lemma-ai
```

## Quick Start

```bash
# Scan your papers directory
lemma scan ~/Documents/Papers

# List indexed papers
lemma list

# Search by keyword
lemma search "machine learning"

# Generate embeddings for semantic search (runs locally)
lemma embed

# Organize files with smart renaming
lemma organize --dry-run

# Verify database integrity
lemma verify
```

## Optional: AI Features

For AI-powered Q&A, set up an API key (completely optional):

```bash
# Set environment variable
export GROQ_API_KEY="your_key_here"

# Or create .env file
echo "GROQ_API_KEY=your_key_here" > .env

# Ask questions across your library
lemma ask "What are the main approaches to polyp segmentation?"
```

Get free API keys:
- [Groq](https://console.groq.com/) - Fast and generous free tier (recommended)
- [Google Gemini](https://makersuite.google.com/) - Alternative option

## Requirements

- Python 3.10 or higher
- macOS (Linux/Windows support coming soon)
- ~500MB disk space for embedding models

## Privacy

- **All papers stay on your machine** - never uploaded anywhere
- **Embeddings generated locally** - no external API calls
- **Cloud APIs only used for optional Q&A feature** - and only if you configure keys
- **Database stored locally** at `~/.lemma/lemma.db`

## Commands

| Command | Description |
|---------|-------------|
| `lemma scan <dir>` | Scan directory for PDFs |
| `lemma list` | List all indexed papers |
| `lemma search <query>` | Search papers by keyword |
| `lemma info <id>` | Show paper details |
| `lemma embed` | Generate embeddings (local) |
| `lemma ask <question>` | Ask questions (requires API key) |
| `lemma organize` | Rename files based on metadata |
| `lemma verify` | Check database integrity |

## License

MIT License - see LICENSE file for details

## Support

- Report issues: [GitHub Issues](https://github.com/alMohimanul/lemma/issues)
- Questions: Open a discussion on GitHub
