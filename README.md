# 📚 Lemma - Local-First Research Paper Manager

> A privacy-first research paper manager with local semantic search and optional AI-powered insights.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🔒 **Privacy-First**: All papers stored locally, no cloud uploads
- 🚀 **Fast Semantic Search**: Local vector search across all papers
- 🤖 **AI Q&A (Optional)**: Ask questions using cloud LLMs
- 📊 **Paper Comparison**: Compare multiple papers side-by-side with intelligent caching
- 🔗 **Similar papers**: Ask for related work in your library; optional arXiv suggestions (`--arxiv` or `LEMMA_ARXIV_RELATED=1`)
- 📈 **Auto-Processing**: One command to scan, rename, and embed
- 🔄 **Incremental Updates**: 70-90% faster re-embedding
- 📂 **Smart Cleanup**: Automatically removes deleted papers from database
- 👀 **Watch Mode**: Auto-process new papers as they're added

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Set Your Papers Folder (One-Time Setup)

```bash
# Set default papers directory and process all papers
lemma sync ~/Papers --set-default

# That's it! Your papers are now:
# ✓ Scanned and indexed
# ✓ Renamed with metadata
# ✓ Embedded for semantic search
# ✓ Ready for questions
```

### 3. Add New Papers (Automatic)

**Option A: Manual Sync**
```bash
# Just drop PDFs into ~/Papers, then run:
lemma sync
```

**Option B: Auto-Sync (Watch Mode)**
```bash
# Start watching (leave running in terminal)
lemma sync --watch

# Now just drop PDFs into ~/Papers
# They're automatically processed in seconds!
```

### 4. Query Your Papers

```bash
# Ask questions
lemma ask "What are the main findings?"

# Compare papers (NEW!)
lemma ask "Compare papers 1 and 5"
lemma ask "Compare the methodology in papers [2], [7], and [12]"
```

## 📖 Common Workflows

### First-Time Setup
```bash
# 1. Set your papers folder and sync everything
lemma sync ~/Papers --set-default

# 2. Query your papers
lemma ask "What is the main contribution?"
```

### Daily Use
```bash
# Download new papers to ~/Papers, then:
lemma sync

# Or enable auto-processing:
lemma sync --watch  # Leave running
```

### Browse Your Library
```bash
lemma list                    # List all papers
lemma search "transformers"   # Search by keyword
lemma show 5                  # Show paper details
```

## 🔧 Advanced Usage

### Sync Options
```bash
lemma sync                    # Use default directory
lemma sync ~/Papers           # Specify directory
lemma sync --no-rename        # Skip automatic renaming
lemma sync --no-embed         # Skip embedding (faster)
lemma sync --watch            # Continuous monitoring
```

### Manual Control (If Needed)
```bash
lemma scan ~/Papers           # Just scan (no rename/embed)
lemma organize                # Rename existing files
lemma embed                   # Generate embeddings only
lemma embed-status            # Check embedding coverage
lemma verify --remove         # Clean up missing files
```

## 🤖 AI Q&A Setup (Optional)

Set up an API key to enable question answering:

```bash
# Option 1: Environment variable
export GROQ_API_KEY="your_key_here"

# Option 2: .env file
echo "GROQ_API_KEY=your_key_here" > ~/.lemma/.env

# Then ask questions
lemma ask "What are the main approaches discussed?"
```

**Get Free API Keys:**
- [Groq](https://console.groq.com/) - Fast and generous free tier (recommended)
- [Google Gemini](https://makersuite.google.com/) - Alternative option

## 📋 Key Commands

| Command | Description |
|---------|-------------|
| `lemma sync` | Auto-process papers (scan + rename + embed) |
| `lemma sync --watch` | Monitor folder and auto-process new papers |
| `lemma list` | List all indexed papers |
| `lemma ask <question>` | Ask questions across papers (requires API key) |
| `lemma ask "Compare papers X and Y"` | Compare multiple papers side-by-side |
| `lemma ask "Similar papers on …"` | Related papers in your library; add `--arxiv` for arXiv API suggestions |
| `lemma search <query>` | Search papers by keyword |
| `lemma show <id>` | Show paper details |
| `lemma embed-status` | Check embedding coverage |

## 🔒 Privacy

- **All papers stay on your machine** - never uploaded anywhere
- **Embeddings generated locally** - no external API calls
- **Cloud APIs only for Q&A** - and only if you configure them
- **Similar papers + arXiv (optional)** - If you run `lemma ask "…similar…" --arxiv` or set `LEMMA_ARXIV_RELATED=1`, Lemma sends a **keyword search query** (derived from your question or seed paper title/abstract) to **export.arxiv.org**. No PDFs are uploaded. Responses are cached locally in `lemma.db` for 24 hours to reduce repeat traffic.
- **Database stored locally** at `~/.lemma/lemma.db`

## 📦 Requirements

- Python 3.10 or higher
- ~500MB disk space for embedding models
- Internet connection only for optional AI Q&A and optional arXiv similar-paper search

## License

MIT License - see LICENSE file for details

## Support

- Report issues: [GitHub Issues](https://github.com/alMohimanul/lemma/issues)
- Questions: Open a discussion on GitHub
