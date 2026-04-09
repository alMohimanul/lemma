# lemma Usage Guide

Complete guide to using lemma for managing your research papers.

---

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install lemma in development mode
pip install -e .
```

### 2. Configure API Keys (Optional)

For LLM features (`lemma ask`), set up at least one provider:

```bash
# Copy example config
cp .env.example .env

# Edit .env and add your API keys
# GROQ_API_KEY=your_key_here
# GEMINI_API_KEY=your_key_here
# OPENROUTER_API_KEY=your_key_here
```

### 3. Basic Workflow

```bash
# Step 1: Scan your papers folder
lemma scan ~/Papers

# Step 2: Generate embeddings for semantic search
lemma embed

# Step 3: Ask questions across all papers
lemma ask "What are the main approaches to visual question answering?"

# Optional: Organize files with smart renaming
lemma organize --dry-run  # Preview
lemma organize            # Apply
```

---

## Command Reference

### `lemma scan <directory>`

Scan a directory for PDF files and extract metadata.

**Options:**
- `--recursive/--no-recursive`: Scan subdirectories (default: `true`)
- `--db PATH`: Database path (default: `~/.lemma/lemma.db`)

**Examples:**

```bash
# Scan current directory
lemma scan .

# Scan specific folder recursively
lemma scan ~/Documents/Research

# Scan without recursion
lemma scan ~/Papers --no-recursive

# Use custom database
lemma scan ~/Papers --db ~/custom.db
```

**What it does:**
- Finds all PDF files in the directory
- Computes SHA256 hash for deduplication
- Extracts metadata (title, authors, year, DOI, arXiv ID) using regex + PDF metadata
- Stores everything in SQLite database
- Skips duplicates automatically

**Output:**
```
✓ Scanning for PDFs...
Processing 50 PDFs... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

╭─────── Scan Results ───────╮
│ Total PDFs found: 50       │
│ New papers added: 47       │
│ Duplicates skipped: 3      │
╰────────────────────────────╯
```

---

### `lemma list`

List indexed papers in a table.

**Options:**
- `--limit N`: Max papers to show (default: `50`)
- `--offset N`: Skip first N papers (default: `0`)
- `--sort-by FIELD`: Sort by `indexed_at`, `year`, or `title` (default: `indexed_at`)
- `--db PATH`: Database path

**Examples:**

```bash
# List 50 most recent papers
lemma list

# List papers sorted by year
lemma list --sort-by year

# Show next 50 papers
lemma list --offset 50

# Show 100 papers sorted by title
lemma list --limit 100 --sort-by title
```

**Output:**
```
                              Papers
┌────┬──────────────────────────────┬────────────────┬──────┬──────────┐
│ ID │ Title                        │ Authors        │ Year │ Status   │
├────┼──────────────────────────────┼────────────────┼──────┼──────────┤
│  1 │ Attention Is All You Need    │ Vaswani et al. │ 2017 │ pending  │
│  2 │ BERT: Pre-training of Deep…  │ Devlin et al.  │ 2019 │ pending  │
└────┴──────────────────────────────┴────────────────┴──────┴──────────┘
```

---

### `lemma search <query>`

Search papers by keyword (searches title, authors, abstract).

**Options:**
- `--db PATH`: Database path

**Examples:**

```bash
# Search for papers about transformers
lemma search "transformer"

# Search by author
lemma search "Yoshua Bengio"

# Search by topic
lemma search "visual question answering"
```

**Output:**
```
Search results for: "transformer"

                              Papers
┌────┬──────────────────────────────┬────────────────┬──────┬──────────┐
│ ID │ Title                        │ Authors        │ Year │ Status   │
├────┼──────────────────────────────┼────────────────┼──────┼──────────┤
│  1 │ Attention Is All You Need    │ Vaswani et al. │ 2017 │ completed│
│  5 │ Vision Transformer (ViT)     │ Dosovitskiy…   │ 2021 │ completed│
└────┴──────────────────────────────┴────────────────┴──────┴──────────┘
```

---

### `lemma info <paper_id>`

Show detailed information about a specific paper.

**Options:**
- `--db PATH`: Database path

**Examples:**

```bash
# View details for paper ID 5
lemma info 5
```

**Output:**
```
╭─────────────── Paper Details ───────────────╮
│ ID: 5                                       │
│ Title: Vision Transformer (ViT)            │
│ Authors: Dosovitskiy, A., et al.           │
│ Year: 2021                                  │
│ arXiv: 2010.11929                           │
│ File: /Users/me/Papers/vit.pdf             │
│ Size: 2.34 MB                               │
│ Indexed: 2024-02-25 14:30:15               │
│                                             │
│ Abstract:                                   │
│ While the Transformer architecture has...  │
╰─────────────────────────────────────────────╯
```

---

### `lemma embed`

Generate embeddings for semantic search.

**Options:**
- `--force`: Re-embed papers that already have embeddings
- `--db PATH`: Database path
- `--index-path PATH`: FAISS index path (default: `~/.lemma/search.index`)

**Examples:**

```bash
# Generate embeddings for new papers
lemma embed

# Re-generate all embeddings
lemma embed --force

# Use custom paths
lemma embed --db ~/custom.db --index-path ~/custom.index
```

**What it does:**
- Extracts full text from each PDF
- Chunks text into 500-word segments with 50-word overlap
- Generates embedding vectors using all-MiniLM-L6-v2 (384D)
- Stores embeddings in SQLite + FAISS index
- Saves FAISS index to disk for persistence

**Output:**
```
ℹ Found 10 papers to embed
✓ Loaded embedding model: all-MiniLM-L6-v2 (384D)
ℹ Created new FAISS index

Embedding 10 papers... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

✓ Saved FAISS index to /Users/me/.lemma/search.index

ℹ Embedding complete: 10 succeeded, 0 failed
✓ You can now use 'lemma ask' to query your papers!
```

**Note:** First run will download the embedding model (~80MB). Subsequent runs are fast.

---

### `lemma ask <question>`

Ask a question across all papers using semantic search + LLM.

**Options:**
- `--top-k N`: Number of papers to retrieve (default: `5`)
- `--db PATH`: Database path
- `--index-path PATH`: FAISS index path

**Examples:**

```bash
# Ask a general question
lemma ask "What are the main approaches to visual question answering?"

# Retrieve more papers for context
lemma ask "How does attention mechanism work?" --top-k 10

# Ask about specific topics
lemma ask "What datasets are used for VQA evaluation?"
```

**What it does:**
1. Encodes your question into a 384D vector
2. Searches FAISS index for top-k most similar paper chunks
3. Retrieves paper metadata + text from database
4. Builds a prompt with context from relevant papers
5. Queries LLM (Groq → Gemini → OpenRouter) with caching
6. Displays answer with sources

**Output:**
```
ℹ Loaded search index with 250 vectors
ℹ Found 5 relevant papers

╭────────── Question ──────────╮
│ What are the main approaches │
│ to visual question answering?│
╰──────────────────────────────╯

╭────────── Answer ────────────╮
│ The main approaches to VQA   │
│ include:                      │
│                               │
│ 1. Attention-based models... │
│ 2. Graph neural networks...  │
│ 3. Pre-trained vision-lang...│
╰──────────────────────────────╯

Sources:
  • [12] VQA: Visual Question Answering (2015)
  • [18] Bottom-Up and Top-Down Attention (2018)
  • [25] LXMERT: Learning Cross-Modality (2019)

ℹ Provider: groq | Model: llama-3.3-70b-versatile | Tokens: 487
```

**Graceful Degradation:**
- If no LLM API keys are set, displays relevant papers instead of generating an answer
- Caches responses for 30 days (instant retrieval for repeated questions)
- Tries multiple providers automatically if one fails

---

### `lemma organize`

Rename PDF files based on metadata with pattern-based naming.

**Options:**
- `--dry-run`: Preview changes without applying
- `--pattern TEXT`: Filename pattern (default: `{year}_{first_author}_{short_title}.pdf`)
- `--db PATH`: Database path

**Pattern Placeholders:**
- `{year}` - Publication year
- `{first_author}` - First author's last name
- `{authors}` - All authors (shortened)
- `{title}` - Full title (sanitized)
- `{short_title}` - Title truncated to 50 chars
- `{doi}` - DOI (sanitized)
- `{arxiv_id}` - arXiv ID

**Examples:**

```bash
# Preview renames (safe, no changes)
lemma organize --dry-run

# Apply default pattern
lemma organize

# Custom pattern with arXiv ID
lemma organize --pattern "{arxiv_id}_{short_title}.pdf"

# Year + authors pattern
lemma organize --pattern "{year}_{authors}.pdf"
```

**Output (dry run):**
```
                 [DRY RUN] File Organization Preview
┌────┬─────────────────────┬──────────────────┬────────────────────────┐
│ ID │ Title               │ Current Name     │ New Name               │
├────┼─────────────────────┼──────────────────┼────────────────────────┤
│  1 │ Attention Is All... │ paper1.pdf       │ 2017_Vaswani_Attent... │
│  2 │ BERT: Pre-traini... │ downloaded.pdf   │ 2019_Devlin_BERT_Pr... │
└────┴─────────────────────┴──────────────────┴────────────────────────┘

ℹ Total files to rename: 2

⚠ This is a dry run. Use without --dry-run to apply changes.
```

**Output (actual rename):**
```
Proceed with renaming? [y/N]: y

Renaming 2 files... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

ℹ Renaming complete: 2 succeeded, 0 failed
✓ Files have been organized!
ℹ Use the database logs to rollback if needed
```

**Safety Features:**
- Dry run by default shows preview
- Confirmation prompt before applying changes
- Logs all operations to database for rollback
- Handles duplicate names automatically
- Updates database with new file paths

---

## Workflows

### Workflow 1: Initial Setup

```bash
# 1. Scan your papers
lemma scan ~/Documents/Research

# 2. Review what was found
lemma list

# 3. Generate embeddings
lemma embed

# 4. Start asking questions!
lemma ask "What are the key findings?"
```

### Workflow 2: Adding New Papers

```bash
# Scan for new papers only (skips duplicates)
lemma scan ~/Downloads

# Embed the new papers
lemma embed

# Query including new papers
lemma ask "Latest research on transformers?"
```

### Workflow 3: Organizing a Messy Collection

```bash
# 1. Scan everything
lemma scan ~/Papers

# 2. Preview organization
lemma organize --dry-run

# 3. Apply rename
lemma organize

# 4. Check results
lemma list --sort-by title
```

### Workflow 4: Research Across Multiple Topics

```bash
# Find papers on topic A
lemma search "attention mechanisms"

# Find papers on topic B
lemma search "visual grounding"

# Ask comparative question
lemma ask "How do attention mechanisms differ between NLP and vision tasks?"
```

---

## Tips & Best Practices

### Metadata Extraction
- **Better source PDFs = better metadata**. Papers from arXiv typically have good metadata.
- If metadata is missing, check the PDF manually and update using `lemma info <id>`
- DOI and arXiv ID are extracted reliably; author names are heuristic-based

### Embeddings
- Run `lemma embed` after adding papers (doesn't re-embed existing ones)
- Use `--force` only when changing models or fixing errors
- Embeddings take ~2-5 seconds per paper on average hardware
- All processing is local; no data sent to cloud

### LLM Usage
- Set up multiple providers for reliability (Groq is fastest, Gemini is most reliable)
- Responses are cached for 30 days - repeated questions are instant
- Free tiers support ~200 papers easily (under limits)
- If you hit rate limits, the system automatically tries the next provider

### File Organization
- **Always** use `--dry-run` first to preview changes
- Keep original filenames if papers are referenced elsewhere
- Use `{arxiv_id}` for preprints, `{doi}` for published papers
- Shorter patterns are better (filesystem limits are 255 chars)

### Performance
- Database is optimized for up to 10,000 papers
- FAISS index loads in <1 second for most collections
- Semantic search is instant (<100ms for top-5 results)
- LLM responses take 2-5 seconds (unless cached)

---

## Troubleshooting

### "No embeddings found. Please run 'lemma embed' first."
**Solution:** Run `lemma embed` to generate embeddings.

### "No LLM providers available"
**Solution:** Set up at least one API key in `.env`:
```bash
export GROQ_API_KEY="your_key_here"
# or
export GEMINI_API_KEY="your_key_here"
```

### "Insufficient text extracted from PDF"
**Solution:** PDF may be scanned or corrupted. Try:
- Re-downloading the paper
- Using a different PDF source
- Manually adding metadata via database

### Embeddings taking too long
**Solution:**
- First run downloads the model (~80MB) - this is one-time
- Subsequent runs are much faster
- Consider processing in batches if you have 1000+ papers

### Search returns irrelevant papers
**Solution:**
- Try more specific questions
- Increase `--top-k` to see more results
- Check if embeddings were generated (`lemma list` shows status)

---

## Advanced Usage

### Custom Database Location

```bash
# Use a project-specific database
lemma scan ~/ProjectA --db ~/ProjectA/papers.db
lemma embed --db ~/ProjectA/papers.db
lemma ask "questions?" --db ~/ProjectA/papers.db
```

### Batch Processing

```bash
# Scan multiple directories
for dir in ~/Papers/*; do
  lemma scan "$dir"
done

# Generate embeddings for all
lemma embed
```

### Export to Bibliography

*(Coming in Week 2)*

```bash
# Export to BibTeX
lemma export --format bibtex > references.bib

# Export specific papers
lemma export --ids 1,5,10 --format bibtex
```

---

## Architecture Overview

```
User Question
     ↓
[Embedding Encoder] → 384D vector
     ↓
[FAISS Index] → Top-K similar paper chunks
     ↓
[SQLite DB] → Retrieve paper metadata + text
     ↓
[Prompt Builder] → Context + question
     ↓
[LLM Router] → Groq → Gemini → OpenRouter (with cache)
     ↓
Answer + Sources
```

**Data Flow:**
1. PDFs → Metadata extraction (local regex + PDF parsing)
2. Text → Chunking → Embeddings (local sentence-transformers)
3. Embeddings → FAISS index (in-memory, persisted to disk)
4. Metadata → SQLite (persistent storage)
5. Questions → Semantic search → LLM synthesis (cloud API)

**Storage:**
- `~/.lemma/lemma.db` - SQLite database (metadata, embeddings backup, cache)
- `~/.lemma/search.index.faiss` - FAISS index file
- `~/.lemma/search.index.pkl` - Paper ID mapping

---

## Next Steps

### Week 2 Features (Coming Soon)

- **Citation extraction**: Parse references from papers
- **ArXiv integration**: Auto-fetch metadata from arXiv API
- **Paper comparison**: `lemma compare <id1> <id2>`
- **Collection summaries**: `lemma summarize --all`
- **Export to BibTeX/RIS**: For citation managers

### Contribute

Found a bug or want a feature? Open an issue at:
https://github.com/yourusername/lemma/issues
