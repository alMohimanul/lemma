<div align="center">
  <img width="1529" height="704" alt="mavyn" src="https://github.com/user-attachments/assets/a77adb6c-ed25-4499-a1d4-3607187e4aa0" />
  <h3>Chat with your research papers — locally, privately, for free.</h3>

  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

<!-- Record with: vhs demo.tape  or  terminalizer record demo -->
<!-- Then place the output at assets/demo.gif -->
<div align="center">
  <img src="assets/demo.gif" alt="MAVYN demo" width="800" />
</div>

---

## What it does

MAVYN is a terminal-based research assistant. Drop your PDFs into a folder, run `mavyn`, and start asking questions in plain English. Everything runs locally — no uploads, no subscriptions.

- **Q&A** — ask anything about one or many papers
- **Summarize** — get structured summaries by section
- **Compare** — side-by-side comparison of methodologies, findings, etc.
- **Literature review** — auto-generate a full review from your library
- **Semantic search** — local FAISS + BM25 hybrid retrieval
- **Similar papers** — find related work in your library (+ optional arXiv)

---

## Install

```bash
pip install -r requirements.txt
```

Set a free [Groq API key](https://console.groq.com/) (takes 30 seconds):

```bash
echo "GROQ_API_KEY=your_key_here" >> ~/.MAVYN/.env
```

---

## Quick start

```
$ mavyn

╭─────────────────────────────────────────────╮
│  Welcome to MAVYN 🤖                        │
│                                             │
│  Commands: /sync ~/Papers  │  /list  │  /help │
│  Example:  tell me about paper 5           │
╰─────────────────────────────────────────────╯

MAVYN> /sync ~/Papers
# Scans, renames, and embeds all PDFs — one command

MAVYN> /list
# Shows your papers with IDs

MAVYN> tell me about the methodology in paper 3
MAVYN> compare papers 1 and 5
MAVYN> find papers similar to paper 2
MAVYN> /review                        # generate a literature review
```

---

## Commands

| Command | What it does |
|---|---|
| `/sync [dir]` | Scan, rename, and embed all PDFs in a directory |
| `/sync --watch` | Watch for new papers automatically |
| `/list` | List all indexed papers with IDs |
| `/review` | Generate a full literature review |
| `/model` | Show LLM status and rate-limit cooldowns |
| `/help` | Full command reference |
| `/exit` | Quit |

Natural language is always on — just type your question without any prefix.

---

## Privacy

All papers and embeddings stay on your machine. The only outbound calls are:
- **Groq / Gemini / OpenRouter** — only for LLM answers, only if you set an API key
- **arXiv** — only if you use `--arxiv` (sends a keyword query, never your PDFs)

No account required to use local features.

---

## Requirements

- Python 3.10+
- ~500 MB disk space (embedding model)
- API key optional — search and indexing work without one

---

## License

MIT — see [LICENSE](LICENSE)
