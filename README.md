# Getting Started with Learn-AI

This guide walks you through setting up and running the project from scratch. No prior experience with AI or Python is required.

---

## What This Project Does

- **RAG Chat** (`rag_chat.py`) — A command-line chatbot that answers questions using a knowledge base. It uses local AI (via Ollama) and can cite sources.
- **Prettify JSONL** (`prettify_jsonl.py`) — Converts a `.jsonl` file into readable, pretty-printed JSON.
- **Enrich RAG Dataset** (`enrich_rag_dataset.py`) — Turns a Q&A CSV file into a RAG-ready dataset (with optional AI enrichment and embeddings).

---

## What You Need to Install

### 1. Python (3.10 or newer)

The project runs on **Python 3**. If you’re not sure whether you have it:

- **Windows**: Open PowerShell and run:
  ```powershell
  python --version
  ```
  If you see something like `Python 3.10.x` or higher, you’re good. Otherwise, download and install from [python.org](https://www.python.org/downloads/). During setup, check **“Add Python to PATH”**.

- **macOS / Linux**: Run `python3 --version`. Install via your package manager or [python.org](https://www.python.org/downloads/) if needed.

### 2. Ollama (local AI)

The chatbot and enrichment scripts use **Ollama** to run AI models on your machine. You must install and run Ollama first.

1. **Download Ollama**  
   Go to [ollama.com](https://ollama.com) and download the installer for your operating system.

2. **Install and start Ollama**  
   Run the installer. On Windows/macOS, Ollama usually starts in the background. On Linux you may need to start the service (see [Ollama docs](https://github.com/ollama/ollama)).

3. **Pull the required models**  
   Open a terminal (PowerShell, Command Prompt, or your system terminal) and run:
   ```bash
   ollama pull gemma3:1b
   ollama pull embeddinggemma:latest
   ```
   This downloads the chat model and the embedding model. It can take a few minutes depending on your connection.

---

## Step-by-Step Setup

### 1. Open a terminal in the project folder

- **Windows**: In File Explorer, go to the `Learn-AI` folder, then in the address bar type `powershell` and press Enter.  
  Or open PowerShell and run: `cd "C:\Users\Devtac-Shawty\Documents\GitHub\Learn-AI"` (adjust the path if yours is different).

- **macOS / Linux**: Open Terminal and run:  
  `cd /path/to/Learn-AI`  
  (replace with your actual path.)

### 2. (Recommended) Create a virtual environment

This keeps the project’s dependencies separate from the rest of your system.

```bash
python -m venv venv
```

Then activate it:

- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**:
  ```cmd
  venv\Scripts\activate.bat
  ```
- **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```

You should see `(venv)` at the start of your command line.

### 3. Install Python dependencies

With the terminal still in the `Learn-AI` folder (and `venv` activated if you use it), run:

```bash
pip install -r requirements.txt
```

This installs:

- `ollama` — Python client to talk to Ollama
- `pandas` — Used by the CSV enrichment script

### 4. Make sure Ollama is running

Before running the chat or enrichment:

- **Windows / macOS**: If you installed Ollama from the website, it’s usually already running (check the system tray / menu bar).
- **Linux**: Start the Ollama service if needed.

You can test it by running:

```bash
ollama list
```

You should see `gemma3:1b` and `embeddinggemma` (or similar) in the list.

---

## How to Run the Programs

### RAG Chat (main chatbot)

1. You need a knowledge base file. The default is `devtac_rag_pretty.json` (or you can use a `.jsonl` file).
2. In the project folder, run:

   ```bash
   python rag_chat.py
   ```

   First run may take a bit longer while it builds and caches embeddings. After that, type your question at the `You:` prompt. Type `quit` or `exit` to stop.

   **Optional arguments:**

   | Option          | Description                                      |
   |-----------------|--------------------------------------------------|
   | `--kb FILE`     | Use a different knowledge base (e.g. `--kb my_data.jsonl`) |
   | `--top-k 5`     | Number of chunks to use for each answer (default 5) |
   | `--no-stream`   | Print the full answer at once instead of streaming |
   | `--regenerate`  | Rebuild embeddings and ignore cache              |
   | `--no-cache`    | Don’t use or save embedding cache                |

   **Examples:**

   ```bash
   python rag_chat.py --kb devtac_rag_pretty.json
   python rag_chat.py --kb devtac_rag.jsonl --top-k 5
   ```

---

### Prettify JSONL (convert JSONL → JSON)

Turns a `.jsonl` file into a single, indented JSON file.

**Default:** reads `devtac_rag.jsonl` and writes `devtac_rag_pretty.json`.

```bash
python prettify_jsonl.py
```

**Custom input/output:**

```bash
python prettify_jsonl.py --input myfile.jsonl --output myfile_pretty.json
```

---

### Enrich RAG Dataset (CSV → JSONL for RAG)

Takes a Q&A CSV and produces a RAG-ready JSONL file. With default settings it uses Ollama to enrich each row (titles, summaries, etc.).

**Prerequisites:** Ollama running, and `gemma3:1b` + `embeddinggemma:latest` pulled.

**Default:** reads `devtac_info_full_v2.csv`, writes `devtac_rag_enriched.jsonl`.

```bash
python enrich_rag_dataset.py
```

**Options:**

| Option       | Description                                      |
|-------------|--------------------------------------------------|
| `--input X` | Input CSV path                                   |
| `--output Y`| Output JSONL path                                |
| `--embed`   | Also compute and store embeddings (needs embedding model) |
| `--no-enrich`| Skip AI enrichment; faster, simpler output     |
| `--limit N` | Process only the first N rows (e.g. `--limit 10` to test) |

**Examples:**

```bash
python enrich_rag_dataset.py --input my_qa.csv --output my_rag.jsonl
python enrich_rag_dataset.py --no-enrich --output devtac_rag_fast.jsonl
python enrich_rag_dataset.py --input my_qa.csv --output my_rag.jsonl --embed --limit 5
```

---

## Quick Checklist

- [ ] Python 3.10+ installed (`python --version` or `python3 --version`)
- [ ] Ollama installed and running ([ollama.com](https://ollama.com))
- [ ] Models pulled: `ollama pull gemma3:1b` and `ollama pull embeddinggemma:latest`
- [ ] In the project folder: `pip install -r requirements.txt`
- [ ] For chat: a knowledge base file (e.g. `devtac_rag_pretty.json`) in the project folder
- [ ] Run: `python rag_chat.py` (or one of the other commands above)

---

## Troubleshooting

| Problem | What to try |
|--------|-------------|
| `python` not found | Use `python3` instead, or reinstall Python and check “Add to PATH”. |
| `pip install -r requirements.txt` fails | Run `python -m pip install -r requirements.txt`. Make sure you’re in the `Learn-AI` folder. |
| “Could not reach Ollama” / “Is Ollama running?” | Start the Ollama app (or service). Run `ollama list` to confirm it’s working. |
| “model not found” or “pull gemma3:1b” | Run `ollama pull gemma3:1b` and `ollama pull embeddinggemma:latest`. |
| “Knowledge base not found” | Use `--kb path/to/yourfile.json` or put `devtac_rag_pretty.json` in the project folder. |
| “Input file not found” (enrich script) | Use `--input path/to/your.csv` or place the default CSV in the project folder. |
| PowerShell won’t run `.\venv\Scripts\Activate.ps1` | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then try again. |

---

## Where to Go Next

- Edit `rag_chat.py` to change the system prompt or default model.
- Add your own `.json` / `.jsonl` knowledge base and run the chat with `--kb yourfile.json`.
- Use `enrich_rag_dataset.py` with your own Q&A CSV, then use the output as the knowledge base for `rag_chat.py`.

If you run into something not covered here, check that Ollama is running and that both models are pulled, then re-read the error message—it often suggests the exact command to fix the issue (e.g. `ollama pull <model>`).
