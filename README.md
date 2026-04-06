# SARAS — System for Autonomous Research and Academic Synthesis

## Architecture Overview

```
User Input (Streamlit)
      │
      ├── Research Query ──────────────────────────────────────────┐
      │                                                            │
      └── PDF Upload → PyMuPDF → Chunks → ChromaDB                │
                                              │                    │
                                              ▼                    ▼
                                    ┌─────────────────────────────────┐
                                    │       LangGraph State Machine   │
                                    │                                 │
                                    │  Planner → Web Scripter ────►  │
                                    │               ▲     │          │
                                    │    (retry)    │     ▼          │
                                    │           Analyst              │
                                    │               │ (approved)     │
                                    │               ▼                │
                                    │          Synthesizer           │
                                    │               │                │
                                    │               ▼                │
                                    │      Source Aggregator         │
                                    └─────────────────────────────────┘
                                              │
                                              ▼
                             ┌────────────────────────────┐
                             │     Streamlit Dashboard     │
                             │  • Full report (Markdown)   │
                             │  • Source list + scores     │
                             │  • Analyst notes            │
                             │  • Download .md             │
                             └────────────────────────────┘
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or higher |
| pip | Latest |
| Google AI Studio API key | [Get free key](https://aistudio.google.com/) |
| Tavily API key | [Get free key](https://tavily.com/) |

---

## Setup Instructions

### 1. Clone / extract the project

```bash
cd saras
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
# Copy the example file
cp .env.example .env

# Open .env and fill in your keys:
# GOOGLE_API_KEY=your_actual_google_key
# TAVILY_API_KEY=your_actual_tavily_key
```

**Where to get your keys:**
- **Google API key**: https://aistudio.google.com/ → Get API key (free tier available)
- **Tavily API key**: https://tavily.com/ → Sign up → Dashboard → API key (free tier: 1000 searches/month)

### 5. Run the application

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## How to Use

### Basic research (web only)
1. Open http://localhost:8501
2. Type your research query in the text box
3. Click **Run Research**
4. Watch the pipeline progress in real time
5. View the report, sources, and analyst notes in the tabs
6. Download the report as `.md`

### Research with your own documents (hybrid RAG)
1. In the sidebar, upload one or more PDF files
2. Click **Ingest Documents** — wait for the success message
3. Enter your query and click **Run Research**
4. The system will cross-reference your documents with live web data
5. Uploaded document sources will appear with 📎 in the Sources tab

### Clear session
Click **Clear Session & Documents** in the sidebar to wipe ChromaDB and start fresh.

---

## Project File Structure

```
saras/
├── app.py                  # Streamlit UI dashboard
├── config.py               # Settings loader (reads .env)
├── graph.py                # LangGraph state machine definition
├── state.py                # Shared ResearchState schema
├── requirements.txt
├── .env.example            # Copy to .env and fill in keys
│
├── agents/
│   ├── __init__.py
│   ├── planner.py          # Decomposes query into sub-tasks
│   ├── web_scripter.py     # Tavily search + ChromaDB retrieval
│   ├── analyst.py          # Cross-reference & contradiction check
│   ├── synthesizer.py      # Writes the structured report
│   └── source_aggregator.py# Ranks & deduplicates sources
│
└── utils/
    ├── __init__.py
    ├── pdf_parser.py       # PyMuPDF + ChromaDB ingestion & retrieval
    └── credibility.py      # Domain-based source trust scoring
```

---

## Agent Roles (for viva reference)

| Agent | Role | Thinking budget |
|---|---|---|
| **Planner** | Decomposes query into 3–5 sub-tasks | Off (speed) |
| **Web Scripter** | Runs Tavily searches + ChromaDB retrieval | N/A (no LLM) |
| **Analyst** | Cross-references, finds contradictions, approves/retries | 2048 tokens (deep) |
| **Synthesizer** | Writes structured report with inline citations | 1024 tokens |
| **Source Aggregator** | Deduplicates, scores, and formats bibliography | N/A (rule-based) |

---

## Key Technical Decisions

- **LangGraph** over plain LangChain: native state management and conditional edges enable the Analyst→Scripter feedback loop without manual while-loops
- **Gemini 2.5 Flash**: per-agent thinking budgets (Analyst gets 2048, Synthesizer 1024, Planner 0) optimise cost vs. reasoning depth
- **ChromaDB**: persistent vector store with cosine similarity — survives app restarts within a session
- **PyMuPDF (fitz)**: fastest pure-Python PDF text extractor; handles scanned docs better than pypdf
- **Credibility scoring**: rule-based (no LLM call needed) — TLD + domain pattern matching → 0.0–1.0 score

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Missing required environment variables` | Copy `.env.example` to `.env` and fill in both keys |
| `TavilyClient error` | Check your Tavily API key and internet connection |
| `chromadb.errors` | Delete the `./chroma_db` folder and restart |
| `thinking_budget` parameter error | Update `langchain-google-genai`: `pip install -U langchain-google-genai` |

---

## Dependencies

| Package | Purpose |
|---|---|
| `langgraph` | Multi-agent graph orchestration |
| `langchain-google-genai` | Gemini 2.5 Flash via LangChain |
| `chromadb` | Local vector database for session memory |
| `pymupdf` | PDF text extraction |
| `tavily-python` | LLM-native web search |
| `streamlit` | Web UI dashboard |
| `python-dotenv` | Environment variable loading |
| `pydantic` | Typed state schema |

## MIT License

Copyright (c) 2026 Mishi Jain

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
