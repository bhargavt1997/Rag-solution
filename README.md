# 🤖 RAG HR Policy Assistant

A **Retrieval-Augmented Generation (RAG)** system that answers employee questions about HR policies using grounded document retrieval — with both a CLI and a beautiful web chat interface.

Built as an assessment tool for evaluating junior engineers' proficiency in RAG and LLM development.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![Flask](https://img.shields.io/badge/Flask-Web_UI-black?logo=flask&logoColor=white)

---

## ✨ Features

- **Modular RAG Pipeline** — Clean separation of ingestion, chunking, embedding, retrieval, and generation
- **Web Chat Interface** — Beautiful dark-mode chat UI with live context display
- **Source Citations** — Every answer cites its source document(s)
- **Context Transparency** — Expandable panels showing retrieved chunks with relevance scores
- **Edge Case Handling** — Relevance threshold filtering + "I don't know" responses
- **CLI Support** — Run sample queries or interactive Q&A from the terminal

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PHASE                         │
│                                                             │
│  📄 .txt Files → DocumentLoader → RecursiveChunker          │
│                    (load & parse)   (500 chars, 100 overlap) │
│                                          │                  │
│                                          ▼                  │
│                    OpenAI Embeddings → ChromaDB              │
│                  (text-embedding-3-small) (cosine similarity)│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      QUERY PHASE                            │
│                                                             │
│  🧑 User Query → Embed → Similarity Search → Top-3 Chunks   │
│                                                  │          │
│                                                  ▼          │
│                              Prompt Builder (System + Context│
│                              + Query) → GPT-4o-mini          │
│                                            │                │
│                                            ▼                │
│                              📤 Answer + Source Citations     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
rag-hr-assistant/
├── README.md                  # This file
├── PROBLEM_STATEMENT.md       # Assessment problem statement (for candidates)
├── SOLUTION.md                # Solution & answer key (for evaluators)
├── requirements.txt           # Python dependencies
├── rag_pipeline.py            # Core RAG pipeline (CLI)
├── app.py                     # Flask web server
├── sample_output.txt          # Sample CLI output for all 6 queries
├── .gitignore
├── static/
│   └── index.html             # Web chat interface
└── policy_documents/          # HR policy documents
    ├── leave_policy.txt
    ├── insurance_policy.txt
    └── code_of_conduct.txt
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/rag-hr-assistant.git
cd rag-hr-assistant

python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

### 3. Run

#### Option A: Web Chat Interface (Recommended)

```bash
python app.py
```

Open **http://localhost:5050** in your browser.

#### Option B: CLI — Run All Sample Queries

```bash
python rag_pipeline.py
```

#### Option C: CLI — Interactive Mode

```bash
python rag_pipeline.py --interactive
```

---

## 💬 Web Chat Interface

The web interface provides a real-time chat experience with:

- 🗨️ **Chat bubbles** with typing animation
- 📄 **Source badges** on every answer
- 🔍 **"View retrieved chunks"** button — expands to show the exact text retrieved from the vector DB
- 📊 **Relevance score bars** showing match quality for each chunk
- 💡 **Suggestion chips** for quick sample queries

---

## 🧪 Sample Queries & Expected Answers

| # | Query | Expected Answer | Source |
|---|-------|----------------|--------|
| 1 | How many days of paternity leave can I take? | **10 working days** | `leave_policy.txt` |
| 2 | Is LASIK surgery covered? | **No, not covered** | `insurance_policy.txt` |
| 3 | What happens if I moonlight for a competitor? | **Strictly prohibited** without CTO/HR approval | `code_of_conduct.txt` |
| 4 | Can I carry forward unused sick leave? | **No**, cannot be carried forward or encashed | `leave_policy.txt` |
| 5 | What is the sum insured? | **INR 5,00,000** base + top-up options | `insurance_policy.txt` |
| 6 | How do I report an ethics violation? | **Ethics Hotline** or **Compliance Officer** | `code_of_conduct.txt` |

---

## 🛠️ Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| **Chunking** | Recursive, 500 chars, 100 overlap | Respects paragraph structure; prevents boundary info loss |
| **Embedding** | `text-embedding-3-small` | Cost-effective, 1536 dims, high quality |
| **Vector DB** | ChromaDB (in-memory) | Zero-config, Python-native, ideal for prototyping |
| **LLM** | `gpt-4o-mini`, temp=0.1 | Low temperature for factual consistency |
| **Relevance filter** | Cosine distance threshold 0.8 | Filters irrelevant chunks before LLM sees them |
| **Grounding** | System prompt + threshold | Two-layer defense against hallucinations |

---

## 📋 Assessment Materials

This repository serves as both a working solution and an assessment toolkit:

| Document | Audience | Purpose |
|----------|----------|---------|
| [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md) | Candidates | What to build (3-hour assessment) |
| [SOLUTION.md](SOLUTION.md) | Evaluators | Answer key, grading rubric, expected outputs |
| Source code | Both | Reference implementation |

---

## 🔮 Future Improvements

- **Re-ranking** — Add Cohere Rerank or ColBERT after initial retrieval
- **Hybrid Search** — Combine BM25 sparse + dense embeddings via reciprocal rank fusion
- **Conversational Memory** — Support multi-turn follow-up questions
- **Evaluation Framework** — Implement RAGAS metrics (faithfulness, answer relevance)
- **Incremental Ingestion** — Support document updates without full re-indexing
- **Authentication** — Add API key management for multi-user deployment

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
