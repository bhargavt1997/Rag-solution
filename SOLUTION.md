# 🔑 RAG & LLM Assessment — Solution & Answer Key

> **⚠️ FOR EVALUATORS ONLY** — Do not share this document with candidates.

---

## Architecture Overview

```
.txt Files → Document Loader → Chunker (Recursive, 500 chars) → OpenAI Embeddings → ChromaDB
                                                                         ↕
User Query → Embedding → Cosine Similarity Search → Prompt Builder → GPT-4o-mini → Response + Sources
```

| Component | Implementation | Rationale |
|-----------|---------------|-----------|
| **Document Loader** | `DocumentLoader` class | Loads all `.txt` files with metadata |
| **Chunker** | `RecursiveChunker` (500 chars, 100 overlap) | Respects paragraph structure |
| **Embedding** | `text-embedding-3-small` (1536 dims) | Cost-effective, high quality |
| **Vector DB** | ChromaDB (in-memory, cosine) | Zero-config, Python-native |
| **LLM** | `gpt-4o-mini` (temp=0.1) | Low temperature for factual accuracy |
| **Relevance Filter** | Distance threshold 0.8 | Filters irrelevant chunks |

---

## Approach — What a Good Answer Looks Like

### 1. Document Ingestion
- Read all `.txt` files from a directory.
- Store each document's content along with metadata (filename, path).

### 2. Chunking Strategy
- **Method**: Recursive character text splitting.
- **Chunk size**: ~500 characters (small enough for precision, large enough for context).
- **Overlap**: ~100 characters (ensures no information is lost at boundaries).
- **Why not sentence-based?** Policy documents have numbered sections; recursive splitting respects paragraph structure better.

### 3. Embedding
- **Model**: `text-embedding-3-small` (OpenAI) — cost-effective, 1536 dimensions, high quality.
- **Alternative**: `all-MiniLM-L6-v2` (HuggingFace/sentence-transformers) — free, runs locally.

### 4. Vector Database
- **ChromaDB** — in-memory, no setup, Python-native, perfect for prototyping.
- **Alternative**: FAISS (Facebook) for larger scale.

### 5. Retrieval
- Embed the user query with the same model.
- Perform cosine similarity search, retrieve top-3 chunks.
- Include source metadata with each chunk.
- **Relevance threshold**: Chunks with cosine distance > 0.8 are filtered out.

### 6. Answer Generation
- Build a prompt with: system instructions + retrieved context + user query.
- Instruct the LLM to answer **only** from context and cite source documents.
- If no relevant context found, respond with "I don't have enough information."

---

## Full Working Code

The complete implementation is in `rag_pipeline.py` — a single, well-documented file with 6 modular classes:

| Class | Responsibility |
|-------|---------------|
| `DocumentChunk` | Data model for a document chunk with metadata |
| `RetrievedContext` | Data model for a retrieved chunk with relevance score |
| `RAGResponse` | Data model for the final pipeline response |
| `DocumentLoader` | Loads `.txt` files from a directory |
| `RecursiveChunker` | Splits documents using recursive strategy |
| `VectorStore` | Manages embeddings + ChromaDB storage and search |
| `AnswerGenerator` | Builds prompts and calls the LLM |
| `RAGPipeline` | Orchestrates the entire pipeline |

### Web Chat Interface

`app.py` provides a Flask-based web chat interface (`static/index.html`) with:
- Real-time chat with typing indicators
- Expandable context panels showing retrieved chunks
- Relevance score visualization
- Source citation badges

---

## Expected Outputs

### Query 1: *"How many days of paternity leave can I take?"*

> **Expected Answer**: Male employees are entitled to **10 working days** of paid paternity leave, which must be availed within **6 months** of the child's birth.
>
> **Source**: `leave_policy.txt`

---

### Query 2: *"Is LASIK surgery covered under the insurance plan?"*

> **Expected Answer**: No, **LASIK surgery is not covered** under the group health insurance plan. However, vision correction (spectacles/lenses) is covered up to INR 5,000 per year.
>
> **Source**: `insurance_policy.txt`

---

### Query 3: *"What happens if I moonlight for a competitor?"*

> **Expected Answer**: Moonlighting or freelancing for competitors is **strictly prohibited** without written approval from the CTO and HR. Violations may lead to disciplinary action following the 3-step process (Written Warning → Suspension → Termination), or immediate termination for severe violations.
>
> **Sources**: `code_of_conduct.txt`

---

### Query 4: *"Can I carry forward my unused sick leave?"*

> **Expected Answer**: No, **sick leave cannot be carried forward** or encashed. However, unused **annual leave** can be carried forward up to a maximum of 10 days to the next year.
>
> **Source**: `leave_policy.txt`

---

### Query 5: *"What is the sum insured under the health insurance?"*

> **Expected Answer**: The base sum insured is **INR 5,00,000** per family per year. Employees can opt for a top-up plan of **INR 10,00,000 or INR 25,00,000** at a subsidized premium.
>
> **Source**: `insurance_policy.txt`

---

### Query 6: *"How do I report an ethics violation?"*

> **Expected Answer**: Employees can report violations through the **anonymous Ethics Hotline (ethics@healthbot.com)** or directly to the **Compliance Officer**. Retaliation against whistleblowers is strictly prohibited and is a terminable offense.
>
> **Source**: `code_of_conduct.txt`

---

## Expected Write-up Answers

### Chunking Strategy
> Recursive character text splitting with chunk_size=500 and overlap=100. This works well for policy documents because:
> - 500 chars keeps each chunk focused on one policy point
> - 100-char overlap prevents information loss at boundaries
> - Recursive strategy respects paragraph/section boundaries before doing hard splits

### Embedding Model Choice
> `text-embedding-3-small` — OpenAI's latest small embedding model. Good balance of quality and cost. 1536 dimensions. Alternative: `all-MiniLM-L6-v2` for a free, local option.

### Vector DB Choice
> ChromaDB — in-memory, zero-config, Python-native. Perfect for prototyping. For production, would consider Pinecone (managed) or Weaviate (self-hosted).

### No-Context Handling
> The system prompt explicitly instructs the LLM to say "I don't have enough information" if the context doesn't contain the answer. Additionally, a relevance threshold on the cosine distance score filters out low-quality retrievals before they reach the LLM.

### Improvements with More Time
> - Add re-ranking (ColBERT, Cohere Rerank) after initial retrieval
> - Implement hybrid search (BM25 + dense embeddings)
> - Add a conversational memory for follow-up questions
> - Build a Streamlit/Gradio UI
> - Add document update pipeline (incremental ingestion)
> - Implement evaluation metrics (faithfulness, relevance via RAGAS)

---

## Grading Notes for Evaluators

| What to Check | Red Flags 🚩 | Green Flags ✅ |
|----------------|-------------|----------------|
| **Chunking** | No overlap; arbitrarily large chunks (>2000 chars) | Thoughtful size + overlap; respects document structure |
| **Embedding** | Using deprecated `text-embedding-ada-002` only; no explanation | Justified choice; considered alternatives |
| **Prompt** | No grounding instruction; LLM can freely hallucinate | Explicit "only use context"; citation required |
| **Temperature** | `temperature=1.0` or default | Low temperature (0-0.3) for factual accuracy |
| **Modularity** | Everything in one function/file | Separate classes/modules for each stage |
| **Edge Cases** | Crashes on empty dir; no handling of irrelevant queries | Graceful errors; "I don't know" responses |
| **Source Citation** | No source tracking | Metadata preserved through pipeline; cited in output |

---

> **💡 Bonus Points** for candidates who:
> - Add a relevance score threshold to filter low-quality chunks
> - Implement a simple evaluation harness
> - Use `metadata` filtering (e.g., query only `leave_policy.txt`)
> - Add a `--interactive` CLI mode
> - Discuss trade-offs between chunk sizes
> - Build a web chat interface
