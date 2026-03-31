"""
RAG Pipeline for HR Policy Q&A
================================
A modular Retrieval-Augmented Generation system that answers
employee questions using internal HR policy documents.

Architecture:
    .txt Files → Document Loader → Chunker → Embedding Model → ChromaDB
    User Query → Embedding → Similarity Search → Prompt Builder → LLM → Response

Usage:
    export OPENAI_API_KEY="sk-..."
    python rag_pipeline.py

    # Interactive mode:
    python rag_pipeline.py --interactive

    # Custom documents directory:
    python rag_pipeline.py --docs-dir /path/to/docs
"""

import os
import sys
import glob
import argparse
from dataclasses import dataclass
from openai import OpenAI
import chromadb


# ─────────────────────────────────────────────
# 1. DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    text: str
    source_file: str
    chunk_index: int
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            base = os.path.basename(self.source_file).replace(".", "_")
            self.chunk_id = f"{base}_chunk_{self.chunk_index}"


@dataclass
class RetrievedContext:
    """Represents a retrieved chunk with its relevance score."""
    text: str
    source_file: str
    distance: float


@dataclass
class RAGResponse:
    """The final response from the RAG pipeline."""
    answer: str
    sources: list[str]
    contexts: list[RetrievedContext]


# ─────────────────────────────────────────────
# 2. DOCUMENT LOADER
# ─────────────────────────────────────────────

class DocumentLoader:
    """Loads .txt documents from a directory."""

    @staticmethod
    def load_documents(directory: str) -> list[dict]:
        """
        Load all .txt files from the given directory.

        Args:
            directory: Path to the directory containing .txt files.

        Returns:
            List of dicts with 'content' and 'source' keys.

        Raises:
            FileNotFoundError: If no .txt files are found in the directory.
        """
        documents = []
        txt_files = glob.glob(os.path.join(directory, "*.txt"))

        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found in '{directory}'. "
                f"Please ensure policy documents are placed in this directory."
            )

        for filepath in sorted(txt_files):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                print(f"  ⚠️  Skipped empty file: {os.path.basename(filepath)}")
                continue
            documents.append({
                "content": content,
                "source": os.path.basename(filepath),
            })
            print(f"  ✅ Loaded: {os.path.basename(filepath)} "
                  f"({len(content)} chars)")

        print(f"\n📂 Loaded {len(documents)} document(s)\n")
        return documents


# ─────────────────────────────────────────────
# 3. CHUNKER
# ─────────────────────────────────────────────

class RecursiveChunker:
    """
    Splits documents into overlapping chunks using a recursive
    strategy that tries to split on paragraph > newline > sentence
    > word boundaries, in that order.

    Design decisions:
    - chunk_size=500: Small enough for retrieval precision, large
      enough to capture a complete policy point.
    - chunk_overlap=100: Prevents information loss at chunk boundaries.
    - Recursive splitting respects paragraph/section structure in
      policy documents better than naive fixed-size splitting.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " "]

    def chunk_documents(
        self, documents: list[dict]
    ) -> list[DocumentChunk]:
        """Split all documents into chunks."""
        all_chunks = []
        for doc in documents:
            chunks = self._split_text(doc["content"])
            for i, chunk_text in enumerate(chunks):
                all_chunks.append(DocumentChunk(
                    text=chunk_text.strip(),
                    source_file=doc["source"],
                    chunk_index=i,
                ))
        print(f"🔪 Created {len(all_chunks)} chunks from "
              f"{len(documents)} documents\n")
        return all_chunks

    def _split_text(self, text: str) -> list[str]:
        """Recursively split text into chunks."""
        return self._recursive_split(text, self.separators)

    def _recursive_split(
        self, text: str, separators: list[str]
    ) -> list[str]:
        """
        Core recursive splitting logic.
        
        Tries each separator in order of preference (paragraph break
        first, then newline, then sentence, then word). If the text
        fits within chunk_size, return it as-is. Otherwise, split
        using the best available separator and merge small parts.
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Try each separator in order of preference
        separator = separators[0] if separators else ""
        remaining_separators = separators[1:] if len(separators) > 1 else []

        if separator and separator in text:
            parts = text.split(separator)
        else:
            # Fall back to next separator
            if remaining_separators:
                return self._recursive_split(text, remaining_separators)
            # Last resort: hard split by character count
            chunks = []
            step = max(1, self.chunk_size - self.chunk_overlap)
            for i in range(0, len(text), step):
                chunks.append(text[i : i + self.chunk_size])
            return chunks

        # Merge small parts into chunks respecting chunk_size
        chunks = []
        current = ""
        for part in parts:
            candidate = (
                current + separator + part if current else part
            )
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part

        if current:
            chunks.append(current)

        # Add overlap between consecutive chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1][-self.chunk_overlap :]
                overlapped.append(prev_tail + separator + chunks[i])
            chunks = overlapped

        return chunks


# ─────────────────────────────────────────────
# 4. VECTOR STORE (ChromaDB + OpenAI Embeddings)
# ─────────────────────────────────────────────

class VectorStore:
    """
    Manages document embeddings using OpenAI's embedding model
    and ChromaDB for storage and similarity search.

    Design decisions:
    - text-embedding-3-small: Cost-effective, 1536 dimensions, good quality.
    - ChromaDB: In-memory, zero-config, Python-native — ideal for prototyping.
    - Cosine similarity: Industry standard for text similarity.
    """

    RELEVANCE_THRESHOLD = 0.8  # Chunks with distance > this are considered irrelevant

    def __init__(
        self,
        openai_client: OpenAI,
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = "hr_policies",
    ):
        self.client = openai_client
        self.embedding_model = embedding_model

        # Initialize ChromaDB (in-memory for simplicity)
        self.chroma_client = chromadb.Client()

        # Delete existing collection if it exists (for re-runs)
        try:
            self.chroma_client.delete_collection(collection_name)
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts using OpenAI API."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Embed and store document chunks in ChromaDB."""
        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [{"source": c.source_file} for c in chunks]

        # Batch embed (OpenAI API supports up to ~2048 inputs per call)
        embeddings = self._get_embeddings(texts)

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"💾 Stored {len(chunks)} chunks in ChromaDB\n")

    def query(
        self, query_text: str, top_k: int = 3
    ) -> list[RetrievedContext]:
        """
        Retrieve the top-K most relevant chunks for a query.
        
        Filters out chunks whose cosine distance exceeds the
        RELEVANCE_THRESHOLD to avoid returning irrelevant results.
        """
        query_embedding = self._get_embeddings([query_text])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        contexts = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # Filter out low-relevance chunks
            if distance > self.RELEVANCE_THRESHOLD:
                continue
            contexts.append(RetrievedContext(
                text=results["documents"][0][i],
                source_file=results["metadatas"][0][i]["source"],
                distance=distance,
            ))
        return contexts


# ─────────────────────────────────────────────
# 5. ANSWER GENERATOR (LLM)
# ─────────────────────────────────────────────

class AnswerGenerator:
    """
    Generates grounded answers using an LLM with retrieved context.
    
    The system prompt enforces strict grounding — the model must
    only use the provided context and must cite sources. Temperature
    is set low (0.1) for factual consistency.
    """

    SYSTEM_PROMPT = """You are an HR Policy Assistant for HealthBot Inc.

RULES:
1. Answer the employee's question ONLY using the provided context.
2. If the context does not contain enough information to answer,
   say: "I don't have enough information in the policy documents
   to answer this question."
3. Always cite which document(s) your answer comes from at the end,
   formatted as: **Source(s):** `filename.txt`
4. Be concise and direct.
5. Do NOT use any knowledge outside the provided context.
6. If the question is ambiguous, answer based on the most relevant
   interpretation and note any assumptions.
"""

    def __init__(
        self,
        openai_client: OpenAI,
        model: str = "gpt-4o-mini",
    ):
        self.client = openai_client
        self.model = model

    def generate(
        self, query: str, contexts: list[RetrievedContext]
    ) -> RAGResponse:
        """Generate an answer grounded in the retrieved contexts."""

        # Handle case where no relevant context was found
        if not contexts:
            return RAGResponse(
                answer=(
                    "I don't have enough information in the policy "
                    "documents to answer this question."
                ),
                sources=[],
                contexts=[],
            )

        # Build context string with source attribution
        context_parts = []
        sources = set()
        for i, ctx in enumerate(contexts, 1):
            context_parts.append(
                f"[Source: {ctx.source_file}]\n{ctx.text}"
            )
            sources.add(ctx.source_file)

        context_str = "\n\n---\n\n".join(context_parts)

        # Build user message
        user_message = f"""CONTEXT:
{context_str}

QUESTION:
{query}

Provide a clear, concise answer based ONLY on the context above.
Cite the source document(s) at the end of your answer."""

        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,  # Low temp for factual accuracy
            max_tokens=500,
        )

        answer = response.choices[0].message.content

        return RAGResponse(
            answer=answer,
            sources=sorted(sources),
            contexts=contexts,
        )


# ─────────────────────────────────────────────
# 6. RAG PIPELINE (Orchestrator)
# ─────────────────────────────────────────────

class RAGPipeline:
    """
    Main orchestrator that ties together all components
    of the RAG pipeline: Loading → Chunking → Embedding →
    Retrieval → Generation.
    """

    def __init__(self, openai_api_key: str, docs_directory: str):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.docs_directory = docs_directory

        # Initialize components
        self.loader = DocumentLoader()
        self.chunker = RecursiveChunker(
            chunk_size=500, chunk_overlap=100
        )
        self.vector_store = VectorStore(self.openai_client)
        self.generator = AnswerGenerator(self.openai_client)

        self._is_ingested = False

    def ingest(self) -> None:
        """Load, chunk, embed, and store all documents."""
        print("=" * 55)
        print("📥  INGESTION PHASE")
        print("=" * 55)

        # Step 1: Load documents
        documents = self.loader.load_documents(self.docs_directory)

        # Step 2: Chunk documents
        chunks = self.chunker.chunk_documents(documents)

        # Step 3: Embed and store
        self.vector_store.add_chunks(chunks)

        self._is_ingested = True
        print("✅ Ingestion complete!\n")

    def query(self, question: str, top_k: int = 3) -> RAGResponse:
        """Process a user question through the RAG pipeline."""
        if not self._is_ingested:
            raise RuntimeError(
                "Documents not ingested yet. Call ingest() first."
            )

        print(f"\n❓ Query: {question}")
        print("-" * 55)

        # Step 1: Retrieve relevant chunks
        contexts = self.vector_store.query(question, top_k=top_k)

        if contexts:
            print(f"🔍 Retrieved {len(contexts)} relevant chunk(s):")
            for i, ctx in enumerate(contexts, 1):
                preview = ctx.text[:80].replace("\n", " ")
                print(f"   {i}. [{ctx.source_file}] "
                      f"(distance: {ctx.distance:.4f})")
                print(f"      \"{preview}...\"")
        else:
            print("🔍 No relevant chunks found (all below relevance threshold)")

        # Step 2: Generate answer
        response = self.generator.generate(question, contexts)

        print(f"\n💬 Answer:\n{response.answer}")
        if response.sources:
            print(f"\n📄 Sources: {', '.join(response.sources)}")
        print("=" * 55)

        return response


# ─────────────────────────────────────────────
# 7. MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_sample_queries(pipeline: RAGPipeline) -> None:
    """Run the pre-defined sample test queries."""
    test_queries = [
        "How many days of paternity leave can I take?",
        "Is LASIK surgery covered under the insurance plan?",
        "What happens if I moonlight for a competitor?",
        "Can I carry forward my unused sick leave?",
        "What is the sum insured under the health insurance plan?",
        "How do I report an ethics violation?",
    ]

    print("\n" + "=" * 55)
    print("🧪  RUNNING SAMPLE QUERIES")
    print("=" * 55)

    for query in test_queries:
        pipeline.query(query)
        print()


def run_interactive(pipeline: RAGPipeline) -> None:
    """Run an interactive Q&A session."""
    print("\n" + "=" * 55)
    print("💬  INTERACTIVE MODE")
    print("    Type your question and press Enter.")
    print("    Type 'quit' or 'exit' to stop.")
    print("=" * 55)

    while True:
        try:
            question = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            break

        pipeline.query(question)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="RAG-based Q&A for HR Policy Documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_pipeline.py                          # Run sample queries
  python rag_pipeline.py --interactive            # Interactive Q&A mode
  python rag_pipeline.py --docs-dir ./my_docs     # Custom docs directory
        """,
    )
    parser.add_argument(
        "--docs-dir",
        default="./policy_documents",
        help="Directory containing .txt policy documents (default: ./policy_documents)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (ask questions one at a time)",
    )
    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable is not set.")
        print("   Please set it:  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    # Initialize and ingest
    pipeline = RAGPipeline(
        openai_api_key=api_key,
        docs_directory=args.docs_dir,
    )
    pipeline.ingest()

    # Run queries
    if args.interactive:
        run_interactive(pipeline)
    else:
        run_sample_queries(pipeline)


if __name__ == "__main__":
    main()
