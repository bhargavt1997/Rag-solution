"""
Web Chat Interface for RAG HR Policy Q&A
=========================================
A Flask web app that provides a beautiful chat interface
for the RAG pipeline with live context display.

Usage:
    export OPENAI_API_KEY="sk-..."
    python app.py
"""

import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from rag_pipeline import RAGPipeline

app = Flask(__name__, static_folder="static")

# Global pipeline instance (initialized once)
pipeline = None


def init_pipeline():
    """Initialize the RAG pipeline on startup."""
    global pipeline
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    docs_dir = os.path.join(os.path.dirname(__file__), "policy_documents")
    pipeline = RAGPipeline(openai_api_key=api_key, docs_directory=docs_dir)
    pipeline.ingest()


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/query", methods=["POST"])
def query():
    """Process a user query and return the RAG response."""
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        response = pipeline.query(question)
        return jsonify({
            "answer": response.answer,
            "sources": response.sources,
            "contexts": [
                {
                    "text": ctx.text,
                    "source": ctx.source_file,
                    "distance": round(ctx.distance, 4),
                    "relevance": round((1 - ctx.distance) * 100, 1),
                }
                for ctx in response.contexts
            ],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    init_pipeline()
    print("\n🌐 Chat interface ready at: http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
