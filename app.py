"""
app.py
=============================================================================
Flask Server & REST API for CodeAlpha Hybrid AI FAQ Chatbot
-----------------------------------------------------------------------------
Endpoints:
- GET  /              : Web UI Chat Interface
- POST /chat          : Hybrid Query Processing (Auto, FAQ Only, AI Assistant)
- GET  /faqs          : Browse / Search Local Knowledge Base
- POST /add-faq       : Dynamically Add Custom FAQ & Live Vectorizer Re-fit
- POST /edit-faq      : Dynamically Edit Existing FAQ & Live Vectorizer Re-fit
- POST /delete-faq    : Dynamically Remove FAQ & Live Vectorizer Re-fit
- GET  /stats         : System, Engine, Model & Corpus Statistics
- GET  /health        : Health Check & Uptime Heartbeat
=============================================================================
"""

import logging
import os
import time
from flask import Flask, jsonify, render_template, request

import config
from nlp_engine import HybridNLPEngine

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FlaskServer")

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Record startup timestamp
START_TIME = time.time()

# Initialize Hybrid NLP Engine singleton
engine = HybridNLPEngine()


@app.route("/")
def index():
    """Renders the ChatGPT / Gemini-style Glassmorphic Chatbot Interface."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Primary chat processing endpoint.
    JSON Payload:
      - message (str): User prompt / query (Required)
      - mode (str): 'auto' (default), 'faq', or 'ai'
      - threshold (float): Similarity threshold (0.10 to 0.80, default 0.35)
      - category_filter (str): Optional category filter ('all', 'university_campus', etc.)
      - history (list): Recent conversation turns for multi-turn context
    """
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({
            "error": "Bad Request. JSON payload must contain a 'message' field."
        }), 400

    query = str(data.get("message", "")).strip()
    mode = str(data.get("mode", "auto")).lower()
    threshold = data.get("threshold", config.DEFAULT_SIMILARITY_THRESHOLD)
    category_filter = data.get("category_filter", "all")
    history = data.get("history", [])

    if not query:
        return jsonify({
            "response": "Please enter a question or select one of the suggestions!",
            "source": "System",
            "confidence": 0.0,
            "matched_faq": None,
            "latency_ms": 0.0,
            "mode": mode,
            "threshold": threshold,
            "candidates": []
        })

    try:
        threshold = float(threshold)
    except (ValueError, TypeError):
        threshold = config.DEFAULT_SIMILARITY_THRESHOLD

    # Process query through Hybrid Routing Architecture
    result = engine.process_query(
        query=query,
        mode=mode,
        threshold=threshold,
        category_filter=category_filter,
        history=history
    )

    return jsonify(result)


@app.route("/faqs", methods=["GET"])
def get_faqs():
    """Returns all FAQs or filtered by category/keyword."""
    category = request.args.get("category")
    search = request.args.get("search", "").strip()

    faqs = engine.faq_engine.get_all_faqs(category=category, search=search)

    return jsonify({
        "total": len(faqs),
        "category": category or "all",
        "faqs": faqs
    })


@app.route("/add-faq", methods=["POST"])
def add_faq():
    """
    Dynamically adds a custom FAQ to dataset and re-fits TF-IDF live!
    JSON Payload:
      - question (str)
      - answer (str)
      - category (str)
      - tags (list of str or comma-separated string)
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    question = str(data.get("question", "")).strip()
    answer = str(data.get("answer", "")).strip()
    category = str(data.get("category", "custom")).strip() or "custom"
    tags = data.get("tags", [])

    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    elif not isinstance(tags, list):
        tags = []

    if not question or not answer:
        return jsonify({"error": "Both 'question' and 'answer' fields are required."}), 400

    try:
        result = engine.add_custom_faq(
            question=question,
            answer=answer,
            category=category,
            tags=tags
        )
        return jsonify(result), 201
    except Exception as e:
        logger.error(f"Error adding FAQ: {e}")
        return jsonify({"error": f"Failed to add FAQ: {str(e)}"}), 500


@app.route("/edit-faq", methods=["POST"])
def edit_faq():
    """
    Dynamically updates an existing FAQ by ID and re-fits TF-IDF live!
    JSON Payload:
      - faq_id (str)
      - question (str)
      - answer (str)
      - category (str, optional)
      - tags (list of str or comma-separated string, optional)
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    faq_id = str(data.get("faq_id", "")).strip()
    question = str(data.get("question", "")).strip()
    answer = str(data.get("answer", "")).strip()
    category = str(data.get("category", "")).strip() or None
    tags = data.get("tags")

    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    if not faq_id or not question or not answer:
        return jsonify({"error": "Fields 'faq_id', 'question', and 'answer' are required."}), 400

    try:
        result = engine.edit_custom_faq(
            faq_id=faq_id,
            question=question,
            answer=answer,
            category=category,
            tags=tags
        )
        return jsonify(result), 200
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error editing FAQ: {e}")
        return jsonify({"error": f"Failed to edit FAQ: {str(e)}"}), 500


@app.route("/delete-faq", methods=["POST"])
def delete_faq():
    """
    Dynamically deletes an FAQ by ID and re-fits TF-IDF live!
    JSON Payload:
      - faq_id (str)
    """
    data = request.get_json(silent=True)
    if not data or "faq_id" not in data:
        return jsonify({"error": "Missing 'faq_id' parameter."}), 400

    faq_id = str(data.get("faq_id", "")).strip()
    try:
        result = engine.delete_custom_faq(faq_id=faq_id)
        return jsonify(result), 200
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error deleting FAQ: {e}")
        return jsonify({"error": f"Failed to delete FAQ: {str(e)}"}), 500


@app.route("/stats", methods=["GET"])
def stats():
    """Returns engine metadata, corpus statistics, and model status."""
    engine_stats = engine.get_stats()
    engine_stats["uptime_seconds"] = round(time.time() - START_TIME, 2)
    engine_stats["internship"] = "CodeAlpha AI Internship - Capstone Project"
    return jsonify(engine_stats)


@app.route("/health", methods=["GET"])
def health():
    """Liveness health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "CodeAlpha Hybrid AI FAQ Chatbot",
        "timestamp": time.time(),
        "uptime": round(time.time() - START_TIME, 2)
    }), 200


if __name__ == "__main__":
    port = config.PORT
    debug = config.FLASK_DEBUG
    logger.info(f"Starting CodeAlpha AI FAQ Chatbot on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
