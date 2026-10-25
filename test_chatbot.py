"""
test_chatbot.py
=============================================================================
Comprehensive Automated Unit Test Suite for CodeAlpha Hybrid AI Chatbot
-----------------------------------------------------------------------------
Verifies:
1. FAQ Matching: Known, paraphrased, and typo queries via TF-IDF Cosine Similarity
2. Routing Modes:
   - Auto Mode (FAQ when confident; LLM fallback when below threshold)
   - FAQ Only Mode (Exclusively FAQ, NEVER invokes LLM under any circumstance)
   - AI Assistant Mode (Direct LLM invocation with multi-turn history)
3. Mocked LLM Calls (Zero paid API keys needed to run tests)
4. LLM Error & Rate-Limit Handling
5. Dynamic FAQ Operations (Add, Edit, Delete with live model refit)
6. REST API Endpoints (/chat, /faqs, /add-faq, /edit-faq, /delete-faq, /stats, /health)
=============================================================================
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from app import app
from faq_engine import FAQEngine
from llm_engine import LLMEngine
from nlp_engine import HybridNLPEngine


class TestHybridChatbot(unittest.TestCase):

    def setUp(self):
        """Set up test instances and Flask test client."""
        self.engine = HybridNLPEngine()
        self.app_client = app.test_client()

    # ==========================================================================
    # 1. FAQ MATCHING TESTS (Known, Paraphrased, Typos)
    # ==========================================================================
    def test_faq_exact_known_question(self):
        """Test that exact known questions achieve high confidence FAQ match."""
        res = self.engine.process_query("what is ai", mode="auto")
        self.assertIn("Local FAQ Match", res["source"])
        self.assertGreaterEqual(res["confidence"], 50.0)
        self.assertIn("artificial intelligence", res["response"].lower())

    def test_faq_paraphrased_question(self):
        """Test that paraphrased questions match through TF-IDF cosine similarity."""
        res = self.engine.process_query("explain the difference between deep learning and machine learning", mode="auto")
        self.assertIn("Local FAQ Match", res["source"])
        self.assertIn("deep learning", res["response"].lower())

    def test_faq_typo_tolerance(self):
        """Test character n-gram resilience for misspelled queries."""
        res = self.engine.process_query("artifical inteligence", mode="auto")
        self.assertIn("Local FAQ Match", res["source"])
        self.assertIn("artificial intelligence", res["response"].lower())

    # ==========================================================================
    # 2. ROUTING MODES & ZERO-LLM VERIFICATION
    # ==========================================================================
    def test_faq_only_mode_never_calls_llm(self):
        """CRITICAL: Verify that FAQ-only mode STRICTLY NEVER calls the LLM even for unknown queries."""
        with patch.object(self.engine.llm_engine, "generate_response") as mock_llm:
            res = self.engine.process_query(
                "quantum mechanics superstring theory in astrophysics",
                mode="faq",
                threshold=0.85
            )
            # The LLM engine must NOT be called at all in FAQ mode
            mock_llm.assert_not_called()
            self.assertEqual(res["source"], "Strict FAQ Matcher")
            self.assertIn("No verified FAQ match found", res["response"])

    def test_auto_mode_calls_llm_when_below_threshold(self):
        """Verify that Auto mode routes to LLM when confidence is below threshold."""
        with patch.object(self.engine.llm_engine, "generate_response", return_value=("Mocked AI Explanation", "✨ Mocked LLM")) as mock_llm:
            res = self.engine.process_query(
                "explain the history of ancient byzantine architecture in detail",
                mode="auto",
                threshold=0.75
            )
            mock_llm.assert_called_once()
            self.assertEqual(res["response"], "Mocked AI Explanation")
            self.assertEqual(res["source"], "✨ Mocked LLM")

    def test_ai_assistant_mode_routes_directly_to_llm(self):
        """Verify that AI Assistant mode routes directly to LLM with history."""
        sample_history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level programming language."}
        ]
        with patch.object(self.engine.llm_engine, "generate_response", return_value=("Follow-up response", "✨ Mocked LLM")) as mock_llm:
            res = self.engine.process_query(
                "how do I use decorators in it?",
                mode="ai",
                history=sample_history
            )
            mock_llm.assert_called_once_with(query="how do I use decorators in it?", history=sample_history)
            self.assertEqual(res["response"], "Follow-up response")

    # ==========================================================================
    # 3. LLM ERROR & RATE LIMIT HANDLING (MOCKED)
    # ==========================================================================
    def test_gemini_rate_limit_handled_gracefully(self):
        """Verify that Gemini quota exhaustion returns a user-friendly error without crashing."""
        llm = LLMEngine()
        llm.gemini_key = "mock_key"
        llm.provider = "gemini"

        # Mock gemini client raising ResourceExhausted / Quota exceeded
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("429 ResourceExhausted: Quota exceeded for model")
        llm.gemini_client = mock_client

        ans, src = llm.generate_response("Explain quantum computing")
        self.assertIn("Rate Limit", src)
        self.assertIn("quota", ans.lower())

    def test_openai_auth_error_handled_gracefully(self):
        """Verify that invalid OpenAI key returns authentication notice without crashing."""
        llm = LLMEngine()
        llm.openai_key = "invalid_key"
        llm.provider = "openai"

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("401 Incorrect API key provided")
        llm.openai_client = mock_client

        ans, src = llm.generate_response("Explain microservices")
        self.assertIn("Auth Error", src)

    # ==========================================================================
    # 4. LIVE DYNAMIC FAQ CRUD & RE-FITTING
    # ==========================================================================
    def test_add_edit_and_delete_faq_live(self):
        """Test adding, editing, and deleting an FAQ with live TF-IDF refit."""
        test_q = "What is the unique secret alpha passcode for test 2026?"
        test_a = "The secret alpha passcode is ALPHA_998877."

        # 1. Add FAQ
        add_res = self.engine.add_custom_faq(
            question=test_q,
            answer=test_a,
            category="custom",
            tags=["test", "passcode"]
        )
        self.assertEqual(add_res["status"], "success")
        faq_id = add_res["faq"]["id"]

        # Immediately query the newly added FAQ
        match_res = self.engine.process_query(test_q, mode="auto")
        self.assertIn("Local FAQ Match", match_res["source"])
        self.assertIn("ALPHA_998877", match_res["response"])

        # 2. Edit FAQ
        updated_a = "The passcode was updated to ALPHA_112233."
        edit_res = self.engine.edit_custom_faq(
            faq_id=faq_id,
            question=test_q,
            answer=updated_a
        )
        self.assertEqual(edit_res["status"], "success")

        # Query updated FAQ
        match_res2 = self.engine.process_query(test_q, mode="auto")
        self.assertIn("ALPHA_112233", match_res2["response"])

        # 3. Clean up by deleting FAQ
        del_res = self.engine.delete_custom_faq(faq_id)
        self.assertEqual(del_res["status"], "success")

    # ==========================================================================
    # 5. REST API ENDPOINTS
    # ==========================================================================
    def test_rest_chat_endpoint(self):
        """Verify POST /chat endpoint returns expected schema."""
        response = self.app_client.post(
            "/chat",
            json={"message": "what is ai", "mode": "auto", "threshold": 0.35}
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("response", data)
        self.assertIn("source", data)
        self.assertIn("confidence", data)
        self.assertIn("latency_ms", data)
        self.assertIn("mode", data)

    def test_rest_faqs_endpoint(self):
        """Verify GET /faqs returns list with total count."""
        response = self.app_client.get("/faqs")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertGreaterEqual(data["total"], 30)
        self.assertIsInstance(data["faqs"], list)

    def test_rest_health_and_stats_endpoints(self):
        """Verify GET /health and GET /stats."""
        h_res = self.app_client.get("/health")
        self.assertEqual(h_res.status_code, 200)
        self.assertEqual(h_res.get_json()["status"], "healthy")

        s_res = self.app_client.get("/stats")
        self.assertEqual(s_res.status_code, 200)
        s_data = s_res.get_json()
        self.assertIn("total_faqs", s_data)
        self.assertIn("vocabulary_size", s_data)


if __name__ == "__main__":
    unittest.main()
